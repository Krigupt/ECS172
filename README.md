# Hidden-State Token Injection for LLM-Based Recommendation

## Overview

This project enhances LLM-based movie recommendation by injecting collaborative filtering signals directly into the transformer's intermediate representations. Rather than modifying the model's weights or replacing input text, we insert an **invisible layer** — a forward hook — at a target decoder layer that additively steers the model's hidden states using projected SASRec embeddings.

```
User History (text)                    SASRec Collaborative Embeddings (50-dim)
       │                                              │
       ▼                                              ▼
┌──────────────┐                            ┌───────────────────┐
│  Tokenizer   │                            │   Adapter (MLP)   │
│  + Embedding │                            │  50 → 1024 → 2048 │
└──────┬───────┘                            └─────────┬─────────┘
       │                                              │
       ▼                                              │
  ┌─────────┐                                         │
  │ Layer 0 │                                         │
  ├─────────┤                                         │
  │ Layer 1 │                                         │
  ├─────────┤                                         │
  │   ...   │                                         │
  ├─────────┤         ┌──────────────────┐            │
  │ Layer 8 │ ◄───────│  Forward Hook    │◄───────────┘
  │         │         │  h += α · z      │   z = adapter(e_movie)
  ├─────────┤         │  (invisible add) │
  │   ...   │         └──────────────────┘
  ├─────────┤
  │ Layer 15│
  └────┬────┘
       │
       ▼
  Letter logprobs (A-J) → Ranking
```

The model sees the full text prompt ("A. The Matrix (1999)...") and processes it normally through all layers. At layer 8, the hook fires and adds the adapter's projected vector to the hidden state at each candidate's position. The model then continues processing with this collaborative signal baked in, producing better-informed ranking predictions.

---

## The Adapter MLP

### Problem

SASRec produces 50-dimensional collaborative filtering embeddings that encode user viewing patterns. LLaMA-3.2-1B operates in a 2048-dimensional hidden space. We need a bridge between these two spaces.

### Architecture

The adapter is a lightweight MLP that projects SASRec embeddings into LLaMA's representation space:

```
Input: e ∈ ℝ⁵⁰  (SASRec item embedding)
                │
        ┌───────▼────────┐
        │ Linear(50→1024)│     Expand to intermediate dimension
        ├────────────────┤
        │     SiLU()     │     Non-linear activation (smooth ReLU)
        ├────────────────┤
        │  LayerNorm(1024)│    Stabilize activations
        ├────────────────┤
        │ Linear(1024→2048)│   Project to LLM hidden dimension
        └───────┬────────┘
                │
Output: z ∈ ℝ²⁰⁴⁸  (vector in LLaMA's representation space)
```

This is defined in `src/adapter.py` as `EmbeddingAdapter`:

```python
self.projector = nn.Sequential(
    nn.Linear(sasrec_dim, hidden_dim),    # 50 → 1024
    nn.SiLU(),                             # smooth non-linearity
    nn.LayerNorm(hidden_dim),              # stabilization
    nn.Linear(hidden_dim, llm_dim),        # 1024 → 2048
)
```

**Parameter count**: 50×1024 + 1024 + 1024 + 1024×2048 + 2048 = **2,153,472 trainable parameters** — less than 0.2% of LLaMA-1B's 1.24 billion parameters.

### Why this architecture?

The two-stage projection (50→1024→2048) with SiLU activation allows the adapter to learn non-linear mappings between the collaborative filtering space and the language model's hidden space. LayerNorm ensures the projected vectors have stable magnitudes that won't destabilize the transformer when added to hidden states. A direct linear projection (50→2048) would be too constrained to capture the complex relationship between collaborative and semantic spaces.

---

## Token Injection via Forward Hook ("Invisible Layer")

### What is a Forward Hook?

PyTorch allows registering callback functions on any module that execute during the forward pass. We use this mechanism to intercept the output of a specific decoder layer and modify its hidden states before they flow to the next layer. From the model's perspective, it's as if an invisible additional computation happens between layers.

### How the Injection Works

```python
# 1. Register hook on decoder layer 8
layer = llm.model.layers[8]
handle = layer.register_forward_hook(hook_fn)

# 2. Hook function fires after layer 8 computes its output
def hook_fn(module, input, output):
    hidden_states = output[0]  # (batch, seq_len, 2048)
    
    # Add adapter vector at each candidate's position
    for pos, z in zip(candidate_positions, adapter_vectors):
        hidden_states[:, pos, :] += alpha * z
    
    return (hidden_states,) + output[1:]

# 3. Normal forward pass — hook fires automatically
outputs = llm(input_ids=...)  # hook modifies layer 8's output in-flight
```

### Finding Candidate Positions

The text prompt contains candidates labeled A through J:

```
A user has rated the following movies:
- The Shawshank Redemption: 5/5
- Pulp Fiction: 4/5
...

From the list below, rank which movie this user would most likely enjoy:
A. The Matrix (1999) (Action|Sci-Fi)        ← position of "A" token
B. Titanic (1997) (Drama|Romance)            ← position of "B" token
...
J. Toy Story (1995) (Animation|Children's)   ← position of "J" token

Reply with just the letter (A-J) of the movie they would rate highest.
Answer:
```

We tokenize the prompt, find where each letter token (A, B, ..., J) sits in the token sequence, and inject the corresponding movie's adapter vector at that position. The injection equation at layer L is:

```
h_new[pos_A] = h[pos_A] + α · adapter(e_matrix)
h_new[pos_B] = h[pos_B] + α · adapter(e_titanic)
...
```

Where `α` (alpha) controls the injection strength.

### Why "Invisible"?

The hook does not add any parameters to the model, does not change the model's architecture, and does not appear in the model's `state_dict`. It is invisible to serialization, to the optimizer (the LLM stays frozen), and to any code that inspects the model structure. The only evidence of its existence is the modified hidden states flowing to the next layer.

---

## Training the Adapter

### Training Objective

The adapter is trained with **listwise ranking cross-entropy**: given 10 candidate movies (A-J), the loss encourages the model to assign the highest log-probability to the letter token corresponding to the true answer.

```
Loss = CrossEntropy(softmax(logits[A, B, ..., J]), target_letter)
```

### Training Loop

```
For each training example:
    1. Build text prompt with user history + 10 candidates (Mode A format)
    2. Tokenize and find candidate letter positions
    3. Project each candidate's SASRec embedding: z_i = adapter(e_i)  ← gradients flow here
    4. Install hook: h[pos_i] += α · z_i at target layer
    5. Forward pass through frozen LLaMA
    6. Compute ranking CE loss over letter logprobs at "Answer:" position
    7. Backward: loss → layers 16..L+1 → hook → z_i → adapter weights
    8. Update adapter parameters (Adam optimizer)
```

The gradient path is shorter than training at the input layer — it only flows back through layers above the hook point. This focuses the adapter on producing vectors that are meaningful in the target layer's representation space.

### Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | Adam | lr=1e-4 |
| Batch size | 4 | Gradient accumulation |
| Epochs | 8 | Best checkpoint saved by val HR@1 |
| Training examples | 4000 | Leave-one-out from train.csv |
| Validation split | 10% | 400 examples |
| Alpha (injection scale) | 1.0 | During training |
| Target layer | 8 | Middle of 16-layer LLaMA |
| Warm start | adapter.pt | Pre-trained reconstruction adapter |

### Autograd Through the Hook

For training, the hook must use **out-of-place operations** so PyTorch's autograd can backpropagate through it:

```python
# ✗ In-place (breaks autograd during training)
hidden_states[:, pos, :] += alpha * z

# ✓ Out-of-place (autograd-safe)
delta = torch.zeros_like(hidden_states)
delta = delta.scatter_add(0, positions, stacked_vectors)
hidden_states = hidden_states + delta  # new tensor, gradient flows
```

At evaluation time, in-place operations are fine since no backward pass occurs.

---

## File Structure

```
scripts/
├── train_hs_adapter.py          # Train injection adapter
├── eval_hidden_state.py          # Evaluate token injection
├── train_adapter.py              # Train base adapter
├── train_adapter_ranking.py      # Train ranking adapter
├── eval_ranking.py               # Evaluate Mode A/C
└── eval_ac.py                    # Compare all modes

src/
├── adapter.py                    # MLP projection network
├── data.py                       # Dataset loading utilities
├── inject_llm.py                 # LLM injection pipeline
├── metrics.py                    # Ranking evaluation metrics
├── ranking_data.py               # Ranking example builder
└── kv_adapter.py                 # KV prefix injection

checkpoints/
├── item_embeddings.pt            # SASRec learned embeddings
├── adapter.pt                    # Reconstruction-trained adapter
├── adapter_ranking.pt            # Ranking-trained adapter
├── adapter_hs_L8.pt              # Layer-8 injection adapter
├── adapter_hs_L10.pt             # Layer-10 injection adapter
├── projected_embeddings.pt       # Cached adapter projections
├── sasrec.pt                     # SASRec model weights
└── id_maps.json                  # Movie↔index mappings
```

---

## Quick Start

```bash
# Evaluate hidden-state injection (base model, layer 8)
python scripts/eval_hidden_state.py --target-layer 8 --alpha 1.0

# Evaluate with LoRA fine-tuned model
python scripts/eval_hidden_state.py \
    --model ./lora \
    --adapter checkpoints/adapter_hs_L8.pt \
    --target-layer 8 --alpha 1.0

# Train hidden-state adapter (Colab A100, ~15-30 min)
python scripts/train_hs_adapter.py \
    --model unsloth/Llama-3.2-1B-Instruct \
    --target-layer 8 --alpha 1.0 --epochs 8

# Sweep alpha values
for A in 0.5 1.0 2.0 5.0; do
    python scripts/eval_hidden_state.py --target-layer 8 --alpha $A
done
```

---

## Results

| Condition | HR@1 | HR@3 | HR@5 | MRR |
|-----------|------|------|------|-----|
| Random | 0.088 | 0.280 | 0.497 | 0.281 |
| Mode A (zero-shot, no injection) | 0.121 | 0.376 | 0.545 | 0.333 |
| HS-Add L8 α=1.0 (base model) | 0.270 | 0.543 | 0.710 | 0.464 |
| Mode C (soft token replacement) | 0.325 | 0.637 | 0.790 | 0.568 |
| Mode A (LoRA fine-tuned) | 0.454 | 0.730 | 0.853 | 0.665 |
| **HS-Add L10 α=1.0 (LoRA)** | **0.488** | **0.767** | **0.884** | **0.651** |

Hidden-state injection with the LoRA fine-tuned model at layer 10 achieves **HR@1=0.488**, outperforming all other conditions including Mode A LoRA (0.454) and Mode C soft token replacement (0.325). The approach keeps the full text prompt intact while steering the model's internal representations with collaborative filtering signals — no text replacement, no architecture changes, just an additive perturbation at the right layer.
