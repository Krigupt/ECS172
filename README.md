# LLM4Rec — Phase 1: SASRec

Implements **SASRec** (Kang & McAuley, 2018, [arXiv:1808.09781](https://arxiv.org/abs/1808.09781)) on your MovieLens CSVs.

**Paper-aligned settings:**
- Implicit feedback (`rating >= 4`)
- BCE loss + **1 random negative** per timestep
- All valid positions in each sequence (shifted next-item targets)
- `max_len=200`, `embed_dim=50`, `batch_size=128`, `dropout=0.2` (ML-1M)
- Pre-norm self-attention + **ReLU** FFN + residual connections
- Eval: rank **1 ground-truth + 100 negatives** → Hit@10, NDCG@10

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python scripts/train_sasrec.py --smoke
python scripts/train_sasrec.py --epochs 15
```

### Outputs (`checkpoints/`)

| File | Description |
|------|-------------|
| `sasrec.pt` | Model weights + config |
| `item_embeddings.pt` | `(n_items, embed_dim)` shared item matrix M |
| `id_maps.json` | `MovieID` ↔ index |

| Flag | Default | Paper (ML-1M) |
|------|---------|---------------|
| `--max-len` | 200 | n = 200 |
| `--embed-dim` | 50 | d ∈ {10..50} |
| `--batch-size` | 128 | 128 |
| `--min-rating` | 4.0 | implicit positives |
| `--num-negatives` | 100 | val ranking pool |

---

## Phase 2: Adapter (LLaMA-grounded alignment)

Maps SASRec vectors into Llama hidden space by teacher-forcing movie title
reconstruction through frozen LLaMA:

```
e (50) --[MLP]--> z (2048)  ← prepended as soft prefix token to LLaMA
[z | embed("Toy Story (Animation)")]  → frozen LLaMA forward
Loss: CrossEntropy over title tokens   (gradient → adapter only)
```

Train (after Phase 1 checkpoints exist):

```bash
python scripts/train_adapter.py --smoke
python scripts/train_adapter.py --epochs 10
```

| Output | Description |
|--------|-------------|
| `checkpoints/adapter.pt` | Projector weights (Linear→SiLU→LayerNorm→Linear) |

Defaults: `sasrec_dim=50`, `hidden_dim=1024`, `llm_dim=2048` (Llama-3.2-1B). Uses best val CE checkpoint.

---

## Phase 3: Your fine-tuned Llama + optional SASRec injection

Place your **MovieLens-tuned LoRA** in `llama31-1b-movielens-full-final/` (PEFT adapter on `unsloth/Llama-3.2-1B-Instruct`).  
Eval **does not train the LLM** — it loads your checkpoint and runs ranking on `test_ranking.csv`.

```bash
pip install -r requirements.txt

# Rank with your fine-tuned LLM only (prompt text, no SASRec vectors)
python scripts/eval_ranking.py --no-injection

# Rank with your LLM + SASRec embedding injection (adapter.pt or adapter_llm.pt)
python scripts/eval_ranking.py --use-adapter-llm

# Optional: train only the small embedding adapter (LLM stays frozen)
python scripts/finetune_ranking.py --epochs 3 --max-train 2000

python scripts/predict_ranking.py --user-id 5 --no-injection
```

| Script | Purpose |
|--------|---------|
| `finetune_ranking.py` | Train adapter with frozen Llama + CE on answer digit |
| `eval_ranking.py` | HR@10 / NDCG@10 on `test_ranking.csv` |
| `predict_ranking.py` | Single-user inference (`--user-id` or custom movie IDs) |

Use `--no-injection` for text-only baseline. `--n-history 10` matches CSV (change when you add more history columns).


## Quick Reference: Training Commands

Here are the commands to run the updated training pipeline:

**1. Train SASRec to generate the base item embeddings**
```bash
python scripts/train_sasrec.py --epochs 15
```

**2. Train the Adapter (Grounded to Base LLaMA)**
```bash
python scripts/train_adapter.py --epochs 10 --llm-model unsloth/Llama-3.2-1B-Instruct
```

**3. Train the Adapter (Grounded to Fine-Tuned LLaMA)**
```bash
python scripts/train_adapter.py --epochs 10 --llm-model ./llama31-1b-movielens-full-final
```

**4. Run Evaluation (Inference)**
```bash
# Compare against Text-Only Baseline
python scripts/eval_ranking.py --no-injection --model ./llama31-1b-movielens-full-final

# Run with the newly grounded adapter injection
python scripts/eval_ranking.py --use-adapter-llm --model ./llama31-1b-movielens-full-final
```
