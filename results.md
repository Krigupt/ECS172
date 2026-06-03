# ECS172 LLM4Rec — Experimental Results

## Setup

**Dataset:** MovieLens-1M — 604 test users, 10 candidates each (1 true positive + 9 sampled negatives).

**Pipeline (three phases):**
1. **SASRec** (Phase 1) — sequential recommendation model trained on user interaction history. Produces a 50-dim item embedding for each movie.
2. **Adapter** (Phase 2) — a two-layer MLP (`Linear(50→1024) → SiLU → LayerNorm → Linear(1024→2048)`) that projects SASRec embeddings into LLaMA's 2048-dim embedding space.
3. **LLM Ranker** (Phase 3) — frozen `unsloth/Llama-3.2-1B-Instruct` scores candidates by log-probability of letter tokens A–J at the answer position.

**Two adapter training strategies compared:**

| Adapter | Training objective | Output file |
|---|---|---|
| Reconstruction | Teacher-forced title CE through frozen LLaMA | `adapter.pt` |
| **Ranking (ours)** | **Listwise ranking CE over A–J letter logits** | `adapter_ranking.pt` |

The ranking adapter warm-starts from the reconstruction adapter and is validated each epoch by HR@1 on a held-out split. Best checkpoint saved by val HR@1.

**Two evaluation modes:**

- **Mode A (text baseline)** — candidates listed as lettered text titles (`A. Title... J. Title`). No adapter involved. Pure zero-shot LLM ranking over text.
- **Mode C (soft tokens)** — history stays as text; each candidate is replaced by its adapter-projected soft token (a single 2048-dim vector prepended as an embedding). The LLM ranks based on collaborative filtering signal rather than titles.

Scoring: single forward pass through frozen LLaMA, read `logits[:, -1, :]` at the `Answer:` position, softmax over the 10 letter token IDs → full ranked list.

---

## Main Results (604 test examples, 10 candidates)

| Condition | HR@1 | HR@3 | HR@5 | NDCG@1 | NDCG@3 | NDCG@5 |
|---|---|---|---|---|---|---|
| Random baseline | 0.088 | 0.280 | 0.497 | 0.088 | 0.196 | 0.284 |
| Mode A (text, zero-shot) | 0.126 | 0.374 | 0.556 | 0.126 | 0.263 | 0.338 |
| **Mode C — reconstruction adapter** | 0.135 | 0.290 | 0.475 | — | — | — |
| **Mode C — ranking adapter (ours)** | **0.320** | **0.631** | **0.791** | **0.320** | **0.499** | **0.565** |

**Key numbers:**
- Ranking adapter Mode C vs random: **+264% HR@1**
- Ranking adapter Mode C vs text baseline (Mode A): **+154% HR@1**
- Ranking adapter Mode C vs reconstruction adapter Mode C: **+137% HR@1**

The reconstruction adapter performed near-random (HR@1=0.135) and showed severe position-A bias (~95% of predictions). Switching to a ranking-aligned training objective fixed both.

---

## Position Bias Analysis

The letter probability distributions reveal why the reconstruction adapter failed and why the ranking adapter works.

### Mode A (text baseline)
```
Letter   AvgP    PredFreq%   TrueFreq%   HR@1_cond%
A        0.192    66.6%       11.1%        74.6%
J        0.170    26.0%       10.4%        28.6%
C        0.110     2.8%       10.9%         6.1%
G        0.107     3.0%        9.6%         5.2%
B–H      0.05–0.09  <1% each   ~10% each    0–2%
```

Mode A predicts A or J in **92.6%** of cases. The LLM reads titles but uses position as its primary signal (primacy/recency bias). HR@1=0.126 is driven almost entirely by the ~11% of examples where the true item happens to be at position A.

### Mode C — Reconstruction adapter (failed)
Predicted A in ~95% of cases (P(A) ≈ 0.45–0.57 uniformly across users). The LLM had no way to differentiate the anonymous soft tokens produced by a reconstruction-trained adapter, so it defaulted to picking the first option with high confidence.

### Mode C — Ranking adapter (ours)
```
Letter   AvgP    PredFreq%   TrueFreq%   HR@1_cond%
A        0.082     6.5%       11.1%        23.9%
B        0.095    10.6%        7.9%        37.5%
C        0.075     7.6%       10.9%        24.2%
D        0.139    15.4%       10.1%        45.9%
E        0.123    15.1%        8.8%        47.2%
F        0.108    12.3%       11.1%        23.9%
G        0.108    10.6%        9.6%        37.9%
H        0.104    10.4%       10.9%        30.3%
I        0.072     5.6%        9.1%        21.8%
J        0.095     6.0%       10.4%        31.7%
```

Max PredFreq drops from **~95% (reconstruction)** to **15.4% (ranking)**. PredFreq% is roughly uniform across all 10 letters (ideal = 10% each), confirming the adapter is producing genuinely distinguishable per-candidate representations.

Mild residual bias toward D and E (mid-list positions) remains. TrueFreq% for those letters is 10.1% and 8.8% — the over-representation in predictions is small and likely diminishes with more training epochs (loss was still declining at epoch 8).

---

## Sample Predictions

### Mode C — ranking adapter
```
user   5: pred=C  true=F  [A:0.075 B:0.058 C:0.231 D:0.204 E:0.052 F:0.204 G:0.075 H:0.028 I:0.015 J:0.058]
user  17: pred=A  true=F  [A:0.772 B:0.018 C:0.030 D:0.011 E:0.018 F:0.056 G:0.021 H:0.018 I:0.030 J:0.026]
user  35: pred=D  true=J  [A:0.024 B:0.080 C:0.038 D:0.246 E:0.043 F:0.149 G:0.217 H:0.090 I:0.033 J:0.080]
user  65: pred=D  true=C  [A:0.058 B:0.051 C:0.180 D:0.336 E:0.035 F:0.031 G:0.123 H:0.085 I:0.015 J:0.085]
user  99: pred=C  true=C  [A:0.022 B:0.022 C:0.744 D:0.012 E:0.033 F:0.014 G:0.018 H:0.020 I:0.054 J:0.061]
```

User 17's P(A)=0.772 is an outlier — likely caused by an OOV candidate at position A (zero vector), which the LLM interprets as a strong signal. This accounts for some of the remaining failures in Mode C.

User 99 (P(C)=0.744, correct) and user 35 (probability distributed across D/G/F) show the two regimes: confident correct prediction when the adapter produces a well-grounded soft token, and reasonable uncertainty spread when the ranking is genuinely ambiguous.

---

## Interpretation

**Why Mode C beats Mode A:** SASRec embeddings encode user-interaction patterns across the full MovieLens training set. Two movies with similar titles but very different viewer demographics will have SASRec embeddings that reflect those differences — information that title text alone does not contain. The ranking adapter learns to project these collaborative signals into LLaMA's embedding space in a way that is directly useful for ranking, giving Mode C access to a richer signal than Mode A's text.

**Why training objective alignment was the critical fix:** The reconstruction adapter was trained to make LLaMA predict a movie's title from its soft token — a generation task. At ranking time, the LLM is asked to compare 10 soft tokens against each other and pick the best — a discrimination task. The adapter had no training signal for discrimination, so all soft tokens landed in semantically equivalent regions of embedding space and the LLM fell back to position. The ranking adapter trains the exact same forward pass that is used at evaluation (build 10-candidate soft-token prompt → read letter logits → CE loss), eliminating the train/eval objective mismatch.

---

## Artifacts

| File | Description |
|---|---|
| `checkpoints/adapter_ranking.pt` | Ranking adapter weights + training config |
| `checkpoints/projected_embeddings_ranking.pt` | Pre-cached projections (3510, 2048) |
| `checkpoints/item_embeddings.pt` | SASRec item embeddings (3510, 50) |
| `checkpoints/id_maps.json` | movie_id ↔ internal index mapping |
| `results_mode_A.json` | Mode A predictions with per-letter probabilities |
| `results_mode_C.json` | Mode C predictions with per-letter probabilities |
| `test_ranking_prompts.json` | 604 test examples with candidates and true positives |

All checkpoints produced on Google Colab T4 GPU.
