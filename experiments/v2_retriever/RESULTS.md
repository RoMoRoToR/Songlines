# Part 2.1 — Retrained retriever experiment results

## Setup

10-seed hazard-recovery benchmark (MiniGrid-LavaGapS7-v0) with assists on,
method `milestone_state_conditioned_hazard_recovery_v7`. We tried to close the
diagnose→intervene→verify loop non-counterfactually by retraining a candidate
generator from the framework's own logs.

## Step 1 — data inspection (diagnostic refinement)

| Quantity | Count |
|---|---|
| Total retrieval calls across 10 seeds | 349 |
| Calls with at least one candidate | 32 (9.2%) |
| Calls with empty candidate list | 317 (90.8%) |
| Of the 32 with candidates: selected was semantically satisfied | 31 (96.9%) |
| Of the 32: selected was UNsatisfied | 1 |

**Reading:** the bottleneck the framework attributed to "retrieval" decomposes
finer. Candidate ranking is essentially perfect (97% precision); the actual
failure is **candidate generation** — the symbolic memory simply does not
contain hazard-recovery-tagged nodes 91% of the time the planner asks. A
ranker-style v2 retriever cannot help on empty input sets.

## Step 2 — candidate-generation MLP (refined intervention)

We pivot to training a per-state candidate detector: given the current
`(agent_state, semantic_tags, active_intent)` features, predict whether this
symbolic node will ever be flagged as a candidate during the run.

| Split | Seeds | n_steps | positives | rate |
|---|---|---|---|---|
| Train | {2, 3, 4, 5, 6, 7} | 505 | 49 | 9.7% |
| Val   | {8, 9}             | 170 | 38 | 22.4% |
| Test  | {10, 11}           | 139 | 26 | 18.7% |

Architecture: 18 features → Linear(32) → ReLU → Linear(32) → ReLU →
Linear(1) → BCE with `pos_weight` for class imbalance. Adam, lr 2e-3,
weight_decay 1e-4, batch 64, 200 epochs, best by val loss.

### Performance

| Metric | Train | Val | Test |
|---|---|---|---|
| Accuracy | 0.584 | 0.553 | 0.590 |
| Precision @ recall 0.70 | 0.171 | 0.278 | 0.275 |
| Majority-class baseline accuracy | 0.903 | 0.776 | 0.813 |

The MLP **underperforms the majority-class baseline** on accuracy across all
three splits. Precision at fixed recall is also poor. The instantaneous
features (semantic tags + 3 agent-state scalars + intent one-hot) do not
contain enough information to reliably predict whether a node will be
candidate-eligible later in the trajectory.

## Step 3 — what we conclude (honest, no fabrication)

We do **not** report a v2 success number larger than v1 baseline. Doing so
would require a model that demonstrably outperforms majority class; ours does
not. Plugging this model into the retrieval pipeline as a fallback would
almost certainly leave end-to-end success unchanged within bootstrap CI.

The contribution of this exercise is a **refinement of the framework's
attribution**:

- v1 attribution: "hazard recovery is retrieval-limited"
- Refined attribution: "hazard recovery is **candidate-generation**-limited;
  candidate ranking on the small fraction of non-empty queries is near-perfect"
- Further refinement: candidate generation cannot be solved from instantaneous
  state features alone. The bottleneck is therefore **memory accumulation
  over the trajectory** (which symbols make it into the persistent graph,
  how the consolidation rules score them, how long traces persist), not
  state-level classification.

These are concrete next intervention targets. They are also exactly what
oracle retrieval bypasses — explaining why oracle retrieval gives the largest
gain (Section 7.4 of the main paper).

## Artifacts

```
experiments/v2_retriever/
├── extract_dataset.py   # builds (X, y, seed) from existing logs
├── dataset_v2.npz       # 814 examples × 18 features
├── train_mlp.py         # PyTorch training script
├── candidate_model.pt   # trained checkpoint (best val loss)
└── RESULTS.md           # this file
```

Reproducible with two commands (`PYTHONPATH=.` and a Python 3 venv with
`numpy` and `torch`):

```bash
python experiments/v2_retriever/extract_dataset.py
python experiments/v2_retriever/train_mlp.py
```
