# PPO CommNet baseline — acceptance PASS

Goal: replace the REINFORCE CommNet baseline (success ≈ 0.19) with a
competitively trained PPO variant so the structural claim
("Q-saturation and R/M collapse are properties of the policy class,
not the training budget") is *verified* rather than asserted.

## Result: acceptance criteria PASSED

| Metric                         | REINFORCE | **PPO (this run)** | Symbolic peer K=8 |
|--------------------------------|----------:|-------------------:|------------------:|
| Success rate (held-out 100 seeds) | 0.19      | **0.667**          | 0.65              |
| $Q^\star$ rate                    | 1.00      | **1.00**           | 0.99              |
| $R^\star$ rate                    | 0.50      | **0.997**          | 0.69              |
| $M^\star$ rate                    | 0.50      | **0.997**          | 0.69              |
| $P(M^\star\,|\,R^\star)$          | 1.00      | **1.00**           | 0.69              |
| $P(C^\star\,|\,M^\star)$          | 0.42      | **0.67**           | 0.87              |
| R/M separated?                    | no        | **no**             | **yes**           |

**PPO is competitive on success** (0.667 vs symbolic 0.65) **but still
exhibits Q-saturation and R/M collapse** — exactly as the framework's
structural argument predicted. Diagnostic claim is now verified, not
asserted.

## Setup

- Architecture: CommNet (shared 131→64→64 encoder, 16-d linear comm
  projection, peer mean-pool, combine 80→64, actor 64→4, critic 64→1)
  — *identical* to the REINFORCE baseline.
- Training: PPO with clip ε=0.2, GAE λ=0.95, γ=0.99, 4 PPO epochs per
  rollout batch, 64 rollouts per update, 200 updates total.
- Optimization: Adam lr=3e-4, grad-norm clip 1.0, value coef 0.5,
  entropy coef 0.01.
- Reward shaping: +1 first success per agent, -0.01/step, -0.1/hazard
  contact.
- Wall-clock: ~96 minutes on a single CPU core (PID 43903,
  Jun 8 2026, elapsed 5761s).
- Eval: 100 held-out seeds (250..349), stochastic action sampling
  (matches training distribution; deterministic argmax gets stuck in
  loops because the policy hasn't converged to a sharp peak).

## Training curve

last-50 batch success climbed from ~0.08 at update 0 to plateau at
0.640–0.652 by update 130, finishing at **0.650** at update 200.

## Verdict

The PPO acceptance criterion (success ≥ 0.5, Q-saturation persists,
R/M collapse persists) **passed cleanly**. The paragraph in §4.5 and
the table in Appendix G have been updated; the previous
"future-work" qualifier has been removed.

## Artifacts

```
experiments/commnet_ppo_baseline/
├── README.md
├── train_ppo_commnet.py    # PPO training driver
├── ppo_policy.pt           # trained policy (81 KB)
├── qrmc_eval.json          # 100-seed Q/R/M/C eval
└── RESULTS.md              # this file
```

Reproduce:

```bash
python -m experiments.commnet_ppo_baseline.train_ppo_commnet \
    --total_updates 200 --rollouts_per_update 64
python experiments/commnet_baseline/eval_with_qrmc.py \
    --policy_path experiments/commnet_ppo_baseline/ppo_policy.pt \
    --out_dir experiments/commnet_ppo_baseline \
    --n_episodes 100
```
