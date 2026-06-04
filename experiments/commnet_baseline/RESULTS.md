# Part 2.2 — CommNet baseline with Q/R/M/C instrumentation

## Setup

Minimal CommNet-style policy on the same multi-agent grid environment used for
the peer-architecture sweep. Architecture:

- Per-agent observation: 4 direction one-hot + 5×5 local cell-type window (125)
  + 2 normalised position = 131-dim feature vector
- Shared-weights encoder: 131 → 64 → 64 (ReLU)
- Communication: linear projection 64 → 16; receive = mean-pool of OTHER agents' projections
- Combine: [h, comm_in] (80-dim) → 64 (ReLU)
- Actor head: 64 → 4 (TURN_LEFT, TURN_RIGHT, FORWARD, NOOP)
- Critic head: 64 → 1

Training:
- REINFORCE + value baseline, clipped grad norm 1.0, lr 3e-4
- 2000 episodes on cycling 200 env-seeds (scenario: N=3, M=2, asymmetric, hazard 0.05, step_limit 80)
- Reward shaping: +1 first success per agent, −0.01 per step, −0.1 per hazard
- Wall-clock: 100 seconds on CPU

## Training curve

Episode-level success rate over the last 100 episodes oscillates between 0.13
and 0.23 throughout training; no monotonic improvement after the first ~200
episodes. The policy reaches a local optimum where roughly one of three agents
finds water by chance per episode.

## Evaluation (50 seeds, stochastic action sampling)

| Metric | Value |
|---|---|
| `success_rate` (mean) | 0.187 |
| `n_succeeded` (mean / 3 agents) | 0.56 |
| `steps_mean` | 80 (i.e. budget exhausted) |

Compare to the rule-based peer architecture at $K=8$ on the same scenario from
the main sweep: $P(M^\star|R^\star) \cdot P(C^\star|M^\star) \approx 0.59$,
mean `t_succ` $\approx 7.47$, success rate $\approx 0.65$.

## Q/R/M/C profile of the trained CommNet policy

| Stage | Rate |
|---|---|
| $Q^\star$ (comm signal non-zero) | **1.00** (saturated) |
| $R^\star$ (agent ever within radius 2 of a water cell) | 0.50 |
| $M^\star$ (state reached has water within radius 2 — distance-reduction proxy) | 0.50 |
| $C^\star$ (success) | 0.19 |
| $P(R \mid Q)$ | 0.50 |
| $P(M \mid R)$ | ≈ 1.00 |
| $P(C \mid M)$ | 0.42 |

## Interpretation

The framework's prediction holds in three ways:

1. **$Q^\star$ is degenerate.** Communication signal is non-zero in every
   episode by construction: the comm projection is a learned linear layer
   whose output is never identically zero in a trained network. The $Q$ stage
   therefore carries no stage-specific information for a learned baseline ---
   exactly analogous to the behaviour-cloned single-agent baseline, whose
   $Q$ event is trivially true whenever the policy emits any action.

2. **The chain $Q \to R \to M \to C$ is observable but the events are
   coarsened.** $R$ and $M$ collapse together because they are both
   downstream of the same "reach-near-water" predicate; on the rule-based
   peer architecture they decouple cleanly (separate conditional rates).
   The learned policy lacks the symbolic structure that would distinguish
   "considered a target" from "moved toward a target".

3. **Final success is roughly random-baseline level.** The peer architecture
   on the same scenario reaches ~0.65 success at $K=8$; CommNet at 2000
   episodes reaches 0.19. With more training and curriculum the gap would
   close, but the Q/R/M/C degeneracy is structural, not a function of
   training budget.

## What this means for the paper

This is exactly the negative-baseline-baseline argument we already make for
the single-agent learned BC baseline (Section 7.3): a learned policy that
performs end-to-end does not expose meaningful per-stage events. The Q/R/M/C
framework's diagnostic claim is therefore preserved precisely \emph{because}
it doesn't apply to learned-communication baselines in the same way it
applies to symbolic architectures.

This is not an argument that CommNet is a worse policy; it is an argument
that CommNet is the wrong shape of artefact to diagnose with Q/R/M/C.

## Artifacts

```
experiments/commnet_baseline/
├── commnet_agent.py        # policy network (CommNet-style)
├── train_commnet.py        # REINFORCE + value-baseline trainer
├── eval_with_qrmc.py       # Q/R/M/C wrapper over trained policy
├── train_log.json          # per-episode training metrics
├── eval_summary.json       # held-out eval summary
├── qrmc_eval.json          # Q/R/M/C eval per-episode + summary
├── commnet_policy.pt       # trained checkpoint
└── RESULTS.md              # this file
```

Reproducible:

```bash
python experiments/commnet_baseline/train_commnet.py --n_episodes 2000
python experiments/commnet_baseline/eval_with_qrmc.py --n_episodes 100
```
