# PPO CommNet baseline — acceptance checklist

This experiment converts the REINFORCE CommNet baseline (success ≈ 0.19)
into a competitively trained PPO variant (target success ≥ 0.5) so that
the diagnostic claim ("Q-saturation and R/M collapse are structural,
not budget-limited") is verified empirically rather than asserted.

## Steps

1. **Train**: `python -m experiments.commnet_ppo_baseline.train_ppo_commnet \
   --total_updates 200 --rollouts_per_update 64`
   Wall-clock budget: 1-3 hours CPU.

2. **Evaluate Q/R/M/C** (reuses existing wrapper unchanged):
   `python experiments/commnet_baseline/eval_with_qrmc.py \
   --policy_path experiments/commnet_ppo_baseline/ppo_policy.pt \
   --out_dir experiments/commnet_ppo_baseline --n_episodes 100`

3. **Accept-or-iterate**:
   - PASS if held-out success ≥ 0.5, Q* ≥ 0.95, R/M collapse holds
     (|P(M|R) − 1.0| < 0.05).
   - If success < 0.5: increase `--total_updates` (try 400, 800);
     add curriculum (start with N=2, T=1).
   - If success ≥ 0.5 but R/M *separate*: that contradicts the
     structural claim; this is a publishable negative result for the
     paper — the structural argument needs revision.

4. **Patch paper** (Appendix G):
   Replace
       "A reviewer who would prefer a stronger baseline is correct
        that one exists..."
   with
       "We verified this empirically: a PPO-trained variant reaching
        {success:.2f} success on the same scenario exhibits the same
        Q-saturation and R/M collapse (Table~\ref{tab:ppo-commnet})."

   Add a 5-row table next to the existing CommNet Q/R/M/C profile
   table showing PPO vs REINFORCE side-by-side.

## What is *not* done in this scaffold

- Sophisticated PPO tricks (linear lr decay, value clipping, KL early
  stopping) are omitted. If basic PPO does not clear 0.5 success in
  3 hours, add them in this order: linear lr decay → value clipping
  → KL early-stop.

- No curriculum or env-randomization. Add if the basic loop plateaus
  below 0.5.

- No GPU support. CPU is sufficient at this scale (≈131-dim obs,
  ≈10k params).
