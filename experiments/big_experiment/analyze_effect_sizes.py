"""
Cluster-robust re-analysis of the cadence sweeps. Real data, deterministic.
Usage:  python analyze_effect_sizes.py [main|n12]   (default: main, the N<=8 sweep)
Reported in the paper as effect sizes with cluster-robust CIs (Appendix E.3),
replacing the uninformative p<1e-4 figures. N=12 CIs feed Table (tab:scale-n12);
there the M|R slope CI [-0.28,+0.02] crosses zero (not resolved at N=12).
"""
import sys
import numpy as np, pandas as pd
from scipy import stats

RNG = np.random.default_rng(20260701)
B = 4000
_which = sys.argv[1] if len(sys.argv) > 1 else "main"
_path = ("tmp/big_experiment_N12/runs.csv" if _which == "n12"
         else "tmp/paper1_clean_experiments_full/multiagent_cadence_full/runs.csv")
peer = pd.read_csv(_path)
peer = peer[peer.architecture == "peer"].copy()
CELL = ["n_agents", "n_waters", "layout", "hazard_density"]
peer["cell"] = peer[CELL].astype(str).agg("|".join, axis=1)
cells = peer["cell"].unique()
by_cell = {c: peer[peer.cell == c] for c in cells}
print(f"[{_which}] peer rows={len(peer)}  independent design cells={len(cells)}  seeds={peer.seed.nunique()}")
print(f"block bootstrap resamples the {len(cells)} config cells, not the correlated runs.\n")

def cluster_boot(stat_fn):
    n = len(cells); vals = []
    for _ in range(B):
        pick = RNG.choice(cells, size=n, replace=True)
        sub = pd.concat([by_cell[c] for c in pick], ignore_index=True)
        v = stat_fn(sub)
        if v is not None and np.isfinite(v): vals.append(v)
    vals = np.array(vals)
    return vals.mean(), np.percentile(vals, 2.5), np.percentile(vals, 97.5)

def sp(s, col):
    d = s[["broadcast_every_k", col]].dropna()
    return stats.spearmanr(d["broadcast_every_k"], d[col]).correlation

print("="*72)
print("#3  EFFECT SIZES: Spearman rho of conditional rate vs cadence K (cluster-robust)")
print("="*72)
for col, name in [("p_M_given_R","P(M*|R*)"), ("p_C_given_M","P(C*|M*)")]:
    rn = sp(peer, col)
    m, lo, hi = cluster_boot(lambda s: sp(s, col))
    mag = "weak" if abs(m)<0.3 else "moderate" if abs(m)<0.5 else "strong"
    print(f"  {name:9s}: rho={m:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  ({mag} effect; naive point est {rn:+.3f})")

print("\n"+"="*72)
print("#8  ROBUSTNESS OF K OPTIMUM (paired cluster bootstrap on mean_t_succ diffs)")
print("="*72)
def tdiff(s, k):
    a = s[s.broadcast_every_k==8]["mean_t_succ"].mean()
    b = s[s.broadcast_every_k==k]["mean_t_succ"].mean()
    return a-b
for k in [1,4,16,64]:
    m, lo, hi = cluster_boot(lambda s: tdiff(s,k))
    sig = "sig (excludes 0)" if (lo<0)==(hi<0) else "NOT sig (CI incl 0)"
    print(f"  t_succ(K=8) - t_succ(K={k:2d}) = {m:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  {sig}")
peer["eff"] = peer["mean_t_succ"]/peer["success_rate"].clip(lower=1e-6)
alt = peer.groupby("broadcast_every_k")["eff"].mean()
print(f"  success-weighted cost argmin K = {int(alt.idxmin())}  (raw t_succ argmin K = {int(peer.groupby('broadcast_every_k')['mean_t_succ'].mean().idxmin())})")
print("  => optimum SHAPE robust (K=8 both metrics), but 8-vs-4 and 8-vs-1 gaps are within noise: curve is shallow.")

print("\n"+"="*72)
print("#4  Y vs C* VALIDITY GAP")
print("="*72)
print("  Multi-agent sweep: success == reaching water == C*, so here corr(Y,C*)=1.0, gap=0 by construction.")
ov = pd.read_csv("tmp/article_revision_10seeds_20260501/article_overview.csv")
ov = ov[ov.assist_mode.astype(str).str.contains("off", case=False, na=False)]
# semantic method rows (exclude baselines random/graph_only/sptm/bc)
sem = ov[~ov.method.astype(str).str.lower().str.contains("random|graph_only|sptm|bc|behavior")]
sem = sem.copy()
sem["chain"] = sem["query_satisfaction_rate"]*sem["semantic_target_materialization_rate"]*sem["post_retrieval_completion_rate"]
sem["gap"] = sem["success_rate"] - sem["chain"]
print("  Single-agent (assist off, semantic methods): Y = success_rate, chain = Qsat*Mat*Compl")
cols=["task_name","method","success_rate","chain","gap"]
print(sem[cols].round(3).to_string(index=False))
print(f"\n  mean Y={sem.success_rate.mean():.3f}  mean chain(C*)={sem.chain.mean():.3f}  mean gap(Y-C*)={sem.gap.mean():+.3f}")
if len(sem)>3 and sem["chain"].std()>0:
    r=np.corrcoef(sem.success_rate, sem.chain)[0,1]
    print(f"  corr(Y, chain) across task/method rows = {r:+.3f}  (n={len(sem)} rows)")
print("  => where gap>0, task success bypasses the semantic path; the chain diagnoses a partially")
print("     parallel channel. Honest fix: report gap; per-episode stage-resolved mediation needs re-instrumented logs.")
