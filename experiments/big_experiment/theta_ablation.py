"""
Threshold (theta) ablation for the empty-candidate-set diagnosis [reviewer #9].

The single-agent analysis attributes hazard-recovery / goal-region failure to a
"memory-accumulation limit" (91% empty candidate sets). A reviewer asks whether
that is an artefact of the required-tag threshold theta (default 0.5).

Wiring confirmed: the benchmark builds its query at scripts/songline_minigrid.py
:1325 via songline_drive.intents.build_planner_query, whose required-tag
threshold and min_confidence read QRMC_REQUIRED_TAG_THETA (this file's override).
The candidacy gate is symbolic_memory._required_matches_profile
(profile[tag] >= theta).

FINDING (reported in the paper, Appendix "Threshold robustness"): the
empty-candidate-set rate is INVARIANT across the full sane range
theta in [0.05, 0.95] (hazard 0.533, goal 0.267). Endpoints confirm the gate is
live: theta=1.0 (accept none) raises empty to 0.75; theta=0.0 (accept all) lowers
it only to 0.50 -- its floor -- because at those decision points there is no
candidate node of ANY confidence. So the "memory-accumulation limit" is absence
of stored evidence, not a threshold artefact. Absolute rates are on a small
hazard/goal sweep and are not the headline 91% (full benchmark); the invariance
is the result.

Run:  PYTHONPATH=. .venv/bin/python experiments/big_experiment/theta_ablation.py
"""
import csv, os, subprocess, sys

THETAS = [0.05, 0.15, 0.25, 0.50, 0.75, 0.95]
TASKS = ["hazard_recovery", "goal_region"]
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_ROOT = "tmp/theta_ablation"


def run_one(theta):
    out = f"{OUT_ROOT}/theta_{int(theta*100)}"
    env = dict(os.environ, QRMC_REQUIRED_TAG_THETA=str(theta), PYTHONPATH=BASE)
    cmd = [sys.executable, "scripts/benchmark_symbolic_memory_article.py",
           "--tasks", *TASKS, "--num_seeds", "3", "--episodes", "5",
           "--assist_modes", "off", "--out_dir", out]
    subprocess.run(cmd, cwd=BASE, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def read_empty_rate(out, task):
    """Empty-candidate-set rate for the semantic (milestone_*) method on a task."""
    path = os.path.join(BASE, out, "article_failure_taxonomy.csv")
    rows = list(csv.DictReader(open(path)))
    sem = [r for r in rows if r["task_name"] == task
           and r["method"].startswith("milestone_semantic")]
    if not sem:
        return None, None
    r = sem[0]
    return float(r["retrieval_failure_empty_rate"]), float(r["success_rate"])


print(f"theta ablation: tasks={TASKS}, 3 seeds x 5 episodes, semantic method\n")
print(f"{'theta':>6} | " + " | ".join(f"{t}: empty% / succ" for t in TASKS))
print("-"*70)
for th in THETAS:
    out = run_one(th)
    cells = []
    for t in TASKS:
        e, s = read_empty_rate(out, t)
        cells.append(f"{e*100:5.1f}% / {s:.2f}" if e is not None else "  n/a")
    star = "  <- deployed" if th == 0.50 else ""
    print(f"{th:>6.2f} | " + " | ".join(cells) + star)
print("\nReading: if the empty-set rate is roughly flat across theta, the")
print("'memory-accumulation limit' is not a threshold artefact; if it collapses")
print("at lower theta, the diagnosis is threshold-sensitive and must be qualified.")
