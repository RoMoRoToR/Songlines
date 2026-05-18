"""Run all 8 collective-memory smoke scripts and summarize pass/fail.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/run_all_smokes.py
    PYTHONPATH=. .venv/bin/python scripts/run_all_smokes.py --out_dir tmp/all_smokes
    PYTHONPATH=. .venv/bin/python scripts/run_all_smokes.py --include phase4

The runner shells out to each smoke script via subprocess so they
inherit a clean Python state.  Each smoke writes its own ``*_summary.json``
to a phase-specific output directory under ``--out_dir``.

Exit code: 0 if all included smokes pass, 1 if any fail.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


SMOKES: List[Tuple[str, str]] = [
    ("phase1_collective", "scripts/multiagent_smoke_collective.py"),
    ("phase2_basic",      "scripts/multiagent_smoke_phase2.py"),
    ("phase2_ab",         "scripts/multiagent_smoke_phase2_ab.py"),
    ("phase3",            "scripts/multiagent_smoke_phase3.py"),
    ("phase4a",           "scripts/multiagent_smoke_phase4a.py"),
    ("phase4b_ab",        "scripts/multiagent_smoke_phase4b_ab.py"),
    ("phase4c",           "scripts/multiagent_smoke_phase4c.py"),
    ("phase4d",           "scripts/multiagent_smoke_phase4d.py"),
]


def run_one(name: str, script: str, out_dir: str) -> Dict:
    phase_out = os.path.join(out_dir, name)
    os.makedirs(phase_out, exist_ok=True)
    cmd = [
        sys.executable, script,
        "--out_dir", phase_out,
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=300,
    )
    duration = time.time() - t0

    last_line = ""
    if proc.stdout:
        lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
        if lines:
            last_line = lines[-1] if "Summary" not in lines[-1] else (
                lines[-2] if len(lines) > 1 else lines[-1]
            )

    return {
        "name": name,
        "script": script,
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "duration_sec": round(duration, 2),
        "last_line": last_line,
        "stderr_tail": (proc.stderr.strip().splitlines()[-5:] if proc.stderr else []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="tmp/all_smokes")
    parser.add_argument(
        "--include", default="",
        help="Substring filter; only run smokes whose name contains this. "
             "Empty = run all.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    selected = [s for s in SMOKES if args.include in s[0]]
    if not selected:
        print(f"No smokes match filter '{args.include}'", file=sys.stderr)
        sys.exit(2)

    results: List[Dict] = []
    print(f"Running {len(selected)} smokes → {args.out_dir}")
    print("=" * 70)

    for name, script in selected:
        print(f"  [{name}] ...", end=" ", flush=True)
        r = run_one(name, script, args.out_dir)
        results.append(r)
        marker = "✓" if r["passed"] else "✗"
        print(f"{marker}  ({r['duration_sec']}s)")
        if not r["passed"] and r["stderr_tail"]:
            for ln in r["stderr_tail"]:
                print(f"      {ln}")

    summary = {
        "n_run": len(results),
        "n_passed": sum(1 for r in results if r["passed"]),
        "n_failed": sum(1 for r in results if not r["passed"]),
        "total_sec": round(sum(r["duration_sec"] for r in results), 2),
        "results": results,
    }
    summary_path = os.path.join(args.out_dir, "all_smokes_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print(
        f"Total: {summary['n_passed']}/{summary['n_run']} passed  "
        f"({summary['total_sec']}s)  →  {summary_path}"
    )
    sys.exit(0 if summary["n_failed"] == 0 else 1)


if __name__ == "__main__":
    main()
