"""
Q/R/M/C on ALFWorld with a real LLM agent (reviewer #2: external, non-toy substrate).

The agent plays standard ALFWorld text games (ALFRED tasks) with an LLM policy
(HF transformers on a GPU node, greedy). The same four-stage contract is logged:

  Q* -- the LLM declares the target object for the task ("TARGET: <object>");
  R* -- the episodic object->receptacle memory (built from past observations)
        contains a candidate location for the declared target;
  M* -- the agent commits to that retrieved candidate ("go to <receptacle>"
        toward the memory-retrieved location);
  C* -- the environment reports task success (won).

Nested by construction at episode level. Failure taxonomy = first missing stage.

Setup (login node): pip install alfworld pyyaml; alfworld-download
  export ALFWORLD_DATA=/mnt/tank/scratch/rzamotaev/alfworld_data
  config: official base_config.yaml (env var ALFWORLD_CONFIG)

Run:
  PYTHONPATH=. python experiments/alfworld_qrmc/run_alfworld_qrmc.py \
      --model Qwen/Qwen2.5-3B-Instruct --episodes 25 --out_dir tmp/cluster/alfworld_qwen3b
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

SYSTEM = (
    "You are an expert household agent playing a text game. Always reply with "
    "EXACTLY ONE admissible command, verbatim from the list, and nothing else."
)

TARGET_SYSTEM = (
    "You extract the key object from a household task. Reply with exactly one "
    "line: TARGET: <object name>, using the simplest noun for the object."
)


def load_env(config_path: str, split: str):
    import yaml
    from alfworld.agents.environment import get_environment

    with open(config_path) as f:
        config = yaml.safe_load(f)
    env_type = config["env"]["type"]
    env = get_environment(env_type)(config, train_eval=split)
    return env.init_env(batch_size=1)


SEE_RE = re.compile(
    r"(?:on|in) the ([a-z]+ \d+),? you see(.*?)(?:\.|$)", re.I | re.S)
ITEM_RE = re.compile(r"\ba (?:[a-z]+ )*?([a-z]+) \d+", re.I)


def update_memory(memory: Dict[str, set], text: str) -> None:
    """Parse 'On the shelf 1, you see a mug 2 ...' into object -> receptacles."""
    for m in SEE_RE.finditer(text):
        recept, items = m.group(1).strip(), m.group(2)
        for it in ITEM_RE.finditer(items):
            memory.setdefault(it.group(1).lower(), set()).add(recept)


def extract_target(backend, task: str, seed: int) -> Optional[str]:
    out = backend.complete(
        f"TASK: {task}\nReply with one line: TARGET: <object>",
        system=TARGET_SYSTEM, seed=seed, max_tokens=24)
    m = re.search(r"TARGET:\s*([a-z][a-z ]*)", out, re.I)
    if not m:
        return None
    # simplest noun: last word
    return m.group(1).strip().lower().split()[-1]


def choose_action(backend, task, obs, admissible, mem_hint, seed, step) -> str:
    cmds = admissible[:40]
    prompt = (
        f"TASK: {task}\n\nOBSERVATION:\n{obs[-900:]}\n\n"
        + (f"MEMORY: {mem_hint}\n\n" if mem_hint else "")
        + "ADMISSIBLE COMMANDS:\n" + "\n".join(f"- {c}" for c in cmds)
        + "\n\nReply with exactly one command from the list."
    )
    out = backend.complete(prompt, system=SYSTEM, seed=seed * 1000 + step,
                           max_tokens=24)
    out_l = out.strip().lower()
    for c in cmds:                       # exact or prefix match
        if c.lower() == out_l or out_l.startswith(c.lower()):
            return c
    for c in cmds:                       # substring fallback
        if c.lower() in out_l:
            return c
    return cmds[0] if cmds else "look"   # deterministic fallback


def run_episode(env, backend, seed: int, step_limit: int) -> Dict[str, Any]:
    obs, info = env.reset()
    obs0 = obs[0]
    task = obs0.split("Your task is to:")[-1].strip() if "Your task is to:" in obs0 else obs0[-200:]

    memory: Dict[str, set] = {}
    update_memory(memory, obs0)

    target = extract_target(backend, task, seed)
    q_star = target is not None
    r_star = m_star = c_star = False
    r_tick = m_tick = None

    text = obs0
    for step in range(step_limit):
        admissible = list(info["admissible_commands"][0])
        # R*: memory holds a candidate receptacle for the target
        cand = sorted(memory.get(target, set())) if (target and target in memory) else []
        if cand and not r_star:
            r_star, r_tick = True, step
        mem_hint = f"the {target} was seen at: {', '.join(cand)}" if cand else ""

        action = choose_action(backend, task, text, admissible, mem_hint, seed, step)
        # M*: commitment to the retrieved candidate
        if cand and action.lower().startswith("go to") and not m_star:
            dest = action.lower().replace("go to", "").strip()
            if any(dest == c.lower() for c in cand):
                m_star, m_tick = True, step

        obs, scores, dones, infos = env.step([action])
        text = obs[0]
        info = infos
        update_memory(memory, text)
        if dones[0]:
            c_star = bool(infos["won"][0])
            break

    stage = ("success" if c_star else
             "C_fail" if m_star else
             "M_fail" if r_star else
             "R_fail" if q_star else "Q_fail")
    return {
        "seed": seed, "task": task[:80], "target": target or "",
        "q_star": int(q_star), "r_star": int(r_star),
        "m_star": int(m_star), "c_star": int(c_star),
        "success": int(c_star), "first_missing_stage": stage,
        "steps": step + 1, "r_tick": r_tick, "m_tick": m_tick,
        "mem_objects": len(memory),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--step_limit", type=int, default=40)
    ap.add_argument("--split", default="eval_out_of_distribution")
    ap.add_argument("--config", default=os.environ.get("ALFWORLD_CONFIG", ""))
    ap.add_argument("--out_dir", default="tmp/cluster/alfworld_qrmc")
    a = ap.parse_args()
    assert a.config and os.path.exists(a.config), "set --config / $ALFWORLD_CONFIG"
    os.makedirs(a.out_dir, exist_ok=True)

    from experiments.llm_collective.hf_backend import HFBackend
    backend = HFBackend(model=a.model,
                        cache_dir=os.path.join(a.out_dir, ".cache_llm"))
    env = load_env(a.config, a.split)

    rows: List[Dict[str, Any]] = []
    for ep in range(a.episodes):
        row = run_episode(env, backend, seed=ep, step_limit=a.step_limit)
        row["model"] = a.model
        rows.append(row)
        print(f"[{a.model}] ep={ep} {row['first_missing_stage']:9s} "
              f"Q/R/M/C={row['q_star']}/{row['r_star']}/{row['m_star']}/{row['c_star']} "
              f"steps={row['steps']} task={row['task'][:50]}", flush=True)
        with open(os.path.join(a.out_dir, "runs.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    n = len(rows)
    agg = {k: sum(r[k] for r in rows) / n for k in
           ("q_star", "r_star", "m_star", "c_star", "success")}
    from collections import Counter
    agg["stage_taxonomy"] = dict(Counter(r["first_missing_stage"] for r in rows))
    agg["episodes"] = n
    agg["model"] = a.model
    agg["backend"] = backend.summary()
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
