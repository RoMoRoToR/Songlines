"""L1 — the LLM end-to-end UCSM cycle.

Closes the loop the frontier document promised:

    LLM observation -> candidate memory -> utility/analogy controller
    -> schema graph -> retrieved ACTIVE CONTEXT -> LLM action.

Division of labour follows the series' W4 protocol: the LLM makes the
SEMANTIC decision (which remembered target to commit to, given the
schema graph's certificates and exceptions), a deterministic waypoint
primitive walks (small models are unreliable at coordinate
arithmetic; the motor skill is not the subject). Shared frame on the
LLM substrate (frame-free identity is proven separately in the route
part).

Per layout: a world family is experienced over four episodes ---
three visits to the base world (water at W1), then a conflict episode
(water secretly moved to W2). The deterministic UCSM controller turns
this stream into a schema graph: parent schema (support 3) with a
RECORDED FAILURE, plus an EXCEPTION schema superseding it. A fresh
LLM session must then commit to a water target in the current world
state, under three context arms:

  none  -- no memory in the prompt (control);
  raw   -- the full chronological episode transcript (long context;
           in conflict layouts the stale target holds the 3:1
           majority of the text);
  ucsm  -- the reconstructed active context: schemas + certificates +
           exceptions, a few lines.

Half the layouts are evaluated in the BASE state (no conflict ever
happened: stream = 3 base episodes; all memory arms should agree),
half in the CONFLICT state (the exception must override the stale
majority).

Registered predictions:
  L1.1 (memory is needed): the no-memory arm's lock precision <= 0.2.
  L1.2 (reconstruction beats replay): the ucsm arm's overall lock
       precision >= the raw arm's, at <= 30% of raw's prompt size.
  L1.3 (exceptions work in context): on conflict layouts the ucsm
       arm's precision > the raw arm's (the 3:1 stale majority in the
       transcript must mislead more often than the explicit
       exception).

Backends: --backend {stub, ollama, hf}. `stub` is a deterministic
plumbing mock (commits to the FIRST water mentioned in the prompt ---
deliberately naive so conflict layouts expose the arms' ordering);
`hf` is for the cluster GPU nodes (Qwen, offline weights).

Usage::

    PYTHONPATH=. python experiments/song_grammar/exp_l1_llm_endtoend.py \
        --backend hf --model Qwen/Qwen2.5-3B-Instruct \
        --layouts 12 --out tmp/cluster/song_grammar/l1_qwen3b
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from experiments.song_grammar.exp_s0_song_smoke import TRAVELER_START
from experiments.song_grammar.u7_common import (
    ROLES, dijkstra, family_world, valid_world, witness_song)
from multiagent_env import WATER

GridXY = Tuple[int, int]
ROLE = ROLES["robust"]
MAX_TOKENS = 512

SYSTEM = (
    "You are a navigation agent in a grid world. You must choose the "
    "most likely CURRENT location of the water source using your "
    "memory. Trust recorded failures and exceptions over sheer "
    "repetition: an exception that supersedes a schema reflects the "
    "latest verified state of the world. Answer with a single JSON "
    "object exactly like {\"target\": [x, y]} and nothing else.")


# ── serialisation: songs -> transcript, schema graph -> context ────

def song_text(song, water: GridXY, k: int) -> str:
    lines = [f"Episode {k}:"]
    for c in song:
        tags = sorted({key.split("@")[0] for key in c["sig"]})
        dx, dy = c["beat"]
        lines.append(f"  passed landmark [{','.join(tags) or 'open'}]"
                     f" after moving ({dx:+d},{dy:+d})")
    lines.append(f"  reached WATER at ({water[0]}, {water[1]}).")
    return "\n".join(lines)


def raw_context(episodes: List[Tuple[Any, GridXY]]) -> str:
    return ("FULL MEMORY TRANSCRIPT (chronological):\n"
            + "\n".join(song_text(s, w, k + 1)
                        for k, (s, w) in enumerate(episodes)))


def ucsm_context(schemas: List[Dict[str, Any]]) -> str:
    lines = ["ACTIVE MEMORY (consolidated schemas with certificates):"]
    for s in schemas:
        head = (f"Schema {s['name']}: water at "
                f"({s['water'][0]}, {s['water'][1]}); support "
                f"{s['support']}; confidence {1.0/s['support']:.2f} "
                f"uncertainty.")
        lines.append(head)
        for fail in s.get("failures", []):
            lines.append(f"  RECORDED FAILURE: {fail}")
        if s.get("supersedes"):
            lines.append(f"  EXCEPTION: supersedes {s['supersedes']} "
                         "(latest verified state of this place).")
    return "\n".join(lines)


# ── formation: deterministic UCSM over the episode stream ──────────

def form_memory_long(fam: int, chain: bool):
    """L1b: a 12-episode history. Chain layouts move the water twice
    (6 visits at W0, 4 at W1, 2 at W2 -- the majority of the raw
    transcript favours the STALE targets 10:2); base layouts are 12
    benign visits to one water through varying hazard textures (the
    long-but-honest transcript control). Every visit uses a fresh
    appearance variant, so raw transcripts do not compress."""
    counts = (6, 4, 2) if chain else (12,)
    envs, waters, songs = [], [], []
    for idx in range(len(counts)):
        env, w = family_world(fam, 0, idx)
        if not (valid_world(env, w) and w not in waters):
            return None
        envs.append(env)
        waters.append(w)
    episodes = []
    visit = 0
    for phase, n in enumerate(counts):
        for _ in range(n):
            env_v, w_v = family_world(fam, visit, phase)
            if w_v != waters[phase] or not valid_world(env_v, w_v):
                return None
            song = witness_song(env_v, w_v, ROLE)
            if song is None:
                return None
            episodes.append((song, w_v))
            visit += 1
    schemas = []
    for phase, n in enumerate(counts):
        sch = {"name": f"S{phase+1}", "water": waters[phase],
               "support": n, "failures": []}
        if phase > 0:
            sch["supersedes"] = f"S{phase}"
        if phase + 1 < len(counts):
            nxt = waters[phase + 1]
            sch["failures"].append(
                f"a later episode found NO water at "
                f"({waters[phase][0]}, {waters[phase][1]}) -- "
                "superseded below.")
        schemas.append(sch)
    true_phase = len(counts) - 1
    return {"episodes": episodes, "schemas": schemas,
            "env": envs[true_phase], "water": waters[true_phase]}


def form_memory(fam: int, conflict: bool):
    env_a, w1 = family_world(fam, 0, 0)
    env_v, w2 = family_world(fam, 0, 1)
    if not (valid_world(env_a, w1) and valid_world(env_v, w2)
            and w1 != w2):
        return None
    song_a = witness_song(env_a, w1, ROLE)
    song_v = witness_song(env_v, w2, ROLE)
    if song_a is None or song_v is None:
        return None
    episodes = [(song_a, w1)] * 3
    schemas = [{"name": "S1", "water": w1, "support": 3,
                "failures": []}]
    if conflict:
        episodes = episodes + [(song_v, w2)]
        schemas[0]["failures"].append(
            "episode 4 found NO water at "
            f"({w1[0]}, {w1[1]}) -- the schema misled there.")
        schemas.append({"name": "E1", "water": w2, "support": 1,
                        "supersedes": "S1"})
        true_env, true_water = env_v, w2
    else:
        true_env, true_water = env_a, w1
    return {"episodes": episodes, "schemas": schemas,
            "env": true_env, "water": true_water}


# ── the LLM step ───────────────────────────────────────────────────

def observation_text() -> str:
    x, y = TRAVELER_START
    return (f"You are at ({x}, {y}). Your local view shows open "
            "cells, some hazards and walls; NO water is visible from "
            "here. You must commit to a target from memory.")


def parse_target(text: str) -> Optional[GridXY]:
    # thinking models emit prose before the answer: take the LAST
    # valid {"target": ...} object, falling back to the last [x, y]
    for m in reversed(re.findall(r"\{[^{}]*\}", text, re.S)):
        try:
            obj = json.loads(m)
            t = obj.get("target")
            if isinstance(t, list) and len(t) == 2:
                return (int(t[0]), int(t[1]))
        except (ValueError, TypeError):
            continue
    ms = re.findall(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", text)
    return (int(ms[-1][0]), int(ms[-1][1])) if ms else None


class StubBackend:
    """Plumbing mock: commits to the FIRST water coordinate mentioned
    in the prompt (deliberately naive recency-blind reader)."""
    model = "stub"

    def complete(self, prompt, *, system="", seed=0, max_tokens=64,
                 stop=None):
        m = re.search(r"water(?:\s+source)?\s+at\s*\(\s*(-?\d+)\s*,"
                      r"\s*(-?\d+)\s*\)", prompt, re.I)
        if m:
            return json.dumps({"target": [int(m.group(1)),
                                          int(m.group(2))]})
        return json.dumps({"target": [0, 0]})

    def summary(self):
        return {"model": self.model}


def run_layout(backend, fam: int, conflict: bool, seed: int,
               mode: str = "short") -> Optional[Dict[str, Any]]:
    formed = (form_memory_long(fam, conflict) if mode == "long"
              else form_memory(fam, conflict))
    if formed is None:
        return None
    contexts = {
        "none": "You have no stored memory of this world.",
        "raw": raw_context(formed["episodes"]),
        "ucsm": ucsm_context(formed["schemas"]),
    }
    row: Dict[str, Any] = {"fam": fam, "conflict": conflict,
                           "water": list(formed["water"]), "arms": {}}
    for arm, ctx in contexts.items():
        prompt = (observation_text() + "\n\n" + ctx
                  + "\n\nWhere is the water NOW? Answer with JSON.")
        out = backend.complete(prompt, system=SYSTEM, seed=seed,
                               max_tokens=MAX_TOKENS)
        target = parse_target(out)
        lock_correct = target == formed["water"]
        completed = False
        if target is not None:
            path, _ = dijkstra(formed["env"], TRAVELER_START, target,
                               ROLE)
            completed = (path is not None
                         and formed["env"].cell(*target) == WATER)
        row["arms"][arm] = {
            "target": list(target) if target else None,
            "lock_correct": lock_correct, "completed": completed,
            "prompt_chars": len(SYSTEM) + len(prompt),
            "ctx_chars": len(ctx),
            "raw_reply": out[-160:]}
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["stub", "ollama", "hf"],
                    default="stub")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--layouts", type=int, default=12)
    ap.add_argument("--mode", choices=["short", "long"],
                    default="short")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--no-think", action="store_true",
                    help="append the qwen3 /no_think switch to the "
                         "system prompt (thinking models exhaust the "
                         "answer budget on rumination)")
    ap.add_argument("--out", type=str, default="tmp/song_grammar/l1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "l1_registered.json"), "w") as f:
        json.dump({
            "mode": a.mode,
            "L1b_note": "long mode: 12-episode histories; chain "
                        "layouts move the water twice (stale majority "
                        "10:2 in the transcript); registered L1b.1 = "
                        "L1.2 with payload <= 15%; L1b.2 = L1.3",
            "L1.1": "no-memory lock precision <= 0.2",
            "L1.2": "ucsm precision >= raw overall, at <= 30% of "
                    "raw's MEMORY PAYLOAD chars (context section; the "
                    "fixed prompt scaffolding is constant and was "
                    "excluded before any LLM run -- the stub run "
                    "exercised plumbing only)",
            "L1.3": "conflict layouts: ucsm precision > raw's",
            "protocol": "W4 division of labour: LLM commits the "
                        "semantic target, waypoint primitive walks; "
                        "shared frame on the LLM substrate",
        }, f, indent=2)

    global SYSTEM, MAX_TOKENS
    MAX_TOKENS = a.max_tokens
    if a.no_think:
        SYSTEM = SYSTEM + " /no_think"
    if a.backend == "hf":
        from experiments.llm_collective.hf_backend import HFBackend
        backend = HFBackend(model=a.model,
                            cache_dir=os.path.join(a.out, ".cache"))
    elif a.backend == "ollama":
        from experiments.llm_collective.llm_backend import OllamaBackend
        backend = OllamaBackend(model=a.model,
                                cache_dir=os.path.join(a.out, ".cache"))
    else:
        backend = StubBackend()

    rows: List[Dict[str, Any]] = []
    fam, tries = 5000, 0
    while len(rows) < a.layouts and tries < a.layouts * 20:
        tries += 1
        fam += 1
        conflict = (len(rows) % 3 != 0 if a.mode == "long"
                    else len(rows) % 2 == 1)
        row = run_layout(backend, fam, conflict, seed=len(rows),
                         mode=a.mode)
        if row is not None:
            rows.append(row)
            arms = row["arms"]
            print(f"layout {len(rows)} (conflict={conflict}): "
                  + " ".join(f"{k}:{'OK' if v['lock_correct'] else 'x'}"
                             for k, v in arms.items()), flush=True)
    with open(os.path.join(a.out, "l1_rows.json"), "w") as f:
        json.dump(rows, f, indent=1)

    def prec(arm, subset=None):
        sel = [r for r in rows
               if subset is None or r["conflict"] == subset]
        return (sum(r["arms"][arm]["lock_correct"] for r in sel)
                / max(1, len(sel)))

    chars = {arm: sum(r["arms"][arm]["prompt_chars"] for r in rows)
             / len(rows) for arm in ("none", "raw", "ucsm")}
    ctx = {arm: sum(r["arms"][arm]["ctx_chars"] for r in rows)
           / len(rows) for arm in ("none", "raw", "ucsm")}
    l11 = prec("none") <= 0.2
    payload_cap = 0.15 if a.mode == "long" else 0.30
    l12 = (prec("ucsm") >= prec("raw")
           and ctx["ucsm"] <= payload_cap * ctx["raw"])
    l13 = prec("ucsm", True) > prec("raw", True)
    summary = {
        "model": backend.summary() if hasattr(backend, "summary")
        else {"model": a.backend},
        "n_layouts": len(rows),
        "lock_precision": {arm: prec(arm)
                           for arm in ("none", "raw", "ucsm")},
        "conflict_precision": {arm: prec(arm, True)
                               for arm in ("raw", "ucsm")},
        "base_precision": {arm: prec(arm, False)
                           for arm in ("raw", "ucsm")},
        "mean_prompt_chars": chars,
        "mean_ctx_chars": ctx,
        "ucsm_vs_raw_payload": ctx["ucsm"] / ctx["raw"],
    }
    verdict = {"L1.1_memory_is_needed": l11,
               "L1.2_reconstruction_beats_replay": l12,
               "L1.3_exceptions_work_in_context": l13}
    with open(os.path.join(a.out, "l1_results.json"), "w") as f:
        json.dump({"summary": summary, "verdict": verdict}, f,
                  indent=2)
    print(json.dumps(summary, indent=2))
    for k, v in verdict.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"Saved: {a.out}/l1_results.json")


if __name__ == "__main__":
    main()
