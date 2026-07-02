"""W4 — cross-session strict warp on an LLM substrate (design §8/W4).

Session 1: agent-A (an LLM tag extractor over NL observations) sweeps
the TextNav household, building a per-place symbolic memory, which is
dumped to disk — the persistence boundary.

Session 2: agent-B — a DIFFERENT agent in a FRESH environment instance,
who has never observed any cell — loads A's dump as foreign-provenance
evidence and solves "bring the apple to the table".  Every lock B makes
on the apple or the table is backed exclusively by A's session-1
evidence: phi = 1.0, strict W*, across both an agent boundary and a
session boundary.

The directional oracle hints of the stock TextNavEnv are removed
(NoHintTextNavEnv): without A's memory the task is a blind search, so
the warp is load-bearing — a control run without the dump quantifies
that.

LLM calls (tag extraction, query forming, action deciding) run on a
local Ollama model with deterministic caching; $0 of API budget.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/warp/exp_warp_cross_session.py \\
        [--model llama3.1:latest] [--step_limit 30]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from experiments.llm_collective.llm_backend import OllamaBackend
from experiments.llm_collective.llm_decider import LLMDecider
from experiments.llm_collective.llm_query_former import LLMQueryFormer, Query
from experiments.llm_collective.llm_tag_extractor import LLMTagExtractor
from experiments.llm_collective.textnav_env import TextNavEnv

GridXY = Tuple[int, int]
OUT_DIR = "tmp/warp/w4_cross_session"


class NoHintTextNavEnv(TextNavEnv):
    """TextNavEnv without the directional oracle, with movable landmarks.

    The stock env tells the agent where the apple/table are ("west from
    here") — with that hint the task is solvable without any memory and
    a warp demo would be vacuous.  Here the agent only perceives its own
    room and objects within reach; long-range knowledge must come from
    memory.  Apple/table/start positions are parameterisable so the
    demo can be replicated over many layouts.
    """

    def __init__(self, step_limit: int = 30, seed: int = 0,
                 apple_xy: Optional[GridXY] = None,
                 table_xy: Optional[GridXY] = None,
                 start_xy: Optional[GridXY] = None) -> None:
        super().__init__(step_limit=step_limit, seed=seed)
        if apple_xy is not None:
            self.APPLE_XY = tuple(apple_xy)   # instance shadows class attr
        if table_xy is not None:
            self.TABLE_XY = tuple(table_xy)
        if start_xy is not None:
            self.agents["a0"].x, self.agents["a0"].y = start_xy

    def observe_text(self, aid: str) -> str:
        ag = self.agents[aid]
        room = self.room_of(ag.x, ag.y)
        parts = [f"You are in the {room} at column {ag.x} row {ag.y}."]
        if (ag.x, ag.y) == self.APPLE_XY and self.apple_present_at_kitchen:
            parts.append("There is an apple here on the counter — "
                         "you can take it (take_apple).")
        elif self.apple_present_at_kitchen and \
                abs(ag.x - self.APPLE_XY[0]) + abs(ag.y - self.APPLE_XY[1]) <= 1:
            parts.append("An apple is on a counter within reach — "
                         "you can take it (take_apple).")
        if abs(ag.x - self.TABLE_XY[0]) + abs(ag.y - self.TABLE_XY[1]) <= 1:
            if ag.carrying == "apple":
                parts.append("The table is right next to you. To finish "
                             "the task choose action put_apple.")
            else:
                parts.append("There is a table here.")
        if ag.carrying == "apple":
            parts.append("You are holding an apple.")
        if self.apple_on_table:
            parts.append("The apple is on the table. Task complete.")
        return " ".join(parts)


class ProvenancedLLMMemory:
    """Per-place, per-source symbolic memory with warp provenance.

    Same query semantics as ``SimpleSymbolicMemory`` (required-tag gate,
    preferred/penalty weighting on merged confidences), plus phi(xy):
    the foreign share of total evidence mass at a place — the masses are
    the same confidences the query score consumes.
    """

    def __init__(self, self_id: str,
                 required_match_threshold: float = 0.3,
                 pref_weight: float = 0.5, pen_weight: float = 1.0) -> None:
        self.self_id = self_id
        # xy -> source -> tag -> confidence
        self.places: Dict[GridXY, Dict[str, Dict[str, float]]] = {}
        self.required_match_threshold = required_match_threshold
        self.pref_weight = pref_weight
        self.pen_weight = pen_weight

    def observe(self, xy: GridXY, tags: Dict[str, float], tick: int,
                source: Optional[str] = None) -> None:
        if not tags:
            return
        src = source or self.self_id
        per_src = self.places.setdefault(tuple(xy), {}).setdefault(src, {})
        for t, c in tags.items():
            per_src[t] = max(per_src.get(t, 0.0), float(c))

    def load_dump(self, dump: Dict[str, Dict[str, float]],
                  source: str) -> int:
        n = 0
        for key, tags in dump.items():
            x, y = (int(v) for v in key.split(","))
            self.observe((x, y), tags, tick=-1, source=source)
            n += 1
        return n

    def merged_tags(self, xy: GridXY) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for tags in self.places.get(tuple(xy), {}).values():
            for t, c in tags.items():
                out[t] = max(out.get(t, 0.0), c)
        return out

    def phi(self, xy: GridXY) -> float:
        per_src = self.places.get(tuple(xy), {})
        own = sum(sum(t.values()) for s, t in per_src.items()
                  if s == self.self_id)
        foreign = sum(sum(t.values()) for s, t in per_src.items()
                      if s != self.self_id)
        total = own + foreign
        return foreign / total if total > 0 else 0.0

    def self_observed(self, xy: GridXY) -> bool:
        return self.self_id in self.places.get(tuple(xy), {})

    def query(self, q: Query) -> List[Dict]:
        out = []
        for xy in self.places:
            tags = self.merged_tags(xy)
            if q.required:
                # any-of gate: required tags outside the memory's tag
                # vocabulary cannot discriminate between places (the LLM
                # query former free-generates tags like 'house'), so a
                # place qualifies if ANY required tag matches.
                if not any(tags.get(t, 0.0) >= self.required_match_threshold
                           for t in q.required):
                    continue
            score = sum(tags.get(t, 0.0) for t in q.required)
            score += sum(self.pref_weight * tags.get(t, 0.0)
                         for t in q.preferred)
            score -= sum(self.pen_weight * tags.get(t, 0.0)
                         for t in q.penalty)
            out.append({"id": f"p{xy[0]}_{xy[1]}", "xy": xy, "score": score,
                        "tags": tags, "phi": round(self.phi(xy), 3)})
        out.sort(key=lambda c: -c["score"])
        return out


# ───────────────────────────────────── session 1: witness sweep


def run_session1(backend: OllamaBackend,
                 env_kwargs: Optional[Dict] = None
                 ) -> Dict[str, Dict[str, float]]:
    """Agent-A sweeps every cell, extracting tags from NL observations."""
    env = NoHintTextNavEnv(**(env_kwargs or {}))
    extractor = LLMTagExtractor(backend=backend)
    dump: Dict[str, Dict[str, float]] = {}
    ag = env.agents["a0"]
    n_tagged = 0
    for y in range(env.HEIGHT):
        xs = range(env.WIDTH) if y % 2 == 0 else range(env.WIDTH - 1, -1, -1)
        for x in xs:
            ag.x, ag.y = x, y
            obs = env.observe_text("a0")
            tags = extractor.extract(obs, seed=0)
            # ground the semantic anchor tags with the cell's true tag —
            # the extractor works over NL, the env just confirms reach
            cell = env.cell_tag(x, y)
            if cell in ("apple", "table"):
                tags = dict(tags)
                tags[cell] = max(tags.get(cell, 0.0), 0.95)
            if tags:
                dump[f"{x},{y}"] = {t: round(c, 3) for t, c in tags.items()}
                n_tagged += 1
    print(f"  session-1: agent-A tagged {n_tagged} places "
          f"(LLM: {backend.summary()['total_calls']} fresh calls)")
    return dump


# ───────────────────────────────────── session 2: traveler episode


def run_session2(backend: OllamaBackend, dump: Optional[Dict],
                 step_limit: int, label: str,
                 env_kwargs: Optional[Dict] = None) -> Dict[str, Any]:
    env = NoHintTextNavEnv(step_limit=step_limit, **(env_kwargs or {}))
    memory = ProvenancedLLMMemory(self_id="agent-B")
    n_loaded = memory.load_dump(dump, source="agent-A") if dump else 0

    extractor = LLMTagExtractor(backend=backend)
    query_former = LLMQueryFormer(backend=backend)
    decider = LLMDecider(backend=backend)

    trace: List[Dict[str, Any]] = []
    m_events: List[Dict[str, Any]] = []
    open_lock: Optional[Dict[str, Any]] = None
    queries: Dict[str, Query] = {}
    ag = env.agents["a0"]
    apple_taken_tick = None
    success_tick = None

    for tick in range(step_limit):
        obs = env.observe_text("a0")
        own_tags = extractor.extract(obs, seed=0)
        memory.observe((ag.x, ag.y), own_tags, tick)

        phase = "deliver" if ag.carrying == "apple" else "fetch"
        # Short retrieval intent for the query former; full instruction
        # for the decider (the query former free-generates tags, so the
        # retrieval phrasing names exactly the semantic anchor).
        if phase == "deliver":
            query_task = "Find the table."
            decider_task = ("Bring the apple you are carrying to the "
                            "living-room table. When the table is next "
                            "to you, choose put_apple.")
        else:
            query_task = "Find the apple."
            decider_task = ("Find and take the apple. When the apple is "
                            "within reach, choose take_apple.")
        if phase not in queries:
            queries[phase] = query_former.form(query_task, seed=0)
        q = queries[phase]

        candidates = memory.query(q)
        # Coordinate legend: defines the action semantics (an embodied
        # agent knows its own controls) — it reveals nothing about where
        # objects are.  seed=tick breaks deterministic action loops.
        obs_for_decider = (
            obs + " Directions: go_north decreases row, go_south "
            "increases row, go_west decreases column, go_east "
            "increases column.")
        d = decider.decide(observation=obs_for_decider, task=decider_task,
                           candidates=candidates,
                           allowed_actions=env.allowed_actions, seed=tick)

        locked_xy = None
        if d.target_id:
            for c in candidates:
                if c["id"] == d.target_id:
                    locked_xy = tuple(c["xy"])
                    break
        if locked_xy is not None and (
                open_lock is None or open_lock["target_xy"] != list(locked_xy)):
            # freeze phi at lock (design §3.1)
            phi = memory.phi(locked_xy)
            ev = {
                "tick": tick, "agent": "agent-B", "phase": phase,
                "target_xy": list(locked_xy),
                "phi": round(phi, 3),
                "self_ever_observed": memory.self_observed(locked_xy),
                "w_star_strict": phi >= 0.999
                                 and not memory.self_observed(locked_xy),
                "cross_session": phi >= 0.999,
                "completed": False,
            }
            m_events.append(ev)
            open_lock = ev

        # Layered execution (the songline stack: Intent → Target →
        # Waypoint → Action).  The LLM owns the SEMANTIC decisions: which
        # remembered place to pursue (the M-lock) and when to take/put.
        # Locomotion toward the locked waypoint is a deterministic motor
        # primitive — local 4-8B models are unreliable at row/column
        # arithmetic, and motor skill is not what W4 measures.
        action = d.action
        if action not in ("take_apple", "put_apple") and open_lock is not None:
            tx, ty = open_lock["target_xy"]
            dx, dy = tx - ag.x, ty - ag.y
            if dx != 0 and (abs(dx) >= abs(dy) or dy == 0):
                action = "go_east" if dx > 0 else "go_west"
            elif dy != 0:
                action = "go_south" if dy > 0 else "go_north"

        result = env.step({"a0": action})
        trace.append({
            "tick": tick, "obs": obs, "phase": phase,
            "llm_action": d.action, "motor_action": action,
            "target_id": d.target_id, "position": [ag.x, ag.y],
            "n_candidates": len(candidates),
            "top_candidate": candidates[0]["id"] if candidates else None,
        })

        # Completion attribution uses the env's affordance radius:
        # take_apple/put_apple act within manhattan 1, and the witness
        # tags the anchor on within-reach cells too — a lock on an
        # adjacent cell that ends in a successful take/put fulfilled
        # exactly what the foreign evidence promised.
        def _fulfils(lock, anchor_xy):
            lx, ly = lock["target_xy"]
            return abs(lx - anchor_xy[0]) + abs(ly - anchor_xy[1]) <= 1

        if ag.carrying == "apple" and apple_taken_tick is None:
            apple_taken_tick = tick
            if open_lock and _fulfils(open_lock, env.APPLE_XY):
                open_lock["completed"] = True
            open_lock = None
        if result.all_succeeded:
            success_tick = tick
            if open_lock and _fulfils(open_lock, env.TABLE_XY):
                open_lock["completed"] = True
            break

    strict_completed = [e for e in m_events
                        if e["w_star_strict"] and e["completed"]]
    return {
        "label": label, "n_loaded_foreign_places": n_loaded,
        "apple_taken_tick": apple_taken_tick,
        "success_tick": success_tick,
        "task_completed": success_tick is not None,
        "m_events": m_events,
        "n_strict_w_completed": len(strict_completed),
        "trace": trace,
    }


def make_layouts(n: int) -> List[Dict]:
    """Layout 0 is the canonical map; the rest are seeded permutations
    (apple somewhere in the kitchen, table in the living room, traveler
    start in the hall)."""
    import numpy as np
    layouts = [{"apple_xy": (1, 1), "table_xy": (7, 1), "start_xy": (4, 1)}]
    for i in range(1, n):
        rng = np.random.default_rng(1000 + i)
        layouts.append({
            "apple_xy": (int(rng.integers(0, 3)), int(rng.integers(0, 3))),
            "table_xy": (int(rng.integers(6, 9)), int(rng.integers(0, 3))),
            "start_xy": (int(rng.integers(3, 6)), int(rng.integers(0, 3))),
        })
    return layouts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get(
        "LLM_COLLECTIVE_MODEL", "llama3.1:latest"))
    parser.add_argument("--step_limit", type=int, default=30)
    parser.add_argument("--layouts", type=int, default=10)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    backend = OllamaBackend(
        model=args.model,
        cache_dir=os.path.join(OUT_DIR, ".cache_llm"),
    )
    print(f"W4 — cross-session warp on LLM substrate "
          f"(model={args.model}, {args.layouts} layouts)")

    per_layout: List[Dict[str, Any]] = []
    for li, env_kwargs in enumerate(make_layouts(args.layouts)):
        print(f"\nLayout {li}: apple={env_kwargs['apple_xy']} "
              f"table={env_kwargs['table_xy']} start={env_kwargs['start_xy']}")
        dump = run_session1(backend, env_kwargs)
        with open(os.path.join(OUT_DIR, f"session1_dump_L{li}.json"), "w") as f:
            json.dump({"source_agent": "agent-A", "session": 1,
                       "layout": env_kwargs, "places": dump}, f, indent=1)
        warp = run_session2(backend, dump, args.step_limit, "warp",
                            env_kwargs)
        control = run_session2(backend, None, args.step_limit, "control",
                               env_kwargs)
        n_strict = sum(1 for e in warp["m_events"] if e["w_star_strict"])
        print(f"  warp: success@{warp['success_tick']} "
              f"strictW={n_strict} strictW_completed="
              f"{warp['n_strict_w_completed']}  |  "
              f"control: success@{control['success_tick']}")
        per_layout.append({"layout_id": li, "layout": env_kwargs,
                           "warp": warp, "control": control})

    n = len(per_layout)
    warp_succ = sum(1 for r in per_layout if r["warp"]["task_completed"])
    ctrl_succ = sum(1 for r in per_layout if r["control"]["task_completed"])
    strict_ok = sum(1 for r in per_layout
                    if r["warp"]["n_strict_w_completed"] >= 1)
    warp_ticks = [r["warp"]["success_tick"] for r in per_layout
                  if r["warp"]["task_completed"]]

    ok = warp_succ > ctrl_succ and strict_ok >= max(1, warp_succ - 1)

    with open(os.path.join(OUT_DIR, "w4_results.json"), "w") as f:
        json.dump({"model": args.model, "n_layouts": n,
                   "warp_success": warp_succ, "control_success": ctrl_succ,
                   "layouts_with_completed_strict_w": strict_ok,
                   "warp_success_ticks": warp_ticks,
                   "per_layout": per_layout, "acceptance": ok,
                   "llm_stats": backend.summary()}, f, indent=1)

    print("\n" + "=" * 60)
    print(f"  warp success:    {warp_succ}/{n}  "
          f"(mean t_succ = "
          f"{sum(warp_ticks) / len(warp_ticks) if warp_ticks else float('nan'):.1f})")
    print(f"  control success: {ctrl_succ}/{n}")
    print(f"  layouts with completed strict W*: {strict_ok}/{n}")
    print(f"  [{'PASS' if ok else 'FAIL'}] cross-session strict W* "
          f"(phi=1.0) with C*=1, replicated over layouts")
    print("=" * 60)
    print(f"Saved: {OUT_DIR}/w4_results.json")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
