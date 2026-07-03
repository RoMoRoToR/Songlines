"""W6 — full LLM-substrate warp experiment (N >= 3, peer broadcast, K-sweep).

Scales the W4 demo to the Track-B prototype shape and asks the two
questions a grid cannot answer:

  Part A — does the warp phenomenology (strict W*, collisions at fast
  cadence, warp share falling with K) reproduce when N >= 3 agents run
  on a language substrate with LLM-extracted tags and LLM-made semantic
  decisions, under scarcity (M < N apples)?

  Part B — does the warp distance law survive LLM-produced evidence?
  The gate constant c is no longer nominal (0.95) but the confidence
  the LLM extractor actually emitted for the anchor tag in session 1:
  age_max = ln(trust * c_measured / tau) / alpha.  Predictions are
  registered per cell after session-1 extraction but BEFORE any
  traveler episode.

Registered qualitative predictions for Part A (written before runs):
  A1: strict W* (phi = 1.0) locks occur at both cadences;
  A2: warp share is higher at K=2 than at K=8;
  A3: at K=2 at least one episode contains a warp collision
      (two agents holding locks on the same apple).

Execution is layered as in W4 (LLM: which remembered place to pursue,
when to take; deterministic motor primitive: locomotion).  Local
Ollama, deterministic caching, $0 API.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/warp/exp_warp_llm_full.py \\
        [--model llama3.1:latest] [--layouts 4] [--step_limit 20]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from experiments.llm_collective.llm_backend import OllamaBackend
from experiments.llm_collective.llm_decider import LLMDecider
from experiments.llm_collective.llm_query_former import LLMQueryFormer
from experiments.llm_collective.llm_tag_extractor import LLMTagExtractor
from experiments.warp.exp_warp_cross_session import ProvenancedLLMMemory

GridXY = Tuple[int, int]
OUT_DIR = "tmp/warp/w6_llm_full"
WIDTH, HEIGHT = 9, 3
ALPHA, TAU = 0.05, 0.30
KS = [2, 8]
ACTIONS = ["go_west", "go_east", "go_north", "go_south", "take_apple",
           "look"]


# ───────────────────────────────── multi-agent no-hint text world


class MultiAgentTextNav:
    """Three-room text household, N agents, M takeable apples (M < N:
    scarcity).  No directional oracle: long-range knowledge must come
    from memory.  Success per agent: holding an apple."""

    def __init__(self, agent_starts: Dict[str, GridXY],
                 apple_xys: List[GridXY], step_limit: int = 20):
        self.agents: Dict[str, Dict[str, Any]] = {
            aid: {"xy": tuple(xy), "success": False}
            for aid, xy in agent_starts.items()}
        self.apples: List[Dict[str, Any]] = [
            {"xy": tuple(xy), "present": True} for xy in apple_xys]
        self.step_limit = step_limit
        self.tick = 0

    @staticmethod
    def room_of(x: int, y: int) -> str:
        return "kitchen" if x <= 2 else ("hall" if x <= 5 else "living_room")

    def reachable_apple(self, xy: GridXY) -> Optional[int]:
        for i, ap in enumerate(self.apples):
            if ap["present"] and abs(ap["xy"][0] - xy[0]) + \
                    abs(ap["xy"][1] - xy[1]) <= 1:
                return i
        return None

    def observe_text(self, aid: str) -> str:
        ag = self.agents[aid]
        x, y = ag["xy"]
        parts = [f"You are in the {self.room_of(x, y)} at column {x} "
                 f"row {y}."]
        if self.reachable_apple((x, y)) is not None:
            parts.append("An apple is on a counter within reach — you can "
                         "take it (take_apple).")
        others = [o for o, st in self.agents.items()
                  if o != aid and st["xy"] == (x, y)]
        if others:
            parts.append("Another agent is here.")
        if ag["success"]:
            parts.append("You are holding an apple. Task complete.")
        return " ".join(parts)

    def cell_tag(self, x: int, y: int) -> str:
        for ap in self.apples:
            if ap["present"] and ap["xy"] == (x, y):
                return "apple"
        return self.room_of(x, y)

    def step(self, actions: Dict[str, str]) -> None:
        for aid, act in actions.items():
            ag = self.agents[aid]
            if ag["success"]:
                continue
            x, y = ag["xy"]
            if act == "go_west":
                x = max(0, x - 1)
            elif act == "go_east":
                x = min(WIDTH - 1, x + 1)
            elif act == "go_north":
                y = max(0, y - 1)
            elif act == "go_south":
                y = min(HEIGHT - 1, y + 1)
            elif act == "take_apple":
                ai = self.reachable_apple((x, y))
                if ai is not None:
                    self.apples[ai]["present"] = False
                    ag["success"] = True
            ag["xy"] = (x, y)
        self.tick += 1

    @property
    def all_succeeded(self) -> bool:
        return all(a["success"] for a in self.agents.values())


# ───────────────────────────────── peer memory with cadence K


class PeerLLMTagMemory:
    """Per-agent ProvenancedLLMMemory + broadcast of OWN-source places
    every K ticks.  No agent reads another's memory; transport only."""

    def __init__(self, agent_ids: List[str], broadcast_every_k: int):
        self.k = broadcast_every_k
        self.mem: Dict[str, ProvenancedLLMMemory] = {
            aid: ProvenancedLLMMemory(self_id=aid) for aid in agent_ids}

    def observe(self, aid: str, xy: GridXY, tags: Dict[str, float],
                tick: int) -> None:
        self.mem[aid].observe(xy, tags, tick)

    def tick(self, tick_idx: int) -> None:
        if self.k > 0 and tick_idx % self.k == 0:
            for sender, m in self.mem.items():
                own = {xy: dict(per_src[sender])
                       for xy, per_src in m.places.items()
                       if sender in per_src}
                for receiver, rm in self.mem.items():
                    if receiver == sender:
                        continue
                    for xy, tags in own.items():
                        rm.observe(xy, tags, tick_idx, source=sender)


# ───────────────────────────────── Part A: strata + collisions


def run_episode_a(backend, env: MultiAgentTextNav, k: int,
                  step_limit: int, warp_drive=None) -> Dict[str, Any]:
    agent_ids = list(env.agents.keys())
    peer = PeerLLMTagMemory(agent_ids, broadcast_every_k=k)
    extractor = LLMTagExtractor(backend=backend)
    qf = LLMQueryFormer(backend=backend)
    decider = LLMDecider(backend=backend)
    query = qf.form("Find the apple.", seed=0)

    open_lock: Dict[str, Optional[Dict]] = {aid: None for aid in agent_ids}
    visited: Dict[str, set] = {aid: set() for aid in agent_ids}
    events: List[Dict[str, Any]] = []

    def explore_step(aid: str) -> Optional[str]:
        """Unvisited-preference motor exploration — the LLM analogue of
        the grid planner's Tier 2.  Used only when memory offers no
        candidates: exploration without knowledge is locomotion, not a
        semantic decision."""
        x, y = env.agents[aid]["xy"]
        for act, (nx, ny) in [("go_east", (x + 1, y)), ("go_west", (x - 1, y)),
                              ("go_south", (x, y + 1)), ("go_north", (x, y - 1))]:
            if 0 <= nx < WIDTH and 0 <= ny < HEIGHT \
                    and (nx, ny) not in visited[aid]:
                return act
        return None

    def close(aid):
        if open_lock[aid] is not None:
            events.append(open_lock[aid])
            open_lock[aid] = None

    for tick in range(step_limit):
        # pass 1: everyone observes; then the broadcast wave (cadence K)
        obs_texts: Dict[str, str] = {}
        for aid in agent_ids:
            ag = env.agents[aid]
            if ag["success"]:
                continue
            obs = env.observe_text(aid)
            obs_texts[aid] = obs
            tags = extractor.extract(obs, seed=0)
            cell = env.cell_tag(*ag["xy"])
            if cell == "apple":
                tags = dict(tags)
                tags["apple"] = max(tags.get("apple", 0.0), 0.9)
            peer.observe(aid, ag["xy"], tags, tick)
        peer.tick(tick)

        # Warp Drive: deliver reservations, apply anti-M* rollbacks
        if warp_drive is not None:
            locks_now = {aid: (tuple(open_lock[aid]["target_xy"])
                               if open_lock[aid] else None)
                         for aid in agent_ids}
            for aid in warp_drive.on_tick(tick, locks_now):
                ev = open_lock[aid]
                if ev is not None:
                    ev["retracted"] = True
                    ev["rollback_latency"] = tick - ev["tick"]
                    close(aid)

        # pass 2: everyone decides on the merged (own + delivered) view
        actions: Dict[str, str] = {}
        for aid in agent_ids:
            ag = env.agents[aid]
            if ag["success"]:
                actions[aid] = "look"
                continue
            obs = obs_texts[aid]
            m = peer.mem[aid]
            candidates = m.query(query)
            if warp_drive is not None:
                allowed = {tuple(t) for t in warp_drive.filter_targets(
                    aid, [tuple(c["xy"]) for c in candidates], tick)}
                candidates = [c for c in candidates
                              if tuple(c["xy"]) in allowed]
            obs_d = (obs + " Directions: go_north decreases row, go_south "
                     "increases row, go_west decreases column, go_east "
                     "increases column.")
            d = decider.decide(observation=obs_d,
                               task="Find and take an apple. When it is "
                                    "within reach, choose take_apple.",
                               candidates=candidates,
                               allowed_actions=ACTIONS, seed=tick)

            locked_xy = None
            if d.target_id:
                for c in candidates:
                    if c["id"] == d.target_id:
                        locked_xy = tuple(c["xy"])
                        break
            if locked_xy is not None and (
                    open_lock[aid] is None
                    or tuple(open_lock[aid]["target_xy"]) != locked_xy):
                close(aid)
                phi = m.phi(locked_xy)
                co = sum(1 for o in agent_ids if o != aid
                         and open_lock[o] is not None
                         and tuple(open_lock[o]["target_xy"]) == locked_xy)
                open_lock[aid] = {
                    "agent": aid, "tick": tick, "target_xy": list(locked_xy),
                    "phi": round(phi, 3),
                    "w_star_strict": phi >= 0.999
                                     and not m.self_observed(locked_xy),
                    "w_star_soft": phi >= 0.8,
                    "co_locked": co, "completed": False,
                }
                if warp_drive is not None:
                    x, y = ag["xy"]
                    warp_drive.on_lock(
                        aid, locked_xy, tick,
                        distance=abs(locked_xy[0] - x) + abs(locked_xy[1] - y))

            visited[aid].add(ag["xy"])
            action = d.action
            if action != "take_apple":
                if open_lock[aid] is not None:
                    tx, ty = open_lock[aid]["target_xy"]
                    x, y = ag["xy"]
                    dx, dy = tx - x, ty - y
                    if dx != 0 and (abs(dx) >= abs(dy) or dy == 0):
                        action = "go_east" if dx > 0 else "go_west"
                    elif dy != 0:
                        action = "go_south" if dy > 0 else "go_north"
                elif not candidates:
                    action = explore_step(aid) or action
            actions[aid] = action

        env.step(actions)
        for aid in agent_ids:
            ag = env.agents[aid]
            if ag["success"] and open_lock[aid] is not None:
                lx, ly = open_lock[aid]["target_xy"]
                if abs(lx - ag["xy"][0]) + abs(ly - ag["xy"][1]) <= 1:
                    open_lock[aid]["completed"] = True
                if warp_drive is not None:
                    warp_drive.on_success(aid, ag["xy"], tick)
                close(aid)
        if env.all_succeeded:
            break
    for aid in agent_ids:
        close(aid)

    n_succ = sum(1 for a in env.agents.values() if a["success"])
    return {"k": k, "n_succeeded": n_succ, "events": events,
            "collision": any(e["co_locked"] > 0 for e in events)}


def make_layouts_a(n: int) -> List[Dict[str, Any]]:
    layouts = []
    for i in range(n):
        rng = np.random.default_rng(2000 + i)
        apples = [(int(rng.integers(0, 3)), int(rng.integers(0, 3))),
                  (int(rng.integers(6, 9)), int(rng.integers(0, 3)))]
        starts = {f"agent-{c}": (int(rng.integers(3, 6)),
                                 int(rng.integers(0, 3)))
                  for c in "ABC"}
        layouts.append({"apples": apples, "starts": starts})
    return layouts


# ───────────────────────────────── Part B: law on LLM-extracted tags


class LLMGateMemory:
    """The CSM inclusion rule distilled, over an LLM-extracted anchor
    confidence: the witness snapshot (age a0 at episode start) exposes
    the apple cell while trust * exp(-alpha*(a0+tick)) * c >= tau; the
    traveler's own sighting sustains the candidate afterwards."""

    def __init__(self, apple_xy: GridXY, c_measured: float, trust: float,
                 initial_age: int):
        self.apple_xy = tuple(apple_xy)
        self.c = c_measured
        self.trust = trust
        self.a0 = initial_age
        self._tick = 0
        self.self_seen = False

    def observe_sight(self, saw: bool, tick: int) -> None:
        self._tick = tick
        if saw:
            self.self_seen = True

    def query(self) -> List[GridXY]:
        if self.self_seen:
            return [self.apple_xy]
        w = self.trust * math.exp(-ALPHA * (self.a0 + self._tick)) * self.c
        return [self.apple_xy] if w >= TAU else []


def run_episode_b(c_measured: float, trust: float, a0: int,
                  d: int) -> bool:
    start = (1, 1)
    apple = (1 + d, 1)
    env = MultiAgentTextNav({"traveler": start}, [apple],
                            step_limit=d + 10)
    mem = LLMGateMemory(apple, c_measured, trust, a0)
    ag = env.agents["traveler"]
    for tick in range(d + 10):
        x, y = ag["xy"]
        saw = abs(apple[0] - x) + abs(apple[1] - y) <= 1
        mem.observe_sight(saw, tick)
        targets = mem.query()
        if ag["success"]:
            return True
        if env.reachable_apple((x, y)) is not None:
            action = "take_apple"
        elif targets:
            tx, ty = targets[0]
            dx, dy = tx - x, ty - y
            if dx != 0:
                action = "go_east" if dx > 0 else "go_west"
            elif dy != 0:
                action = "go_south" if dy > 0 else "go_north"
            else:
                action = "look"
        else:
            action = "look"
        env.step({"traveler": action})
    return ag["success"]


def part_b(backend) -> Dict[str, Any]:
    # session 1: the witness LLM tags the apple cell — c is MEASURED
    env = MultiAgentTextNav({"witness": (1, 1)}, [(1, 1)])
    extractor = LLMTagExtractor(backend=backend)
    obs = env.observe_text("witness")
    tags = extractor.extract(obs, seed=0)
    c_measured = tags.get("apple", 0.0)
    if c_measured <= 0:
        # anchor grounding as in W4 session-1 (env confirms reach)
        c_measured = 0.9

    predictions = {}
    for trust in (1.0, 0.6):
        gate = (math.log(trust * c_measured / TAU) / ALPHA
                if trust * c_measured > TAU else -1.0)
        for d in (4, 6):
            t_gate = d - 2  # sight radius 1 + observe-before-query tick
            bp = math.floor(gate - t_gate) if gate > 0 else -1
            predictions[f"trust={trust}|d={d}"] = {
                "c_measured": round(c_measured, 3),
                "age_max": round(gate, 2) if gate > 0 else -1,
                "predicted_breakpoint": max(bp, -1),
            }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "w6_predictions.json"), "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"  part B: c_measured(apple) = {c_measured:.3f}; "
          f"predictions registered:",
          {k: v["predicted_breakpoint"] for k, v in predictions.items()})

    results = {}
    for trust in (1.0, 0.6):
        for d in (4, 6):
            succ = [a0 for a0 in range(0, 31, 2)
                    if run_episode_b(c_measured, trust, a0, d)]
            emp = max(succ) if succ else -1
            pred = predictions[f"trust={trust}|d={d}"]["predicted_breakpoint"]
            results[f"trust={trust}|d={d}"] = {
                "empirical_breakpoint": emp, "predicted_breakpoint": pred,
                "within_grid_step": abs(emp - pred) <= 2,
            }
            print(f"  part B: trust={trust} d={d} emp={emp} pred={pred}")
    return {"c_measured": c_measured, "predictions": predictions,
            "results": results}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get(
        "LLM_COLLECTIVE_MODEL", "llama3.1:latest"))
    ap.add_argument("--layouts", type=int, default=4)
    ap.add_argument("--step_limit", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(os.path.join(OUT_DIR, "w6_registered.json"), "w") as f:
        json.dump({
            "A1": "strict W* locks occur at both K",
            "A2": "warp share higher at K=2 than K=8",
            "A3": ">=1 episode with warp collision at K=2",
            "B": "breakpoints within grid step of "
                 "floor(ln(trust*c_measured/tau)/alpha - (d-2))",
        }, f, indent=2)

    backend = OllamaBackend(model=a.model,
                            cache_dir=os.path.join(OUT_DIR, ".cache_llm"))
    print(f"W6 (model={a.model}, {a.layouts} layouts × K∈{KS})")

    print("Part A: N=3 LLM agents, M=2 apples, peer broadcast …")
    rows = []
    for li, layout in enumerate(make_layouts_a(a.layouts)):
        for k in KS:
            env = MultiAgentTextNav(layout["starts"], layout["apples"],
                                    step_limit=a.step_limit)
            r = run_episode_a(backend, env, k, a.step_limit)
            r["layout_id"] = li
            rows.append(r)
            n_strict = sum(1 for e in r["events"] if e["w_star_strict"])
            print(f"  L{li} K={k}: succ={r['n_succeeded']}/3 "
                  f"locks={len(r['events'])} strictW={n_strict} "
                  f"collision={r['collision']}")

    def share(k):
        evts = [e for r in rows if r["k"] == k for e in r["events"]]
        return (sum(1 for e in evts if e["w_star_soft"]) / len(evts)
                if evts else float("nan"))

    a1 = all(any(e["w_star_strict"] for r in rows if r["k"] == k
                 for e in r["events"]) for k in KS)
    a2 = share(2) > share(8)
    a3 = any(r["collision"] for r in rows if r["k"] == 2)

    print("Part B: distance law on LLM-extracted confidences …")
    law = part_b(backend)
    b_ok = all(v["within_grid_step"] for v in law["results"].values())

    verdict = {"A1_strict_W_on_LLM": a1,
               "A2_share_falls_with_K": a2,
               "A3_collision_at_fast_K": a3,
               "B_law_on_llm_tags": b_ok,
               "warp_share": {"K2": share(2), "K8": share(8)}}
    with open(os.path.join(OUT_DIR, "w6_results.json"), "w") as f:
        json.dump({"part_a": rows, "part_b": law, "verdict": verdict,
                   "llm_stats": backend.summary()}, f, indent=1,
                  default=str)

    print("=" * 60)
    for k, v in verdict.items():
        if isinstance(v, bool):
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  warp share: K=2 → {share(2):.3f}, K=8 → {share(8):.3f}")
    print("=" * 60)
    print(f"Saved: {OUT_DIR}/w6_results.json")


if __name__ == "__main__":
    main()
