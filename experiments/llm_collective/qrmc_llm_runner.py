"""Q/R/M/C runner over an LLM-driven agent.

Identical event semantics to experiments/big_experiment/runner.py:
  Q*  = query non-empty
  R*  = at least one candidate satisfies the task within eps of a real target
  M*  = decider locked onto a candidate that is within eps of a real target
  C*  = agent reached success (env returns success flag)

The point of this module is to demonstrate that the Q/R/M/C measurement
contract is agent-class agnostic: the runner reads off the same
per-tick events from an LLM agent, without modifying the canonical
runner. We deliberately re-implement the event loop here (rather than
monkey-patching) because the canonical runner expects the env's
custom multi-agent interface; the text-env interface differs slightly,
and the diagnostic claim is about *the events*, not about reusing the
same Python function.
"""

from __future__ import annotations

import dataclasses as dc
import time
from typing import Any, Dict, List, Optional, Tuple

from experiments.llm_collective.llm_backend import OllamaBackend
from experiments.llm_collective.peer_llm_agent import PeerLLMAgent
from experiments.llm_collective.textnav_env import TextNavEnv


EPS = 1.0  # adjacency-eps for the TextNav substrate (the multi-agent
           # grid uses 0.6 because targets sit on integer cells with a
           # continuous reach radius; TextNav is discrete with reach 1)


def _xy_close(a: Tuple[float, float], b: Tuple[float, float],
              tol: float = EPS) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) <= tol


@dc.dataclass
class TickLog:
    tick: int
    aid: str
    obs_text: str
    extracted_tags: Dict[str, float]
    query_req: List[str]
    query_pref: List[str]
    query_pen: List[str]
    n_candidates: int
    top_candidate_id: Optional[str]
    locked_target_xy: Optional[Tuple[int, int]]
    chosen_action: str
    Q: bool
    R: bool
    M: bool
    C_so_far: bool


@dc.dataclass
class EpisodeResult:
    seed: int
    succeeded: bool
    n_ticks: int
    q_star: bool
    r_star: bool
    m_star: bool
    c_star: bool
    n_Q: int
    n_R: int
    n_M: int
    ticks_played: int
    tick_logs: List[TickLog]
    wall_clock_s: float

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "succeeded": self.succeeded,
            "n_ticks": self.n_ticks,
            "q_star": int(self.q_star),
            "r_star": int(self.r_star),
            "m_star": int(self.m_star),
            "c_star": int(self.c_star),
            "n_Q": self.n_Q,
            "n_R": self.n_R,
            "n_M": self.n_M,
            "ticks_played": self.ticks_played,
            "wall_clock_s": round(self.wall_clock_s, 2),
        }


def run_one_episode(
    seed: int = 0,
    step_limit: int = 30,
    verbose: bool = False,
    backend: Optional[OllamaBackend] = None,
) -> EpisodeResult:
    env = TextNavEnv(step_limit=step_limit, seed=seed)
    agent = PeerLLMAgent(aid="a0", backend=backend)

    Q_star = False
    R_star = False
    M_star = False
    C_star = False
    n_Q = n_R = n_M = 0
    tick_logs: List[TickLog] = []

    real_targets: List[Tuple[int, int]] = [env.APPLE_XY, env.TABLE_XY]

    t0 = time.time()
    last_action_was_putapple = False
    for tick in range(step_limit):
        info_xy = env.agents["a0"].x, env.agents["a0"].y
        obs_text = env.observe_text("a0")

        # OBSERVE: tags go into memory at the current cell.
        # Also write low-confidence symbolic tags into neighbouring cells
        # the agent can see — this is the standard "observation radius"
        # convention used by the symbolic stack.
        tags = agent.observe(info_xy, obs_text, tick, seed=seed * 1000 + tick)
        for nxy, ntag in env.visible_neighborhood("a0"):
            if nxy != info_xy and ntag not in ("hall", "kitchen", "living_room"):
                # Inject a simple symbolic observation about the neighbour
                agent.memory.observe(nxy, {ntag: 0.7}, tick)

        # DECIDE: form query (cached after first formation), retrieve, decide
        action, query, candidates, locked = agent.decide(
            observation_text=obs_text,
            task_text=env.task_text,
            allowed_actions=env.allowed_actions,
            seed=seed * 1000 + tick,
        )

        # ── Q event: query non-empty (any of req/pref/pen)
        q_held = not query.is_empty()
        # ── R event: at least one candidate satisfies the task
        #             (within eps of a real semantic target)
        r_held = False
        top_cid = candidates[0]["id"] if candidates else None
        if candidates:
            for c in candidates[:5]:
                for w in real_targets:
                    if _xy_close((float(c["xy"][0]), float(c["xy"][1])),
                                 (float(w[0]), float(w[1]))):
                        r_held = True
                        break
                if r_held:
                    break
        # ── M event: decider locked target is within eps of a real target
        m_held = False
        lock_xy = None
        if locked is not None:
            lock_xy = locked["xy"]
            for w in real_targets:
                if _xy_close((float(lock_xy[0]), float(lock_xy[1])),
                             (float(w[0]), float(w[1]))):
                    m_held = True
                    break

        if q_held:
            n_Q += 1; Q_star = True
        if r_held:
            n_R += 1; R_star = True
        if m_held:
            n_M += 1; M_star = True

        if verbose:
            print(f"  t{tick:02d} pos={info_xy} obs={obs_text[:60]!r} "
                  f"tags={list(tags)[:3]} q_held={q_held} r_held={r_held} "
                  f"m_held={m_held} act={action}")

        tick_logs.append(TickLog(
            tick=tick, aid="a0", obs_text=obs_text,
            extracted_tags=tags,
            query_req=list(query.required), query_pref=list(query.preferred),
            query_pen=list(query.penalty),
            n_candidates=len(candidates),
            top_candidate_id=top_cid,
            locked_target_xy=lock_xy,
            chosen_action=action,
            Q=q_held, R=r_held, M=m_held,
            C_so_far=C_star,
        ))

        result = env.step({"a0": action})
        if result.all_succeeded:
            C_star = True
            break

    elapsed = time.time() - t0
    return EpisodeResult(
        seed=seed,
        succeeded=C_star,
        n_ticks=tick + 1,
        q_star=Q_star, r_star=R_star, m_star=M_star, c_star=C_star,
        n_Q=n_Q, n_R=n_R, n_M=n_M, ticks_played=tick + 1,
        tick_logs=tick_logs,
        wall_clock_s=elapsed,
    )
