"""LLM action decider.

Contract: (observation, task, retrieved_candidates) -> chosen action.

The decider's output is a string action selected from the env's
allowed action vocabulary. We expose the candidate list (with their
tag profiles) to the LLM so the agent can reason over the materialized
target before acting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from experiments.llm_collective.llm_backend import OllamaBackend


SYSTEM_PROMPT = """You are an embodied agent's action selector. Choose
the single best action that makes progress on the task. Look at:
  - the observation (it tells you the room and direction hints)
  - the task (your goal)
  - the memory candidates (places you have visited and what tags they carry)

Decision rules:
  - If the observation says the goal object is here and can be taken,
    choose the matching take action.
  - If the observation says the goal destination is next to you and you
    are holding the goal object, choose the matching put action.
  - Otherwise follow explicit direction hints with the matching go action.

Output strictly two lines, no prose, no code fences:
ACTION=<one of the allowed actions, exactly>
TARGET=<id of the candidate you are heading towards, or NONE>

Examples:
ACTION=go_west
TARGET=p1_1

ACTION=go_east
TARGET=p1_1

ACTION=take_apple
TARGET=p1_1

ACTION=put_apple
TARGET=p7_1
"""


_ACTION_RE = re.compile(r"ACTION\s*=\s*([^\n;]+)", re.IGNORECASE)
_TARGET_RE = re.compile(r"TARGET\s*=\s*([^\n;]+)", re.IGNORECASE)


@dataclass
class Decision:
    action: str
    target_id: Optional[str]
    raw: str


class LLMDecider:
    def __init__(self, backend: Optional[OllamaBackend] = None) -> None:
        self.backend = backend or OllamaBackend()

    def decide(
        self,
        observation: str,
        task: str,
        candidates: List[Dict],
        allowed_actions: List[str],
        *,
        seed: int = 0,
    ) -> Decision:
        cand_lines = []
        for c in candidates[:5]:
            tags = ",".join(
                f"{k}:{v:.2f}" for k, v in sorted(
                    c.get("tags", {}).items(), key=lambda kv: -kv[1]
                )[:5]
            )
            cand_lines.append(
                f"  - id={c.get('id', '?')} place=({c.get('xy', (0,0))}) tags=[{tags}]"
            )
        cand_block = "\n".join(cand_lines) if cand_lines else "  (none)"
        allowed = ", ".join(allowed_actions)
        prompt = (
            f"Task: {task.strip()}\n"
            f"Observation: {observation.strip()}\n"
            f"Memory candidates (id, place, tags):\n{cand_block}\n"
            f"Allowed actions: {allowed}\n"
            "ACTION="
        )
        raw = self.backend.complete(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            seed=seed,
            max_tokens=80,
        )
        return self._parse(raw, allowed_actions)

    def _parse(self, raw: str, allowed: List[str]) -> Decision:
        m_act = _ACTION_RE.search(raw)
        m_tgt = _TARGET_RE.search(raw)
        chosen = (m_act.group(1).strip() if m_act else "").lower()
        # Snap to allowed action vocab (case-insensitive prefix match)
        action = ""
        for a in allowed:
            if chosen == a.lower() or chosen.startswith(a.lower()):
                action = a
                break
        if not action and allowed:
            # Permissive: pick first allowed action token appearing in raw
            for a in allowed:
                if re.search(rf"\b{re.escape(a)}\b", raw, re.IGNORECASE):
                    action = a
                    break
        if not action and allowed:
            # Prefer a benign no-op fallback if the LLM produced ambiguous text
            for noop in ("look", "noop", "stay"):
                if noop in [a.lower() for a in allowed]:
                    for a in allowed:
                        if a.lower() == noop:
                            action = a; break
                    if action:
                        break
            if not action:
                action = allowed[0]
        tgt_raw = (m_tgt.group(1).strip() if m_tgt else "")
        target_id = tgt_raw if tgt_raw and tgt_raw.upper() != "NONE" else None
        return Decision(action=action, target_id=target_id, raw=raw)
