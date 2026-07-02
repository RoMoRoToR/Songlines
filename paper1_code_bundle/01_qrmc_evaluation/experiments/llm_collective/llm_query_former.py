"""LLM-driven query former.

Contract: task description (NL) -> (required, preferred, penalty) tags.

These three sets are the same query interface as the symbolic planner.
The downstream consumer is the existing semantic retrieval weighted
log-score, which is reused without modification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from experiments.llm_collective.llm_backend import OllamaBackend


SYSTEM_PROMPT = """You are a precise query former for a memory-driven agent.
Given the agent's current task, produce a small retrieval query of
three short tag lists: required, preferred, penalty. Each list has up
to 4 snake_case tags. Output strict format on one line, no prose:

REQ=[t1,t2];PREF=[t3,t4];PEN=[t5]
"""


FEWSHOT = """\
Task: Bring an apple from the kitchen to the table in the living room.
REQ=[apple,kitchen];PREF=[graspable,edible];PEN=[locked,broken]

Task: Find a way out of the locked office.
REQ=[exit,door];PREF=[key,openable];PEN=[locked]

Task: Put the cold soda on the counter.
REQ=[soda,counter];PREF=[cold,graspable];PEN=[hot,broken]
"""


_LIST_RE = re.compile(r"\[([^\]]*)\]")
_FIELD_RE = re.compile(
    r"REQ\s*=\s*\[([^\]]*)\]\s*;\s*PREF\s*=\s*\[([^\]]*)\]\s*;\s*PEN\s*=\s*\[([^\]]*)\]",
    re.IGNORECASE,
)


@dataclass
class Query:
    required: List[str]
    preferred: List[str]
    penalty: List[str]

    def as_tuple(self) -> Tuple[List[str], List[str], List[str]]:
        return self.required, self.preferred, self.penalty

    def is_empty(self) -> bool:
        return not (self.required or self.preferred or self.penalty)


class LLMQueryFormer:
    def __init__(
        self,
        backend: Optional[OllamaBackend] = None,
        max_per_list: int = 4,
    ) -> None:
        self.backend = backend or OllamaBackend()
        self.max_per_list = int(max_per_list)

    def form(self, task_text: str, *, seed: int = 0) -> Query:
        prompt = FEWSHOT + f"\nTask: {task_text.strip()}\n"
        raw = self.backend.complete(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            seed=seed,
            max_tokens=80,
        )
        return self._parse(raw)

    def _parse(self, raw: str) -> Query:
        raw = raw.replace("```", " ").strip()
        first_line = next((line.strip() for line in raw.splitlines() if line.strip()), "")
        # Prefer the first generated line. Raw completions can continue with
        # copied few-shot examples after the answer; those must not override it.
        for text in (first_line, raw):
            m = _FIELD_RE.search(text)
            if m:
                req, pref, pen = m.group(1), m.group(2), m.group(3)
                return Query(
                    required=self._split(req),
                    preferred=self._split(pref),
                    penalty=self._split(pen),
                )
            # Permissive fallback: take first three [ ... ] groups as REQ/PREF/PEN
            groups = _LIST_RE.findall(text)
            if len(groups) >= 3:
                return Query(
                    required=self._split(groups[0]),
                    preferred=self._split(groups[1]),
                    penalty=self._split(groups[2]),
                )
        return Query(required=[], preferred=[], penalty=[])

    def _split(self, s: str) -> List[str]:
        out: List[str] = []
        for part in s.split(","):
            t = part.strip().lower()
            t = re.sub(r"[^a-z0-9_]", "", t)
            if t:
                out.append(t)
            if len(out) >= self.max_per_list:
                break
        return out
