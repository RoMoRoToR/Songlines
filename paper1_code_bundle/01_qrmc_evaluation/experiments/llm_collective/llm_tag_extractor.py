"""LLM-based symbolic tag extractor for natural-language observations.

Contract: NL scene text -> Dict[tag, confidence_in_0_to_1].

This is the only point where the LLM enters the memory pathway.
The downstream consumer is peer_memory.PeerAgent.observe(),
which already accepts a Dict[tag, float] without modification.

The extractor uses few-shot prompting and parses a strict
comma-separated list "tag:confidence,tag:confidence,...".
Output is robust to LLM stylistic variation (extra prose, code fences).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from experiments.llm_collective.llm_backend import OllamaBackend


# Canonical ALFWorld-flavoured vocabulary used as a soft hint to the LLM.
# Not enforced — LLM may emit other tags, which is intentional (open-set).
DEFAULT_VOCABULARY = [
    # locations
    "kitchen", "bedroom", "bathroom", "living_room", "office", "garage",
    # containers
    "fridge", "cabinet", "drawer", "shelf", "table", "counter", "sink",
    "microwave", "oven", "trashcan", "box",
    # objects
    "apple", "bread", "knife", "fork", "spoon", "plate", "cup", "mug",
    "book", "lamp", "key", "remote", "phone", "candle", "soap",
    # affordances / states
    "openable", "closeable", "graspable", "edible", "hot", "cold",
    "dirty", "clean", "broken", "locked", "lit", "off",
    # task-relevant
    "goal_target", "obstacle", "danger", "exit",
]


SYSTEM_PROMPT = """You are a precise symbolic tag extractor for an embodied agent.
Given a short scene description, you emit a list of semantic tags that
the agent's memory should record. Tags must be short snake_case words.
Each tag carries a confidence in [0,1]. Emit only the comma-separated
list, no prose, no code fences.

Example format:
tag1:0.9,tag2:0.7,tag3:0.5
"""


FEWSHOT_EXAMPLES = """\
Scene: You are in the kitchen. You see a fridge, an apple on the counter, and a sink.
Tags: kitchen:0.95,fridge:0.9,apple:0.9,counter:0.8,sink:0.85

Scene: A locked wooden door is in front of you. There is a small brass key on the floor.
Tags: door:0.95,locked:0.9,wooden:0.7,key:0.95,brass:0.7,floor:0.6

Scene: You enter the bathroom. A toothbrush is on the sink and a towel hangs from a rack.
Tags: bathroom:0.95,toothbrush:0.9,sink:0.85,towel:0.85,rack:0.7
"""


_TAG_PAIR_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9_\-]{0,30})\s*[:=]\s*([01](?:\.\d+)?|\.\d+)"
)


class LLMTagExtractor:
    def __init__(
        self,
        backend: Optional[OllamaBackend] = None,
        vocabulary: Optional[List[str]] = None,
        min_confidence: float = 0.3,
        max_tags: int = 12,
    ) -> None:
        self.backend = backend or OllamaBackend()
        self.vocabulary = vocabulary or DEFAULT_VOCABULARY
        self.min_confidence = float(min_confidence)
        self.max_tags = int(max_tags)

    # --- public ---------------------------------------------------------

    def extract(
        self,
        observation_text: str,
        *,
        seed: int = 0,
    ) -> Dict[str, float]:
        prompt = self._build_prompt(observation_text)
        raw = self.backend.complete(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            seed=seed,
            max_tokens=160,
        )
        return self._parse(raw)

    # --- internals ------------------------------------------------------

    def _build_prompt(self, observation_text: str) -> str:
        vocab_hint = ", ".join(self.vocabulary[:32])
        return (
            FEWSHOT_EXAMPLES
            + f"\nVocabulary hint (you may use these or others, snake_case): {vocab_hint}\n\n"
            + f"Scene: {observation_text.strip()}\n"
            + "Tags:"
        )

    def _parse(self, raw: str) -> Dict[str, float]:
        # Strip code fences if any
        raw = raw.replace("```", " ").strip()
        # First newline-truncated chunk only (model can ramble)
        head = raw.split("\n", 1)[0]
        # Find all tag:conf pairs in the head (and the tail as fallback)
        matches = _TAG_PAIR_RE.findall(head) or _TAG_PAIR_RE.findall(raw)
        out: Dict[str, float] = {}
        for tag, conf in matches:
            try:
                c = float(conf)
            except ValueError:
                continue
            if c < self.min_confidence:
                continue
            tag = tag.lower().strip()
            if not tag:
                continue
            # Keep the strongest occurrence
            if tag not in out or c > out[tag]:
                out[tag] = max(0.0, min(1.0, c))
            if len(out) >= self.max_tags:
                break
        return out
