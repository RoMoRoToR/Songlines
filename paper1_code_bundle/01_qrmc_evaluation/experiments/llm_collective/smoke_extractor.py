"""Smoke test for the LLM tag extractor on 10 ALFWorld-style observations.

Acceptance:
  (a) Mean number of tags per scene >= 2
  (b) >= 50% of scenes mention at least one canonical vocabulary tag
  (c) Cache hit rate = 100% on the second pass (determinism)
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.llm_collective.llm_backend import OllamaBackend
from experiments.llm_collective.llm_tag_extractor import (
    LLMTagExtractor, DEFAULT_VOCABULARY,
)


SCENES = [
    "You are in the kitchen. You see a fridge, an apple on the counter, and a sink.",
    "You enter the bathroom. A toothbrush is on the sink and a towel hangs from a rack.",
    "A locked wooden door is in front of you. There is a small brass key on the floor.",
    "You are in the bedroom. The bed is unmade, a lamp is off on the nightstand.",
    "You see a microwave on the counter. There is bread inside it.",
    "You are in the living room. A remote control is on the table next to a candle.",
    "A trashcan stands by the wall. It is half full with paper.",
    "You enter the office. A book sits on the desk, the lamp is lit.",
    "A drawer is closed. A spoon and a fork rest on the kitchen counter.",
    "The garage is dark. You see a broken bicycle and an empty box.",
]


def main():
    backend = OllamaBackend()
    extractor = LLMTagExtractor(backend=backend)

    print("== Pass 1: extract from 10 scenes ==")
    results = []
    for i, scene in enumerate(SCENES):
        tags = extractor.extract(scene, seed=42 + i)
        results.append((scene, tags))
        print(f"  [{i:2d}] tags={tags}")

    print("\n== Pass 2: re-extract (cache check) ==")
    all_same = True
    for i, scene in enumerate(SCENES):
        again = extractor.extract(scene, seed=42 + i)
        if again != results[i][1]:
            all_same = False
            print(f"  MISMATCH on {i}: was {results[i][1]} now {again}")

    print("\n== Acceptance ==")
    total_tags = sum(len(t) for _, t in results)
    mean_tags = total_tags / len(results)
    canonical_hit_count = sum(
        1 for _, t in results
        if any(tag in DEFAULT_VOCABULARY for tag in t)
    )
    print(f"  Mean tags per scene: {mean_tags:.2f} (need >= 2)")
    print(f"  Canonical-hit scenes: {canonical_hit_count}/{len(SCENES)} "
          f"(need >= 5)")
    print(f"  Determinism (cache): {'OK' if all_same else 'FAIL'}")
    print(f"  Backend stats: {backend.summary()}")

    ok = (mean_tags >= 2.0 and canonical_hit_count >= 5 and all_same)
    print("\n  ACCEPTANCE:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
