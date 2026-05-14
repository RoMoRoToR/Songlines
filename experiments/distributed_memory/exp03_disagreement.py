"""Experiment 03 — inter-agent disagreement detection.

Two agents observe the same physical place but assign incompatible
dominant tags (water_source vs hazard_edge).  The consensus layer should:

  1. Still merge them spatially into one distributed concept
  2. Flag the disagreement explicitly
  3. Lower the inter_agent_agreement
  4. Lower the consensus_confidence

Setup
-----
scout-A sees CONTESTED as water_source (8 observations).
scout-B sees CONTESTED as hazard_edge (8 observations).
Spatial centroid will be the same place.

Expectations
------------
  - 1 aligned distributed concept at CONTESTED
  - disagreement_flags is non-empty
  - inter_agent_agreement < 0.5 (severe disagreement)
  - consensus_confidence < confidence of any agent's local belief
  - report.disagreements lists scout-A vs scout-B with water vs hazard tags

Usage::

    PYTHONPATH=. .venv/bin/python experiments/distributed_memory/exp03_disagreement.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from distributed_memory import DistributedRuntime

ENV_ID = "exp03-grid"
WATER_TAG = "water_source"
HAZARD_TAG = "hazard_edge"
CONTESTED: Tuple[int, int] = (4, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="tmp/distributed_exp03")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rt = DistributedRuntime(env_id=ENV_ID, consensus_radius=4.0,
                            consensus_profile_threshold=0.0)  # allow cross-tag merge
    rt.spawn_agent("scout-A", trust=1.0)
    rt.spawn_agent("scout-B", trust=1.0)

    # scout-A: water-only observations
    for i in range(8):
        rt.observe("scout-A", CONTESTED, {WATER_TAG: 0.95},
                   episode_id=1, step_idx=i, confidence=0.95)
    # scout-B: hazard-only observations of the SAME place
    for i in range(8):
        rt.observe("scout-B", CONTESTED, {HAZARD_TAG: 0.92},
                   episode_id=1, step_idx=i, confidence=0.92)

    report = rt.tick()
    out_path = os.path.join(args.out_dir, "exp03_summary.json")
    with open(out_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    # Find aligned concept
    aligned = [c for c in report.distributed_concepts if c.n_agents == 2]
    assert aligned, (
        f"Expected aligned concept despite tag conflict. "
        f"All: {[(c.consensus_id, c.n_agents, c.consensus_dominant_tag) for c in report.distributed_concepts]}"
    )
    dc = aligned[0]

    assert dc.disagreement_flags, "disagreement_flags must be non-empty"
    assert dc.inter_agent_agreement < 0.6, (
        f"Agreement should be low for incompatible tags: got "
        f"{dc.inter_agent_agreement:.3f}"
    )
    assert report.disagreements, "report.disagreements must list pairs"

    pair = report.disagreements[0]
    tags = {pair.tag_a, pair.tag_b}
    assert WATER_TAG in tags and HAZARD_TAG in tags, (
        f"Disagreement should be water_source vs hazard_edge, got {tags}"
    )

    # Confidence must be reduced relative to a hypothetical full-trust baseline.
    # Both agents have local_confidence around 0.9; with agreement < 0.6 the
    # consensus_confidence should be visibly lower.
    local_conf_max = max(c.local_confidence for c in dc.contributions)
    assert dc.consensus_confidence < local_conf_max * 0.75, (
        f"Consensus_conf {dc.consensus_confidence:.3f} should be << "
        f"local_conf_max {local_conf_max:.3f} due to disagreement"
    )

    print(
        f"✓ Exp03 passed  "
        f"disagreements={len(report.disagreements)}  "
        f"agreement={dc.inter_agent_agreement:.3f}  "
        f"consensus_conf={dc.consensus_confidence:.3f}  "
        f"flags={dc.disagreement_flags[0]}"
    )
    print(f"Summary → {out_path}")


if __name__ == "__main__":
    main()
