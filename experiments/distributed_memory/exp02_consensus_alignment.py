"""Experiment 02 — consensus alignment across agents.

Two agents observe the same physical place with compatible tags.  After
``tick()``, the consensus layer should merge their local concepts into
ONE distributed concept with both as contributors.

Setup
-----
scout-A observes place (3, 4) with ``water_source`` (8 times).
scout-B observes the same place (3, 4) with ``water_source`` (6 times).
Both also observe a private place to verify isolated concepts pass through.

Expectations
------------
  - ConsensusReport has at least 1 distributed concept with n_agents == 2
  - That concept has consensus_dominant_tag == "water_source"
  - Its centroid is near (3, 4)
  - n_aligned >= 1; n_isolated >= 2 (the two private concepts)
  - avg_agreement ≈ 1.0 (no disagreement)

Usage::

    PYTHONPATH=. .venv/bin/python experiments/distributed_memory/exp02_consensus_alignment.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from distributed_memory import DistributedRuntime

ENV_ID = "exp02-grid"
WATER_TAG = "water_source"
SHARED_PLACE: Tuple[int, int] = (3, 4)
PRIVATE_A: Tuple[int, int] = (0, 0)
PRIVATE_B: Tuple[int, int] = (9, 7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="tmp/distributed_exp02")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rt = DistributedRuntime(env_id=ENV_ID, consensus_radius=4.0)
    rt.spawn_agent("scout-A", trust=1.0)
    rt.spawn_agent("scout-B", trust=0.9)

    # Shared observation of SHARED_PLACE
    for i in range(8):
        rt.observe("scout-A", SHARED_PLACE, {WATER_TAG: 0.95},
                   episode_id=1, step_idx=i, confidence=0.95)
    for i in range(6):
        rt.observe("scout-B", SHARED_PLACE, {WATER_TAG: 0.90},
                   episode_id=1, step_idx=i, confidence=0.90)

    # Private observations
    for i in range(5):
        rt.observe("scout-A", PRIVATE_A, {WATER_TAG: 0.92},
                   episode_id=2, step_idx=i, confidence=0.92)
        rt.observe("scout-B", PRIVATE_B, {WATER_TAG: 0.88},
                   episode_id=2, step_idx=i, confidence=0.88)

    report = rt.tick()
    out_path = os.path.join(args.out_dir, "exp02_summary.json")
    with open(out_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    # Find the aligned (multi-agent) concept
    aligned = [c for c in report.distributed_concepts if c.n_agents == 2]
    assert aligned, (
        f"Expected ≥1 aligned concept, got {report.n_aligned}. "
        f"All concepts: {[(c.consensus_id, c.n_agents) for c in report.distributed_concepts]}"
    )
    shared_concept = aligned[0]
    cx, cy = shared_concept.centroid_xy
    assert abs(cx - SHARED_PLACE[0]) < 0.5 and abs(cy - SHARED_PLACE[1]) < 0.5, (
        f"Aligned centroid {shared_concept.centroid_xy} not near {SHARED_PLACE}"
    )
    assert shared_concept.consensus_dominant_tag == WATER_TAG
    assert shared_concept.inter_agent_agreement >= 0.99, (
        f"Compatible observations should give full agreement, got "
        f"{shared_concept.inter_agent_agreement:.3f}"
    )

    # Isolated concepts: PRIVATE_A and PRIVATE_B
    assert report.n_isolated >= 2, (
        f"Expected ≥2 isolated concepts, got {report.n_isolated}"
    )
    assert report.avg_agreement >= 0.99

    print(
        f"✓ Exp02 passed  "
        f"aligned={report.n_aligned}  "
        f"isolated={report.n_isolated}  "
        f"shared_centroid={shared_concept.centroid_xy}  "
        f"consensus_conf={shared_concept.consensus_confidence:.3f}  "
        f"agreement={shared_concept.inter_agent_agreement:.3f}"
    )
    print(f"Summary → {out_path}")


if __name__ == "__main__":
    main()
