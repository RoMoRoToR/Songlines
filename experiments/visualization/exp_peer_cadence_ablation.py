"""Peer broadcast-cadence ablation.

Same scenario as ``exp_4way_walk.py`` but instead of comparing the four
memory architectures, this script holds the architecture FIXED at peer
and varies the broadcast cadence ``K``:

  k=1   — broadcast every tick (close to centralized-immediate)
  k=2   — fast gossip
  k=4   — moderate (default in exp_4way_walk)
  k=10  — slow gossip (peer info is stale most of the time)

The point is to show that within ONE communication pattern (peer-to-peer
broadcast) the *cadence* alone produces visibly different path patterns
and outcomes — coordination emerges from latency, not from a central
authority.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/visualization/exp_peer_cadence_ablation.py \\
        --n_ticks 60 --out_dir tmp/visualization_peer_cadence
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Reuse all infrastructure from exp_4way_walk
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_4way_walk as base  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_ticks", type=int, default=60)
    parser.add_argument("--out_dir", default="tmp/visualization_peer_cadence")
    parser.add_argument("--cadences", type=str, default="1,2,4,10",
                        help="Comma-separated K values to compare (2-4 of them).")
    parser.add_argument("--frame_duration_ms", type=int, default=350)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    frames_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    cadences = [int(k) for k in args.cadences.split(",") if k]
    assert 2 <= len(cadences) <= 4, "give 2 to 4 cadence values"

    print(f"Running peer × {len(cadences)} cadences × {args.n_ticks} ticks ...")
    snaps_per_cadence: Dict[int, List[base.TickSnapshot]] = {}
    for k in cadences:
        print(f"  cadence k={k}")
        snaps_per_cadence[k] = base.run_peer(args.n_ticks, broadcast_every_k=k)

    print("Rendering frames ...")
    trails_per_cadence: Dict[int, Dict[str, List[Tuple[int, int]]]] = {
        k: {spec["id"]: [] for spec in base.AGENT_SPEC}
        for k in cadences
    }
    frame_paths: List[str] = []
    for tick in range(args.n_ticks):
        snaps_now: Dict[int, base.TickSnapshot] = {}
        for k, snaps in snaps_per_cadence.items():
            snap = snaps[tick]
            snaps_now[k] = snap
            for aid, xy in snap.agent_positions.items():
                trail = trails_per_cadence[k].setdefault(aid, [])
                if not trail or trail[-1] != xy:
                    trail.append(xy)
        out_png = os.path.join(frames_dir, f"frame_{tick:03d}.png")
        _render_frame(tick, snaps_now, trails_per_cadence, cadences, out_png)
        frame_paths.append(out_png)
        if tick % 10 == 0:
            print(f"  rendered tick {tick}/{args.n_ticks}")

    # GIF
    gif_path = os.path.join(args.out_dir, "peer_cadence.gif")
    try:
        import imageio.v2 as imageio_v2
    except ImportError:
        import imageio as imageio_v2
    images = [imageio_v2.imread(p) for p in frame_paths]
    duration_s = args.frame_duration_ms / 1000.0
    try:
        imageio_v2.mimsave(gif_path, images, duration=duration_s, loop=0)
    except Exception:
        imageio_v2.mimsave(gif_path, images, duration=args.frame_duration_ms)

    # Stats per cadence
    stats: Dict[int, Any] = {}
    for k, snaps in snaps_per_cadence.items():
        last = snaps[-1]
        first_success: Dict[str, Optional[int]] = {}
        for aid in last.agent_success:
            first_success[aid] = None
            for t, snap in enumerate(snaps):
                if snap.agent_success.get(aid):
                    first_success[aid] = t
                    break
        succ_times = [v for v in first_success.values() if v is not None]
        stats[k] = {
            "n_succeeded": sum(1 for v in last.agent_success.values() if v),
            "first_success_tick": first_success,
            "mean_success_tick": (sum(succ_times) / len(succ_times)
                                  if succ_times else None),
        }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"cadences": cadences, "stats": stats,
                   "n_ticks": args.n_ticks}, f, indent=2)

    print("=" * 75)
    print(f"✓ Cadence ablation done")
    print(f"  Frames:  {frames_dir}  ({len(frame_paths)} PNGs)")
    print(f"  GIF:     {gif_path}")
    print()
    print(f"{'k':>4}  {'n_succ':>7}  {'mean_tick':>10}  {'first_success_tick':>30}")
    print("-" * 75)
    for k in cadences:
        s = stats[k]
        fs = s["first_success_tick"]
        fs_str = ", ".join(f"{aid[-1]}={t}" for aid, t in fs.items())
        mean_str = f"{s['mean_success_tick']:.1f}" if s["mean_success_tick"] else "-"
        print(f"{k:>4}  {s['n_succeeded']}/3{'':<3} {mean_str:>10}  {fs_str:>30}")


def _render_frame(
    tick: int,
    snaps: Dict[int, base.TickSnapshot],
    trails: Dict[int, Dict[str, List[Tuple[int, int]]]],
    cadences: List[int],
    out_path: str,
) -> None:
    n = len(cadences)
    if n == 2:
        nrows, ncols = 1, 2
    elif n == 3:
        nrows, ncols = 1, 3
    else:
        nrows, ncols = 2, 2

    fig, axs = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.5 * nrows))
    if nrows == 1 and ncols == 1:
        axs = [[axs]]
    elif nrows == 1:
        axs = [axs]
    elif ncols == 1:
        axs = [[a] for a in axs]
    # Flatten with iteration
    panels = [axs[i // ncols][i % ncols] for i in range(nrows * ncols)]

    plt.subplots_adjust(top=0.91, bottom=0.10, left=0.05, right=0.97,
                        hspace=0.30, wspace=0.18)

    for idx, k in enumerate(cadences):
        title = f"peer (broadcast каждые k={k} тиков)"
        base.render_panel(panels[idx], snaps[k], title, trails[k])

    counts = []
    for k in cadences:
        snap = snaps[k]
        n_known = [len(v) for v in snap.known_concepts_per_agent.values()]
        n_succ = sum(1 for v in snap.agent_success.values() if v)
        counts.append(f"k={k}: known={n_known} succ={n_succ}")

    fig.suptitle(
        f"Peer broadcast cadence ablation — tick {tick:02d}\n"
        f"  {' | '.join(counts)}",
        fontsize=11, y=0.985,
    )

    legend_handles = [
        mpatches.Patch(facecolor="#aed6f1", edgecolor="#3498db", label="water"),
        mpatches.Patch(facecolor="#f5b7b1", edgecolor="#c0392b", label="hazard"),
        mpatches.Patch(facecolor="none", edgecolor="purple", label="known concept"),
    ]
    for spec in base.AGENT_SPEC:
        legend_handles.append(
            mpatches.Patch(facecolor=spec["color"], edgecolor="black",
                           label=spec["id"])
        )
    legend_handles.append(
        plt.Line2D([0], [0], marker="*", color="black",
                   markerfacecolor="gold", markersize=11, linewidth=0,
                   label="success")
    )
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(legend_handles), fontsize=8,
               bbox_to_anchor=(0.5, 0.01), frameon=False)

    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
