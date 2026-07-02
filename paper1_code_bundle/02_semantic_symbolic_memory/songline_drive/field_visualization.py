"""Phase 4 — field visualisation helpers.

Lightweight export utilities with no required external dependencies.
matplotlib is imported lazily for optional PNG output.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from songline_drive.semantic_field import SemanticField


def activation_table(
    field: SemanticField,
    channels: Optional[List[str]] = None,
    concept_tag_map: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """Return per-concept activation table sorted by mean activation descending.

    Each row: ``{concept_id, dominant_tag, base_conflict, <channel>: activation, ...}``.
    """
    channels = channels or field.channels
    rows: List[Dict] = []
    for cid, cell in field.cells.items():
        row: Dict[str, Any] = {
            "concept_id": cid,
            "dominant_tag": (concept_tag_map or {}).get(cid, "?"),
            "base_confidence": round(cell.base_confidence, 4),
            "base_conflict": round(cell.base_conflict, 4),
            "support_count": cell.support_count,
        }
        total = 0.0
        for ch_name in channels:
            ch = cell.channels.get(ch_name)
            act = ch.activation if ch is not None else 0.0
            row[ch_name] = round(act, 4)
            total += act
        row["_mean_activation"] = round(total / max(1, len(channels)), 4)
        rows.append(row)
    rows.sort(key=lambda r: r["_mean_activation"], reverse=True)
    return rows


def save_snapshot(
    field: SemanticField,
    path: str,
    label: str = "",
) -> str:
    """Save a JSON snapshot of the field to ``path``."""
    snap = field.to_snapshot()
    snap["label"] = label
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(snap, fh, ensure_ascii=False, indent=2)
    return path


def export_channel_activations(
    field: SemanticField,
    channel: str,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """Return top-k (concept_id, activation) pairs for a channel."""
    return field.top_k_for_channel(channel, k=top_k)


def channel_summary(
    field: SemanticField,
    channel: str,
    concept_tag_map: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """Human-readable summary of one channel: concept_id, tag, activation, explain."""
    items = []
    for cid, act in field.top_k_for_channel(channel, k=len(field.cells)):
        exp = field.explain_score(cid, channel)
        items.append({
            "concept_id": cid,
            "dominant_tag": (concept_tag_map or {}).get(cid, "?"),
            "activation": round(act, 5),
            **{k: round(v, 4) for k, v in exp.items() if k != "activation"},
        })
    return items


def plot_activation_heatmap(
    field: SemanticField,
    channel: str,
    out_path: str,
    title: str = "",
    top_k: int = 15,
) -> bool:
    """Render an activation bar chart to PNG.  Returns True if matplotlib is available."""
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return False

    items = field.top_k_for_channel(channel, k=top_k)
    if not items:
        return False

    labels = [cid[:30] for cid, _ in items]
    values = [act for _, act in items]

    fig, ax = plt.subplots(figsize=(10, max(3, len(items) * 0.4)))
    ax.barh(labels[::-1], values[::-1])
    ax.set_xlabel("Activation")
    ax.set_title(title or f"Semantic field — channel: {channel}")
    ax.set_xlim(0, max(values) * 1.15 if values else 1.0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return True
