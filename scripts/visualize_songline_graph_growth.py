import argparse
import json
import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np


SEMANTIC_COLORS = {
    "goal_region": "#2E7D32",
    "water_source": "#1976D2",
    "safe_rest_zone": "#388E3C",
    "hazard_recovery_route": "#F9A825",
    "safe_exit": "#00897B",
    "hazard_edge": "#D32F2F",
    "room_center": "#6A1B9A",
    "corridor": "#5D4037",
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def dominant_semantic_tag(node):
    confs = dict(node.get("semantic_tag_confidence", {}))
    if not confs:
        return "default"
    tag_name, tag_value = max(confs.items(), key=lambda item: float(item[1]))
    if float(tag_value) <= 0.0:
        return "default"
    return str(tag_name)


def node_color(node):
    tag = dominant_semantic_tag(node)
    return SEMANTIC_COLORS.get(tag, "#455A64")


def normalize_layout(layout):
    if not layout:
        return {}
    coords = np.asarray(list(layout.values()), dtype=np.float64)
    x_vals = coords[:, 0]
    y_vals = coords[:, 1]
    x_min, x_max = float(x_vals.min()), float(x_vals.max())
    y_min, y_max = float(y_vals.min()), float(y_vals.max())
    x_span = max(1.0, x_max - x_min)
    y_span = max(1.0, y_max - y_min)
    normalized = {}
    for node_id, pos in layout.items():
        normalized[node_id] = np.array(
            [
                (float(pos[0]) - x_min) / x_span,
                (float(pos[1]) - y_min) / y_span,
            ],
            dtype=np.float64,
        )
    return normalized


def build_layered_layout(nodes, edges):
    node_ids = sorted(int(node["node_id"]) for node in nodes)
    if not node_ids:
        return {}

    undirected = {node_id: set() for node_id in node_ids}
    for edge in edges:
        src = int(edge["src"])
        dst = int(edge["dst"])
        if src in undirected and dst in undirected:
            undirected[src].add(dst)
            undirected[dst].add(src)

    root = min(node_ids)
    levels = {root: 0}
    q = deque([root])
    while q:
        cur = q.popleft()
        for nxt in sorted(undirected[cur]):
            if nxt in levels:
                continue
            levels[nxt] = levels[cur] + 1
            q.append(nxt)

    max_level = 0
    for node_id in node_ids:
        if node_id not in levels:
            max_level += 1
            levels[node_id] = max(levels.values()) + max_level

    by_level = {}
    for node in nodes:
        node_id = int(node["node_id"])
        by_level.setdefault(levels[node_id], []).append(node)

    layout = {}
    sorted_levels = sorted(by_level.keys())
    for level_idx, level in enumerate(sorted_levels):
        items = sorted(
            by_level[level],
            key=lambda node: (
                -float(node.get("utility_cached", 0.0)),
                int(node["node_id"]),
            ),
        )
        count = len(items)
        if count == 1:
            ys = [0.0]
        else:
            ys = np.linspace(-1.0, 1.0, num=count)
        for y, node in zip(ys, items):
            node_id = int(node["node_id"])
            x = float(level_idx)
            layout[node_id] = np.array([x, float(y)], dtype=np.float64)

    return normalize_layout(layout)


def _dedupe_positions(points):
    out = []
    prev = None
    for point in points:
        cur = (float(point[0]), float(point[1]))
        if prev is None or abs(cur[0] - prev[0]) > 1e-9 or abs(cur[1] - prev[1]) > 1e-9:
            out.append(np.array(cur, dtype=np.float64))
            prev = cur
    return out


def build_spatial_layout_from_demo(nodes, demo_metadata):
    if not demo_metadata:
        return {}

    points = []
    for row in demo_metadata:
        pos = row.get("pos")
        if not isinstance(pos, list) or len(pos) != 2:
            continue
        points.append([float(pos[0]), float(pos[1])])
    points = _dedupe_positions(points)
    if not points:
        return {}

    node_ids = sorted(int(node["node_id"]) for node in nodes)
    if not node_ids:
        return {}

    point_count = len(points)
    layout = {}
    used_counts = {}
    for order_idx, node_id in enumerate(node_ids):
        if len(node_ids) == 1:
            point_idx = 0
        else:
            alpha = float(order_idx) / float(max(1, len(node_ids) - 1))
            point_idx = int(round(alpha * float(point_count - 1)))
        base = np.array(points[point_idx], dtype=np.float64)
        overlap_rank = int(used_counts.get(point_idx, 0))
        used_counts[point_idx] = overlap_rank + 1
        if overlap_rank > 0:
            angle = 0.85 * float(overlap_rank)
            radius = 0.12 * float((overlap_rank + 1) // 2)
            base = base + np.array(
                [np.cos(angle) * radius, np.sin(angle) * radius],
                dtype=np.float64,
            )
        layout[node_id] = base
    return normalize_layout(layout)


def choose_layout(nodes, edges, demo_metadata):
    spatial_layout = build_spatial_layout_from_demo(nodes, demo_metadata)
    if spatial_layout:
        return spatial_layout, "spatial_demo_trajectory"
    return build_layered_layout(nodes, edges), "layered_graph"


def frame_schedule(num_nodes: int, max_frames: int):
    if num_nodes <= 0:
        return []
    frame_count = min(max_frames, num_nodes)
    values = np.linspace(1, num_nodes, num=frame_count, dtype=int)
    out = []
    seen = set()
    for value in values:
        value = int(value)
        if value not in seen:
            out.append(value)
            seen.add(value)
    if out[-1] != num_nodes:
        out.append(num_nodes)
    return out


def render_growth_frames(nodes, edges, layout, layout_mode, run_summary, out_dir, max_frames=32):
    try:
        import imageio.v2 as imageio
    except Exception:
        import imageio
    from PIL import Image

    ensure_dir(out_dir)
    node_map = {int(node["node_id"]): node for node in nodes}
    thresholds = frame_schedule(len(nodes), max_frames=max_frames)
    frames = []
    metadata = []

    title = "Songline Graph Growth"
    if run_summary is not None:
        task_mode = str(run_summary.get("task_mode", "default"))
        intent_type = str(run_summary.get("intent_type", "none"))
        env_id = str(run_summary.get("env_id", "env"))
        title = f"{env_id} | {task_mode} | {intent_type}"

    for threshold in thresholds:
        visible_nodes = [node_map[node_id] for node_id in sorted(node_map.keys()) if node_id < threshold]
        visible_ids = {int(node["node_id"]) for node in visible_nodes}
        visible_edges = [
            edge for edge in edges
            if int(edge["src"]) in visible_ids and int(edge["dst"]) in visible_ids
        ]
        newest_node_id = max(visible_ids) if visible_ids else None

        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        ax.set_title(title)
        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.08, 1.08)
        ax.axis("off")

        for edge in visible_edges:
            src = int(edge["src"])
            dst = int(edge["dst"])
            src_xy = layout.get(src)
            dst_xy = layout.get(dst)
            if src_xy is None or dst_xy is None:
                continue
            ax.plot(
                [src_xy[0], dst_xy[0]],
                [src_xy[1], dst_xy[1]],
                color="#B0BEC5",
                linewidth=0.8 + 0.25 * float(edge.get("weight", 1)),
                alpha=0.65,
                zorder=1,
            )

        xs = []
        ys = []
        cs = []
        ss = []
        for node in visible_nodes:
            node_id = int(node["node_id"])
            pos = layout.get(node_id)
            if pos is None:
                continue
            xs.append(pos[0])
            ys.append(pos[1])
            cs.append(node_color(node))
            visits = max(1, int(node.get("visits", 1)))
            ss.append(80.0 + 22.0 * np.sqrt(visits))

        if xs:
            ax.scatter(xs, ys, c=cs, s=ss, edgecolors="#263238", linewidths=0.8, zorder=2)

        if newest_node_id is not None:
            newest_pos = layout.get(newest_node_id)
            if newest_pos is not None:
                ax.scatter(
                    [newest_pos[0]],
                    [newest_pos[1]],
                    s=260,
                    facecolors="none",
                    edgecolors="#FF6F00",
                    linewidths=2.0,
                    zorder=3,
                )

        for node in visible_nodes:
            node_id = int(node["node_id"])
            pos = layout.get(node_id)
            if pos is None:
                continue
            if node_id == newest_node_id or int(node.get("visits", 0)) >= 3:
                ax.text(
                    pos[0],
                    pos[1] + 0.025,
                    str(node_id),
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    color="#111111",
                    zorder=4,
                )

        info_lines = [
            f"layout: {layout_mode}",
            f"visible nodes: {len(visible_nodes)} / {len(nodes)}",
            f"visible edges: {len(visible_edges)} / {len(edges)}",
            f"newest node: {newest_node_id if newest_node_id is not None else '-'}",
        ]
        if newest_node_id is not None:
            node = node_map[newest_node_id]
            info_lines.append(
                f"tag={dominant_semantic_tag(node)} visits={int(node.get('visits', 0))} utility={float(node.get('utility_cached', 0.0)):.2f}"
            )
        ax.text(
            0.01,
            0.02,
            "\n".join(info_lines),
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.90, edgecolor="#CFD8DC"),
        )

        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()
        plt.close(fig)

        frame_path = os.path.join(out_dir, f"frame_{len(frames):03d}.png")
        imageio.imwrite(frame_path, frame)
        frames.append(frame)
        metadata.append(
            {
                "frame_idx": len(frames) - 1,
                "visible_nodes": len(visible_nodes),
                "visible_edges": len(visible_edges),
                "newest_node_id": newest_node_id,
                "threshold": int(threshold),
                "layout_mode": layout_mode,
            }
        )

    gif_path = os.path.join(out_dir, "graph_growth.gif")
    imageio.mimsave(gif_path, frames, duration=0.45)

    with open(os.path.join(out_dir, "graph_growth_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    key_count = min(6, len(frames))
    indices = np.linspace(0, len(frames) - 1, num=key_count, dtype=int)
    unique_indices = []
    seen = set()
    for idx in indices:
        idx = int(idx)
        if idx not in seen:
            unique_indices.append(idx)
            seen.add(idx)
    tiles = [Image.fromarray(frames[idx]) for idx in unique_indices]
    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    sheet = Image.new("RGB", (tile_w * len(tiles), tile_h), (255, 255, 255))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (i * tile_w, 0))
    sheet.save(os.path.join(out_dir, "graph_growth_storyboard.png"))

    final_frame = Image.fromarray(frames[-1])
    final_frame.save(os.path.join(out_dir, "graph_growth_final.png"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory containing summary.json and env_0/")
    parser.add_argument("--env_idx", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--out_subdir", type=str, default="graph_growth_viz")
    return parser.parse_args()


def main():
    args = parse_args()
    env_dir = os.path.join(args.run_dir, f"env_{int(args.env_idx)}")
    phrases_path = os.path.join(env_dir, "phrases.json")
    edges_path = os.path.join(env_dir, "graph_edges.json")
    summary_path = os.path.join(args.run_dir, "summary.json")
    demo_metadata_path = os.path.join(args.run_dir, "demo", "demo_metadata.json")
    if not os.path.exists(phrases_path):
        raise FileNotFoundError(f"Missing phrases.json: {phrases_path}")
    if not os.path.exists(edges_path):
        raise FileNotFoundError(f"Missing graph_edges.json: {edges_path}")

    nodes = load_json(phrases_path)
    edges = load_json(edges_path)
    run_summary = load_json(summary_path) if os.path.exists(summary_path) else None
    demo_metadata = load_json(demo_metadata_path) if os.path.exists(demo_metadata_path) else None
    layout, layout_mode = choose_layout(nodes, edges, demo_metadata)

    out_dir = os.path.join(args.run_dir, args.out_subdir)
    render_growth_frames(
        nodes=nodes,
        edges=edges,
        layout=layout,
        layout_mode=layout_mode,
        run_summary=run_summary,
        out_dir=out_dir,
        max_frames=int(args.max_frames),
    )


if __name__ == "__main__":
    main()
