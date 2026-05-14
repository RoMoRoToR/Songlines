from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


GridPos = Tuple[int, int]


@dataclass
class PlaceObservation:
    agent_id: str
    step_idx: int
    pos: GridPos
    semantic_tags: Dict[str, float]


@dataclass
class LocalNode:
    node_id: int
    pos: GridPos
    visits: int = 0
    semantic_sum: Dict[str, float] = field(default_factory=dict)
    semantic_count: Dict[str, int] = field(default_factory=dict)

    def update(self, tags: Dict[str, float]) -> None:
        self.visits += 1
        for tag, value in tags.items():
            if value <= 0.0:
                continue
            self.semantic_sum[tag] = self.semantic_sum.get(tag, 0.0) + float(value)
            self.semantic_count[tag] = self.semantic_count.get(tag, 0) + 1

    @property
    def semantic_profile(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for tag, total in self.semantic_sum.items():
            cnt = self.semantic_count.get(tag, 1)
            out[tag] = total / max(1, cnt)
        return out

    @property
    def dominant_tag(self) -> str:
        profile = self.semantic_profile
        if not profile:
            return "unknown"
        return max(profile.items(), key=lambda kv: kv[1])[0]


@dataclass
class LocalGraph:
    agent_id: str
    nodes_by_pos: Dict[GridPos, LocalNode] = field(default_factory=dict)
    edges: List[Tuple[GridPos, GridPos]] = field(default_factory=list)
    last_pos: Optional[GridPos] = None

    def observe(self, step_idx: int, pos: GridPos, tags: Dict[str, float]) -> None:
        node = self.nodes_by_pos.get(pos)
        if node is None:
            node = LocalNode(node_id=len(self.nodes_by_pos), pos=pos)
            self.nodes_by_pos[pos] = node
        node.update(tags)
        if self.last_pos is not None and self.last_pos != pos:
            edge = (self.last_pos, pos)
            if edge not in self.edges:
                self.edges.append(edge)
        self.last_pos = pos


@dataclass
class CollectiveBelief:
    pos: GridPos
    observations: List[PlaceObservation] = field(default_factory=list)

    def add(self, obs: PlaceObservation) -> None:
        self.observations.append(obs)

    @property
    def semantic_profile(self) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for obs in self.observations:
            for tag, value in obs.semantic_tags.items():
                if value <= 0.0:
                    continue
                totals[tag] = totals.get(tag, 0.0) + float(value)
                counts[tag] = counts.get(tag, 0) + 1
        return {tag: totals[tag] / counts[tag] for tag in totals}

    @property
    def agents(self) -> List[str]:
        return sorted({obs.agent_id for obs in self.observations})


@dataclass
class SharedConcept:
    concept_id: str
    member_positions: List[GridPos]
    semantic_profile: Dict[str, float]
    supporting_agents: List[str]
    centroid_xy: Tuple[float, float]
    confidence: float
    freshness: float
    conflict_score: float

    @property
    def dominant_tag(self) -> str:
        if not self.semantic_profile:
            return "unknown"
        return max(self.semantic_profile.items(), key=lambda kv: kv[1])[0]


@dataclass
class FieldCell:
    concept_id: str
    centroid_xy: Tuple[float, float]
    channels: Dict[str, float]


def euclidean(a: GridPos, b: GridPos) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def cosine_similarity(left: Dict[str, float], right: Dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    keys = set(left.keys()) & set(right.keys())
    if not keys:
        return 0.0
    num = sum(left[k] * right[k] for k in keys)
    left_norm = math.sqrt(sum(v * v for v in left.values()))
    right_norm = math.sqrt(sum(v * v for v in right.values()))
    if left_norm <= 1e-9 or right_norm <= 1e-9:
        return 0.0
    return num / (left_norm * right_norm)


def profile_conflict_score(profile: Dict[str, float]) -> float:
    water = float(profile.get("water_source", 0.0))
    hazard = float(profile.get("hazard_edge", 0.0))
    if water <= 0.0 or hazard <= 0.0:
        return 0.0
    return min(water, hazard) / max(water, hazard)


class CollectiveBus:
    def __init__(self) -> None:
        self.beliefs: Dict[GridPos, CollectiveBelief] = {}
        self.events: List[dict] = []

    def publish(self, obs: PlaceObservation) -> None:
        belief = self.beliefs.get(obs.pos)
        if belief is None:
            belief = CollectiveBelief(pos=obs.pos)
            self.beliefs[obs.pos] = belief
        belief.add(obs)
        self.events.append(
            {
                "agent_id": obs.agent_id,
                "step_idx": obs.step_idx,
                "pos": list(obs.pos),
                "semantic_tags": dict(obs.semantic_tags),
            }
        )


def build_concepts(
    bus: CollectiveBus,
    current_step: int,
    spatial_radius: float = 1.5,
    semantic_similarity_threshold: float = 0.55,
    decay_factor: float = 0.97,
) -> List[SharedConcept]:
    beliefs = list(bus.beliefs.values())
    used = set()
    concepts: List[SharedConcept] = []
    counters: Dict[str, int] = {}

    for idx, seed in enumerate(beliefs):
        if idx in used:
            continue
        seed_profile = seed.semantic_profile
        seed_dom = max(seed_profile.items(), key=lambda kv: kv[1])[0] if seed_profile else "unknown"
        cluster_idxs = [idx]
        used.add(idx)
        for j, other in enumerate(beliefs):
            if j in used or j == idx:
                continue
            other_profile = other.semantic_profile
            other_dom = max(other_profile.items(), key=lambda kv: kv[1])[0] if other_profile else "unknown"
            if other_dom != seed_dom:
                continue
            if euclidean(seed.pos, other.pos) > spatial_radius:
                continue
            if cosine_similarity(seed_profile, other_profile) < semantic_similarity_threshold:
                continue
            cluster_idxs.append(j)
            used.add(j)

        members = [beliefs[k] for k in cluster_idxs]
        member_positions = [m.pos for m in members]
        agents = sorted({agent for m in members for agent in m.agents})
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        last_seen = -1
        for m in members:
            for tag, value in m.semantic_profile.items():
                totals[tag] = totals.get(tag, 0.0) + float(value)
                counts[tag] = counts.get(tag, 0) + 1
            for obs in m.observations:
                last_seen = max(last_seen, int(obs.step_idx))
        profile = {tag: totals[tag] / counts[tag] for tag in totals}
        centroid = (
            sum(pos[0] for pos in member_positions) / len(member_positions),
            sum(pos[1] for pos in member_positions) / len(member_positions),
        )
        freshness = decay_factor ** max(0, current_step - last_seen)
        confidence = min(1.0, 0.35 + 0.20 * math.log1p(len(member_positions)) + 0.12 * len(agents))
        conflict = profile_conflict_score(profile)
        dom = max(profile.items(), key=lambda kv: kv[1])[0] if profile else "unknown"
        counters[dom] = counters.get(dom, 0) + 1
        concept_id = f"{dom}_cluster_{counters[dom]-1}"
        concepts.append(
            SharedConcept(
                concept_id=concept_id,
                member_positions=member_positions,
                semantic_profile=profile,
                supporting_agents=agents,
                centroid_xy=centroid,
                confidence=confidence,
                freshness=freshness,
                conflict_score=conflict,
            )
        )
    return concepts


CHANNEL_AFFINITIES: Dict[str, Dict[str, float]] = {
    "water_source": {
        "water_source": 1.0,
        "water_candidate": 0.5,
        "safe_neutral": 0.1,
        "hazard_edge": -0.9,
    },
    "hazard_edge": {
        "hazard_edge": 1.0,
        "water_source": -0.8,
        "safe_neutral": -0.2,
    },
    "safe_neutral": {
        "safe_neutral": 1.0,
        "water_source": 0.15,
        "hazard_edge": -0.3,
    },
}


def build_field(
    concepts: Sequence[SharedConcept],
    lambda_decay: float = 0.95,
    alpha_belief: float = 0.60,
    eta_conflict: float = 0.30,
    previous: Optional[Dict[str, FieldCell]] = None,
) -> Dict[str, FieldCell]:
    previous = previous or {}
    out: Dict[str, FieldCell] = {}
    for concept in concepts:
        base_belief = 0.50 * concept.confidence + 0.30 * concept.freshness + 0.20 * min(1.0, len(concept.supporting_agents) / 2)
        channels: Dict[str, float] = {}
        for channel, weights in CHANNEL_AFFINITIES.items():
            affinity = 0.0
            for tag, weight in weights.items():
                affinity += weight * float(concept.semantic_profile.get(tag, 0.0))
            raw = max(0.0, alpha_belief * base_belief * affinity - eta_conflict * concept.conflict_score)
            prev = previous.get(concept.concept_id)
            prev_act = 0.0 if prev is None else float(prev.channels.get(channel, 0.0))
            channels[channel] = lambda_decay * prev_act + (1.0 - lambda_decay) * raw
        out[concept.concept_id] = FieldCell(
            concept_id=concept.concept_id,
            centroid_xy=concept.centroid_xy,
            channels=channels,
        )
    return out


class Agent:
    def __init__(self, agent_id: str, route: Sequence[GridPos], color: str) -> None:
        self.agent_id = agent_id
        self.route = list(route)
        self.color = color
        self.local_graph = LocalGraph(agent_id=agent_id)
        self.current_pos = self.route[0]
        self.current_target_concept: Optional[str] = None

    def step(self, step_idx: int, world: "World") -> PlaceObservation:
        pos = self.route[min(step_idx, len(self.route) - 1)]
        self.current_pos = pos
        tags = world.observe(pos)
        self.local_graph.observe(step_idx, pos, tags)
        return PlaceObservation(agent_id=self.agent_id, step_idx=step_idx, pos=pos, semantic_tags=tags)


class World:
    def __init__(self, width: int = 10, height: int = 8) -> None:
        self.width = width
        self.height = height
        self.water_cells = {(2, 2), (2, 3), (7, 5), (7, 6)}
        self.hazard_cells = {(4, 3), (4, 4), (5, 3)}

    def observe(self, pos: GridPos) -> Dict[str, float]:
        x, y = pos
        tags = {
            "safe_neutral": 0.55,
            "water_source": 0.0,
            "water_candidate": 0.0,
            "hazard_edge": 0.0,
        }
        if pos in self.water_cells:
            tags["water_source"] = 1.0
            tags["water_candidate"] = 0.8
            tags["safe_neutral"] = 0.15
        if pos in self.hazard_cells:
            tags["hazard_edge"] = 1.0
            tags["safe_neutral"] = 0.10
        for wx, wy in self.water_cells:
            d = euclidean(pos, (wx, wy))
            if 0.0 < d <= 1.5:
                tags["water_candidate"] = max(tags["water_candidate"], 0.35)
        for hx, hy in self.hazard_cells:
            d = euclidean(pos, (hx, hy))
            if 0.0 < d <= 1.5:
                tags["hazard_edge"] = max(tags["hazard_edge"], 0.35)
        if tags["water_source"] > 0 and tags["hazard_edge"] > 0:
            tags["safe_neutral"] = 0.0
        return tags


def best_water_concept(concepts: Sequence[SharedConcept], field: Dict[str, FieldCell]) -> Optional[str]:
    ranked: List[Tuple[float, str]] = []
    for concept in concepts:
        cell = field.get(concept.concept_id)
        if cell is None:
            continue
        score = cell.channels.get("water_source", 0.0)
        ranked.append((score, concept.concept_id))
    ranked.sort(reverse=True)
    return None if not ranked else ranked[0][1]


def draw_world(ax, world: World, agents: Sequence[Agent], title: str) -> None:
    ax.set_xlim(-0.5, world.width - 0.5)
    ax.set_ylim(-0.5, world.height - 0.5)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xticks(range(world.width))
    ax.set_yticks(range(world.height))
    ax.grid(True, alpha=0.25)
    for (x, y) in world.water_cells:
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="#7ec8ff", alpha=0.35, edgecolor="none"))
    for (x, y) in world.hazard_cells:
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="#ff8f8f", alpha=0.35, edgecolor="none"))
    for agent in agents:
        x, y = agent.current_pos
        ax.scatter([x], [y], s=90, color=agent.color, edgecolor="black", zorder=5)
        ax.text(x + 0.12, y + 0.12, agent.agent_id, fontsize=8)


def draw_local_graph(ax, graph: LocalGraph, title: str, world: World, color: str) -> None:
    draw_world(ax, world, [], title)
    for a, b in graph.edges:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, alpha=0.4, linewidth=1.5)
    for pos, node in graph.nodes_by_pos.items():
        x, y = pos
        dom = node.dominant_tag
        if dom == "water_source":
            face = "#2b8cff"
        elif dom == "hazard_edge":
            face = "#d62728"
        else:
            face = "#7f7f7f"
        ax.scatter([x], [y], s=30 + 6 * node.visits, color=face, edgecolor="black", linewidth=0.5, zorder=4)


def draw_shared_memory(ax, world: World, concepts: Sequence[SharedConcept], field: Dict[str, FieldCell], consumer: Agent) -> None:
    draw_world(ax, world, [], "Общая память: concepts + field")
    for concept in concepts:
        cx, cy = concept.centroid_xy
        dom = concept.dominant_tag
        if dom == "water_source":
            color = "#2b8cff"
        elif dom == "hazard_edge":
            color = "#d62728"
        else:
            color = "#7f7f7f"
        radius = 0.18 + 0.05 * len(concept.member_positions)
        circ = plt.Circle((cx, cy), radius=radius, fill=False, color=color, linewidth=2.2, alpha=0.9)
        ax.add_patch(circ)
        water_act = field.get(concept.concept_id).channels.get("water_source", 0.0) if concept.concept_id in field else 0.0
        hazard_act = field.get(concept.concept_id).channels.get("hazard_edge", 0.0) if concept.concept_id in field else 0.0
        ax.text(cx + 0.1, cy + 0.1, f"{concept.concept_id}\nW={water_act:.2f} H={hazard_act:.2f}", fontsize=7)
        for px, py in concept.member_positions:
            ax.plot([px, cx], [py, cy], color=color, alpha=0.18, linewidth=1)

    if consumer.current_target_concept:
        target = next((c for c in concepts if c.concept_id == consumer.current_target_concept), None)
        if target is not None:
            x0, y0 = consumer.current_pos
            x1, y1 = target.centroid_xy
            ax.plot([x0, x1], [y0, y1], color=consumer.color, linestyle="--", linewidth=2)
            ax.scatter([x1], [y1], s=120, facecolors="none", edgecolors=consumer.color, linewidth=2)


def draw_field_table(ax, concepts: Sequence[SharedConcept], field: Dict[str, FieldCell]) -> None:
    ax.set_title("Water / Hazard field")
    ax.axis("off")
    rows = []
    for concept in concepts:
        cell = field.get(concept.concept_id)
        if cell is None:
            continue
        rows.append(
            [
                concept.concept_id,
                f"{cell.channels.get('water_source', 0.0):.2f}",
                f"{cell.channels.get('hazard_edge', 0.0):.2f}",
                f"{cell.channels.get('safe_neutral', 0.0):.2f}",
                ",".join(concept.supporting_agents),
            ]
        )
    rows = rows[:8]
    table = ax.table(
        cellText=rows,
        colLabels=["concept", "water", "hazard", "neutral", "agents"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)


def make_routes() -> Dict[str, List[GridPos]]:
    scout_a = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3), (4, 3), (4, 4), (5, 4), (6, 4)]
    scout_b = [(9, 7), (8, 7), (7, 7), (7, 6), (7, 5), (6, 5), (5, 5), (4, 5), (4, 4), (4, 3), (3, 3)]
    consumer = [(5, 0), (5, 1), (5, 2), (4, 2), (3, 2), (2, 2), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3)]
    return {"scout-A": scout_a, "scout-B": scout_b, "consumer-C": consumer}


def run_experiment(out_dir: str, n_steps: int = 11) -> None:
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    world = World()
    routes = make_routes()
    agents = [
        Agent("scout-A", routes["scout-A"], "#1f77b4"),
        Agent("scout-B", routes["scout-B"], "#ff7f0e"),
        Agent("consumer-C", routes["consumer-C"], "#2ca02c"),
    ]
    bus = CollectiveBus()
    prev_field: Dict[str, FieldCell] = {}
    history: List[dict] = []

    for step_idx in range(n_steps):
        for agent in agents:
            obs = agent.step(step_idx, world)
            bus.publish(obs)

        concepts = build_concepts(bus, current_step=step_idx)
        field = build_field(concepts, previous=prev_field)
        prev_field = field

        consumer = next(agent for agent in agents if agent.agent_id == "consumer-C")
        consumer.current_target_concept = best_water_concept(concepts, field)

        history.append(
            {
                "step_idx": step_idx,
                "concepts": [
                    {
                        "concept_id": c.concept_id,
                        "dominant_tag": c.dominant_tag,
                        "member_positions": [list(p) for p in c.member_positions],
                        "supporting_agents": c.supporting_agents,
                        "confidence": c.confidence,
                        "freshness": c.freshness,
                        "conflict_score": c.conflict_score,
                        "centroid_xy": list(c.centroid_xy),
                    }
                    for c in concepts
                ],
                "consumer_target_concept": consumer.current_target_concept,
            }
        )

        fig = plt.figure(figsize=(15.5, 9.0))
        gs = fig.add_gridspec(2, 3, hspace=0.22, wspace=0.20)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[1, 2])

        draw_world(ax0, world, agents, f"Локация, step={step_idx}")
        draw_local_graph(ax1, agents[0].local_graph, "Scout-A: локальный граф памяти", world, agents[0].color)
        draw_local_graph(ax2, agents[1].local_graph, "Scout-B: локальный граф памяти", world, agents[1].color)
        draw_local_graph(ax3, agents[2].local_graph, "Consumer-C: локальный граф памяти", world, agents[2].color)
        draw_shared_memory(ax4, world, concepts, field, consumer)
        draw_field_table(ax5, concepts, field)

        fig.suptitle(
            "Collective Memory Cinema: локальные графы → shared concepts → semantic field",
            fontsize=16,
            y=0.98,
        )
        frame_path = os.path.join(frames_dir, f"frame_{step_idx:03d}.png")
        fig.savefig(frame_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    summary = {
        "n_steps": n_steps,
        "n_events": len(bus.events),
        "n_collective_places": len(bus.beliefs),
        "final_concepts": history[-1]["concepts"],
        "consumer_final_target_concept": history[-1]["consumer_target_concept"],
        "frames_dir": frames_dir,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as fh:
        json.dump(history, fh, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visual multi-agent local-memory vs collective-memory experiment")
    parser.add_argument("--out_dir", type=str, default="/tmp/memory_cinema_demo")
    parser.add_argument("--steps", type=int, default=11)
    args = parser.parse_args()

    run_experiment(out_dir=args.out_dir, n_steps=args.steps)
