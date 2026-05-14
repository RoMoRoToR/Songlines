from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

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

    def observe(self, pos: GridPos, tags: Dict[str, float]) -> None:
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


@dataclass
class Agent:
    agent_id: str
    route: Sequence[GridPos]
    color: str
    local_graph: LocalGraph = field(init=False)
    current_pos: GridPos = field(init=False)
    current_target_concept: Optional[str] = None

    def __post_init__(self) -> None:
        self.local_graph = LocalGraph(agent_id=self.agent_id)
        self.current_pos = self.route[0]

    def step(self, step_idx: int, world: "World") -> PlaceObservation:
        pos = self.route[min(step_idx, len(self.route) - 1)]
        self.current_pos = pos
        tags = world.observe(pos)
        self.local_graph.observe(pos, tags)
        return PlaceObservation(agent_id=self.agent_id, step_idx=step_idx, pos=pos, semantic_tags=tags)


class World:
    def __init__(self, width: int = 10, height: int = 8) -> None:
        self.width = width
        self.height = height
        self.water_cells = {(2, 2), (2, 3), (7, 5), (7, 6)}
        self.hazard_cells = {(4, 3), (4, 4), (5, 3)}

    def observe(self, pos: GridPos) -> Dict[str, float]:
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
        return tags


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


def build_concepts(
    bus: CollectiveBus,
    current_step: int,
    spatial_radius: float = 1.6,
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
        if not seed_profile:
            continue
        seed_dom = max(seed_profile.items(), key=lambda kv: kv[1])[0]
        cluster_idxs = [idx]
        used.add(idx)
        for j, other in enumerate(beliefs):
            if j in used or j == idx:
                continue
            other_profile = other.semantic_profile
            if not other_profile:
                continue
            other_dom = max(other_profile.items(), key=lambda kv: kv[1])[0]
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
        dom = max(profile.items(), key=lambda kv: kv[1])[0]
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
        base_belief = (
            0.50 * concept.confidence
            + 0.30 * concept.freshness
            + 0.20 * min(1.0, len(concept.supporting_agents) / 2)
        )
        channels: Dict[str, float] = {}
        for channel, weights in CHANNEL_AFFINITIES.items():
            affinity = 0.0
            for tag, weight in weights.items():
                affinity += weight * float(concept.semantic_profile.get(tag, 0.0))
            raw = max(0.0, alpha_belief * base_belief * affinity - eta_conflict * concept.conflict_score)
            prev = previous.get(concept.concept_id)
            prev_act = 0.0 if prev is None else float(prev.channels.get(channel, 0.0))
            channels[channel] = lambda_decay * prev_act + (1.0 - lambda_decay) * raw
        out[concept.concept_id] = FieldCell(concept_id=concept.concept_id, centroid_xy=concept.centroid_xy, channels=channels)
    return out


def best_water_concept(concepts: Sequence[SharedConcept], field: Dict[str, FieldCell]) -> Optional[str]:
    ranked: List[Tuple[float, str]] = []
    for concept in concepts:
        cell = field.get(concept.concept_id)
        if cell is None:
            continue
        water = cell.channels.get("water_source", 0.0)
        hazard = cell.channels.get("hazard_edge", 0.0)
        score = water - 0.5 * hazard
        ranked.append((score, concept.concept_id))
    ranked.sort(reverse=True)
    return None if not ranked else ranked[0][1]


def concept_importance(concept: SharedConcept, field: Dict[str, FieldCell]) -> float:
    cell = field.get(concept.concept_id)
    if cell is None:
        return 0.0
    return max(cell.channels.values())


def color_for_tag(tag: str) -> str:
    if tag == "water_source":
        return "#2b8cff"
    if tag == "hazard_edge":
        return "#d62728"
    return "#7f7f7f"


def draw_world(ax, world: World, agents: Sequence[Agent], step_idx: int) -> None:
    ax.set_xlim(-0.5, world.width - 0.5)
    ax.set_ylim(-0.5, world.height - 0.5)
    ax.set_aspect("equal")
    ax.set_title(f"World state • step {step_idx}")
    ax.set_xticks(range(world.width))
    ax.set_yticks(range(world.height))
    ax.grid(True, alpha=0.25)
    for (x, y) in world.water_cells:
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="#7ec8ff", alpha=0.35, edgecolor="none"))
    for (x, y) in world.hazard_cells:
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="#ff8f8f", alpha=0.35, edgecolor="none"))
    for agent in agents:
        x, y = agent.current_pos
        ax.scatter([x], [y], s=100, color=agent.color, edgecolor="black", zorder=5)
        ax.text(x + 0.12, y + 0.12, agent.agent_id, fontsize=8)


def draw_local_graph(ax, graph: LocalGraph, title: str, world: World) -> None:
    ax.set_xlim(-0.5, world.width - 0.5)
    ax.set_ylim(-0.5, world.height - 0.5)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xticks(range(world.width))
    ax.set_yticks(range(world.height))
    ax.grid(True, alpha=0.20)
    for (x, y) in world.water_cells:
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="#7ec8ff", alpha=0.18, edgecolor="none"))
    for (x, y) in world.hazard_cells:
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="#ff8f8f", alpha=0.18, edgecolor="none"))
    for a, b in graph.edges:
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#555555", alpha=0.45, linewidth=1.4)
    for pos, node in graph.nodes_by_pos.items():
        x, y = pos
        face = color_for_tag(node.dominant_tag)
        ax.scatter([x], [y], s=24 + 7 * node.visits, color=face, edgecolor="black", linewidth=0.5, zorder=4)


def draw_shared_memory(ax, world: World, concepts: Sequence[SharedConcept], field: Dict[str, FieldCell], consumer: Agent) -> None:
    ax.set_xlim(-0.5, world.width - 0.5)
    ax.set_ylim(-0.5, world.height - 0.5)
    ax.set_aspect("equal")
    ax.set_title("Collective memory: shared concepts")
    ax.set_xticks(range(world.width))
    ax.set_yticks(range(world.height))
    ax.grid(True, alpha=0.20)
    for (x, y) in world.water_cells:
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="#7ec8ff", alpha=0.18, edgecolor="none"))
    for (x, y) in world.hazard_cells:
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="#ff8f8f", alpha=0.18, edgecolor="none"))

    important = sorted(concepts, key=lambda c: concept_importance(c, field), reverse=True)[:5]
    important_ids = {c.concept_id for c in important}

    for concept in concepts:
        cx, cy = concept.centroid_xy
        color = color_for_tag(concept.dominant_tag)
        radius = 0.16 + 0.045 * len(concept.member_positions)
        for px, py in concept.member_positions:
            ax.plot([px, cx], [py, cy], color=color, alpha=0.16, linewidth=1.0)
            ax.scatter([px], [py], s=14, color=color, alpha=0.55, zorder=2)
        circ = plt.Circle((cx, cy), radius=radius, fill=False, color=color, linewidth=2.0 if concept.concept_id in important_ids else 1.2, alpha=0.95)
        ax.add_patch(circ)
        if concept.concept_id in important_ids or concept.concept_id == consumer.current_target_concept:
            cell = field.get(concept.concept_id)
            water = 0.0 if cell is None else cell.channels.get("water_source", 0.0)
            hazard = 0.0 if cell is None else cell.channels.get("hazard_edge", 0.0)
            ax.text(cx + 0.10, cy + 0.10, f"{concept.concept_id}\nW={water:.2f} H={hazard:.2f}", fontsize=7)

    if consumer.current_target_concept:
        target = next((c for c in concepts if c.concept_id == consumer.current_target_concept), None)
        if target is not None:
            x0, y0 = consumer.current_pos
            x1, y1 = target.centroid_xy
            ax.plot([x0, x1], [y0, y1], color=consumer.color, linestyle="--", linewidth=2.2)
            ax.scatter([x0], [y0], s=90, color=consumer.color, edgecolor="black", zorder=5)
            ax.scatter([x1], [y1], s=140, facecolors="none", edgecolors=consumer.color, linewidth=2.2, zorder=6)
            ax.text(x0 + 0.15, y0 + 0.15, "consumer query", fontsize=8, color=consumer.color)


def draw_field_bars(ax, concepts: Sequence[SharedConcept], field: Dict[str, FieldCell]) -> None:
    ranked = sorted(concepts, key=lambda c: concept_importance(c, field), reverse=True)[:6]
    labels = [c.concept_id for c in ranked]
    water = [field[c.concept_id].channels.get("water_source", 0.0) for c in ranked]
    hazard = [field[c.concept_id].channels.get("hazard_edge", 0.0) for c in ranked]
    neutral = [field[c.concept_id].channels.get("safe_neutral", 0.0) for c in ranked]

    y = list(range(len(ranked)))
    ax.barh([v + 0.22 for v in y], water, height=0.22, color="#2b8cff", label="water")
    ax.barh(y, hazard, height=0.22, color="#d62728", label="hazard")
    ax.barh([v - 0.22 for v in y], neutral, height=0.22, color="#7f7f7f", label="neutral")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 0.85)
    ax.set_title("Semantic field: top concept activations")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("activation")


def make_routes() -> Dict[str, List[GridPos]]:
    return {
        "scout-A": [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3), (4, 3), (4, 4), (5, 4), (6, 4)],
        "scout-B": [(9, 7), (8, 7), (7, 7), (7, 6), (7, 5), (6, 5), (5, 5), (4, 5), (4, 4), (4, 3), (3, 3)],
        "consumer-C": [(5, 0), (5, 1), (5, 2), (4, 2), (3, 2), (2, 2), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3)],
    }


def concept_summary(concepts: Sequence[SharedConcept], field: Dict[str, FieldCell]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for c in concepts:
        cell = field.get(c.concept_id)
        if cell is None:
            continue
        out[c.concept_id] = dict(cell.channels)
    return out


def step_events(
    step_idx: int,
    observations: Sequence[PlaceObservation],
    concepts: Sequence[SharedConcept],
    consumer_target: Optional[str],
    state: Dict[str, object],
) -> List[str]:
    events: List[str] = []
    seen_water_agents = state.setdefault("seen_water_agents", set())
    seen_hazard_agents = state.setdefault("seen_hazard_agents", set())
    seen_water_concept = state.setdefault("seen_water_concept", False)
    last_target = state.setdefault("last_target", None)

    for obs in observations:
        if obs.semantic_tags.get("water_source", 0.0) >= 0.95 and obs.agent_id not in seen_water_agents:
            seen_water_agents.add(obs.agent_id)
            events.append(f"{obs.agent_id} confirms water at {obs.pos}")
        if obs.semantic_tags.get("hazard_edge", 0.0) >= 0.95 and obs.agent_id not in seen_hazard_agents:
            seen_hazard_agents.add(obs.agent_id)
            events.append(f"{obs.agent_id} marks hazard at {obs.pos}")

    water_concepts = [c for c in concepts if c.dominant_tag == "water_source"]
    if water_concepts and not seen_water_concept:
        state["seen_water_concept"] = True
        top = max(water_concepts, key=lambda c: len(c.supporting_agents))
        events.append(f"shared concept formed: {top.concept_id} (agents: {', '.join(top.supporting_agents)})")

    if consumer_target and consumer_target != last_target:
        state["last_target"] = consumer_target
        events.append(f"consumer-C selects {consumer_target}")

    if not events:
        events.append(f"memory refresh at step {step_idx}")
    return events


def save_gif(frames: Sequence[str], gif_path: str, fps: float) -> None:
    if not frames:
        return
    duration_ms = int(1000 / max(0.1, fps))
    images = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frames]
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=duration_ms,
        loop=0,
    )


def run_experiment(out_dir: str, n_steps: int = 11, fps: float = 1.5, dpi: int = 140) -> None:
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
    frame_paths: List[str] = []
    event_state: Dict[str, object] = {}

    for step_idx in range(n_steps):
        observations: List[PlaceObservation] = []
        for agent in agents:
            obs = agent.step(step_idx, world)
            observations.append(obs)
            bus.publish(obs)

        concepts = build_concepts(bus, current_step=step_idx)
        field = build_field(concepts, previous=prev_field)
        prev_field = field

        consumer = next(agent for agent in agents if agent.agent_id == "consumer-C")
        consumer.current_target_concept = best_water_concept(concepts, field)
        banner_events = step_events(step_idx, observations, concepts, consumer.current_target_concept, event_state)

        history.append(
            {
                "step_idx": step_idx,
                "events": banner_events,
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
                "field": concept_summary(concepts, field),
                "consumer_target_concept": consumer.current_target_concept,
            }
        )

        fig = plt.figure(figsize=(16.0, 9.4))
        gs = fig.add_gridspec(2, 3, hspace=0.24, wspace=0.22)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[1, 2])

        draw_world(ax0, world, agents, step_idx)
        draw_local_graph(ax1, agents[0].local_graph, "Scout-A local memory graph", world)
        draw_local_graph(ax2, agents[1].local_graph, "Scout-B local memory graph", world)
        draw_local_graph(ax3, agents[2].local_graph, "Consumer-C local memory graph", world)
        draw_shared_memory(ax4, world, concepts, field, consumer)
        draw_field_bars(ax5, concepts, field)

        fig.suptitle("Collective Memory Cinema: local graphs → shared concepts → semantic field", fontsize=17, y=0.985)
        banner = " • ".join(banner_events[:3])
        fig.text(0.5, 0.948, banner, ha="center", va="center", fontsize=10, color="#333333")

        frame_path = os.path.join(frames_dir, f"frame_{step_idx:03d}.png")
        fig.savefig(frame_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(frame_path)

    gif_path = os.path.join(out_dir, "collective_memory_cinema.gif")
    save_gif(frame_paths, gif_path, fps=fps)

    summary = {
        "n_steps": n_steps,
        "n_events": len(bus.events),
        "n_collective_places": len(bus.beliefs),
        "final_concepts": history[-1]["concepts"],
        "consumer_final_target_concept": history[-1]["consumer_target_concept"],
        "frames_dir": frames_dir,
        "gif_path": gif_path,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as fh:
        json.dump(history, fh, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved visual experiment: local graphs, shared concepts and semantic field")
    parser.add_argument("--out_dir", type=str, default="tmp/memory_cinema_demo_v2")
    parser.add_argument("--steps", type=int, default=11)
    parser.add_argument("--fps", type=float, default=1.5)
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()
    run_experiment(out_dir=args.out_dir, n_steps=args.steps, fps=args.fps, dpi=args.dpi)
