import hashlib
import json
import os
from collections import deque

import numpy as np


class SymbolicTokenizer:
    def __init__(self, mode="argmax", proj_dim=32, seed=1):
        self.mode = mode
        self.proj_dim = proj_dim
        self.rng = np.random.RandomState(seed)
        self.proj = None

    def _ensure_proj(self, dim):
        if self.proj is None:
            self.proj = self.rng.normal(0.0, 1.0, size=(dim, self.proj_dim))

    def encode(self, embedding):
        z = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if z.size == 0:
            return 0

        if self.mode == "argmax":
            return int(np.argmax(z))

        if self.mode == "hash_sign":
            self._ensure_proj(z.shape[0])
            z = np.matmul(z, self.proj)
            bits = (z > 0).astype(np.uint8).tobytes()
            token = hashlib.blake2b(bits, digest_size=8).digest()
            return int.from_bytes(token, "little", signed=False)

        raise ValueError("Unknown tokenizer mode: {}".format(self.mode))


class LZMapMemory:
    def __init__(self, min_goal_visits=3):
        self.min_goal_visits = min_goal_visits
        self.dictionary = {}
        self.nodes = {}
        self.edges = {}
        self.current_phrase = []
        self.previous_phrase_id = None
        self.current_phrase_id = None

        self.token_sequence = []
        self.completed_phrases = []
        self.node_growth = []
        self.edge_growth = []
        self.interventions = 0
        self.intervention_attempts = 0
        self.plan_hits = 0
        self.plan_total = 0

    def _new_node(self, phrase):
        node_id = len(self.dictionary)
        self.dictionary[phrase] = node_id
        self.nodes[node_id] = {
            "phrase": phrase,
            "visits": 0,
            "reward_sum": 0.0,
            "reward_count": 0,
            "pose_sum": np.zeros(2, dtype=np.float64),
            "pose_count": 0,
            "goal_sum": np.zeros(2, dtype=np.float64),
            "goal_count": 0,
            "last_seen_step": -1,
        }
        self.edges[node_id] = {}
        return node_id

    def update_token(self, token, step_idx):
        self.token_sequence.append(int(token))
        self.current_phrase.append(int(token))
        phrase_key = tuple(self.current_phrase)

        new_phrase = False
        if phrase_key not in self.dictionary:
            new_phrase = True
            phrase_id = self._new_node(phrase_key)
            self.completed_phrases.append(phrase_id)
            self.current_phrase_id = phrase_id
            if self.previous_phrase_id is not None:
                self.edges[self.previous_phrase_id][phrase_id] = \
                    self.edges[self.previous_phrase_id].get(phrase_id, 0) + 1
            self.previous_phrase_id = phrase_id
            self.current_phrase = []

        self.node_growth.append(len(self.nodes))
        edge_count = 0
        for src in self.edges:
            edge_count += len(self.edges[src])
        self.edge_growth.append(edge_count)
        return new_phrase

    def observe(self, step_idx, pose_xy=None, goal_xy=None, reward=None):
        if self.current_phrase_id is None:
            return

        node = self.nodes[self.current_phrase_id]
        node["visits"] += 1
        node["last_seen_step"] = int(step_idx)

        if pose_xy is not None:
            node["pose_sum"] += np.asarray(pose_xy, dtype=np.float64)
            node["pose_count"] += 1

        if goal_xy is not None:
            node["goal_sum"] += np.asarray(goal_xy, dtype=np.float64)
            node["goal_count"] += 1

        if reward is not None:
            node["reward_sum"] += float(reward)
            node["reward_count"] += 1

    def _mean_reward(self, node_id):
        node = self.nodes[node_id]
        if node["reward_count"] == 0:
            return -np.inf
        return node["reward_sum"] / max(1, node["reward_count"])

    def _bfs_shortest_path_len(self, src, targets):
        if src in targets:
            return 0, src
        seen = set([src])
        q = deque([(src, 0)])
        while q:
            cur, dist = q.popleft()
            for nxt in self.edges.get(cur, {}):
                if nxt in seen:
                    continue
                if nxt in targets:
                    return dist + 1, nxt
                seen.add(nxt)
                q.append((nxt, dist + 1))
        return None, None

    def suggest_subgoal(self, top_k=5):
        self.intervention_attempts += 1
        if self.current_phrase_id is None:
            return None

        scored = []
        for node_id, node in self.nodes.items():
            if node["visits"] < self.min_goal_visits:
                continue
            if node["goal_count"] == 0:
                continue
            mean_reward = self._mean_reward(node_id)
            if np.isneginf(mean_reward):
                continue
            scored.append((mean_reward, node_id))

        if not scored:
            return None

        scored.sort(reverse=True, key=lambda x: x[0])
        candidate_nodes = [nid for _, nid in scored[:top_k]]
        path_len, selected = self._bfs_shortest_path_len(
            self.current_phrase_id, set(candidate_nodes)
        )
        if selected is None:
            selected = candidate_nodes[0]
            path_len = 0

        node = self.nodes[selected]
        goal_xy = node["goal_sum"] / max(1, node["goal_count"])
        self.interventions += 1
        return {
            "goal_xy": goal_xy,
            "node_id": selected,
            "path_len": int(path_len),
            "mean_reward": float(self._mean_reward(selected)),
        }

    def record_plan_outcome(self, improved):
        self.plan_total += 1
        if improved:
            self.plan_hits += 1

    def export(self, out_dir, env_idx):
        env_dir = os.path.join(out_dir, "env_{}".format(env_idx))
        os.makedirs(env_dir, exist_ok=True)

        with open(os.path.join(env_dir, "token_sequence.json"), "w") as f:
            json.dump(self.token_sequence, f)

        phrases = []
        for node_id in sorted(self.nodes.keys()):
            node = self.nodes[node_id]
            phrases.append({
                "node_id": node_id,
                "phrase": list(node["phrase"]),
                "length": len(node["phrase"]),
                "visits": node["visits"],
                "mean_reward": None if node["reward_count"] == 0 else
                node["reward_sum"] / node["reward_count"],
                "last_seen_step": node["last_seen_step"],
            })
        with open(os.path.join(env_dir, "phrases.json"), "w") as f:
            json.dump(phrases, f)

        edge_list = []
        for src, dsts in self.edges.items():
            for dst, w in dsts.items():
                edge_list.append({"src": src, "dst": dst, "weight": int(w)})
        with open(os.path.join(env_dir, "graph_edges.json"), "w") as f:
            json.dump(edge_list, f)

        metrics = {
            "num_tokens": len(self.token_sequence),
            "num_nodes": len(self.nodes),
            "num_edges": len(edge_list),
            "intervention_rate": 0.0 if self.intervention_attempts == 0 else
            self.interventions / self.intervention_attempts,
            "plan_hit_rate": 0.0 if self.plan_total == 0 else
            self.plan_hits / self.plan_total,
        }
        with open(os.path.join(env_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        plt = None
        try:
            import matplotlib.pyplot as plt_mod
            plt = plt_mod
        except Exception:
            plt = None

        if plt is not None and len(self.nodes) > 0:
            lengths = [len(node["phrase"]) for node in self.nodes.values()]
            plt.figure(figsize=(5, 3))
            plt.hist(lengths, bins=min(20, max(1, len(set(lengths)))))
            plt.title("Phrase Lengths")
            plt.xlabel("Length")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(env_dir, "phrase_length_hist.png"))
            plt.close()

        if plt is not None and len(self.node_growth) > 0:
            plt.figure(figsize=(6, 3))
            plt.plot(self.node_growth, label="nodes")
            plt.plot(self.edge_growth, label="edges")
            plt.title("Graph Growth")
            plt.xlabel("Step")
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(env_dir, "growth_curves.png"))
            plt.close()

        try:
            import networkx as nx
            g = nx.DiGraph()
            goal_candidates = []
            for node_id in self.nodes:
                g.add_node(node_id)
                if self.nodes[node_id]["reward_count"] > 0 and \
                        self.nodes[node_id]["visits"] >= self.min_goal_visits:
                    goal_candidates.append(node_id)
            for src, dsts in self.edges.items():
                for dst, w in dsts.items():
                    g.add_edge(src, dst, weight=w)
            if plt is not None and g.number_of_nodes() > 0:
                plt.figure(figsize=(7, 5))
                pos = nx.spring_layout(g, seed=7)
                nx.draw_networkx_edges(g, pos, alpha=0.4, arrows=True)
                nx.draw_networkx_nodes(
                    g, pos, nodelist=list(g.nodes()), node_size=140,
                    node_color="#7ba6d8"
                )
                if goal_candidates:
                    nx.draw_networkx_nodes(
                        g, pos, nodelist=goal_candidates, node_size=180,
                        node_color="#e06666"
                    )
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(os.path.join(env_dir, "graph_layout.png"))
                plt.close()
        except Exception:
            pass
