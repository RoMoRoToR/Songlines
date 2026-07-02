"""Part 2.2 — minimal CommNet-style multi-agent policy.

CommNet (Sukhbaatar et al. 2016) at its core: agents share a hidden state via a
learned mean-pooled communication vector, broadcast every step.  We implement a
minimal version on top of our existing multi-agent grid environment.

Architecture per agent (shared weights):
  obs (radius-2 local view, encoded as flat features) → encoder (MLP) → h_t
  comm_in = mean(h_t over peers)
  h_t' = MLP([h_t, comm_in])
  policy head → action distribution (4 actions: TURN_LEFT, TURN_RIGHT, FORWARD, NOOP)
  value head → scalar
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Observation features per agent (per tick):
#  - one-hot agent direction (4)
#  - cell types in radius-2 cells (5 types × (2*r+1)^2 cells = 5*25 = 125)
#  - normalised (x, y) position (2)
# Total: 4 + 125 + 2 = 131
OBS_DIM = 131
H_DIM = 64
COMM_DIM = 16


def _xy_normalised(x: int, y: int, W: int, H: int) -> np.ndarray:
    return np.array([x / max(1, W - 1), y / max(1, H - 1)], dtype=np.float32)


def _encode_local_window(env, agent_id: str, radius: int = 2) -> np.ndarray:
    from multiagent_env import EMPTY, WALL, WATER, HAZARD, GOAL

    ag = env.agents[agent_id]
    cells_per_side = 2 * radius + 1
    one_hot = np.zeros((cells_per_side, cells_per_side, 5), dtype=np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            cx, cy = ag.x + dx, ag.y + dy
            if 0 <= cx < env.width and 0 <= cy < env.height:
                v = env.cell(cx, cy)
            else:
                v = WALL
            cell_type = {EMPTY: 0, WALL: 1, WATER: 2, HAZARD: 3, GOAL: 4}.get(v, 0)
            one_hot[dy + radius, dx + radius, cell_type] = 1.0
    return one_hot.flatten()


def encode_observation(env, agent_id: str) -> np.ndarray:
    ag = env.agents[agent_id]
    dir_one_hot = np.zeros(4, dtype=np.float32)
    dir_one_hot[ag.direction] = 1.0
    window = _encode_local_window(env, agent_id, radius=2)
    pos = _xy_normalised(ag.x, ag.y, env.width, env.height)
    return np.concatenate([dir_one_hot, window, pos]).astype(np.float32)


class CommNetPolicy(nn.Module):
    """Shared-weights CommNet policy for any number of agents."""

    def __init__(self, n_actions: int = 4, h_dim: int = H_DIM, comm_dim: int = COMM_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(OBS_DIM, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.comm_proj = nn.Linear(h_dim, comm_dim)
        self.combine = nn.Sequential(
            nn.Linear(h_dim + comm_dim, h_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(h_dim, n_actions)
        self.critic = nn.Linear(h_dim, 1)
        self.n_actions = n_actions

    def forward(self, obs_batch: torch.Tensor):
        """obs_batch shape: (N_agents, OBS_DIM).

        Returns:
            action_logits: (N_agents, n_actions)
            value:         (N_agents,)
            comm_vec:      (N_agents, comm_dim) — useful for logging
        """
        h = self.encoder(obs_batch)
        c_each = self.comm_proj(h)
        # Mean of others — exclude self
        N = h.shape[0]
        if N > 1:
            comm_in = (c_each.sum(dim=0, keepdim=True) - c_each) / (N - 1)
        else:
            comm_in = torch.zeros_like(c_each)
        h2 = self.combine(torch.cat([h, comm_in], dim=-1))
        logits = self.actor(h2)
        value = self.critic(h2).squeeze(-1)
        return logits, value, c_each
