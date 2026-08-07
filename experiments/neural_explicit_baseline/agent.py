"""Neural agent with EXPLICIT Q/R/M/C interfaces (reviewer package J).

The reviewer claim being answered: "CommNet does not expose stages because it
has no interface".  This policy is a purely neural (no symbolic pipeline)
agent whose architecture is factored into four learned heads that each emit a
loggable, verifiable event:

  Q -- query head       : Categorical over a discrete query vocabulary
                          (require-tag tokens + NO_QUERY).  Emitting a token
                          != NO_QUERY is a Q-event.
  R -- learned retriever: scores ALL entries of the external append-only
                          memory against the query embedding + agent context;
                          the top-k candidates are an R-event.  "candidate
                          |= query" is checkable (entry.tag == query tag),
                          but is NOT enforced -- selection is fully learned.
  M -- target-lock head : Categorical over [keep-current-lock] + k retrieved
                          candidates.  Choosing a candidate COMMITS a lock,
                          written into env-visible state
                          (env.agents[aid].locked_target) -- an M-event.
                          The lock logit mixes an independent lock scorer
                          with the retriever score, so lock != top-1
                          retrieval structurally (R and M are distinct heads).
  C -- learned controller: low-level motor actions (turn/forward/noop)
                          conditioned on obs encoding + lock features.
                          Arrival at the locked cell is a C-event.

Observation encoding (dim 131) is shared with the CommNet/MAPPO baselines
(experiments/commnet_baseline/commnet_agent.encode_observation) so success
curves are comparable.  There is NO inter-agent communication channel: the
only coupling between agents is the shared external memory.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from experiments.commnet_baseline.commnet_agent import OBS_DIM  # 131
from experiments.neural_explicit_baseline.memory_store import (
    ENTRY_FEAT_DIM, QUERY_VOCAB,
)

H_DIM = 64
QUERY_EMB_DIM = 16
ENTRY_ENC_DIM = 32
LOCK_FEAT_DIM = 5  # has_lock, dx_norm, dy_norm, manhattan_norm, arrived
N_MOTOR_ACTIONS = 4  # TURN_LEFT, TURN_RIGHT, FORWARD, NOOP


class NeuralExplicitPolicy(nn.Module):
    """Four explicit heads over a shared observation encoder."""

    def __init__(self, obs_dim: int = OBS_DIM, h_dim: int = H_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU(),
        )
        # Q head: discrete structured query (len(QUERY_VOCAB) == 5 tokens)
        self.query_head = nn.Linear(h_dim, len(QUERY_VOCAB))
        self.query_emb = nn.Embedding(len(QUERY_VOCAB), QUERY_EMB_DIM)
        # R head: learned retriever scorer over memory entries
        self.entry_enc = nn.Sequential(
            nn.Linear(ENTRY_FEAT_DIM, ENTRY_ENC_DIM), nn.ReLU(),
            nn.Linear(ENTRY_ENC_DIM, ENTRY_ENC_DIM),
        )
        self.retr_scorer = nn.Sequential(
            nn.Linear(ENTRY_ENC_DIM + QUERY_EMB_DIM + h_dim, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        # M head: independent lock scorer (+ retriever score mixed in outside)
        self.lock_scorer = nn.Sequential(
            nn.Linear(ENTRY_ENC_DIM + h_dim, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.no_lock_head = nn.Linear(h_dim, 1)  # logit of "keep / no lock"
        # C head: motor controller conditioned on lock features
        self.controller = nn.Sequential(
            nn.Linear(h_dim + LOCK_FEAT_DIM, 64), nn.ReLU(),
            nn.Linear(64, N_MOTOR_ACTIONS),
        )
        self.value_head = nn.Sequential(
            nn.Linear(h_dim + LOCK_FEAT_DIM, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    # ── shared ─────────────────────────────────────────────────────────
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    # ── Q ──────────────────────────────────────────────────────────────
    def query_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.query_head(h)

    # ── R ──────────────────────────────────────────────────────────────
    def retrieval_scores(self, h: torch.Tensor, query_tok: torch.Tensor,
                         entry_feats: torch.Tensor) -> torch.Tensor:
        """Score entries against (query, context).

        h           : [B, H]        query_tok : [B] long
        entry_feats : [B, E, ENTRY_FEAT_DIM]
        returns     : [B, E]
        """
        B, E, _ = entry_feats.shape
        enc = self.entry_enc(entry_feats)                       # [B, E, 32]
        q = self.query_emb(query_tok).unsqueeze(1).expand(B, E, QUERY_EMB_DIM)
        ctx = h.unsqueeze(1).expand(B, E, h.shape[-1])
        return self.retr_scorer(torch.cat([enc, q, ctx], -1)).squeeze(-1)

    # ── M ──────────────────────────────────────────────────────────────
    def lock_logits(self, h: torch.Tensor, cand_feats: torch.Tensor,
                    cand_retr_scores: torch.Tensor,
                    cand_mask: torch.Tensor) -> torch.Tensor:
        """Logits over [keep/no-lock] + k candidates.

        h [B,H], cand_feats [B,K,F], cand_retr_scores [B,K],
        cand_mask [B,K] bool (True = real candidate).  Returns [B, K+1].
        """
        B, K, _ = cand_feats.shape
        enc = self.entry_enc(cand_feats)                        # [B, K, 32]
        ctx = h.unsqueeze(1).expand(B, K, h.shape[-1])
        own = self.lock_scorer(torch.cat([enc, ctx], -1)).squeeze(-1)
        cand_logits = own + cand_retr_scores                    # gradient -> R
        cand_logits = cand_logits.masked_fill(~cand_mask, float("-inf"))
        keep = self.no_lock_head(h)                             # [B, 1]
        return torch.cat([keep, cand_logits], dim=-1)

    # ── C ──────────────────────────────────────────────────────────────
    def motor_logits(self, h: torch.Tensor,
                     lock_feat: torch.Tensor) -> torch.Tensor:
        return self.controller(torch.cat([h, lock_feat], -1))

    def value(self, h: torch.Tensor, lock_feat: torch.Tensor) -> torch.Tensor:
        return self.value_head(torch.cat([h, lock_feat], -1)).squeeze(-1)


def lock_features(agent_xy: Tuple[int, int],
                  lock_xy: Optional[Tuple[int, int]],
                  width: int, height: int) -> torch.Tensor:
    """LOCK_FEAT_DIM vector describing the current committed lock."""
    f = torch.zeros(LOCK_FEAT_DIM)
    if lock_xy is None:
        return f
    ax, ay = agent_xy
    lx, ly = lock_xy
    f[0] = 1.0
    f[1] = (lx - ax) / max(1, width - 1)
    f[2] = (ly - ay) / max(1, height - 1)
    f[3] = (abs(lx - ax) + abs(ly - ay)) / (width + height)
    f[4] = 1.0 if (lx, ly) == (ax, ay) else 0.0
    return f
