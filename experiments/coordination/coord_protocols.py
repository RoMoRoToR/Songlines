"""Coordination-baseline protocols for the Warp Drive benchmark (Package A, part 1).

Reviewer claim addressed: "reservation is an obvious engineering patch —
compare against standard coordination alternatives."  Five alternative
conflict-resolution arms are implemented against the SAME hook interface
that ``experiments/warp/warp_runner.run_warp_episode`` exposes for
``WarpDriveProtocol`` (on_lock / on_success / on_tick / filter_targets /
stats), so every arm runs on the identical benchmark with identical
information:

  * agents learn about each other's commitments ONLY from announcements
    piggybacked on the broadcast wave (same cadence K as the memory
    broadcasts) — exactly the channel Warp Drive reservations use;
  * no arm ever reads another agent's private state; the only ``env``
    access is an agent reading its OWN position (own knowledge);
  * conflict *detection* is therefore delayed by up to K ticks for every
    arm, including Warp Drive.

Arms
----
  none       measurement-only null protocol (no coordination; identical
             behaviour to running without a protocol)
  wd         Warp Drive (optimistic lock + reservation + anti-M* rollback),
             wrapped with the same message/duplicate accounting
  random     random tie-break: on a co-lock revealed by the broadcast, the
             loser is chosen equiprobably via a deterministic seeded coin
             per (target, agent pair) — both parties compute the same coin
             locally, no extra round trip
  nearest    nearest-agent-wins: the agent FARTHER from the target at lock
             time yields (announced lock distance; ties by agent id)
  greedy     decentralised greedy assignment ("Hungarian-lite"): on each
             broadcast wave every agent locally computes a greedy
             agent→target assignment over the announced (position, lock)
             pairs it knows, and yields if the assignment gives its target
             to someone else
  backoff    random exponential backoff WITHOUT reservations: on a
             detected co-lock the agent retracts and suppresses the target
             for a random duration drawn from a doubling window; no
             foreign-lock table is used for target filtering
  occupancy  soft occupancy penalty: a target with announced foreign locks
             is penalised by +LAMBDA cells in the query score — never
             banned, never retracted

Message accounting
------------------
The memory broadcast itself is identical across arms and is NOT counted;
we count only the coordination payload piggybacked on each wave.  Byte
schema (12x10 grid): agent id = 1 B, coordinate = 1 B each, tick = 2 B,
distance = 1 B.  Per-announcement payloads:

  wd         owner + (x,y) + reserved_at + distance          = 6 B
             release marker: owner + (x,y) + flag            = 4 B
  random     owner + (x,y)                                   = 3 B
  nearest    owner + (x,y) + distance                        = 4 B
  greedy     owner + (x,y) + (px,py)                         = 5 B
  backoff    owner + (x,y)                                   = 3 B
  occupancy  owner + (x,y)                                   = 3 B

``coord_n_msgs`` counts announcements placed on the wave (a broadcast is
one message per announcement, delivered to N-1 peers → ``coord_n_deliveries``).

Duplicate-commitment accounting (all arms) is measurement-only: it uses
the runner-provided ``locks`` dict inside ``on_tick`` for *diagnostics*,
never for decisions.
"""

from __future__ import annotations

import hashlib
import statistics
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from experiments.warp.warp_drive import WarpDriveProtocol

XY_TOL = 0.6

# Soft penalty (in cells of Manhattan distance) added per announced
# foreign lock on a candidate target for the ``occupancy`` arm.
OCCUPANCY_LAMBDA = 6.0

# Exponential-backoff constants (mirror WarpDriveProtocol defaults).
BACKOFF_BASE = 4
BACKOFF_CAP = 32

ANNOUNCE_BYTES = {
    "wd": 6,
    "wd_release": 4,
    "random": 3,
    "nearest": 4,
    "greedy": 5,
    "backoff": 3,
    "occupancy": 3,
    "none": 0,
}


def _close(a, b, tol: float = XY_TOL) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def _key(t) -> Tuple[int, int]:
    return (int(round(t[0])), int(round(t[1])))


def _stable_hash(*parts) -> int:
    key = "|".join(str(p) for p in parts).encode()
    return int(hashlib.md5(key).hexdigest()[:8], 16)


# ─────────────────────────────────────────────── accounting meters


@dataclass
class MessageMeter:
    n_agents: int
    n_waves: int = 0
    n_msgs: int = 0
    n_bytes: int = 0

    def wave(self) -> None:
        self.n_waves += 1

    def add(self, n_msgs: int, n_bytes: int) -> None:
        self.n_msgs += n_msgs
        self.n_bytes += n_bytes

    @property
    def n_deliveries(self) -> int:
        return self.n_msgs * max(0, self.n_agents - 1)


class DuplicateMeter:
    """Counts duplicate commitments from the runner's locks dict.

    Measurement only — never consulted by any protocol decision.
    ``dup_lock_ticks``  = sum over ticks of (holders - 1) per contested
                          target (agent-ticks wasted on duplicated locks).
    ``dup_lock_events`` = number of transitions of a target into a
                          contested (>= 2 holders) state.
    """

    def __init__(self) -> None:
        self.dup_lock_ticks = 0
        self.dup_lock_events = 0
        self._prev_contested: set = set()

    def measure(self, locks: Dict[str, Optional[Tuple[int, int]]]) -> None:
        groups: Dict[Tuple[int, int], int] = {}
        for lock in locks.values():
            if lock is None:
                continue
            k = _key(lock)
            groups[k] = groups.get(k, 0) + 1
        contested = {k for k, n in groups.items() if n >= 2}
        self.dup_lock_ticks += sum(groups[k] - 1 for k in contested)
        self.dup_lock_events += len(contested - self._prev_contested)
        self._prev_contested = contested


# ─────────────────────────────────────────────── announcement / slot


@dataclass
class LockAnnouncement:
    owner: str
    target: Tuple[int, int]
    lock_tick: int
    distance: int                    # manhattan(owner, target) at lock time
    pos: Tuple[int, int]             # owner position at lock time
    delivered_at: int = -1


@dataclass
class CoordSlot:
    """Per-agent protocol state — never reads any other slot."""

    agent_id: str
    own: Optional[LockAnnouncement] = None
    # foreign announcements as THIS agent knows them (owner -> announcement);
    # refreshed wholesale on every broadcast wave (implicit release: an
    # owner that stopped announcing disappears from the next wave).
    table: Dict[str, LockAnnouncement] = field(default_factory=dict)
    # target -> (suppressed_until_tick, n_retractions)
    backoff: Dict[Tuple[int, int], Tuple[int, int]] = field(
        default_factory=dict)

    def backoff_active(self, target, tick: int) -> bool:
        for k, (until, _n) in self.backoff.items():
            if _close(k, target) and tick < until:
                return True
        return False

    def add_backoff(self, target, tick: int, duration: int) -> None:
        k = _key(target)
        _until, n = self.backoff.get(k, (0, 0))
        self.backoff[k] = (tick + duration, n + 1)

    def n_collisions(self, target) -> int:
        return self.backoff.get(_key(target), (0, 0))[1]


# ─────────────────────────────────────────────── base protocol


class CoordProtocolBase:
    """Shared machinery: announcement waves, own-lock sync, accounting.

    Subclasses implement ``_blocked`` (target filtering against the
    foreign-lock table) and ``_resolve_conflict`` (own-lock retraction).
    """

    arm = "base"

    def __init__(self, agent_ids: List[str], broadcast_every_k: int,
                 *, env=None, seed: int = 0):
        self.k = max(1, int(broadcast_every_k))
        self.env = env
        self.seed = seed
        self.slots: Dict[str, CoordSlot] = {
            aid: CoordSlot(aid) for aid in agent_ids}
        self.meter = MessageMeter(len(agent_ids))
        self.dup = DuplicateMeter()
        self.n_retractions = 0
        self.rollback_latencies: List[int] = []

    # own knowledge: an agent may always read ITS OWN position
    def _own_pos(self, aid: str) -> Tuple[int, int]:
        ag = self.env.agents[aid]
        return (ag.x, ag.y)

    # ── hooks consumed by run_warp_episode ────────────────────────

    def on_lock(self, agent_id: str, target, tick: int,
                distance: int = 0) -> None:
        slot = self.slots[agent_id]
        t = _key(target)
        if slot.own is not None and _close(slot.own.target, t):
            return
        slot.own = LockAnnouncement(agent_id, t, tick, int(distance),
                                    self._own_pos(agent_id))

    def on_success(self, agent_id: str, xy, tick: int) -> None:
        pass  # planner's occupied-skip covers completed targets (all arms)

    def filter_targets(self, agent_id: str, targets: List, tick: int) -> List:
        slot = self.slots[agent_id]
        out = []
        for t in targets:
            if slot.backoff_active(t, tick):
                continue
            if self._blocked(slot, t, tick):
                continue
            out.append(t)
        return out

    def on_tick(self, tick: int,
                locks: Dict[str, Optional[Tuple[int, int]]]) -> Dict[str, str]:
        self.dup.measure(locks)          # diagnostics only

        # 1. sync own announcement with own lock (own knowledge only)
        for aid, slot in self.slots.items():
            lock = locks.get(aid)
            if slot.own is not None and (lock is None
                                         or not _close(slot.own.target, lock)):
                slot.own = None

        # 2. broadcast wave — same cadence as the memory broadcasts
        if (tick + 1) % self.k == 0:
            self.meter.wave()
            anns = [slot.own for slot in self.slots.values()
                    if slot.own is not None]
            self.meter.add(len(anns), len(anns) * ANNOUNCE_BYTES[self.arm])
            for slot in self.slots.values():
                slot.table = {
                    a.owner: replace(a, delivered_at=tick)
                    for a in anns if a.owner != slot.agent_id}

        # 3. per-slot conflict resolution: my lock vs my table
        retractions: Dict[str, str] = {}
        for aid, slot in self.slots.items():
            if slot.own is None or locks.get(aid) is None:
                continue
            reason = self._resolve_conflict(slot, tick)
            if reason is not None:
                self.n_retractions += 1
                self.rollback_latencies.append(tick - slot.own.lock_tick)
                self._on_retract(slot, tick)
                slot.own = None
                retractions[aid] = reason
        return retractions

    # ── subclass extension points ─────────────────────────────────

    def _blocked(self, slot: CoordSlot, target, tick: int) -> bool:
        return False

    def _resolve_conflict(self, slot: CoordSlot,
                          tick: int) -> Optional[str]:
        return None

    def _on_retract(self, slot: CoordSlot, tick: int) -> None:
        pass

    # ── diagnostics ───────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        lat = self.rollback_latencies
        return {
            "coord_arm": self.arm,
            "coord_n_retractions": self.n_retractions,
            "coord_mean_rollback_latency": (statistics.mean(lat) if lat
                                            else float("nan")),
            "coord_n_waves": self.meter.n_waves,
            "coord_n_msgs": self.meter.n_msgs,
            "coord_n_bytes": self.meter.n_bytes,
            "coord_n_deliveries": self.meter.n_deliveries,
            "dup_lock_ticks": self.dup.dup_lock_ticks,
            "dup_lock_events": self.dup.dup_lock_events,
        }


# ─────────────────────────────────────────────── arm: none (null)


class NullProtocol(CoordProtocolBase):
    """Measurement-only passthrough — identical behaviour to no protocol."""

    arm = "none"

    def on_lock(self, agent_id, target, tick, distance=0):
        pass

    def filter_targets(self, agent_id, targets, tick):
        return targets

    def on_tick(self, tick, locks):
        self.dup.measure(locks)
        return {}


# ─────────────────────────────────────────────── arm: random tie-break


class RandomTieBreakProtocol(CoordProtocolBase):
    """Loser of a co-lock chosen equiprobably by a deterministic seeded
    random priority per (target, agent): lower hash wins.  Both parties
    compute the same result from broadcast-known information, so the
    resolution needs no extra communication round.  The per-target
    priority is a total random order over agents — equiprobable for any
    pair AND transitive, so multi-way conflicts cannot form
    non-transitive cycles (a pairwise coin can produce i>j>k>i, making
    ALL contenders retract simultaneously and livelock)."""

    arm = "random"

    def _foreign_wins(self, my_id: str, ann: LockAnnouncement) -> bool:
        t = _key(ann.target)
        h_me = _stable_hash("rndtie", self.seed, t, my_id)
        h_other = _stable_hash("rndtie", self.seed, t, ann.owner)
        return (h_other, ann.owner) < (h_me, my_id)

    def _blocked(self, slot, target, tick) -> bool:
        return any(_close(ann.target, target)
                   and self._foreign_wins(slot.agent_id, ann)
                   for ann in slot.table.values())

    def _resolve_conflict(self, slot, tick) -> Optional[str]:
        for ann in slot.table.values():
            if _close(ann.target, slot.own.target) \
                    and self._foreign_wins(slot.agent_id, ann):
                return f"random tie-break lost to {ann.owner}"
        return None


# ─────────────────────────────────────────────── arm: nearest-agent-wins


class NearestWinsProtocol(CoordProtocolBase):
    """The agent farther from the target at lock time yields.  Distances
    are the announced lock distances (broadcast-known to both parties);
    ties break by agent id.  When filtering without an own lock the
    agent's CURRENT distance is the hypothetical lock distance."""

    arm = "nearest"

    def _foreign_wins(self, my_id: str, my_dist: int,
                      ann: LockAnnouncement) -> bool:
        return (ann.distance, ann.owner) < (my_dist, my_id)

    def _blocked(self, slot, target, tick) -> bool:
        pos = self._own_pos(slot.agent_id)
        my_dist = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
        return any(_close(ann.target, target)
                   and self._foreign_wins(slot.agent_id, int(my_dist), ann)
                   for ann in slot.table.values())

    def _resolve_conflict(self, slot, tick) -> Optional[str]:
        for ann in slot.table.values():
            if _close(ann.target, slot.own.target) \
                    and self._foreign_wins(slot.agent_id,
                                           slot.own.distance, ann):
                return (f"nearest-wins lost to {ann.owner} "
                        f"(d={ann.distance} < {slot.own.distance})")
        return None


# ─────────────────────────────────────────────── arm: greedy assignment


class GreedyAssignmentProtocol(CoordProtocolBase):
    """Decentralised greedy assignment (Hungarian-lite).

    On each wave every agent locally computes a greedy agent→target
    assignment over the participants it knows: itself (fresh own
    position) plus all announcers (positions/targets as announced,
    stale by up to K ticks — the honest cost of cadence-limited
    information).  Pairs are sorted by (distance, agent id, target) and
    assigned greedily, one target per agent.  An agent yields its lock
    when the assignment gives its target to someone else, and filters
    out targets assigned to others."""

    arm = "greedy"

    def _assignment(self, slot: CoordSlot) -> Dict[Tuple[int, int], str]:
        agents: Dict[str, Tuple[int, int]] = {
            slot.agent_id: self._own_pos(slot.agent_id)}
        targets = set()
        if slot.own is not None:
            targets.add(_key(slot.own.target))
        for ann in slot.table.values():
            agents[ann.owner] = ann.pos
            targets.add(_key(ann.target))
        pairs = sorted(
            (abs(p[0] - t[0]) + abs(p[1] - t[1]), aid, t)
            for aid, p in agents.items() for t in targets)
        assigned_agent: set = set()
        assignment: Dict[Tuple[int, int], str] = {}
        for _d, aid, t in pairs:
            if aid in assigned_agent or t in assignment:
                continue
            assignment[t] = aid
            assigned_agent.add(aid)
        return assignment

    def _blocked(self, slot, target, tick) -> bool:
        assignment = self._assignment(slot)
        owner = assignment.get(_key(target))
        return owner is not None and owner != slot.agent_id

    def _resolve_conflict(self, slot, tick) -> Optional[str]:
        assignment = self._assignment(slot)
        owner = assignment.get(_key(slot.own.target))
        if owner is not None and owner != slot.agent_id:
            return f"greedy assignment gives {slot.own.target} to {owner}"
        return None


# ─────────────────────────────────────────────── arm: random backoff


class RandomBackoffProtocol(CoordProtocolBase):
    """No reservations: the foreign-lock table is used ONLY to detect a
    co-lock at the wave; on detection the agent retracts and suppresses
    the target for a random duration drawn uniformly from a doubling
    window [1, BASE·2^n] (capped), seeded per (agent, target, round).
    Resolution comes from random desynchronisation plus physical
    arrival — no standing claim is ever honoured."""

    arm = "backoff"

    def _blocked(self, slot, target, tick) -> bool:
        return False        # no reservations — only own backoff applies

    def _resolve_conflict(self, slot, tick) -> Optional[str]:
        for ann in slot.table.values():
            if _close(ann.target, slot.own.target):
                return f"co-lock with {ann.owner} → random backoff"
        return None

    def _on_retract(self, slot, tick) -> None:
        n = slot.n_collisions(slot.own.target)
        window = min(BACKOFF_CAP, BACKOFF_BASE * (2 ** n))
        h = _stable_hash("backoff", self.seed, slot.agent_id,
                         _key(slot.own.target), n)
        duration = 1 + h % window
        slot.add_backoff(slot.own.target, tick, duration)


# ─────────────────────────────────────────────── arm: occupancy penalty


class OccupancyPenaltyProtocol(CoordProtocolBase):
    """Soft penalty, no bans, no retractions: each announced foreign lock
    on a candidate adds +LAMBDA cells to its query score.  The planner
    picks the nearest returned candidate, so the penalised argmin is
    materialised by returning the penalised-best candidate plus every
    strictly-farther one (an exact emulation of ``argmin(dist + penalty)``
    through the existing filter interface — a penalised target CAN still
    be chosen when no alternative comes within LAMBDA per contender)."""

    arm = "occupancy"

    def __init__(self, *args, lam: float = OCCUPANCY_LAMBDA, **kwargs):
        super().__init__(*args, **kwargs)
        self.lam = lam

    def filter_targets(self, agent_id, targets, tick):
        slot = self.slots[agent_id]
        if not targets:
            return targets
        pos = self._own_pos(agent_id)

        def raw(t):
            return abs(pos[0] - t[0]) + abs(pos[1] - t[1])

        def penalty(t):
            return self.lam * sum(1 for ann in slot.table.values()
                                  if _close(ann.target, t))

        best = min(targets, key=lambda t: (raw(t) + penalty(t), raw(t),
                                           _key(t)))
        return [best] + [t for t in targets
                         if t is not best and raw(t) > raw(best)]

    def on_tick(self, tick, locks):
        self.dup.measure(locks)
        for aid, slot in self.slots.items():
            lock = locks.get(aid)
            if slot.own is not None and (lock is None
                                         or not _close(slot.own.target, lock)):
                slot.own = None
        if (tick + 1) % self.k == 0:
            self.meter.wave()
            anns = [slot.own for slot in self.slots.values()
                    if slot.own is not None]
            self.meter.add(len(anns), len(anns) * ANNOUNCE_BYTES[self.arm])
            for slot in self.slots.values():
                slot.table = {a.owner: replace(a, delivered_at=tick)
                              for a in anns if a.owner != slot.agent_id}
        return {}           # never retracts


# ─────────────────────────────────────────────── arm: warp drive (counted)


class CountingWarpDrive(WarpDriveProtocol):
    """WarpDriveProtocol + the same message/duplicate accounting as the
    baseline arms.  Reservation payload = 6 B, release marker = 4 B
    (reservations carry reserved_at + distance — the extra fields the
    baselines do not pay for).  Decision logic is untouched."""

    arm = "wd"

    def __init__(self, agent_ids: List[str], broadcast_every_k: int,
                 *, env=None, seed: int = 0, **kwargs):
        super().__init__(agent_ids, broadcast_every_k, **kwargs)
        self.meter = MessageMeter(len(agent_ids))
        self.dup = DuplicateMeter()

    def on_tick(self, tick, locks):
        self.dup.measure(locks)
        if (tick + 1) % self.k == 0:
            self.meter.wave()
            # releases queued on this tick's wave (own lock dropped/changed)
            n_rel = sum(
                1 for aid, slot in self.slots.items()
                if slot.own is not None
                and (locks.get(aid) is None
                     or not _close(slot.own.target, locks[aid])))
            # reservations that survive the release sync get re-announced
            n_res = sum(
                1 for aid, slot in self.slots.items()
                if slot.own is not None and locks.get(aid) is not None
                and _close(slot.own.target, locks[aid]))
            self.meter.add(n_rel + n_res,
                           n_rel * ANNOUNCE_BYTES["wd_release"]
                           + n_res * ANNOUNCE_BYTES["wd"])
        return super().on_tick(tick, locks)

    def stats(self) -> Dict[str, Any]:
        out = super().stats()
        lat = self.rollback_latencies
        out.update({
            "coord_arm": self.arm,
            "coord_n_retractions": self.n_retractions,
            "coord_mean_rollback_latency": (statistics.mean(lat) if lat
                                            else float("nan")),
            "coord_n_waves": self.meter.n_waves,
            "coord_n_msgs": self.meter.n_msgs,
            "coord_n_bytes": self.meter.n_bytes,
            "coord_n_deliveries": self.meter.n_deliveries,
            "dup_lock_ticks": self.dup.dup_lock_ticks,
            "dup_lock_events": self.dup.dup_lock_events,
        })
        return out


# ─────────────────────────────────────────────── factory


ARMS = ["none", "wd", "random", "nearest", "greedy", "backoff", "occupancy"]

_ARM_CLASSES = {
    "none": NullProtocol,
    "wd": CountingWarpDrive,
    "random": RandomTieBreakProtocol,
    "nearest": NearestWinsProtocol,
    "greedy": GreedyAssignmentProtocol,
    "backoff": RandomBackoffProtocol,
    "occupancy": OccupancyPenaltyProtocol,
}


def make_protocol(arm: str, agent_ids: List[str], broadcast_every_k: int,
                  *, env=None, seed: int = 0):
    if arm not in _ARM_CLASSES:
        raise ValueError(f"Unknown coordination arm: {arm} (choose from {ARMS})")
    return _ARM_CLASSES[arm](agent_ids, broadcast_every_k, env=env, seed=seed)
