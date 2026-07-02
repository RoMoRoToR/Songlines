"""Warp Drive protocol — optimistic lock + reservation + anti-M rollback.

Design §7.  Three primitives, all decentralised:

  1. Optimistic warp lock — locks on foreign evidence stay allowed
     immediately (unchanged planner behaviour), but the protocol records
     them as pending reservations.
  2. Warp reservation — at the owner's NEXT broadcast tick (same cadence
     K as the memory broadcasts; reservations are a field piggybacked on
     the snapshot, no extra channel and no centre) the reservation is
     delivered to every peer.  Receivers apply occupancy suppression to
     reserved targets when querying.
  3. Anti-M* rollback (Time Warp semantics) — when an agent holds a lock
     on a target for which a higher-priority foreign reservation has
     arrived, an anti-M* event retracts the lock: the planner re-queries,
     the target enters an exponential-backoff cooldown (hysteresis), and
     rollback latency is logged.

Priority is (reserved_at_tick, agent_id) — earlier reservation wins,
ties break by agent id.  Oscillation is damped by the backoff (4 ticks,
doubling per repeated retraction of the same target, capped at 32).

Decentralisation invariant: every decision an agent's slot makes uses
ONLY that agent's own inbox/table.  ``WarpDriveProtocol`` itself is a
passive router exactly like ``BroadcastBus`` — it moves messages on the
broadcast cadence and never aggregates.  ``exp_warp_drive.py`` contains
the 'remove the runtime' check: the same protocol driven by direct
per-agent calls produces identical decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

XY_TOL = 0.6


def _close(a, b, tol: float = XY_TOL) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


@dataclass
class Reservation:
    owner: str
    target: Tuple[int, int]
    reserved_at: int          # tick of the original lock
    distance: int = 0         # manhattan(owner, target) at lock time
    delivered_at: int = -1

    @property
    def priority(self) -> Tuple[int, int, str]:
        # Earlier lock wins; simultaneous locks go to the closer agent
        # (minimises wasted travel — the misled far agent should be the
        # one to retract); agent id is the final deterministic tie-break.
        return (self.reserved_at, self.distance, self.owner)


@dataclass
class WarpDriveAgentSlot:
    """Per-agent protocol state.  Never reads any other slot."""

    agent_id: str
    # own pending/active reservation (follows the agent's current lock)
    own: Optional[Reservation] = None
    own_announced: bool = False
    # foreign reservations as THIS agent knows them: target -> Reservation
    table: Dict[Tuple[int, int], Reservation] = field(default_factory=dict)
    # backoff: target -> (suppressed_until_tick, n_retractions)
    backoff: Dict[Tuple[int, int], Tuple[int, int]] = field(
        default_factory=dict)

    def receive(self, res: Reservation, tick: int) -> None:
        """Merge one incoming reservation (or release) into the table."""
        key = res.target
        if res.reserved_at < 0:          # release marker
            cur = self.table.get(key)
            if cur is not None and cur.owner == res.owner:
                del self.table[key]
            return
        cur = self.table.get(key)
        if cur is None or res.priority < cur.priority:
            self.table[key] = Reservation(
                res.owner, res.target, res.reserved_at,
                distance=res.distance, delivered_at=tick)

    def blocking_reservation(self, target, my_priority) -> Optional[Reservation]:
        for key, res in self.table.items():
            if _close(key, target) and res.owner != self.agent_id \
                    and res.priority < my_priority:
                return res
        return None

    def suppressed(self, target, tick: int) -> bool:
        for key, (until, _n) in self.backoff.items():
            if _close(key, target) and tick < until:
                return True
        return False

    def add_backoff(self, target, tick: int,
                    base: int = 4, cap: int = 32) -> None:
        key = (int(round(target[0])), int(round(target[1])))
        _until, n = self.backoff.get(key, (0, 0))
        duration = min(cap, base * (2 ** n))
        self.backoff[key] = (tick + duration, n + 1)


class WarpDriveProtocol:
    """Passive reservation router + per-agent rollback logic.

    Hook interface consumed by ``warp_runner.run_warp_episode``:
      on_lock / on_success / on_tick / filter_targets / stats.
    """

    def __init__(self, agent_ids: List[str], broadcast_every_k: int,
                 *, reservation_ttl_factor: int = 4):
        self.k = max(1, int(broadcast_every_k))
        self.slots: Dict[str, WarpDriveAgentSlot] = {
            aid: WarpDriveAgentSlot(aid) for aid in agent_ids}
        self.ttl = self.k * reservation_ttl_factor
        # diagnostics
        self.n_reservations = 0
        self.n_retractions = 0
        self.rollback_latencies: List[int] = []
        self._retraction_log: List[Dict[str, Any]] = []

    # ── hooks ─────────────────────────────────────────────────────

    def on_lock(self, agent_id: str, target: Tuple[int, int],
                tick: int, distance: int = 0) -> None:
        slot = self.slots[agent_id]
        if slot.own is not None and _close(slot.own.target, target):
            return  # same reservation continues
        slot.own = Reservation(agent_id, (int(round(target[0])),
                                          int(round(target[1]))), tick,
                               distance=distance)
        slot.own_announced = False
        self.n_reservations += 1

    def on_success(self, agent_id: str, xy, tick: int) -> None:
        # Keep the reservation: a completed target IS occupied.  (The
        # planner's own occupied-skip provides the same signal once the
        # winner physically arrives; the reservation delivers it K-early.)
        slot = self.slots[agent_id]
        if slot.own is not None:
            slot.own_announced = False  # re-announce as occupancy

    def filter_targets(self, agent_id: str, targets: List, tick: int) -> List:
        slot = self.slots[agent_id]
        my_pri = (slot.own.priority if slot.own
                  else (tick, 0, agent_id))
        out = []
        for t in targets:
            if slot.suppressed(t, tick):
                continue
            if slot.blocking_reservation(t, my_pri) is not None:
                continue
            out.append(t)
        return out

    def on_tick(self, tick: int, locks: Dict[str, Optional[Tuple[int, int]]]
                ) -> Dict[str, str]:
        """Deliver due reservations, then compute per-slot retractions.

        ``locks`` is only used to keep each slot's OWN reservation in
        sync with its own planner lock (an agent always knows its own
        lock) — no slot ever reads another agent's lock.
        """
        # 1. sync own reservations with own locks; queue releases
        outgoing: List[Reservation] = []
        for aid, slot in self.slots.items():
            lock = locks.get(aid)
            if slot.own is not None and (lock is None
                                         or not _close(slot.own.target, lock)):
                # own lock dropped/retargeted → release on this tick's wave
                outgoing.append(Reservation(aid, slot.own.target, -1))
                slot.own = None
                slot.own_announced = False

        # 2. broadcast wave (same cadence as memory broadcasts)
        if (tick + 1) % self.k == 0:
            for aid, slot in self.slots.items():
                if slot.own is not None:
                    outgoing.append(Reservation(
                        aid, slot.own.target, slot.own.reserved_at,
                        distance=slot.own.distance))
                    slot.own_announced = True
            for res in outgoing:
                for aid, slot in self.slots.items():
                    if aid != res.owner:
                        slot.receive(res, tick)

        # 3. expire stale foreign reservations (owner may be stuck/dead)
        for slot in self.slots.values():
            stale = [key for key, res in slot.table.items()
                     if res.delivered_at >= 0 and tick - res.delivered_at > self.ttl]
            for key in stale:
                del slot.table[key]

        # 4. per-slot anti-M*: my lock vs my table
        retractions: Dict[str, str] = {}
        for aid, slot in self.slots.items():
            lock = locks.get(aid)
            if lock is None or slot.own is None:
                continue
            blocker = slot.blocking_reservation(
                slot.own.target, slot.own.priority)
            if blocker is not None:
                retractions[aid] = (f"anti-M*: {blocker.owner} reserved "
                                    f"{blocker.target} at {blocker.reserved_at}")
                slot.add_backoff(slot.own.target, tick)
                self.n_retractions += 1
                self.rollback_latencies.append(tick - slot.own.reserved_at)
                self._retraction_log.append({
                    "agent": aid, "tick": tick,
                    "target": list(slot.own.target),
                    "blocker": blocker.owner,
                    "latency": tick - slot.own.reserved_at,
                })
                slot.own = None
        return retractions

    # ── diagnostics ───────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        lat = self.rollback_latencies
        return {
            "wd_n_reservations": self.n_reservations,
            "wd_n_retractions": self.n_retractions,
            "wd_mean_rollback_latency": (sum(lat) / len(lat)
                                         if lat else float("nan")),
            "wd_max_rollback_latency": max(lat) if lat else float("nan"),
        }

    @property
    def retraction_log(self) -> List[Dict[str, Any]]:
        return list(self._retraction_log)
