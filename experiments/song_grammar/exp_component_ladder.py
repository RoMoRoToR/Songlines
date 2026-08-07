"""CL — staged component ladder of Songline Memory Runtime v1.

Reviewer: the integrated runtime stacks too many mechanisms
(formation + quarantine + receiver-side admission + reservations +
staleness + provenance) for ONE causal conclusion --- stage them, and
run leave-one-component-out from the full system.  This driver runs
ONE runtime (``songlines`` Config flags, the I1 registry --- no
hand-forked copies) on the I1 substrate with paired assignment
streams per seed (the world's RNG is consumed identically by every
configuration, so arms are compared on the same family/drift
history, seeds 0..N as in S1-S3).

Own-memory FORMATION (utility gate, exceptions, immutable layer) is
identical at every step; only exchange-side components move.  Where
a step coincides with a registry configuration this is noted --- the
step IS that configuration, not a re-implementation.

Ladder:
  L1 independent        no communication (I1 "independent"; the
                        non-comm agent keeps world_clock off on its
                        own memory --- the S3 lesson).
  L2 +raw exchange      broadcast + consolidate, no guards: world
                        clock off, provenance off, admission "none",
                        reservations off (= registry "song_plain";
                        the exchange-side analogue of S1's finding).
  L3 +staleness         + world-clock / referent-version gate at
                        both ends (= S2's contract, without
                        provenance).
  L4 +provenance        + origin-bound records and trust-flip links
                        (= registry "song_wclock").
  L5 +quarantine(hold)  foreign certificates quarantined FOREVER:
                        the quarantine mechanism exists but no
                        admission validation does yet, so nothing
                        foreign is ever consumed.  Equivalence probe:
                        this should be "independent with metadata".
  L6 +admission         + receiver-side validation on own visits,
                        utility measured not told (admission "util";
                        = registry "no_reservations"; ~ S3 adm_util).
  L7 +reservations      the full system (= registry "songline_full").

Leave-one-component-out from L7:
  -admission      admission "none"      (= registry "no_admission")
  -quarantine     validate-in-place: foreign records are admitted
                  and CONSUMABLE immediately, validated
                  retroactively on the receiver's next visit to that
                  family (failures evicted from consumption).
                  Isolates the fail-closed HOLDING property from the
                  validation property.
  -staleness      world_clock off       (= registry "no_worldclock")
  -reservations   identical configuration to L6 by construction; it
                  is run again under its LOO name and the pairwise
                  equality of rows doubles as a determinism check.
  -provenance     provenance off        (= registry "no_provenance")

Registered predictions (frozen before any run; thresholds refer to
the full 12-seed x 300-episode x 6-agent run; the local smoke at
~100 episodes is a direction check --- S1-S3 effects grow with
horizon):
  CL.1 (raw exchange harms): L2 group cost > L1 on both roles.
  CL.2 (staged recovery at the boundaries S2/S3 certified):
       L3 < L2 on both roles; L6 < L3 and L6 < L1 on both roles ---
       exchange turns net-positive only once admission exists.
  CL.3 (quarantine-forever = independent + metadata): L5 within 5%
       of L1 on both roles and foreign_admitted == 0.  Any residual
       gap is the world-clock gate acting on the agent's OWN records
       (version news gossips in even though schemas never leave
       quarantine) --- report it as the price of the gate itself.
  CL.4 (reservations compose): L7 duplicate rate <= 0.5 x L6's at
       group cost within 3%.
  CL.5 (LOO ranking): -admission is the single largest cost
       degradation from full on both roles (I1 measured +22/+38%);
       -staleness second; -quarantine >= full on fail-open or cost
       (phantoms can fire between receipt and validation);
       -provenance and -reservations within noise on cost but worse
       than full on >= 1 of {fail-open, duplicates, foreign volume}.
  CL.6 (expected null, registered as such): L4 ~ L3 on group cost
       --- provenance is unloaded insurance in a non-adversarial
       stream (P1/I1 lesson); its value is NOT claimed here.

Metrics per cell: per-role mean cumulative cost, duplicate rate,
memory items, foreign_seen / foreign_admitted / quarantined_end,
fail-open, first-lock success, wire bits, memory bits.

Usage (seed-sharded)::

    PYTHONPATH=. python experiments/song_grammar/exp_component_ladder.py \
        --seeds 0 4 --episodes 100 --agents 6 \
        --out tmp/component_ladder_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from experiments.song_grammar.exp_i1_integration import (
    CADENCE, CONTENTION, INTENTS, World, build_song_cfg, make_fp,
    walk)
from experiments.song_grammar.exp_s0_song_smoke import (
    BAND, TRAVELER_START)
from experiments.song_grammar.runtime import (
    RESV_BITS, U_THR, Config, Record, SonglineAgent, record_bits,
    song_target)
from experiments.song_grammar.u7_common import ROLES, dijkstra

GridXY = Tuple[int, int]
FULL = Config()


def _c(**kw) -> Config:
    return replace(FULL, **kw)


# name -> (Config, communicating?, adm_mode, ladder_step)
# adm_mode: "cfg" defers to Config.admission; "hold" quarantines
# forever; "in_place" admits immediately + validates retroactively.
CONFIGS: Dict[str, Tuple[Config, bool, str, Optional[int]]] = {
    "l1_independent": (FULL, False, "cfg", 1),
    "l2_raw_exchange": (_c(world_clock=False, provenance=False,
                           admission="none", reservations=False),
                        True, "cfg", 2),
    "l3_staleness": (_c(provenance=False, admission="none",
                        reservations=False), True, "cfg", 3),
    "l4_provenance": (_c(admission="none", reservations=False),
                      True, "cfg", 4),
    "l5_quarantine_hold": (_c(reservations=False), True, "hold", 5),
    "l6_admission": (_c(reservations=False), True, "cfg", 6),
    "l7_full": (FULL, True, "cfg", 7),
    # leave-one-component-out from l7_full
    "loo_no_admission": (_c(admission="none"), True, "cfg", None),
    "loo_no_quarantine": (FULL, True, "in_place", None),
    "loo_no_staleness": (_c(world_clock=False), True, "cfg", None),
    "loo_no_reservations": (_c(reservations=False), True, "cfg",
                            None),
    "loo_no_provenance": (_c(provenance=False), True, "cfg", None),
}


class LadderAgent(SonglineAgent):
    """SonglineAgent + the two admission variants the registry lacks.

    ``hold``: the quarantine mechanism without any admission
    validation --- foreign certificates are held forever (version
    news still travels: metadata is cheap).
    ``in_place``: admission validation without the quarantine ---
    foreign records enter consumption immediately on receipt and are
    validated retroactively at the receiver's next visit to their
    family; failures are evicted from consumption (role_u < 0), not
    deleted, so exception parent links stay valid.
    """

    def __init__(self, aid: int, role_name: str, cfg: Config,
                 adm_mode: str):
        super().__init__(aid, role_name, cfg)
        self.adm_mode = adm_mode
        self.pending: set = set()          # uids awaiting validation
        self.validated = 0
        self.evicted = 0

    def receive(self, rec: Record, sender: Optional[int] = None
                ) -> None:
        if self.adm_mode == "cfg":
            super().receive(rec, sender)
            return
        if rec.uid in self.received:
            return
        if self.cfg.provenance and sender is not None \
                and rec.origin != sender:
            return
        self.received.add(rec.uid)
        self.note_version(rec.family, rec.version)
        if self.adm_mode == "hold":        # held forever, never used
            self.quarantine.setdefault(rec.family, []).append(rec)
            return
        # in_place: straight into consumption, flagged for later
        # validation on the next own visit to this family
        n_before = len(self.records)
        self._admit(rec, rec.role_u)
        if len(self.records) > n_before:
            self.pending.add(rec.uid)

    def on_visit(self, env, fam: int, ver: int, t: int,
                 utility_fn) -> None:
        if self.adm_mode == "cfg":
            super().on_visit(env, fam, ver, t, utility_fn)
            return
        self.visited.add(fam)
        self.note_version(fam, ver)
        if self.adm_mode == "hold":        # nothing ever leaves
            return
        # in_place: retroactive validation of pending records about
        # this family, on the actual current world with the own role
        for r in self.records:
            if r.uid not in self.pending or r.family != fam:
                continue
            self.pending.discard(r.uid)
            if self.cfg.world_clock and r.version < \
                    self.known_version.get(fam, -1):
                r.role_u[self.role_name] = -1.0   # stale: evict
                self.evicted += 1
                continue
            # exclude the record under test from its own baseline
            # (it is already consumable --- that is the point of the
            # ablation); other pending records stay in the baseline,
            # which is the honest price of having no quarantine
            r.role_u[self.role_name] = -1.0
            u = utility_fn(env, self, r.song, r.intent)
            if u < U_THR:
                self.evicted += 1                 # useless: evicted
            else:
                r.role_u[self.role_name] = u      # measured, not told
                self.validated += 1


def run_cell(name: str, seed: int, n_agents: int, n_episodes: int
             ) -> Dict[str, Any]:
    cfg, comm, adm_mode, step = CONFIGS[name]
    rng = np.random.default_rng(seed * 7 + 1)
    roles = ["fragile" if i % 2 == 0 else "robust"
             for i in range(n_agents)]
    agents = [LadderAgent(i, roles[i], cfg, adm_mode)
              for i in range(n_agents)]
    if not comm:
        # the independent baseline stays strong: no world-clock gate
        # on one's own memory (the S3 lesson, as in I1)
        for ag in agents:
            ag.cfg = replace(cfg, world_clock=False)
    fpf = make_fp(0.0, rng)
    world = World(seed)
    stats = {"cost": {r: [] for r in ROLES}, "dup": 0, "commits": 0,
             "phantom": 0, "refused": 0, "succ": 0, "wire_bits": 0,
             "n": 0}

    def utility_fn(env, agent, song, intent):
        band_fps = {xy: fpf(env, xy) for xy in BAND}
        base = agent.targets(band_fps, intent)
        cand = song_target(song, band_fps, agent.cfg.sim_threshold)
        probe = base + ([cand] if cand is not None else [])
        role = ROLES[agent.role_name]
        kind = INTENTS[intent]
        return (walk(env, base, role, kind)["cost"]
                - walk(env, probe, role, kind)["cost"])

    for t in range(n_episodes):
        assignments = [world.assign() for _ in range(n_agents)]
        episode_targets: Dict[GridXY, int] = {}
        for i, (fam, env, tg, ver) in enumerate(assignments):
            ag = agents[i]
            intent = "water" if (t + i) % 3 else "rest"
            ag.now = t
            ag.on_visit(env, fam, ver, t, utility_fn)
            band_fps = {xy: fpf(env, xy) for xy in BAND}
            targets = ag.targets(band_fps, intent)
            if cfg.reservations and comm and targets:
                stats["wire_bits"] += RESV_BITS
                targets = [tt for tt in targets
                           if episode_targets.get(tt) in (None, i)]
            r = walk(env, targets, ROLES[roles[i]], INTENTS[intent])
            cost = r["cost"]
            if targets:
                stats["commits"] += 1
                first = targets[0]
                if episode_targets.get(first, i) != i:
                    stats["dup"] += 1
                    if not (cfg.reservations and comm):
                        cost += CONTENTION
                else:
                    episode_targets[first] = i
            stats["cost"][roles[i]].append(cost)
            stats["phantom"] += int(r["phantom"])
            stats["refused"] += int(r["refused"])
            stats["succ"] += int(r["success_first"])
            stats["n"] += 1
            path, _ = dijkstra(env, TRAVELER_START, tg[intent],
                               ROLES[roles[i]])
            if path is None:
                continue
            song = build_song_cfg(env, path, fpf, ag.cfg)
            role_u = {rn: utility_fn(env, ag, song, intent) if rn ==
                      ag.role_name else 0.0 for rn in ROLES}
            ag.form(song, intent, fam, ver, t, role_u)

        if comm and t % CADENCE == CADENCE - 1:
            for i, ag in enumerate(agents):
                for rec in ag.outbox(t - CADENCE):
                    stats["wire_bits"] += record_bits(rec, ag.cfg)
                    for j, other in enumerate(agents):
                        if j != i:
                            other.receive(rec, sender=i)

    def foreign_admitted(ag: LadderAgent) -> int:
        return sum(1 for r in ag.records if r.origin != ag.aid
                   and r.role_u.get(ag.role_name, 0.0) >= 0.0)

    return {"config": name, "ladder_step": step, "seed": seed,
            "agents": n_agents, "episodes": n_episodes,
            "group_cost": {r: float(np.mean(v))
                           for r, v in stats["cost"].items()},
            "duplicate_rate": stats["dup"] / max(1, stats["commits"]),
            "success_first": stats["succ"] / stats["n"],
            "fail_open": stats["phantom"] / stats["n"],
            "refusal": stats["refused"] / stats["n"],
            "wire_bits": stats["wire_bits"],
            "memory_bits": float(np.mean([a.memory_bits()
                                          for a in agents])),
            "mem_items": float(np.mean([len(a.records)
                                        for a in agents])),
            "foreign_seen": float(np.mean([len(a.received)
                                           for a in agents])),
            "foreign_admitted": float(np.mean([foreign_admitted(a)
                                               for a in agents])),
            "quarantined_end": float(np.mean(
                [sum(len(v) for v in a.quarantine.values())
                 for a in agents])),
            "validated": float(np.mean([a.validated
                                        for a in agents])),
            "evicted": float(np.mean([a.evicted for a in agents]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 4])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--configs", type=str, nargs="*", default=None,
                    help="subset of configuration names (sharding)")
    ap.add_argument("--out", type=str,
                    default="tmp/component_ladder_smoke")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reg = os.path.join(a.out, "cl_registered.json")
    if not os.path.exists(reg):
        with open(reg, "w") as f:
            json.dump({
                "CL.1": "L2 > L1 group cost, both roles",
                "CL.2": "L3 < L2; L6 < L3 and L6 < L1, both roles",
                "CL.3": "L5 within 5% of L1, foreign_admitted == 0",
                "CL.4": "L7 duplicates <= 0.5 x L6 at cost within 3%",
                "CL.5": "LOO: -admission largest cost degradation "
                        "(both roles), -staleness second; "
                        "-quarantine >= full on fail-open or cost; "
                        "-provenance/-reservations worse on >=1 of "
                        "{fail-open, duplicates, foreign volume}",
                "CL.6": "registered null: L4 ~ L3 on cost "
                        "(provenance unloaded without an adversary)",
                "protocol": "paired assignment streams per seed; one "
                            "runtime, arms are Config flags "
                            "(registry) + two admission variants "
                            "(hold, in_place) over the same runtime; "
                            "loo_no_reservations == l6_admission by "
                            "construction (determinism check); "
                            "thresholds frozen for 12 x e300 x N6",
            }, f, indent=2)
    names = a.configs or list(CONFIGS)
    shard = (f"cl_e{a.episodes}_a{a.agents}"
             f"_s{a.seeds[0]}-{a.seeds[1]}.jsonl")
    if a.configs:
        shard = shard.replace(".jsonl",
                              f"_{'_'.join(a.configs)}.jsonl")
    with open(os.path.join(a.out, shard), "w") as f:
        for seed in range(a.seeds[0], a.seeds[1]):
            for name in names:
                row = run_cell(name, seed, a.agents, a.episodes)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"seed {seed} {name}: "
                      f"frag {row['group_cost']['fragile']:.0f} "
                      f"rob {row['group_cost']['robust']:.0f} "
                      f"dup {row['duplicate_rate']:.3f} "
                      f"adm {row['foreign_admitted']:.0f} "
                      f"quar {row['quarantined_end']:.0f}",
                      flush=True)
    print(f"Saved: {a.out}/{shard}")


if __name__ == "__main__":
    main()
