"""Package I --- full resource accounting on the B-unified benchmark.

Reviewer claim addressed: "equal memory/comm budgets are not equal
RESOURCE budgets; graph memory may be more expensive in compute".
This driver re-runs the exact `exp_b_unified.run_cell` cells (same
worlds, same seeds, same arms) with a metering proxy wrapped around
every arm, and reports, per arm and per run:

  1. stored bytes         -- actual serialized (pickle) size of the
                             memory store, sampled every --snap-every
                             episodes (growth curve) + final;
  2. transmitted bytes    -- full-protocol wire bits (song codec +
                             certificate + provenance + time/version
                             for songline_full; song codec + 32-bit
                             time for the raw-sharing baselines ---
                             their protocol genuinely carries no
                             metadata), plus an explicit estimated
                             reservation-traffic column (B-unified
                             resolves contention in the env, so
                             reservation messages are NOT on the wire
                             there; I1 charges RESV_BITS per commit
                             --- we surface the same estimate here);
  3. update time          -- wall-clock inside observe() + receive()
                             (formation, merge, admission,
                             consolidation), time.perf_counter;
  4. query latency        -- wall-clock inside driver-level targets()
                             (retrieval + frame-free alignment);
  5. CPU-seconds          -- time.process_time delta per run;
  6. model/LLM calls      -- 0 by construction for all six arms
                             (deterministic; no LLM client is imported
                             anywhere in baselines.py / songlines/*);
                             recorded explicitly so the table says so;
  7. amortized formation  -- total formation wall-time divided by the
                             number of retrievals that actually
                             consumed memory (non-empty targets), and
                             by the number of stored entries.

No source file is modified: the proxy is installed by monkey-patching
`exp_b_unified.make_policy` for the duration of one cell and restored
in a finally block.  Task-utility columns (team cost, success,
fail-open) come from run_cell's own return row, so the utility side
of the Pareto table is byte-identical to the published benchmark.

Usage (seed-sharded, mirrors exp_b_unified)::

    PYTHONPATH=. python experiments/song_grammar/exp_resource_accounting.py \
        --policy songline_full --seeds 100 103 --episodes 60 \
        --agents 6 --out tmp/resource_accounting_smoke

Then aggregate: scripts/analyze_resource_accounting.py --dir <out>.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# read-only imports of the benchmark under measurement
import experiments.song_grammar.exp_b_unified as bu
from songlines.record import RESV_BITS, bits_of_song

ALL_POLICIES = ("independent", "songline_full", "decision_centric",
                "execution_path", "graph_memory", "learned_formation")


# ── serialization of each arm's memory store ────────────────────────

def stored_state(inner) -> Any:
    """The arm's actual long-term store (what would be persisted).

    SonglineArm: schema records + immutable episodic layer +
    quarantine (all are memory the agent must hold).  Baselines: the
    flat item list.  Protocol bookkeeping (received-uid set,
    known_version) is included as `proto` so dedup state is not
    silently free.
    """
    ag = getattr(inner, "ag", None)
    if ag is not None:                               # SonglineArm
        return {"records": ag.records, "episodic": ag.episodic,
                "quarantine": ag.quarantine,
                "proto": (ag.received, ag.known_version, ag.visited)}
    return {"items": inner.items,
            "proto": getattr(inner, "received", None)}


def stored_bytes(inner) -> int:
    return len(pickle.dumps(stored_state(inner), protocol=4))


def n_entries(inner) -> int:
    ag = getattr(inner, "ag", None)
    if ag is not None:
        q = sum(len(v) for v in ag.quarantine.values())
        return len(ag.records) + len(ag.episodic) + q
    return len(inner.items)


# ── meter + proxy ───────────────────────────────────────────────────

class Meter:
    def __init__(self, snap_every: int):
        self.snap_every = snap_every
        self.observe_s = 0.0
        self.receive_s = 0.0
        self.query_s = 0.0
        self.outbox_s = 0.0
        self.n_observe = 0
        self.n_receive = 0
        self.n_query = 0
        self.n_query_nonempty = 0
        self.n_messages = 0            # broadcasts (records sent once)
        self.n_deliveries = 0          # per-receiver deliveries
        self.wire_bits_payload = 0     # pure song codec
        self.wire_bits_full = 0        # arm's own full protocol
        self.llm_calls = 0             # no arm has an LLM client
        self.snap: Dict[int, List[int]] = {}   # episode -> [bytes/agent]
        self.arms: List[Any] = []


class MeteredArm:
    """Timing/byte-counting proxy with the exact driver-facing API of
    the arms in exp_b_unified.run_cell.  Inner calls made by the
    benchmark's utility_fn go to the RAW inner agent (run_cell passes
    it directly), so counterfactual utility probes are charged to
    formation (observe) time --- where the benchmark incurs them ---
    and never double-counted as query latency."""

    def __init__(self, inner, meter: Meter):
        self._inner = inner
        self._m = meter
        meter.arms.append(self)

    # -- formation / consolidation (update time) --
    def observe(self, env, intent, song, fam, ver, t, role_name,
                utility_fn):
        m = self._m
        t0 = time.perf_counter()
        self._inner.observe(env, intent, song, fam, ver, t, role_name,
                            utility_fn)
        m.observe_s += time.perf_counter() - t0
        m.n_observe += 1
        if m.snap_every and t % m.snap_every == m.snap_every - 1:
            m.snap.setdefault(t, []).append(stored_bytes(self._inner))

    def receive(self, rec, sender):
        m = self._m
        t0 = time.perf_counter()
        self._inner.receive(rec, sender)
        m.receive_s += time.perf_counter() - t0
        m.n_receive += 1
        m.n_deliveries += 1

    # -- retrieval (query latency) --
    def targets(self, band_fps, intent, role_name):
        m = self._m
        t0 = time.perf_counter()
        out = self._inner.targets(band_fps, intent, role_name)
        m.query_s += time.perf_counter() - t0
        m.n_query += 1
        if out:
            m.n_query_nonempty += 1
        return out

    # -- exchange --
    def outbox(self, since):
        m = self._m
        t0 = time.perf_counter()
        recs = self._inner.outbox(since)
        m.outbox_s += time.perf_counter() - t0
        for rec in recs:
            m.n_messages += 1
            m.wire_bits_payload += bits_of_song(rec["song"])
            if hasattr(self._inner, "wire_bits"):     # full protocol
                m.wire_bits_full += self._inner.wire_bits(rec)
            else:                                     # raw baselines
                m.wire_bits_full += bits_of_song(rec["song"]) + 32
        return recs

    def memory_bits(self):
        return self._inner.memory_bits()

    def __getattr__(self, name):
        # exposes wire_bits (and anything else) only when the inner
        # arm has it, so run_cell's hasattr dispatch is preserved
        return getattr(self._inner, name)


# ── one metered cell ────────────────────────────────────────────────

def run_cell_metered(policy: str, seed: int, n_agents: int,
                     n_episodes: int, snap_every: int) -> Dict[str, Any]:
    meter = Meter(snap_every)
    orig = bu.make_policy

    def metered_factory(name, aid, role_name):
        return MeteredArm(orig(name, aid, role_name), meter)

    bu.make_policy = metered_factory
    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    try:
        row = bu.run_cell(policy, seed, n_agents, n_episodes)
    finally:
        bu.make_policy = orig
    cpu_s = time.process_time() - cpu0
    wall_s = time.perf_counter() - wall0

    final_bytes = [stored_bytes(a._inner) for a in meter.arms]
    entries = [n_entries(a._inner) for a in meter.arms]
    update_s = meter.observe_s + meter.receive_s
    uses = meter.n_query_nonempty
    resv_bits_est = (uses * RESV_BITS
                     if policy == "songline_full" else 0)
    total_entries = sum(entries)
    row.update({
        "agents": n_agents, "episodes": n_episodes,
        # 1. stored bytes
        "stored_bytes_final_mean": float(sum(final_bytes)
                                         / max(1, len(final_bytes))),
        "stored_bytes_final_total": int(sum(final_bytes)),
        "stored_bytes_curve": [
            [t, float(sum(v) / len(v))]
            for t, v in sorted(meter.snap.items())],
        "n_entries_final_total": total_entries,
        # 2. transmitted bytes (bits kept too, for the paper tables)
        "wire_bits_payload": meter.wire_bits_payload,
        "wire_bits_full": meter.wire_bits_full,
        "resv_bits_est": resv_bits_est,
        "transmitted_bytes": (meter.wire_bits_full + resv_bits_est) / 8.0,
        "n_messages": meter.n_messages,
        "n_deliveries": meter.n_deliveries,
        # 3. update time
        "update_time_s": update_s,
        "observe_time_s": meter.observe_s,
        "receive_time_s": meter.receive_s,
        "outbox_time_s": meter.outbox_s,
        "n_observe": meter.n_observe, "n_receive": meter.n_receive,
        # 4. query latency
        "query_time_s": meter.query_s,
        "n_query": meter.n_query,
        "n_query_nonempty": uses,
        "query_latency_ms_mean": (1e3 * meter.query_s
                                  / max(1, meter.n_query)),
        # 5. CPU-seconds
        "cpu_time_s": cpu_s,
        "wall_time_s": wall_s,
        # 6. model calls
        "llm_calls": 0,
        "llm_calls_note": "deterministic arms; no LLM client imported "
                          "by baselines.py or songlines/* (0 by "
                          "construction)",
        # 7. amortized formation cost
        "amortized_formation_ms_per_use": (1e3 * update_s
                                           / max(1, uses)),
        "amortized_formation_ms_per_entry": (1e3 * update_s
                                             / max(1, total_entries)),
    })
    return row


# ── CLI ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="all",
                    help="one of %s or 'all'" % (ALL_POLICIES,))
    ap.add_argument("--seeds", type=int, nargs=2, default=[100, 103])
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--snap-every", type=int, default=10,
                    help="sample the serialized store every K episodes")
    ap.add_argument("--out", type=str,
                    default="tmp/resource_accounting_smoke")
    a = ap.parse_args()
    policies = (list(ALL_POLICIES) if a.policy == "all"
                else [a.policy])
    os.makedirs(a.out, exist_ok=True)
    reg = os.path.join(a.out, "resources_registered.json")
    if not os.path.exists(reg):
        with open(reg, "w") as f:
            json.dump({
                "claim": "Package I: equal memory/comm budget is "
                         "checked against equal RESOURCE budget "
                         "(stored/transmitted bytes, update time, "
                         "query latency, CPU-s, LLM calls, amortized "
                         "formation) on the B-unified arms; if the "
                         "graph arm costs more compute, that is the "
                         "honest result and goes to limitations.",
                "policies": list(ALL_POLICIES),
                "utility_source": "exp_b_unified.run_cell row "
                                  "(team_cost/success/fail_open), "
                                  "unmodified",
            }, f, indent=2)
    for pol in policies:
        shard = (f"resources_{pol}_a{a.agents}_e{a.episodes}"
                 f"_s{a.seeds[0]}-{a.seeds[1]}.jsonl")
        with open(os.path.join(a.out, shard), "w") as f:
            for seed in range(a.seeds[0], a.seeds[1]):
                row = run_cell_metered(pol, seed, a.agents,
                                       a.episodes, a.snap_every)
                f.write(json.dumps(row) + "\n")
                print(f"{pol} seed {seed}: team {row['team_cost']:.1f} "
                      f"stored {row['stored_bytes_final_mean']:.0f}B "
                      f"tx {row['transmitted_bytes']:.0f}B "
                      f"upd {row['update_time_s']:.2f}s "
                      f"qry {row['query_latency_ms_mean']:.2f}ms "
                      f"cpu {row['cpu_time_s']:.1f}s", flush=True)
        print(f"Saved: {a.out}/{shard}")


if __name__ == "__main__":
    main()
