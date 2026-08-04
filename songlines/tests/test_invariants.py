"""Runtime invariant tests --- the safety/correctness contracts that
must never regress (reviewer Stage 6).  No pytest dependency: run

    PYTHONPATH=. python -m songlines.tests.test_invariants

Invariants I1--I6 are in-package (runtime/alignment); I7 (reservation
uniqueness) and I8 (rupture bound) are driver-level and validated in
exp_i1_integration / exp_c1_continuous and the route-warp rupture law
respectively --- referenced here, tested there.
"""

from __future__ import annotations

from songlines import (Config, Record, SonglineAgent, Schema,
                       SonglineMemory, decide, song_target)

_checks = []


def check(name, cond):
    _checks.append((name, bool(cond)))


def sig(*keys):
    return {k: 1.0 for k in keys}


def couplet(keys, beat):
    return {"sig": sig(*keys), "beat": beat}


# ── I1: immutable evidence cannot be overwritten ───────────────────
def test_immutable():
    ag = SonglineAgent(0, "robust", Config())
    s1 = [couplet(["a@0,0", "b@1,0"], (1, 0)), couplet([], None)]
    ag.form(s1, "water", 1, 0, 0, {"robust": 99.0, "fragile": 0.0})
    first = ag.episodic[0]["song"]
    s2 = [couplet(["c@0,0", "d@1,0"], (2, 0)), couplet([], None)]
    ag.form(s2, "water", 2, 0, 1, {"robust": 99.0, "fragile": 0.0})
    check("I1_immutable_evidence_preserved",
          ag.episodic[0]["song"] is first and len(ag.episodic) == 2)


# ── I2: provenance origin cannot be laundered ──────────────────────
def test_origin_bound():
    ag = SonglineAgent(1, "robust", Config(admission="none"))
    rec = Record([couplet(["a@0,0"], (1, 0))], "water", 5,
                 {"robust": 9.0}, origin=9, uid=(9, 3), t=3, version=0)
    ag.receive(rec, sender=2)          # claimed origin 9 != sender 2
    check("I2_origin_bound_rejects_launder",
          len(ag.records) == 0 and not ag.quarantine)


# ── I3: quarantined record cannot directly form an action ──────────
def test_quarantine_gates_action():
    ag = SonglineAgent(1, "robust", Config(admission="util"))
    rec = Record([couplet(["a@0,0"], (1, 0))], "water", 7,
                 {"robust": 9.0}, origin=1, uid=(1, 4), t=4, version=0)
    ag.receive(rec, sender=1)          # family 7 never visited
    quarantined = ag.quarantine.get(7)
    tgts = ag.targets({}, "water")     # nothing admitted -> no action
    check("I3_quarantine_no_action",
          quarantined is not None and tgts == [])


# ── I4: expired record gets no authority (world clock) ─────────────
def test_world_clock_expiry():
    ag = SonglineAgent(1, "robust", Config(world_clock=True))
    ag.note_version(7, 5)
    stale = Record([couplet(["a@0,0"], (1, 0))], "water", 7,
                   {"robust": 9.0}, origin=1, uid=(1, 1), t=1,
                   version=2)
    fresh = Record([couplet(["a@0,0"], (1, 0))], "water", 7,
                   {"robust": 9.0}, origin=1, uid=(1, 2), t=1,
                   version=5)
    check("I4_stale_inadmissible",
          (not ag.admissible(stale)) and ag.admissible(fresh))


# ── I5: ambiguous alignment fails closed (refusal, not phantom) ────
def test_ambiguous_fails_closed():
    # one couplet whose signature matches TWO band positions equally
    song = [couplet(["x@0,0", "y@1,0"], (1, 0)), couplet([], None)]
    band = {(3, 3): sig("x@0,0", "y@1,0"),
            (8, 8): sig("x@0,0", "y@1,0")}   # duplicate -> ambiguous
    t = song_target(song, band, sim=0.999)
    check("I5_ambiguous_refusal", t is None)


# ── I6: exception does not destroy its parent schema ───────────────
def test_exception_keeps_parent():
    mem = SonglineMemory(u_thr=5.0, share_thr=0.4, d_thr=3.0)
    base = [couplet(["a@0,0", "b@1,0"], (1, 0)),
            couplet(["c@0,0", "d@1,0"], (1, 0)), couplet([], None)]
    mem.consider(base, utility=99.0, episode_id="e0", conditions={})
    parent = mem.schemas[0]
    # same signatures (simple) but far end-displacement (conflict)
    variant = [couplet(["a@0,0", "b@1,0"], (5, 5)),
               couplet(["c@0,0", "d@1,0"], (5, 5)), couplet([], None)]
    op = mem.consider(variant, utility=99.0, episode_id="e1",
                      conditions={})
    check("I6_exception_preserves_parent",
          op == "EXCEPTION" and mem.schemas[0] is parent
          and len(mem.schemas) == 2)


# ── decision-matrix sanity (the two axes) ──────────────────────────
def test_decision_matrix():
    hi_simple = {"share": 0.9, "D": 0}
    hi_conflict = {"share": 0.9, "D": 9}
    lo_simple = {"share": 0.9, "D": 0}
    check("D_merge", decide(9, hi_simple, 5, 0.4, 3) == "MERGE")
    check("D_exception", decide(9, hi_conflict, 5, 0.4, 3) == "EXCEPTION")
    check("D_new", decide(9, {"share": 0.1, "D": 0}, 5, 0.4, 3)
          == "NEW_SCHEMA")
    check("D_repeat", decide(1, lo_simple, 5, 0.4, 3) == "REPEAT")
    check("D_drop", decide(1, {"share": 0.1, "D": 0}, 5, 0.4, 3)
          == "DROP")


def main():
    for fn in (test_immutable, test_origin_bound,
               test_quarantine_gates_action, test_world_clock_expiry,
               test_ambiguous_fails_closed, test_exception_keeps_parent,
               test_decision_matrix):
        fn()
    ok = sum(1 for _, c in _checks if c)
    for name, c in _checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    print(f"{ok}/{len(_checks)} invariant checks passed")
    print("(I7 reservations, I8 rupture: driver-level --- "
          "exp_i1/exp_c1 and route-warp rupture law)")
    return 0 if ok == len(_checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
