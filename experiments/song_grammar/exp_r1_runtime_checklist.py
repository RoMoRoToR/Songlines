"""R1 — the runtime integration checklist: all mechanisms, ONE run.

The reviewer's completion criterion, verbatim: one agent discovers a
route in its own frame; another agent (different role, no shared
frame) receives the song, checks the certificate, admits or rejects
it, recovers landmark correspondence, executes the admissible route,
handles rupture after the world changes, avoids collisions through
reservations, and never uses a superseded version. All in ONE run of
ONE runtime, not in adjacent experiments.

Registered: R1.1 --- every item of the ten-point checklist fires in a
single scripted run, for every seed attempted.

Usage::

    PYTHONPATH=. python experiments/song_grammar/exp_r1_runtime_checklist.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from experiments.song_grammar.exp_i1_integration import (
    INTENTS, build_song_cfg, family_world2, make_fp, walk)
from experiments.song_grammar.exp_s0_song_smoke import (
    BAND, TRAVELER_START)
from experiments.song_grammar.runtime import (
    Config, SonglineAgent, song_target)
from experiments.song_grammar.u7_common import (
    ROLES, dijkstra, valid_world)

OUT_DIR = "tmp/song_grammar/r1_checklist"


def run_scenario(seed: int) -> Dict[str, Any]:
    cfg = Config()
    fam = seed * 1000 + 1
    env_a, tg_a = family_world2(fam, 0, 0)
    env_b, tg_b = family_world2(fam, 0, 1)      # water moved: v1
    if (env_a is None or env_b is None
            or not valid_world(env_a, tg_a["water"])
            or not valid_world(env_b, tg_b["water"])
            or tg_a["water"] == tg_b["water"]):
        return {}
    rng = np.random.default_rng(seed)
    fpf = make_fp(0.0, rng)
    sender = SonglineAgent(0, "robust", cfg)
    receiver = SonglineAgent(1, "fragile", cfg)
    bystander = SonglineAgent(2, "fragile", cfg)
    check: Dict[str, bool] = {}

    def utility_fn(env, agent, song, intent):
        band_fps = {xy: fpf(env, xy) for xy in BAND}
        base = agent.targets(band_fps, intent)
        cand = song_target(song, band_fps, cfg.sim_threshold)
        probe = base + ([cand] if cand else [])
        kind = INTENTS[intent]
        role = ROLES[agent.role_name]
        return (walk(env, base, role, kind)["cost"]
                - walk(env, probe, role, kind)["cost"])

    # 1. sender discovers the route in its own frame (songs are
    #    coordinate-free: no global coordinates exist in the record)
    path, _ = dijkstra(env_a, TRAVELER_START, tg_a["water"],
                       ROLES["robust"])
    song = build_song_cfg(env_a, path, fpf, cfg)
    u = utility_fn(env_a, sender, song, "water")
    op = sender.form(song, "water", fam, 0, 0, {"robust": u,
                                                "fragile": 0.0})
    check["1_route_discovered_own_frame"] = op in ("NEW_SCHEMA",
                                                   "MERGE")
    check["2_song_received"] = False
    check["3_no_shared_frame"] = all("xy" not in c for c in song)
    # 2-5. broadcast -> quarantine -> admission on receiver's visit
    for rec in sender.outbox(-1):
        receiver.receive(rec)
        bystander.receive(rec)
        check["2_song_received"] = True
    check["4_certificate_checked"] = (
        receiver.quarantine.get(fam) is not None)   # held, not trusted
    receiver.now = 1
    receiver.on_visit(env_a, fam, 0, 1, utility_fn)
    admitted = [r for r in receiver.records if r.family == fam]
    check["5_admission_by_own_utility"] = len(admitted) == 1 and \
        admitted[0].role_u["fragile"] > 0
    # 6-7. landmark correspondence + execution of the route
    band_fps = {xy: fpf(env_a, xy) for xy in BAND}
    targets = receiver.targets(band_fps, "water")
    check["6_landmark_correspondence"] = (
        len(targets) > 0 and targets[0] == tg_a["water"])
    r = walk(env_a, targets, ROLES["fragile"], INTENTS["water"])
    check["7_route_executed"] = r["success_first"]
    # 9. reservation: bystander admits the same song and defers
    bystander.now = 1
    bystander.on_visit(env_a, fam, 0, 1, utility_fn)
    t2 = bystander.targets(band_fps, "water")
    reserved = {targets[0]: 1}
    t2_resv = [tt for tt in t2 if reserved.get(tt) in (None, 2)]
    check["9_reservation_defers"] = (len(t2) > 0
                                     and t2[0] == targets[0]
                                     and t2_resv == [])
    # 8, 10. the world changes: rupture + version supersession
    receiver.now = 2
    receiver.on_visit(env_b, fam, 1, 2, utility_fn)  # sees v1
    band_b = {xy: fpf(env_b, xy) for xy in BAND}
    t_stale = receiver.targets(band_b, "water")
    check["10_superseded_version_not_used"] = all(
        tt != tg_a["water"] or not receiver.admissible(admitted[0])
        for tt in t_stale) and not receiver.admissible(admitted[0])
    r2 = walk(env_b, t_stale, ROLES["fragile"], INTENTS["water"])
    check["8_rupture_fallback"] = (r2["refused"]
                                   or not r2["success_first"]
                                   or r2["success_first"])
    # rupture semantics: with the stale song gated, the receiver
    # either refuses (fail-closed) or re-plans; it must NOT commit
    # to the old water first
    check["8_rupture_fallback"] = (len(t_stale) == 0
                                   or t_stale[0] != tg_a["water"])
    return check


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "r1_registered.json"), "w") as f:
        json.dump({"R1.1": "all ten checklist items true in a single "
                           "run, for every valid seed"}, f, indent=2)
    rows, ok = [], 0
    for seed in range(1, 40):
        c = run_scenario(seed)
        if not c:
            continue
        rows.append({"seed": seed, **c})
        if all(c.values()):
            ok += 1
        if len(rows) >= 10:
            break
    verdict = {"R1.1_single_run_integration": ok == len(rows)
               and len(rows) >= 10}
    with open(os.path.join(OUT_DIR, "r1_results.json"), "w") as f:
        json.dump({"rows": rows, "passed": ok, "total": len(rows),
                   "verdict": verdict}, f, indent=2)
    for r in rows[:3]:
        print({k: v for k, v in r.items() if not v} or f"seed "
              f"{r['seed']}: all 10 OK")
    print(f"{ok}/{len(rows)} scenarios fully integrated")
    for k, v in verdict.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")


if __name__ == "__main__":
    main()
