# `songlines/` — the method as one package (Runtime v1)

Stage 6 of the productionisation plan: the validated logic, previously
spread across `experiments/song_grammar/{runtime,ucsm,u7_common}.py`,
now lives in one substrate-agnostic package. Experiment drivers keep
working through thin re-export shims (`experiments/song_grammar/runtime.py`,
`ucsm.py`) — nothing in the paper's results changed.

## Layout

| Module | Owns | Plan layer |
|---|---|---|
| `record.py` | `Config`, `Record`, `ROLE_NAMES`, bit codec, `record_bits`, `bits_of_song`, `bits_of_snapshot` | memory/record + evaluation/costs |
| `analogy.py` | `Schema`, `Certificate`, `analogy`, `nearest`, `decide`, `SonglineMemory` | formation/analogy |
| `alignment.py` | `song_target` (frame-free landmark matching, consensus, unimodal clustering, loop closure) | alignment/landmarks + frame_recovery |
| `runtime.py` | `SonglineAgent` — the orchestrator composing formation, communication, admission, consumption | (the cycle) |
| `config.py` | named arm & ablation configs (one registry) | config-driven ablations |
| `tests/test_invariants.py` | 8 safety/correctness invariants | — |

## The boundary (a reasoned deviation from the plan's literal dirs)

The plan proposed `memory/ formation/ communication/ alignment/
planning/ evaluation/` subpackages. We adopt a **flatter layout on a
sharper boundary**: `songlines/` owns the *substrate-agnostic method*;
grid and continuous worlds, walkers, and drivers stay under
`experiments/song_grammar/`. This boundary is load-bearing — the
continuous substrate (C1) reuses the runtime unchanged, proving the
method never touches world coordinates. Splitting the stateful
`SonglineAgent` across five subpackages would fragment one coherent
cycle into artificial files without reducing real coupling; the
conceptual layers (formation / communication / planning) are marked
by comment banners inside `runtime.py` instead. This honours the plan's
actual goal — *reduce the distance between spec, code, experiment, and
claim* — over its literal directory names.

## One-runtime discipline

Every benchmark arm and ablation is a `Config` in `config.py`; there
is no hand-forked runtime per arm. Drivers select by name:

```python
from songlines.config import get
cfg, communicating = get("songline_full")   # or "no_admission", "c_safe", ...
```

## Invariants (run before any release)

```bash
PYTHONPATH=. python -m songlines.tests.test_invariants   # 11/11
```

Covers: immutable evidence not overwritten · origin-bound provenance
(no laundering) · quarantine gates action · stale record loses
authority · ambiguous alignment fails closed · exception preserves
parent · two-axis decision matrix. I7 (reservation uniqueness) and I8
(rupture bound) are driver-level (exp_i1/exp_c1, route-warp rupture law).

## Freeze / claims / verdicts

`docs/SONGLINES_V1_FREEZE.md` · `docs/CLAIM_EVIDENCE_MATRIX.md` ·
`docs/SERIES_VERDICTS.md`.
