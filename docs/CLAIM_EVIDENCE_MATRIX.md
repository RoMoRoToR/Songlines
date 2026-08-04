# Claim–Evidence Matrix

**Дата:** 2026-08-03. Самый важный редакционный документ: **в статьях не должно быть ни одного сильного предложения, которого нет здесь.** Каждому claim приписан глагол-модальность (`We define` / `We show` / `We observe` / `We claim` / `We prove` / `We do not claim`), эксперимент, число, ограничение, артефакт. Правило: `prove` — только для теорем; экспериментальное совпадение — это `We show`, тенденция — `We observe`.

Легенда типов: **C** conceptual · **T** theoretical · **M** mechanistic · **E** empirical.

---

## Статья A — Executable Foreign Memory (Parts II+III)

| ID | Claim | Модальность | Тип | Эксперимент | Результат | Ограничение | Артефакт |
|---|---|---|---|---|---|---|---|
| A1 | Foreign-memory value must be provenance-conditioned, not read from success/retrieval | **We claim** | C | warp sweep (W1) | P(C\*\|W\*) 0.004–0.020 vs own 0.22–0.41, ДИ disjoint | grid, 2400 эп. | tmp/warp w1 |
| A2 | Foreign evidence carries value as transport | **We show** | E | taxi chains (W1) | 51.2% own-completions warp-assisted | grid | analyze_warp_chains |
| A3 | Intention horizon is predictable in closed form | **We show** | T+E | distance law (W2) | 6/6 hold-out EXACT, predictions preregistered | exp. trust×staleness gate | w2_age_law/holdout |
| A4 | Fast-sharing loss = contention, not bad evidence | **We show** | E | collision sweep (W6) | loss на foreign commits; WD восстанавливает | grid+VMAS+LLM | tmp/warp w6 |
| A5 | Reservation+rollback makes fast sharing best | **We show** | E | Warp Drive (W3) | 0.614–0.620 vs best fixed 0.583, ДИ disjoint | grid | w3_drive |
| A6 | Place-only transfer collapses at first wall (cliff) | **We show** | E | R1 | success 1.0→0.0 между D=1.00 и 1.38; хуже blind | grid mazes | r1_route_gain |
| A7 | Rupture law predicts exact failing edge | **We show** | T+E | R2 | 110/110 regimes, 21 ruptures, edge-error 0 | seq. edge stamping | r2_route_rupture |
| A8 | Route preserves witness risk over validated prefix | **We show** | E | R3 | 0 hits vs 3.25 place; risk returns at predicted edge | hazard fields | r3_hazard |
| A9 | Identity recovers across private frames, fails closed | **We show** | E | W7–W9 | 100%/80-80 recovery; ambiguity→refusal, не phantom | translations + quarter-turns | w7/w8/w9 |
| A10 | Which features carry identity is measurable, fail-safe in Σ | **We show** | M | W10 | 0 fail-open/252; hazard 0.56/wall 0.44/void 0.00 | fixed vocabulary | w10_landmark_ablation |
| A11 | Landmarks=identity, edges=transport; composition needed | **We show** | M | S0 | beats-only 11/11 fail-open; sigs-only 0 transport | grid | s0_song_smoke |
| A12 | The song is a different object, not a lighter map | **We claim** | C | S0 | pure-codec 366 vs 5690 bits (6.4%) | **pure song codec, excl. metadata** | s0_song_smoke |
| A13 | Full-protocol payload (certs+provenance+time) larger than pure codec | **We show** | M | codec accounting | pure 6.4% / full **9.0%** of snapshot | grid codec | docs/COST_ACCOUNTING.md |

## Статья B — Utility-Certified Analogical Memory (Part V)

| ID | Claim | Модальность | Тип | Эксперимент | Результат | Ограничение | Артефакт |
|---|---|---|---|---|---|---|---|
| B1 | Utility and analogy are independent axes | **We claim** | C | U1 | ops = matrix ×8 worlds | grid, synthetic | u1_ucsm_smoke |
| B2 | Duplicates cost nothing by construction | **We show** | M | U1 | marginal U = 0 exactly | — | u1 |
| B3 | Similarity is not a consolidation criterion | **We show** | M | U1 | sim-only 62.6 vs 12.3 steps, silent corruption | — | u1 |
| B4 | EXCEPTION protects from false generalisation | **We show** | M | U1 | variant 20.5 vs 70.4 без него | synthetic conflict | u1 |
| B5 | The formation matrix is learnable | **We show** | E | U2 | bandit 78.1 < hand 83.5 < random 105.9 | contextual bandit | u2b_results |
| B6 | Learning rediscovers append-not-overwrite | **We observe** | M | U2 | hand-MERGE→NEW_SCHEMA (immutable store) | reward-equiv. quotient | u2b |
| B7 | Bounded song budget is selected, not imposed | **We show** | E | U3 | vmax=10 finite, no cost regression | fixed grammar family | u3_cluster |
| B8 | Grammar **parameters** evolve to designer-neighbourhood | **We show** | E | U3 | share 0.46≈0.4, d 3.9≈3 | **parameters within fixed grammar family** | u3 |
| B9 | Terminal credit suffices (beats dense bandit) | **We show** | E | M1 | meta-ES −97.95 < bandit −99.69 < hand | small policy class, synthetic | m1_final |
| B10 | Graph-matching abstracts appearance variants | **We show** | E | G1 | −24% schemas & bits, cost ±5%, phantom +0.012 | LCS→NW, no roles | g1 (near-miss 0.76 vs 0.75) |
| B11 | Consolidation buys bits; freshness buys success | **We show** | E | U7/U7b/U7c | structure beats mainstream 1.7–4.5×; ladder −14/−3/−0.9% | grid, e10–1000 | u7* |
| B12 | Cross-family motifs compress sublinearly-ish | **We observe** | E | X1 | codebook 7.91× < 8.42×, −16% @e1000 | first-order codebook | u7c |
| B13 | Causal landmark promotion works over a candidate set | **We show** | M | E1 | parity junk caught Δ=−0.67, 0 fail-open | **predefined candidate set** | e1 |
| B14 | Attribution ≠ causal necessity (void guard) | **We observe** | M | E1 | void: 0 mass yet load-bearing | — | e1 |
| B15 | Reconstruction beats replay in LLM context | **We show** | E | L1/L1b | raw 0.42→0.08 (llama), 1.0→0.75 (7B); ucsm 1.0 @6–12% | 3 models, small text worlds | l1* |
| B16 | Unregulated social exchange is harmful at horizon | **We show** | E | S1 | independent beats every exchange arm (148/171 vs 170/238) | grid, e300 | s1 |
| B17 | World-clock staleness is necessary, not sufficient | **We show** | E | S2 | rob −7.4% but independent wins 12/12 | grid | s2 |
| B18 | Receiver-validated admission makes exchange beneficial | **We show** | E | S3 | full beats independent 12/12, −20/−21% | prototype scale | s3 |
| B19 | Utility estimator matches oracle counterfactual | **We show** | E | UE1 | Spearman 0.989, sign 0.97; system on Û ≈ oracle | grid; external calibration open | ue1 |

## Интеграция / acceptance-тест (обе статьи)

| ID | Claim | Модальность | Тип | Эксперимент | Результат | Ограничение | Артефакт |
|---|---|---|---|---|---|---|---|
| X1 | All mechanisms compose in one runtime | **We show** | M | R1 | 10/10 checklist in one run | scripted | r1_checklist |
| X2 | Full method dominates every arm | **We show** | E | I1 | full ≤ all 8 arms, both roles; 12/12 seeds, p=0.0002 | grid, N=6 | i1 |
| X3 | Provenance/exceptions/immutable = free insurance here | **We observe** | M | I1-H1b | cost-free in this substrate; value in U1/S1/P1 | substrate-dependent | i1 |
| X4 | Advantage survives noise and scale | **We show** | E | I1-H2/H3 | −13/15% @noise; −11/13/14% @N=4/6/8 | grid | i1 |
| X5 | Raw doesn't catch up even at 4× budget | **We show** | E | B1 | full 120/125 vs raw 159/201 @1–4× | grid | b1 |
| X6 | Poison degrades full ≤1.2%; adversary=entropy | **We show** + **We observe** | E | P1 | full +1.2%; unprotected already phantom-saturated | 1 poisoner | p1 |
| X7 | Effect transfers to a continuous substrate | **We show** | E | C1 | full 245/71 < raw 257/101 < indep | synthetic continuous | c1e150 |
| X8 | Fail-closed under noise via anchor consensus | **We show** | E | N1v2 | fail-open 0.21→0.000 to 20%×20% | grid, margin softening | n1v2 |
| X9 | Continuous social-scale safety via three layers | **We show** | E | C1.4 | fail-open 0.0014 (554×), 10/12 exactly 0 | continuous, prototype | c1safe3 |
| BU1 | Full method beats every direct modern baseline on team cost | **We show** | E | unified bench (30 seeds) | songline 135.1 < best baseline (DeMem 164.5), **30/30 seeds, −17.9%, p=1.9e-9, bootstrap 95% CI Δ [28.1,30.7]** | grid, natural budget | bench30/v30_verdict.json |
| BU2 | Every communicating baseline is worse than independent | **We show** | E | unified bench (30 seeds) | DeMem/Mage/RIR/Mem-α 164-181 vs indep 137 | modern methods lack admission | bench30 |
| BU3 | Advantage is a Pareto point (cost+safety), paid in bits | **We observe** | E | unified bench | full safer than 3/4 baselines but +memory/wire | not axis-dominance | COST_ACCOUNTING 9.3 |
| X10 | The whole runtime transfers to a real continuous-physics simulator | **We show** | E | V2 (VMAS) | songline_safe 254.5 vs indep 543.5 team steps (−53.2%), 12/12 seeds p=0.0005; transport fail-open 0.0000; reached 44/48 vs 15/48 | VMAS box, 4 agents; substrate density recalibrated, method flags unchanged | tmp/v2_full/v2_verdict.json |

## Явные `We do not claim`

- Мы **не** заявляем, что решили долговременную память LLM-мультиагентов в общем виде.
- 6.4% относится к чистому song-кодеку; полная стоимость протокола = **9.0%** снапшота (A13, COST_ACCOUNTING §9.1) — не путать в тексте.
- Мы **не** заявляем general theory of collective memory.
- Мы **не** заявляем open-ended возникновение признаков (только causal promotion из предзаданного набора — B13).
- Мы **не** заявляем causal/role analogy (только структурную — B10).
- Мы **не** заявляем внешнюю валидность за пределами controlled grid, synthetic continuous и VMAS-физики (B19, X7, X10); ALFWorld / реальные роботы остаются открытыми.
- Мы **не** используем `prove` нигде, кроме стандартных категорных лемм в Part IV (cocompleteness of Pos).

---
**Критерий завершения этапа 3:** каждое сильное предложение статей ↔ строка этой матрицы. **Долги закрыты:** A13 (COST_ACCOUNTING §9.1), прямые baselines (BU1-BU3), внешний end-to-end субстрат (этап 13, VMAS — X10), 30-seed rerun (этап 8, BU1/BU2 на 30 сидах, p=1.9e-9). **Критический путь пуст.**
