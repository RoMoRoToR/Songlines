# Фигуры и таблицы (макеты — заполняются после прогонов)

**Родитель:** [`04_EXPERIMENTS.md`](04_EXPERIMENTS.md), [`03_METRICS.md`](03_METRICS.md). Все числа ниже — **placeholder**, не результаты. Дисциплина серии требует регистрации формы предсказания до прогона (`docs/SERIES_VERDICTS.md`); эти макеты — именно такая регистрация формы, не данных.

---

## Figure 1 — Testimony vs Authority pipeline (концептуальная, для §1 будущей статьи)

```
          COMMUNICATION                   AUTHORITY

A observes X
     │
     ▼
┌──────────┐
│ Testimony│
└────┬─────┘
     │
     ▼
┌────────────┐
│ Quarantine │  ◄── RECEIVED → QUARANTINED (01_FORMAL_MODEL.md §5)
└─────┬──────┘
      │
      ├── provenance (E, origin vs transport — 01_FORMAL_MODEL.md §2)
      ├── validity (закон дальности, valid_until)
      ├── receiver role (role_conditions/state_conditions)
      ├── causal utility (LCB(Û), 01_FORMAL_MODEL.md §4)
      │
      ▼
┌──────────────┐
│   ADMITTED   │
│ action auth. │
└──────┬───────┘
       │
       ▼
     Action  (ActionRecord, 01_FORMAL_MODEL.md §8)
       │
       ▼
 Validation  (ValidationEvent, 01_FORMAL_MODEL.md §6)
       │
   ┌───┴────┐
   ▼        ▼
retain    revoke → REVOKED
```

Подпись (одна фраза, для caption статьи): **«Communication transports testimony; admission grants authority.»**

---

## Figure 2 — Social amplification vs independent corroboration (headline figure программы)

Прямая визуализация Theorem 1 + Theorem 3 на одной оси X, см. также [`02_THEOREMS.md`](02_THEOREMS.md) «Как safety и liveness совмещаются».

```
authority
1.0 |                                     independent confirmations (E2)
    |                                    ╱
    |                                 ╱
    |                              ╱
0.6 |                           ╱
    |                        ╱
    |   naive shared/vector (E1, baseline)
0.4 |  ╱───────────────────
    | ╱
    |╱
    │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   provenance-aware authority (E1, ours) — ПЛОСКАЯ
0.0 +──────────────────────────────────────────────►
      1     2     3     4     5      n (hops ретрансляции ИЛИ независимых подтверждений)
```

Два ряда на одной фигуре: (а) x = hops ретрансляции без нового origin → ожидаем плоскую линию у ours, растущую у baseline; (б) x = число независимых подтверждений → ожидаем растущую линию у ours (обе кривые растут — E2 не про то, что authority никогда не растёт, а про то, что растёт от правильного триггера).

**Числа в этой фигуре — заглушки** до прогона E1/E2. Обязательный элемент: непересекающиеся CI между k=1 и k=5 для baseline (рост), и CI, включающий 0 (отсутствие роста) для ours на панели (а).

---

## Figure 3 — Staleness / revocation (headline figure для E3)

```
world state:  bridge=open ────────────────────╳ bridge=closed ──────────────────►
                                          episode 20         t
authority
A(m,t)
1.0 |████████████████████████
    |                        ╲
    |                          ╲___ ours: L_R мала
0.5 |                              ╲___________________
    |
    |
    |                        ╲___________________________________ baseline: L_R велика
0.0 +────────────────────────────────────────────────────────────►
                              t_invalidated              t
```

Заштрихованная область под кривой baseline ПОСЛЕ `t_invalidated` — это визуализация метрики `stale authority area` (`03_METRICS.md` §3). Подпись: разница площадей — это буквально «сколько недопустимого доверия прожила система».

---

## Figure 4 — Causal utility calibration (для E5)

```
τ̂ (предсказанная причинная полезность)
 │           ●
 │        ●     ●
 │      ●    ●
 │   ●     ●
 │ ●    ●
 │───────────────────────────► τ_true (измеренная через randomized intervention)
 │ ●
 │    ●  ●
 │        ●
```

Стандартный scatter + диагональ идеальной калибровки + доверительная полоса. Два панели рядом: (а) детерминированный grid, replay доступен — сравнение UE1 (существующий replay estimator) vs новый causal estimator; (б) стохастический/LLM субстрат — только causal estimator vs heuristic/LLM-rated baselines (replay недоступен по определению среды, UE1 неприменим).

---

## Figure 5 — LLM long-horizon (headline figure для E9/E10)

Три панели, общая ось X = episode number (0 → H):

```
Panel A: Success rate                Panel B: False Authority Rate         Panel C: Token cost
1.0 |  full authority ────────        0.3 |  shared/naive ─────────         high |  shared/naive (растёт с H)
    |  trust/staleness CSM ─ ─         0.2 |                                     |
0.5 |  shared (падает при corruption)  0.1 |  full authority (низкий, плоский)   |  full authority (сублинейно, X1-эффект)
    |  raw/vector (деградирует)        0.0 +──────────────────────►    low  +──────────────────────►
    +──────────────────────►                0    H/2    H                        0    H/2    H
     0    H/2    H
```

Точки инъекции corruption (§2 `05_BENCHMARK_CORRUPTIONS.md`) отмечаются вертикальными пунктирными линиями на всех трёх панелях одновременно — визуально показывает, что просадка/восстановление синхронизированы с известным моментом инъекции, а не случайны.

---

## Table 1 — Full method vs baselines (главная сравнительная таблица)

Шаблон колонок (заполняется после E9; строки — плечи из [`04_EXPERIMENTS.md`](04_EXPERIMENTS.md) E9):

| Method | Success ↑ | Cost ↓ | FAR ↓ | Revocation Latency ↓ | Tokens ↓ | PAF (≈1 target) |
|---|---:|---:|---:|---:|---:|---:|
| Raw context | | | | | | |
| Shared vector memory | | | | | | |
| Per-agent vector memory | | | | | | |
| Naive shared graph | | | | | | |
| + freshness only | | | | | | |
| + provenance only | | | | | | |
| + authority (full) | | | | | | |

Порядок строк — намеренно ablative (каждая следующая строка добавляет один компонент) — это позволяет Table 1 одновременно служить и сравнительной таблицей, и первой половиной ablation study (вторая половина — Table 2).

## Table 2 — Authority Gating Ablation (для E6)

| Компонент включён | FAR ↓ | Authority Precision ↑ | Authority Recall ↑ | PAF (≈1) | Task success |
|---|---:|---:|---:|---:|---:|
| No gate (baseline) | | | | | |
| E only | | | | | |
| U only | | | | | |
| E + U | | | | | |
| E + provenance | | | | | |
| E + U + provenance | | | | | |
| **FULL** (+ structural exception) | | | | | |

Регистрируемое предсказание формы (не чисел, см. [`04_EXPERIMENTS.md`](04_EXPERIMENTS.md) E6): `E only` закрывает большую часть FAR-эффекта, но не Authority Recall на role-dependent ячейках; `U only` не даёт PAF≈1; только `FULL` проходит по всем пяти колонкам одновременно.

---

## Что НЕ подлежит преждевременному заполнению

Ни одна ячейка таблиц или точка на графиках выше не заполняется до реального прогона соответствующего эксперимента ([`04_EXPERIMENTS.md`](04_EXPERIMENTS.md)). Это прямое продолжение дисциплины серии: «предсказания регистрируются до прогонов; провалы регистраций публикуются с механизмами» (`docs/SERIES_VERDICTS.md`). Данный документ фиксирует **форму** ожидаемого результата и то, какая форма считается acceptance, какая — честным provalом.

---

**Файл:** `docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/06_FIGURES_AND_TABLES.md`
