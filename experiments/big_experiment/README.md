# Big experiment — Cadence Phase Diagram

Многопараметрический sweep по 4 архитектурам коллективной памяти + 5
cadence-значениям peer × 3 layout × 3 hazard density × N×M scenarios ×
20 seeds. Тестирует **Songlines Cadence Hypothesis**.

## We claim

> **The Songlines Cadence Hypothesis.** Communication cadence is a
> first-class architectural parameter for multi-agent collective memory.
> In environments with exclusive resources (M targets, N agents,
> ρ = N/M > 1), there exists an optimal broadcast cadence K\* > 1 such
> that peer-to-peer at K\* achieves lower mean time-to-success than
> both centralized aggregation (K=1 extreme) and full decentralization
> (K=∞ extreme). The phenomenon is invisible to architectures that
> lack a cadence dimension. K\* scales monotonically with ρ.

Тестируемые гипотезы:
- **H1 (existence):** mean_t_succ как функция K имеет минимум при K\* > 1 для ρ > 1
- **H2 (scaling):** K\* монотонно растёт с ρ (Spearman correlation)
- **H3 (degeneracy):** для ρ ≤ 1 K\* = 1 (fast broadcast выигрывает)
- **H4 (Pareto):** peer(K\*) Pareto-доминирует centralized по (mean_t_succ, success_rate)

## Sweep оси

| Ось | Значения | |
|---|---|---|
| N (агентов) | 3, 5, 8 | spawn в 8 предопределённых позициях |
| M (waters) | по N: {2,3}/{2,3,5}/{2,3,5,8} | M ≤ N |
| Layout | symmetric / asymmetric / random | как разместить waters |
| Architecture | independent / shared / centralized / peer | 4 варианта |
| Peer K | 1, 2, 4, 8, 16 | только для peer |
| Hazard density | 0.0 / 0.05 / 0.10 | ablation robustness |
| Seeds | 0..19 | 20 повторов на конфигурацию |

**Total: 12960 runs**. При 80 runs/sec на 8 ядрах ~3 минуты.

## Файлы

| Файл | Что |
|---|---|
| `env_factory.py` | build_env(N, M, layout, hazard%, seed) |
| `memory_factory.py` | adapters для 4 архитектур, единый API |
| `planner.py` | универсальный memory-driven planner |
| `runner.py` | run_one_config() → metrics dict |
| `config.py` | smoke_configs() + full_configs() |
| `exp_cadence_phase.py` | parallel driver, streaming CSV |
| `analyze.py` | aggregation, plots, claim validation |

## Запуск

```bash
# Smoke (24 configs, проверить pipeline)
PYTHONPATH=. .venv/bin/python experiments/big_experiment/exp_cadence_phase.py \
    --mode smoke --out_dir tmp/big_experiment_smoke

# Full sweep
PYTHONPATH=. .venv/bin/python experiments/big_experiment/exp_cadence_phase.py \
    --mode full --workers 8 --out_dir tmp/big_experiment_full

# Анализ + графики + claim validation
PYTHONPATH=. .venv/bin/python experiments/big_experiment/analyze.py \
    --runs_csv tmp/big_experiment_full/runs.csv \
    --out_dir tmp/big_experiment_full \
    --layouts symmetric,asymmetric,random \
    --hazards 0.0,0.05,0.10
```

## Метрики per run

- `success_rate` — доля агентов достигших воды в budget
- `mean_t_succ` — среднее время до success (среди успешных)
- `p95_t_succ` — 95-й перцентиль (worst-case)
- `total_trail` — суммарная длина уникальных траекторий
- `n_hazard_hits`
- `scarcity` = N/M

## Артефакты

```
tmp/big_experiment_full/
├── runs.csv                                 # 12960 rows
├── aggregates.csv                           # per-config means/stds
├── cadence_curves_{layout}_h{X}.png         # H1+H2 visualization
├── phase_diagram_{layout}_h{X}.png          # heatmap K × ρ
├── pareto_{layout}_h{X}.png                 # H4 verification
├── hazard_robustness_{layout}.png           # ablation by hazard
└── claim_validation.json                    # statistical tests
```

## Claim validation logic

- **H1:** Для каждого (N, M, layout) с N > M вычисляем K* = argmin mean_t_succ среди K∈{1,2,4,8,16}. Считаем долю случаев где K* лежит во **внутренней** части диапазона (не 1, не 16). H1 supported если >50%.
- **H2:** Spearman correlation между ρ и K* по всем (N, M, layout). H2 supported если ρ > 0 и p < 0.10.
- **H3:** Для (N, M) с N ≤ M проверяем K* == 1. H3 supported если >50% таких случаев.
- **Peer vs centralized:** Mann-Whitney U test (one-sided) на mean_t_succ при scarcity (N > M). H supported если peer(K*) < centralized и p < 0.05.

## Что увидим (предсказание)

Если гипотезы верны, **cadence_curves** покажут U-образные кривые с минимумом не на краях диапазона. **Phase diagram** будет иметь красные прямоугольники (K*) сдвигающиеся вправо по мере роста ρ. **Pareto** покажет peer-точки в верхне-левой части (быстро + успешно).

Результаты будут заполнены в `RESULTS.md` после прогона.
