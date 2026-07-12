# Q/R/M/C: A Stage-Decomposition Evaluation Protocol for Memory-Based Agents

**Подача на AAAI-27.** Дедлайны: abstract **21 июля 2026**, paper
**28 июля**, supplements **31 июля**. Лимит: 7 стр. технического
контента (+ references и reproducibility checklist сверх лимита).
Текущий PDF: 8 стр., контент до p6.

## О чём

Success-only метрики склеивают качественно разные отказы памяти в один
скаляр. Мы операционализируем четыре события на query–lock цепочке —
Q\*, R\*, M\*, C\* — так, что четыре условных вероятности вычислимы из
обычных логов. События вложены по построению, поэтому
P(C\*) = q·r·m·c — точное тождество; вклад — операционализация (как
precision/recall), а не математика.

## Ключевые результаты

- **Single-agent:** oracle-интервенции разделяют retrieval-limited
  (0.39→0.60, sign test p=0.004) и downstream-limited задачи; лог-форензика:
  ранжирование 96.9% при 91% пустых кандидатов → лимит накопления.
- **Multi-agent (35 640 прогонов):** bottleneck shift M↔C (Spearman
  −0.53/+0.43, cluster-robust CI); интерьерный оптимум K=8 на условном
  времени (с честным цензур-корректным контрастом); механизм —
  провенанс-стратификация (коллапс C сосредоточен в warp-страте).
- **Внешняя валидация:** один tool-trace адаптер на OpenAI SDK /
  LangGraph / AutoGen × 2 модели; структурные отказы локализуются
  идентично (R=0, C=0 в 20/20); поведенческий разъезд атрибутируется
  в M1; слепая meta-evaluation по инъекции 10 типов отказов.

## Файлы

- `songlines_qrmc_aaai27.tex/pdf` — подача (EN); `*_ru.*` — русский дубль;
- `ReproducibilityChecklist_qrmc_aaai27.tex` — заполненный checklist
  (комментарии `% DRAFT` — обоснования на вычитку);
- `aaai2027.sty/.bst`, `aaai27refs.bib` — AAAI-кит; `figures -> ../figures`.

## Связанный код и данные

| Что | Где |
|---|---|
| Мультиагентный свип + effect sizes + CI фигур | `experiments/big_experiment/` (`analyze_effect_sizes.py`, `analyze_fig_cis.py`); данные `tmp/paper1_clean_experiments_full/`, `tmp/big_experiment_*` |
| Single-agent oracle-интервенции | `scripts/benchmark_oracle_stage_interventions.py`, `scripts/analyze_oracle_pairing.py`; данные `tmp/oracle_stage_interventions_final_20260430/` |
| Retrieval-форензика (91%/96.9%) | `experiments/v2_retriever/`; данные `tmp/article_revision_10seeds_20260501/` |
| Внешняя валидация + meta-evaluation | `experiments/qrmc_external/` (см. его README); данные `tmp/qrmc_external/` |
| Learned-базлайны | `experiments/commnet_baseline/`, `experiments/commnet_ppo_baseline/` |
| LLM-коллектив | `experiments/llm_collective/` |
| VMAS-проверка | `experiments/vmas_portability/` |
| Фигуры | `scripts/make_paper_figures_v3.py`, `experiments/qrmc_external/make_external_figure.py` → `papers/figures/` |

## Компиляция

```bash
pdflatex songlines_qrmc_aaai27.tex && bibtex songlines_qrmc_aaai27 \
  && pdflatex songlines_qrmc_aaai27.tex && pdflatex songlines_qrmc_aaai27.tex
```

Supplement подачи = `../qrmc_measurement_framework/` (46 стр.) + Code & Data Supplement.
