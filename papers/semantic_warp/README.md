# Semantic Warp: Provenance-Conditioned Completion in Collective Memory

Companion-статья №1 (10 стр.). **Тезис:** ценность коллективной памяти —
не в информации, а в координационном контракте, прикреплённом к
информации. Мы аннотируем каждый M\*-lock долей чужих свидетельств φ
(из весов самого merge, заморожены в момент lock) и называем lock с
φ≥0.8 «варпом» — коммит к цели, известной агенту только от пиров.

## Ключевые результаты (волны W0–W6)

- **W1. Стратификация:** P(C\*|W\*) ≪ P(C\*|own) (0.004 vs 0.223);
  warp share падает 0.55→0.11 с ростом K — механизм bottleneck shift
  из Q/R/M/C. Устойчиво к порогу φ∈{0.5…0.95} (`phi_sensitivity.py`).
- **«Варп как такси»:** 51.2% собственных завершений — warp-assisted
  (W\*-lock подвозит с r≈9.5 до obs_radius) — разрешение парадокса
  положительного warp gain при почти нулевом P(C\*|W\*).
- **W2. Закон дальности:** age_max = ln(trust·conf/τ)/α — тик-в-тик,
  hold-out 6/6 EXACT с предсказаниями, зарегистрированными до прогонов.
- **W3. Warp Drive:** децентрализованный протокол «резервация +
  anti-M\* rollback + backoff»; recovery 0.90–6.6 от потолка дефицита
  min(m,M)/m; WD@K4 (0.926) > baseline K8 (0.901); безусловное
  превосходство над K=8 (0.614–0.620 vs 0.583).
- **W4–W6. Универсальность:** cross-session на LLM (10/10 vs 0/10 у
  контроля); VMAS (страты 0.012 vs 0.562); LLM-коллектив N=3 — сырой
  обмен снижает success 2/3→1/3, WD возвращает 0.333→0.417.
- **Outlook (2026-07-26):** контракт = память-для-доверия в
  функциональной таксономии (возврат/рассказ/доверие, у каждой своя
  сигнатура отказа); критерий валидности разделяемой памяти —
  экологическая пригодность (P(C*|W*) и закон), а не прямая верификация.

## Файлы

`songlines_semantic_warp.tex/pdf` (EN), `*_ru.*` (RU);
симлинки `figures`, `neurips_2026.sty`.

## Связанный код и данные

| Что | Где |
|---|---|
| Вся программа W0–W6 | `experiments/warp/` (см. его README и RESULTS.md) |
| Инструментация φ | `experiments/warp/warp_instrumentation.py` |
| Warp Drive | `experiments/warp/warp_drive.py`, `exp_warp_drive.py` |
| φ-sensitivity | `experiments/warp/phi_sensitivity.py` |
| Фигуры | `experiments/warp/make_warp_figures.py` → `papers/figures/fig_warp_*` |
| Данные | `tmp/warp/` (w1_gain, w3_drive, w5a_vmas, w6_llm, …) |

Дизайн-док фронтира: `docs/FRONTIER_SEMANTIC_WARP_2026-07-02.md`.

## Компиляция

```bash
pdflatex songlines_semantic_warp.tex   # ×2
```
