# Статьи проекта Songlines

Серия из шести рукописей вокруг одной программы: **Q/R/M/C — стадийная
декомпозиция выполнения задач агентами с семантической памятью**
(Query → Retrieval → Materialization → Completion) и её следствия для
коллективной памяти (провенанс, перенос маршрутов, тождество мест).

## Карта серии

| Папка | Статья | Стр. | Роль / статус |
|---|---|---|---|
| [`qrmc_aaai27/`](qrmc_aaai27/) | Q/R/M/C: A Stage-Decomposition **Evaluation Protocol** for Memory-Based Agents | 8 | **Подача на AAAI-27** (abstract 21.07, paper 28.07.2026) |
| [`qrmc_measurement_framework/`](qrmc_measurement_framework/) | Q/R/M/C: A Stage-Decomposition Measurement Framework for Memory-Based Navigation | 46 | Полная версия = Technical Supplement к AAAI-подаче |
| [`semantic_warp/`](semantic_warp/) | Semantic Warp: Provenance-Conditioned Completion in Collective Memory | 10 | Companion №1 (волны W0–W6) |
| [`route_warp/`](route_warp/) | The Song, Not the Pin: Route Transfer and Meaning-Based Place Identity | 7 | Companion №2 (R0–R3 + W7–W9) |
| [`collective_semantic_memory/`](collective_semantic_memory/) | Collective Semantic Memory: Merge, Trust, and Staleness for Peer Memory | 22 | Companion №3 (CSM-механика + категорная рамка) |
| [`symbolic_memory/`](symbolic_memory/) | Q/R/M/C Framework + Minimal Collective Semantic Memory (ранняя объединённая версия) | 33 | Исторический предшественник qrmc_measurement_framework |

В каждой папке лежат английская и русская (`*_ru.tex/pdf`) версии и
свой `README.md` с тезисом, ключевыми результатами и картой
относящегося кода.

## Общие ресурсы

- `figures/` — все фигуры серии (папки статей ссылаются на неё
  симлинком `figures -> ../figures`, поэтому каждый `.tex` компилируется
  на месте);
- `styles/` — `neurips_2026.sty`, шаблон `neurips_2026.tex` и общий
  `checklist.tex` (симлинки в папках, которым они нужны); AAAI-кит
  (`aaai2027.sty/.bst`) лежит физически в `qrmc_aaai27/` — папка подачи
  самодостаточна.

## Компиляция

Из папки статьи:

```bash
pdflatex songlines_<paper>.tex            # NeurIPS-статьи: 2 прохода
pdflatex songlines_qrmc_aaai27.tex        # AAAI: pdflatex + bibtex + 2×pdflatex
```

Русские версии используют пакет `tempora` (Times-клон с кириллицей T2A).

## Код и данные

Код экспериментов остаётся в `experiments/` и `songline_drive/` (одни и
те же эксперименты питают несколько статей, а модули импортируются как
`experiments.<name>....` — физический перенос сломал бы импорты).
Карта «статья → код → данные» — в README каждой статьи; сырые
результаты прогонов — в `tmp/` (не версионируются как источник истины,
у ключевых экспериментов есть `RESULTS.md`).
