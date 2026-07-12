# Q/R/M/C Framework + Minimal Collective Semantic Memory (ранняя объединённая версия)

Историческая (33 стр.) рукопись — первая версия, в которой Q/R/M/C
framework и минимальная коллективная семантическая память были одной
статьёй. Из неё выросли текущие `../qrmc_measurement_framework/`
(протокол, расширенный внешней валидацией) и
`../collective_semantic_memory/` (механика CSM). Хранится как
исторический артефакт и источник ранних формулировок; **не является
текущей подачей** — актуальные версии см. в соседних папках.

## Файлы

`songlines_symbolic_memory.tex/pdf` (EN, 33 стр.), `*_ru.*` (RU, 30 стр.);
симлинки `figures`, `neurips_2026.sty`, `checklist.tex`.

## Связанный код

Single-agent стек Songlines: `songline_drive/` (интенты, семантические
предикаты, materialization, waypoint-контроллер),
`experiments/v2_retriever/` (retrieval-форензика),
`scripts/benchmark_*` (бенчмарки и oracle-интервенции),
`scripts/make_paper_figures*.py` (фигуры).

## Компиляция

```bash
pdflatex songlines_symbolic_memory.tex   # ×2
```
