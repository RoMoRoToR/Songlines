# Q/R/M/C: A Stage-Decomposition Measurement Framework for Memory-Based Navigation

Полная (46 стр.) версия протокола Q/R/M/C — идёт как **Technical
Supplement** к AAAI-подаче (`../qrmc_aaai27/`). Содержит всё, что не
влезло в 7 страниц: формальные определения и доказательства оценимости
(Appendix «Operational vs. nested stage estimators»), стресс-тест
stage-Markov-допущения, методологию effect sizes и парных
bootstrap-контрастов, θ-робастность, N=12 scale-up, POM/MOP-анализ Φ,
SOTA coverage-матрицу, расширенный related work, а также полную
таблицу второй модели (qwen3:4b) внешней валидации.

## Структура аргумента

Один logging-контракт → пять классов систем (Table portability):
символический single-agent → символический коллектив (4 архитектуры ×
каденция K) → learned policies (наблюдаемость стадий схлопывается) →
LLM-агенты → сторонние стеки (OpenAI SDK / LangGraph / AutoGen). В
каждом классе — локализованный диагноз, которого не даёт success rate.

## Файлы

- `songlines_qrmc_measurement_framework.tex/pdf` (EN, 46 стр.),
  `*_ru.*` (RU, 47 стр.);
- симлинки: `figures -> ../figures`, `neurips_2026.sty`,
  `checklist.tex -> ../styles/`.

## Связанный код и данные

Те же, что у `../qrmc_aaai27/` (см. его README — таблица полная).
Дополнительно здесь используются: приложения по семантическому
матчингу (`experiments/warp/semantic_identity.py`, W7–W9), VMAS
(`experiments/vmas_portability/`), BabyAI-портируемость.

## Компиляция

```bash
pdflatex songlines_qrmc_measurement_framework.tex   # ×2 (библиография встроенная)
```
