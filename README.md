# Songlines

Исследовательская программа про **семантическую память агентов и её
измеримость**: протокол Q/R/M/C (стадийная декомпозиция выполнения
задач по памяти), коллективная семантическая память с настраиваемой
каденцией обмена, провенанс-анализ («семантический варп»), перенос
маршрутов и тождество мест по смыслу. Шесть рукописей, основная —
подача на **AAAI-27**.

*Songlines is a research program on agent semantic memory and its
measurability: the Q/R/M/C stage-decomposition evaluation protocol,
collective semantic memory with tunable sharing cadence, provenance
analysis, route transfer, and meaning-based place identity.*

---

## Главная идея

Success-only метрики склеивают качественно разные отказы памяти в один
скаляр. Мы раскладываем выполнение задачи «по памяти» на четыре
логируемых события —

**Q**uery formation → **R**etrieval satisfaction → target
**M**aterialization → **C**ompletion

— вложенных по построению, так что P(C\*) = q·r·m·c — точное
тождество, а каждое звено оценивается из обычных логов. Один и тот же
контракт инструментирует символические стеки, коллективы с обменом
памятью, LLM-агентов и сторонние фреймворки (OpenAI SDK / LangGraph /
AutoGen) — и в каждом классе локализует отказ, которого не видит
success rate.

## Серия статей → `papers/`

| Папка | Статья | Роль |
|---|---|---|
| [`papers/qrmc_aaai27/`](papers/qrmc_aaai27/) | Q/R/M/C: Evaluation Protocol (8 стр.) | **Подача AAAI-27** (abstract 21.07, paper 28.07.2026) |
| [`papers/qrmc_measurement_framework/`](papers/qrmc_measurement_framework/) | Полная версия (46 стр.) | Technical Supplement |
| [`papers/semantic_warp/`](papers/semantic_warp/) | Semantic Warp | Companion: провенанс, закон дальности, Warp Drive |
| [`papers/route_warp/`](papers/route_warp/) | The Song, Not the Pin | Companion: перенос маршрутов, тождество мест |
| [`papers/collective_semantic_memory/`](papers/collective_semantic_memory/) | Collective Semantic Memory | Companion: merge/trust/staleness + категорная рамка |
| [`papers/symbolic_memory/`](papers/symbolic_memory/) | Ранняя объединённая версия | Исторический артефакт |

В каждой папке: EN + RU версии, README (тезис, результаты, карта
кода), симлинки на общие `papers/figures/` и `papers/styles/`.
Индекс серии — [`papers/README.md`](papers/README.md).

## Структура репозитория

```
papers/                  # Шесть рукописей (EN+RU), общие фигуры и стили
songline_drive/          # Ядро: интенты, семантическая граф-память,
│                        #   scene encoder/tokenizer, collective_*.py (CSM)
experiments/             # Все эксперименты, по теме на папку, в каждой README:
├── big_experiment/      #   мультиагентный свип 35 640 прогонов + effect sizes
├── qrmc_external/       #   внешняя валидация + meta-evaluation (MemoryHouse)
├── warp/                #   программа W0–W9 и route-warp R0–R3
├── llm_collective/      #   коллектив LLM-агентов (Ollama)
├── collective_semantic_memory/, peer_memory/, distributed_memory/,
├── independent_memory/, multiagent_navigation/, visualization/,
├── commnet_baseline/, commnet_ppo_baseline/, v2_retriever/,
└── vmas_portability/, minigrid_multiagent_wrapper/, place_identity_demo/
scripts/                 # Бенчмарки, oracle-интервенции, фигуры статей
docs/                    # Дизайн-доки фронтиров, планы, отчёты, легаси
tmp/                     # Сырые результаты прогонов (не источник истины —
                         #   у ключевых экспериментов есть RESULTS.md)
```

## Слои системы

1. **Single-agent стек** (`songline_drive/`): восприятие →
   семантические теги → граф-память мест (`semantic_tag_counts` /
   `confidence`) → intent-conditioned retrieval → materialization
   (lock цели) → waypoint-контроллер. Пайплайн:
   `AgentState → Intent → Semantic predicate → Graph target → Waypoint → Action`.
2. **Коллективный слой**: четыре архитектуры обмена — independent,
   shared bus, central aggregator, peer-to-peer broadcast с каденцией
   K; merge с trust-весами, temporal decay, staleness-гейт
   (age_max = ln(trust·conf/τ)/α).
3. **Измерительный слой**: события Q\*/R\*/M\*/C\* на query–lock
   цепочке; oracle-интервенции по стадиям; провенанс-аннотация φ
   каждого lock'а (варп-стратификация); registered predictions.
4. **Тождество мест по смыслу**: отпечатки-созвездия tag@(dx,dy),
   взаимно-уникальный матчинг, SE(2)-выравнивание кадров — коллектив
   работает без общего origin и общего севера.

## Быстрый старт

```bash
# окружение (Python 3.9)
python3.9 -m venv .venv && .venv/bin/pip install -r requirements.txt

# конвенция запуска — всегда из корня репо с PYTHONPATH=.
PYTHONPATH=. .venv/bin/python experiments/warp/exp_warp_gain.py
```

Для LLM-экспериментов нужен локальный **Ollama** с моделями
`llama3.1:latest` и `qwen3:4b` (для qwen везде включён think-off —
`extra_body={"think": False}` — ради паритета бюджета).

### Компиляция статей

```bash
cd papers/<paper>/ && pdflatex songlines_<paper>.tex        # NeurIPS: ×2
# AAAI: pdflatex + bibtex + pdflatex ×2
```

Русские версии используют пакет `tempora`.

## Методологические конвенции

- **Registered predictions**: предсказания и решающие правила
  фиксируются в `tmp/<exp>/registered*.json` **до** прогонов; провалы
  регистраций публикуются с механизмами (v1→v2→v3 ревизии семантики —
  задокументированы там же).
- **Долгие прогоны** — через `nohup ... & disown` (фоновые джобы
  оболочки убиваются); у раннеров есть resume по per-episode
  чекпоинтам (`rows_*.json`).
- **Статистика**: cluster-bootstrap CI по дизайн-ячейкам (не по
  прогонам), парные cell-контрасты, Wilson CI при малых n, точные
  sign-тесты; effect sizes вместо голых p-values.
- **Потолок дефицита**: при N агентах и M целях P(C\*|M\*) ≤ M/N —
  recovery любого протокола меряется против него, не против базлайна.

## Ключевые данные в `tmp/`

| Папка | Что |
|---|---|
| `paper1_clean_experiments_full/` | свип 35 640 прогонов (runs.csv) + fig_aggregates.json |
| `big_experiment_{qrmc_40,extraK,oracle}/` | базовый свип, K∈{32..64}, oracle-интервенции |
| `oracle_stage_interventions_final_20260430/` | single-agent oracle (per-seed/episode) |
| `article_revision_10seeds_20260501/` | retrieval-логи (91% empty, query_debug) |
| `warp/` | W0–W9, R0–R3 (per-lock φ, WD, законы) |
| `qrmc_external/` | внешняя валидация + meta-evaluation (registered, rows, verdicts) |

## Статус (июль 2026)

- AAAI-27: контент в лимите 7 стр.; reproducibility checklist заполнен
  (вычитать DRAFT-обоснования + лицензия кода); ответ на внешнюю
  рецензию внесён (формализм вложенности, meta-evaluation по слепой
  инъекции отказов, статистика с CI, related work).
- Идёт: qwen-прогон meta-evaluation (вставка результатов в статьи по
  готовности).
- До подачи: вычитка автором, анонимизированный Code & Data
  Supplement, плоский abstract для формы.

## История

Репозиторий начинался с кодовой базы
[Active Neural SLAM](https://openreview.net/pdf?id=HklXn1BKDH)
(Chaplot et al.); ранний этап проекта — семантический выбор мест в
MiniGrid/MiniWorld — описан в
[`docs/LEGACY_README_semantic_place_selection.md`](docs/LEGACY_README_semantic_place_selection.md).
Дизайн-доки фронтиров: `docs/FRONTIER_SEMANTIC_WARP_2026-07-02.md`,
`docs/FRONTIER_ROUTE_WARP_2026-07-02.md`; сводный отчёт программы:
`docs/PROJECT_REPORT_QRMC_WARP_2026-07-09.md`.
