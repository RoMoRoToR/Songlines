# qrmc_external — внешняя валидация Q/R/M/C на сторонних агентных стеках

Два эксперимента для [`papers/qrmc_aaai27/`](../../papers/qrmc_aaai27/)
и полной версии. Результаты первого — [`RESULTS.md`](RESULTS.md).

## 1. Внешняя валидация (external validation)

Среда **MemoryHouse** (`memory_house.py`): эпизодическая память дома за
инструментами `recall`/`goto`/`take`, единый tool-trace контракт
Q\*/R\*/M\*/C\* (маргинальные частоты — tool trace не даёт lock-цепочки,
произведение не заявляется). Один адаптер на три стека — OpenAI SDK
function-calling loop, LangGraph ReAct, AutoGen classic
(`run_external_validation.py`), 4 варианта × 20 сидов × 2 модели
(llama3.1, qwen3:4b).

Главное: структурные отказы локализуются идентично на всех стеках
(consolidation gap → R=0.00; tight budget → C=0.00; 20/20), а
поведенческий разъезд чистого control (0.50–0.75) атрибутируется в
слабую первокоммитную дисциплину (M1) + разные recovery-петли.

## 2. Meta-evaluation по слепой инъекции отказов

`run_meta_evaluation.py` — протокол проверяется как **классификатор**:
10 типов неисправностей по 2–3 на стадию (структурные И поведенческие,
`FAULT_STAGE` в `memory_house.py`), решающее правило и пороги
зарегистрированы **до** прогонов (`tmp/qrmc_external/registered_meta.json`),
hold-out control меряет false positives, manipulation check исключает
несвязавшиеся фолты. Метрики: stage-level accuracy, confusion matrix,
межфреймворковое согласие.

llama3.1 (720 эпизодов): openai_sdk 8/10, langgraph 8/10, autogen
10/10; структурные 15/15; hold-out → none 3/3; промахи — в
зарегистрированных рисках (M-семейство на слабом коммиттере).

## Данные

`tmp/qrmc_external/`: `registered*.json` (регистрации),
`rows_*/summary_*` (llama20/qwen20), `rows_meta_*/meta_verdict_meta_*`
(мета-оценка), `meta_run.log`.

## Запуск

```bash
PYTHONPATH=. .venv/bin/python experiments/qrmc_external/run_external_validation.py \
    --model llama3.1:latest --seeds 20
PYTHONPATH=. .venv/bin/python experiments/qrmc_external/run_meta_evaluation.py \
    --model qwen3:4b --tag meta_qwen
```

Нужен локальный Ollama. Многочасовые прогоны запускать через
`nohup ... & disown` (есть resume по `rows_*.json`).
