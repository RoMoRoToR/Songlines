# v2_retriever — retrieval-форензика single-agent стека

Анализ retrieval-звена для single-agent секций
[`papers/qrmc_aaai27/`](../../papers/qrmc_aaai27/) и полной версии.
Числа — [`RESULTS.md`](RESULTS.md).

Главный результат серии: на 10-сидовом hazard-recovery прогоне из 349
retrieval-вызовов только 32 (9.2%) вернули хотя бы одного кандидата, и
из них 31 (96.9%) выбрал семантически удовлетворяющего — узкое место
не **ранжирование**, а **генерация кандидатов** (накопление памяти).
Цифра 91% пустых кандидатов устойчива к взвешиванию (pooled 0.908 /
seed 0.909 / episode 0.891 — `scripts/analyze_oracle_pairing.py`) и к
порогу θ (θ-абляция).

Сырые записи запросов: `tmp/article_revision_10seeds_20260501/.../
query_debug.json` (per-query candidate_node_ids + satisfied-флаги).
