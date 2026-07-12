# collective_semantic_memory — фазовые эксперименты CSM

Смоки и A/B-проверки механики коллективной семантической памяти для
[`papers/collective_semantic_memory/`](../../papers/collective_semantic_memory/).
Ядро реализации — `songline_drive/collective_*.py` + `multiagent_runtime`.
Числа — в [`RESULTS.md`](RESULTS.md).

Фазы (все закрыты в мае 2026):

- **Phase 1** — параллельный пакет коллективной памяти (broadcast
  снапшотов, merge);
- **Phase 2** — shared concept consolidation + `ConceptRecallLayer`
  (A/B-смок);
- **Phase 3** — temporal decay + conflict fusion + инкрементальные
  обновления;
- **Phase 4a/4b** — semantic field (дескриптивный) + read-only
  reranking; **4c/4d** — coordinated deconfliction + adaptive
  reweighting.

Связанные направления в соседних папках: `peer_memory/` (p2p без
центра), `distributed_memory/` (изолированная память + консенсус),
`independent_memory/` (no-comm контракт), `multiagent_navigation/`,
`visualization/`.
