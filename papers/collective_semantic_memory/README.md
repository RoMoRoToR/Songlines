# Collective Semantic Memory: Merge, Trust, and Staleness for Peer Memory in Multi-Agent Navigation

Companion-статья №3 (22 стр.). Механика коллективной семантической
памяти: peer-to-peer обмен снапшотами с trust-взвешенным merge,
temporal decay и conflict fusion — и категорная рамка, в которой
«общность интерпретации» параметризована явно.

## Ключевые результаты

- **Гейт, а не обмен:** выигрыш CSM обеспечивает staleness-гейт
  (age_max = ln(trust·conf/τ)/α, hold-out 6/6; trust-flip обнуляет
  варп), а не сам факт обмена.
- **Тождество мест из смысла** (секция Place identity from meaning,
  компакт W7–W9): recovery 100%, fail-closed vs fail-open асимметрия,
  восстановление 88% разрыва oracle/coordinate в полном стеке.
- **Категорная рамка:** alignment-морфизмы **сконструированы**
  (трансляционный и SE(2)-выравниватели), существование копредела —
  эмпирический предикат; 180°-симметричный мир = операционализованная
  категорная несуществуемость. Три параметра общности интерпретации:
  алфавит Σ (предполагается), морфизмы выравнивания (строятся),
  trust×staleness обогащение (градуируется).
- Коалгебраическое чтение вынесено в приложение как «lens, not
  mechanism».
- **Σ-внутренность (2026-07-26):** внутри «предполагаемого» алфавита Σ
  измеримо, какие классы тегов несут соответствие (W10 companion:
  fail-safe, атрибуция hazard/wall/void); отбор Σ координацией — самый
  острый открытый конец параметра (i). Selective consolidation получила
  измеренный пилот: реляционные песни = перенос уровня снапшота за 6.4%
  бит (S0), песни поверх тех же трёх правил CSM — больше не спекуляция.

## Файлы

- `songlines_collective_semantic_memory.tex/pdf` (EN), `*_ru.*` (RU);
- `collective_memory_appendix.tex` — отдельное приложение фаз;
- симлинки `figures`, `neurips_2026.sty`, `checklist.tex`.

## Связанный код и данные

| Что | Где |
|---|---|
| Ядро CSM (merge/trust/decay/fusion) | `songline_drive/collective_*.py` |
| Фазовые эксперименты 1–4 | `experiments/collective_semantic_memory/` |
| Peer-to-peer без центра | `experiments/peer_memory/` |
| Изолированная память + консенсус | `experiments/distributed_memory/` |
| First-class independent-вариант | `experiments/independent_memory/` |
| Тождество мест | `experiments/warp/semantic_identity.py`, `experiments/place_identity_demo/` |
| Навигация/визуализация | `experiments/multiagent_navigation/`, `experiments/visualization/` |

## Компиляция

```bash
pdflatex songlines_collective_semantic_memory.tex   # ×2
```
