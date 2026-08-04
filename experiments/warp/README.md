# warp — программа «Семантический варп» (W0–W10) и route-warp (R0–R3)

Экспериментальная база для двух статей:
[`papers/semantic_warp/`](../../papers/semantic_warp/) (W0–W6) и
[`papers/route_warp/`](../../papers/route_warp/) (R0–R3 + W7–W10).
Полный отчёт с числами — [`RESULTS.md`](RESULTS.md). Все ключевые
эксперименты регистрируют предсказания до прогонов; провалы
регистраций публикуются с механизмами.

## Карта файлов по волнам

| Волна | Что делает | Файлы |
|---|---|---|
| W0 | φ (foreignness) на каждом M\*-lock из весов merge | `warp_instrumentation.py`, `warp_runner.py`, `exp_warp_anchor.py` |
| W1 | Стратификация P(C\*\|W\*) vs P(C\*\|own), warp gain, такси-цепочки | `exp_warp_gain.py`, `analyze_warp_chains.py` |
| W2 | Закон дальности age_max=ln(trust·conf/τ)/α + hold-out | `exp_warp_age_law.py` |
| W3 | Warp Drive: резервация + anti-M\* rollback + backoff | `warp_drive.py`, `exp_warp_drive.py` |
| W4 | Cross-session strict W\* на LLM (Ollama) | `exp_warp_cross_session.py` |
| W5a | VMAS: страты и закон на непрерывной подложке | `exp_warp_vmas.py` |
| W6 / W6-C | LLM-коллектив N=3: обмен, коллизии, WD на LLM | `exp_warp_llm*.py` |
| R0–R3 | Route-warp: строгий RW\*, cliff, закон разрыва, hazard-стратификация | `exp_route_warp_r0.py` … `_r3.py`, `RoutePeerMemory` |
| W7 | Тождество мест: отпечатки-созвездия + mutual-unique матчинг | `semantic_identity.py`, `exp_semantic_identity.py` |
| W8 | Семантика в полном peer/CSM-стеке, закон без общего кадра | `semantic_peer_memory.py`, `exp_semantic_stack.py` |
| W9 | SE(2)-инвариантность (4 гипотезы поворота, corner-trap) | `align_frames_se2` в `semantic_identity.py`, `exp_se2_*.py` |
| W10 | Абляция словаря ориентиров Σ (fail-safe + атрибуция) | `exp_warp_landmark_ablation.py` |
| — | Порог-чувствительность φ (ответ рецензенту) | `phi_sensitivity.py` |
| — | Фигуры статей | `make_warp_figures.py`, `make_route_figures.py` → `papers/figures/` |

## Данные

`tmp/warp/` — сырые прогоны (`w1_gain/w1_rows.jsonl` — 2 400 эпизодов с
per-lock φ; `w3_drive/w3_results.json`; `r1_route_gain/` и т.д.) и
`phi_sensitivity.json`.

## Запуск

```bash
PYTHONPATH=. .venv/bin/python experiments/warp/exp_warp_gain.py
```
(у каждого exp-файла — docstring с usage; LLM-волны требуют локальный
Ollama с llama3.1).
