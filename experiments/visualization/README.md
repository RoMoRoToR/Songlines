# Visualization experiments

Наглядное side-by-side сравнение всех четырёх реализаций коллективной
памяти на одном и том же сценарии. Кадр-за-кадром видно как агенты
двигаются по сетке и как изменяется состояние памяти у каждого из
четырёх вариантов одновременно.

## Запуск

```bash
PYTHONPATH=. .venv/bin/python experiments/visualization/exp_4way_walk.py \
    --n_ticks 28 --out_dir tmp/visualization_4way
```

Длится ~5 секунд. На выходе:
- `tmp/visualization_4way/frames/frame_NNN.png` — 28 PNG кадров (по одному на тик)
- `tmp/visualization_4way/4way_walk.gif` — анимированный GIF, все кадры в одном
- `tmp/visualization_4way/summary.json` — финальное состояние знания у каждого агента

## Сценарий

- Сетка 12×10
- 3 water cells (по одной в NW / NE / S регионах) — `(2,2)`, `(9,2)`, `(5,8)`
- 3 hazards — `(5,4)`, `(5,5)`, `(7,6)`
- 3 агента стартуют в углах:
  - `agent-A` (красный) в NW (0, 0)
  - `agent-B` (синий) в NE (11, 0)
  - `agent-C` (зелёный) в SW (0, 9)
- Каждый агент идёт по **детерминированному** маршруту: A — на восток вдоль верхней кромки потом змейкой, B — на юг потом на запад, C — на север потом на восток. Маршруты захватывают разные регионы.

Маршруты одинаковые во всех четырёх вариантах — env state идентичный. Различается только **состояние памяти**.

## Что показано в каждом кадре

2×2 сетка панелей, по одной на вариант:

| Панель | Реализация |
|---|---|
| (top-left) independent | `independent_memory/` — никакой коммуникации |
| (top-right) shared bus | `songline_drive/` — один event bus + один граф на всех |
| (bottom-left) centralized | `distributed_memory/` — приватные графы + ConsensusLayer |
| (bottom-right) peer | `peer_memory/` — periodic broadcast, свой merged view у каждого |

В каждой панели видно:

- **Цветной фон ячейки** — голубой "W" = water, розовый "X" = hazard
- **Фиолетовые круги** — концепты в памяти (centroids) которые видит этот вариант
- **Цветные кружки со стрелкой** — текущие позиции и направление агентов
- **Бледные цветные линии** — пройденные траектории
- **Пунктирные цветные линии** — связь "агент → концепт который он сейчас знает". Цвет линии = цвет агента
- **Заголовок** — счётчик тиков и сколько концептов знает каждый агент в каждом варианте

## Что видно глазами

Финал после 80 тиков с **асимметричным** layout'ом water cells `[(3, 8), (8, 7), (10, 2)]`:

| Вариант | success | first_success_tick | mean | пояснение |
|---|---|---|---|---|
| independent | 3/3 | A=28, B=4, **C=42** | 24.7 | A и C блуждают долго |
| shared      | 3/3 | **A=74**, B=4, C=5 | 27.7 | **хуже** average — A misled на (3,8), занятое C |
| centralized | 3/3 | **A=74**, B=4, C=5 | 27.7 | то же что shared |
| **peer**    | 3/3 | A=18, B=4, C=10 | **10.7** | **лучше** — delay в broadcast = естественная coordination |

**Нюансированный результат:** коллективная память **не монотонно лучше**. Shared/centralized могут **навредить** при конкуренции за ограниченные ресурсы (A locked onto (3, 8), C занимает первым, A долго не может найти альтернативу). Peer с moderate cadence (k=4) — лучший компромисс.

Визуально:
- **Independent**: длинные периметрические обходы (A и C методом проб и ошибок)
- **Shared / Centralized**: A делает огромный S-образный detour через юг и обратно — типичный misdirection bug
- **Peer**: короткие путь — burst-cadence естественно разводит агентов до того как они конкурируют за один target

## Файлы

- `exp_4way_walk.py` — главный 4-way эксперимент (~600 LOC)
  - Сценарий с **асимметричными** water cells: (3, 8), (8, 7), (10, 2)
  - Memory-driven planner: каждый агент **сам** выбирает действие на основе памяти
  - Tier 1: иди к ближайшей известной незанятой воде
  - Tier 2: exploration — предпочитай невиданные клетки
  - Tier 3: deterministic random fallback
  - Снимает trail + per-agent knowledge + success state
  - Рендерит 2×2 panel-кадр через matplotlib, GIF через imageio
- `exp_peer_cadence_ablation.py` — ablation broadcast cadence (~250 LOC)
  - Тот же сценарий, но фиксирован peer mode
  - Варьирует K ∈ {1, 2, 4, 10} тиков между broadcast'ами
  - Показывает что есть **sweet spot cadence** — ни слишком быстро, ни слишком медленно

## Параметры exp_4way_walk

```bash
--n_ticks N             # сколько тиков (default 40, рекомендую 80 для асимметрии)
--out_dir DIR
--broadcast_every_k K   # как часто peer broadcastит (default 4)
--frame_duration_ms MS  # длительность кадра в GIF (default 350)
--gif / --no-gif        # делать ли GIF (default yes)
```

## Параметры exp_peer_cadence_ablation

```bash
--n_ticks N
--cadences 1,2,4,10     # 2-4 значения K для сравнения
--out_dir DIR
```

## Дополнительные findings

**Sweet spot broadcast cadence** (из `exp_peer_cadence_ablation.py`):

| k | n_succ | mean_tick | пояснение |
|---|---|---|---|
| 1 | 2/3 | 4.5 | A misled to occupied target, не находит alternative |
| 2 | 2/3 | 4.5 | same |
| 4 | 3/3 | 10.7 | **sweet spot** — все доходят |
| 10 | 3/3 | 24.7 | как independent — нет benefit'а |

Это показывает: коллективная память **не всегда монотонно лучше**. Слишком быстрый broadcast = misdirection при конкуренции за ресурс. Слишком медленный = нет coordination. Оптимум — moderate cadence.
