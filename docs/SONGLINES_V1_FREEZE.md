# Songlines Runtime v1 — Freeze Record

**Дата заморозки:** 2026-08-03
**Тег:** `songlines-runtime-v1`
**Назначение:** зафиксировать ровно тот метод и те результаты, которые описывают статьи серии, до любой реорганизации кода и новых сравнений. Любой участник, взяв один коммит, должен получить именно этот метод.

---

## 1. Что входит в Songlines Runtime v1

Единый рантайм — пакет `songlines/` (этап 6 рефактора: `record.py`/`analogy.py`/`alignment.py`/`runtime.py`/`config.py`; `experiments/song_grammar/{runtime,ucsm}.py` — тонкие shim-реэкспорты, все проверенные драйверы работают без правок). Один тип записи `m = (G, C, E, U, P, T, R, F, A)`, один цикл observation→execution, каждый механизм — флаг `Config`.

- **Формирование:** двухосевая матрица (contrfactual utility × structural analogy) → 5 операций MERGE/EXCEPTION/NEW_SCHEMA/REPEAT/DROP; immutable episodic store.
- **Идентичность/перенос:** landmark-констелляции (frame-free), beat-рёбра, mutual-unique + loop-closure; graph-matching аналогия (`exp_g1`).
- **Провенанс:** origin-bound (без ретрансляции; flip-links исключений); certificate = (conditions, ΔV, uncertainty, evidence, analogy map, failures).
- **Коммуникация/допуск:** карантин → валидация на визите по собственной полезности («measured, not told»); world-clock (версия референта); резервации.
- **Safety (continuous):** трёхслойная оборона — anchor consensus ≥ k, commit-top1, safe-prefix verification.
- **Обучение:** LinUCB-контроллер (`exp_u2`), ES по параметрам грамматики (`exp_u3`), long-horizon meta-RL (`exp_m1`), utility-estimator без оракула (`exp_ue1`).

## 2. Что НЕ входит (следующая программа)

Open-ended порождение признаков из выученных представлений · полноценная causal/role graph-analogy (роли, причинность внутри выравнивания) · schema-level слоты и переменные · реальные роботы / фотореалистичный субстрат · большой social-LLM benchmark · общая категорная теория коллективной памяти · continuous SE(2) вариант 2 (метрический registration с RANSAC/covariance) · прямые DeMem/Mage/OAS-бейзлайны на их субстратах.

## 3. Окончательные эксперименты (final)

Part II: W1, W2 (+hold-out), W3, W4, W5a, W6. Part III: R0–R3, W7–W10, S0. Part IV: CSM benchmark, phase diagram. Part V: U1, U3, M1, S3; U7/U7b/U7c/X1; E1; L1/L1b (llama3.1-8B, Qwen2.5-3B/7B); G1; S1, S2. Интеграция: R1, I1 (17 конфигов), UE1, N1v2, B1, P1, C1 (C1.1/C1.3/C1.4).

## 4. Exploratory (не final)

- U2 v1, U1 v1, N1 v1, C1-safe depth-3 — провалившиеся первые регистрации, сохранены с механизмами ревизий.
- co-lock-pressure предиктор (Part II) — помечен exploratory в статье.
- qwen3:4b в L1 — ниже порога задачи, исключён честно.

## 5. Preregistered (предсказания записаны до прогонов)

Все `*_registered.json`. Точные-совпадения: W2 6/6 hold-out, R2 110/110 (21 разрыв, ошибка 0), W9 80/80, W10 0 fail-open/252. Честные FAIL с механизмами: R1-cliff, W9-P1, U2.2, E1.1, G1.1, U7(×4), U7b@e1000, S1(×3), S2, I1-H1b, N1.1, P1.1, C1.4-depth3.

## 6. Не перенастраивать (do-not-retune)

- Константы формирования: U_THR=5, SHARE_THR=0.4, D_THR=3 — заморожены с U1, применялись всюду.
- Закон дальности: α=0.05, τ=0.30 — из Part II/CSM, не менялись.
- Safety-калибровка (consensus, closure, prefix_verify=5): заморожена на dev-семействах 5001+/5101+/200-.
- **Test-сиды 100+ никогда не использовались для настройки любого порога.**

## 7. Соответствие таблиц и файлов

Полная карта claim→эксперимент→файл: `docs/CLAIM_EVIDENCE_MATRIX.md`. Сводка вердиктов: `docs/SERIES_VERDICTS.md`. Сырьё: `tmp/*/*_results.json` (локально) и `sphinx:/mnt/tank/scratch/rzamotaev/songlines/tmp/cluster/song_grammar/` (кластер).

## 8. Окружение

- Локально: Python 3.9.6, numpy 2.0.2, scipy 1.13.1.
- **Кластер (sphinx): numpy 2.5.1** — расхождение версий numpy зафиксировано как caveat воспроизводимости; детерминированные seeds совпадают (U1 воспроизведён 4/4 на обеих машинах), но точные float могут отличаться в 3-м знаке. Финальный rerun (этап 8 плана) должен зафиксировать один `environment.yml`.
- Базовый коммит на момент заморозки: `99689b9` (незакоммиченная рабочая копия содержит всю серию; перед тегом требуется commit — см. §9).

## 9. Действие до тега

Рабочее дерево не закоммичено (серия писалась поверх `99689b9`). Порядок: (1) `git add` кода экспериментов, docs, papers; (2) commit «Songlines Runtime v1»; (3) `git tag songlines-runtime-v1`; (4) записать финальный hash сюда. PDF и tmp-артефакты — по .gitignore-политике репо (не как источник истины; RESULTS.md версионируются).

---
**Критерий завершения этапа 1:** один коммит `songlines-runtime-v1` → ровно метод статьи. **Статус:** документ готов; тег требует явного commit-действия (§9) — не выполняю без запроса, т.к. репо давно не коммитился и это решение автора.
