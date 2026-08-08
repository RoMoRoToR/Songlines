# Бенчмарк: шесть типов controlled corruption + long-horizon протокол

**Родитель:** [`04_EXPERIMENTS.md`](04_EXPERIMENTS.md). Этот документ фиксирует общую конструкцию мира-с-историей, поверх которой E1–E10 выбирают конкретные срезы. Без общего long-horizon мира E3 (staleness), E9 (LLM society) и E10 (long horizon) не были бы сопоставимы друг с другом — каждый использовал бы свой одноразовый сценарий.

---

## 1. Почему нужен один общий мир-с-историей, а не набор изолированных сценариев

Ключевое требование, нарушаемое типичным memory-benchmark'ом (явно отмечено в roadmap 12.06 §2.3 п.4 и в критике из обсуждения): между эпизодами память НЕ сбрасывается. Иначе это не long-term memory benchmark, а серия несвязанных single-episode тестов.

```
Episode 1    Agents explore.
Episode 2    Some world facts change.
Episode 3    Agents exchange information.
Episode 4    One old fact becomes false.
Episode 5    Agent B acts based on A's old testimony.
Episode 6    A malicious/incorrect report appears.
Episode 7    The same false report gets retransmitted.
Episode 8    Independent evidence appears.
   ...
Episode 100+
```

Мир между эпизодами эволюционирует по world-clock (тот же механизм, что уже валидирован в S2/S3, `docs/FRONTIER_UCSM_2026-07-27.md`); агенты не получают привилегированного сигнала «мир изменился» — обнаружение инвалидации целиком лежит на evidential admissibility и revocation-механике ([`02_THEOREMS.md`](02_THEOREMS.md) Theorem 2).

## 2. Шесть типов controlled corruption

Каждый тип — независимо инъецируемый, с известной evaluator-side ground truth ДО прогона (та же дисциплина, что в blinded fault benchmark TAE-трека — предсказание пишется на диск до исполнения).

### A. Staleness

```
A: door X is open.
world changes (door X closes).
B later needs door X, действует по старому certificate.
```
Проверяет: Theorem 2 (revocation), метрика Revocation Latency. Основной эксперимент: [`04_EXPERIMENTS.md`](04_EXPERIMENTS.md) E3.

### B. False testimony

```
A ошибочно (не по злому умыслу — шум/неверная интерпретация observation) сообщает B:
"room 4 contains resource."
Ground truth: room 4 пуста.
```
Проверяет: базовую evidential admissibility (не пропускать явно неподтверждённый claim в ADMITTED без независимой валидации). Отличается от Staleness тем, что claim никогда не был верным — не «устарел», а «был неверен изначально».

### C. Provenance laundering

```
A → B → C → D → A
```
Один факт возвращается к исходному агенту, как будто это независимый social consensus. Самый жёсткий тест Theorem 1 — цикл в графе ретрансляции обязан не создавать illusion независимого подтверждения (`n_eff` должен оставаться равным 1, `03_METRICS.md` §5). Основной эксперимент: [`04_EXPERIMENTS.md`](04_EXPERIMENTS.md) E1, расширенный вариант (замыкание цикла) сверх базового linear chain.

### D. Role-dependent validity

```
Путь безопасен для scout, но не для carrier (переносит тяжёлый груз, не может уклоняться).
```
Проверяет: receiver-specific authority, go/no-go свойство №4 (`README.md` §6.4). Основной эксперимент: [`04_EXPERIMENTS.md`](04_EXPERIMENTS.md) E4.

### E. Context-dependent exception

```
Общее правило "narrow corridor → safe route" верно в 90% случаев,
но неверно при state z = carrying_large_object.
```
Проверяет: ось S (structural assimilability), необходимость EXCEPTION-операции на relational уровне. Основной эксперимент: [`04_EXPERIMENTS.md`](04_EXPERIMENTS.md) E8.

### F. Semantic aliasing

```
Два похожих, но различных объекта (напр. две одинаково выглядящие constellation
landmarks) ошибочно матчатся как один референт.
```
Проверяет: пересечение с уже закрытым слоем идентичности (W7–W10, fail-closed vs fail-open гарантия) — здесь тестируется, распространяется ли эта гарантия на authority-слой: неправильный alias НЕ должен получать ADMITTED статус только потому, что структурный матчер его принял (E-гейт должен ловить то, что S-слой пропустил, — независимость осей E/S — прямая проверка).

## 3. Матрица «тип corruption × эксперимент, где проверяется»

| Corruption | Основной эксперимент | Theorem/свойство | Метрика |
|---|---|---|---|
| A. Staleness | E3, E10 | Theorem 2 | Revocation Latency |
| B. False testimony | E1 (базовый), E6 | базовая E-admissibility | FAR |
| C. Provenance laundering | E1 (жёсткий вариант, цикл) | Theorem 1 | PAF, n_eff |
| D. Role-dependent validity | E4 | receiver-specificity | AP/AR по ролям |
| E. Context-dependent exception | E8 | ось S | catastrophic overwrite rate |
| F. Semantic aliasing | E6 (ablation), связано с W7–W10 | независимость E/S осей | FAR при корректном S-match |

## 4. Смешанные (mixed) corruption — обязательное дополнение

По аналогии с TAE-треком (там: «retrieval noise + tight budget», проверка на неспособность one-stage protocol восстановить несколько simultaneous root causes) — здесь минимум одна-две smoke-ячейки должны комбинировать два типа одновременно, например:

```
Staleness + Provenance laundering:
  устаревший факт циркулирует по циклу retransmission,
  выглядя как множественно подтверждённый И актуальный одновременно.
```

**Честная развязка, а не magic recovery:** не нужно ожидать, что authority-протокол механически «решает» комбинацию — регистрируемое предсказание для mixed-ячейки: протокол либо (а) корректно идентифицирует earliest limiting problem (в данном случае staleness доминирует, потому что revocation срабатывает независимо от provenance-структуры), либо (б) явно сигнализирует ambiguity (переход в `CONTESTED`, не молчаливый `ADMITTED`). Второй исход тоже acceptance, если задокументирован механизм — не нужно делать вид, что протокол умеет магически восстанавливать несколько одновременных root causes.

## 5. Long-horizon протокол (общий каркас для E9/E10)

```
H ∈ {10, 50, 100, 250, 500, 1000} эпизодов
```

Каждый corruption-тип инъецируется на фиксированных, известных заранее эпизодах (регистрируется до прогона — не «когда-то в середине», а конкретный номер эпизода), чтобы:
1. Revocation Latency была измерима относительно точного `t_invalidated`;
2. несколько прогонов с разным H были сопоставимы (не переинъекция corruption на разных относительных позициях).

Между эпизодами:
- world-clock продолжает идти (не сбрасывается);
- persistent memory каждого агента сохраняется (не сбрасывается);
- только **evaluator-side ground truth** (для расчёта FAR/Revocation Latency) обновляется — агенты его не видят.

## 6. Инфраструктурное требование: evaluator ground truth должен быть отделён от observable interface агентов

Это прямое перенесение принципа Q/R/M/C-инструментации (`docs/PROJECT_STATUS_2026-05-09.md` §2.1: observable-query assumption явно обозначена как assumption, не скрытая деталь) — evaluator знает точное значение claim в каждый момент времени; агенты — только то, что видели/получили. Без этого разделения FAR и Revocation Latency неизмеримы (не с чем сравнивать admission-решение агента). Практически: та же архитектура, что у ε-match логгера (`docs/FRONTIER_SONG_GRAMMAR_2026-07-25.md` §3.1 — «уровень референта... доступен только измерению»), расширенная с идентичности мест на истинность произвольных claim.

---

**Файл:** `docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/05_BENCHMARK_CORRUPTIONS.md`
