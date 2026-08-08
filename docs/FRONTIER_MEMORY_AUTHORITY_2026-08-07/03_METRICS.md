# Новый vocabulary метрик: False Authority Rate, Authority Precision/Recall, Revocation Latency, PAF

**Родитель:** [`01_FORMAL_MODEL.md`](01_FORMAL_MODEL.md), [`02_THEOREMS.md`](02_THEOREMS.md). Существующие метрики серии (Q/R/M/C conditional rates, duplicate_target_rate, reservation_conflict_rate, fail-open rate, mislock rate) остаются как есть и переиспользуются без изменений везде, где применимы (`docs/PROJECT_STATUS_2026-07-18.md` §2 — карта код→статья→данные). Ниже — метрики, специфичные для authority-слоя, которых в существующем vocabulary нет.

---

## 1. False Authority Rate (FAR)

Обычного success/fail-open недостаточно, чтобы отличить «система провалилась потому что была неудачная задача» от «система провалилась потому что действовала по недопустимому свидетельству».

```
FAR = #{actions caused by invalid authoritative memories} / #{memory-driven actions}
```

**Invalid** означает: certificate был в состоянии `ADMITTED` в момент действия, но по независимой (evaluator-side) ground truth claim был ложным/устаревшим/неприменимым к роли агента в тот момент. Требует `ActionRecord.supporting_certificate_ids` (§8 `01_FORMAL_MODEL.md`) + evaluator ground truth (аналог ε-match логгера, `docs/FRONTIER_SONG_GRAMMAR_2026-07-25.md` §3.1, уровень референта).

**Интерпретация:** FAR — прямой аналог fail-open rate из geometric-safety слоя (N1v2/C1.4, `docs/SERIES_VERDICTS.md`), но на уровне evidential admissibility, а не идентичности места. Целевой acceptance bar по аналогии с уже принятым в серии стандартом (`<1% safety-critical fail-open`, `docs/REVIEW_RISK_REGISTER.md`) — регистрируется отдельно для authority-слоя, не наследуется автоматически: природа отказа другая (эпистемическая, не геометрическая).

## 2. Authority Precision / Authority Recall

Стандартная пара precision/recall, применённая к решению admission-функции против evaluator ground truth «valid AND useful»:

```
Authority Precision (AP) = P(memory valid and useful | admitted)
Authority Recall    (AR) = P(admitted | memory valid and useful)
```

**Зачем нужна пара, а не одна метрика.** Высокий AP при низком AR = система излишне консервативна (потенциальный провал liveness, Theorem 3). Высокий AR при низком AP = система излишне доверчива (провал safety, Theorem 1/2). Обе кривые обязательны в [`04_EXPERIMENTS.md`](04_EXPERIMENTS.md) §E6 (Authority Gating Ablation) — заменяют одномерный accuracy/F1, которым уже пользуется TAE-трек Q/R/M/C статьи (0.873 accuracy / 0.835 macro-F1 для fault-localization, отдельный трек — не путать, см. `README.md` §5), но здесь по другой оси: не «правильно ли определена стадия отказа», а «правильно ли выдано право действовать».

## 3. Revocation Latency

```
L_R = t_revoked − t_invalidated
```

`t_invalidated` — момент, когда claim перестал соответствовать миру (evaluator-side ground truth, известный только измерению, не агенту — аналог world-version в S2/S3). `t_revoked` — момент, когда authority-слой сам перевёл certificate в `REVOKED`.

**Связь с уже измеренным.** Прямое продолжение S2 (`docs/FRONTIER_UCSM_2026-07-27.md`: «world-clock staleness необходим, но недостаточен» — rob −7.4%, но independent бьёт 12/12) и rupture law R2 (`docs/SERIES_VERDICTS.md`: «110/110, все 21 разрыв на предсказанном ребре, ошибка 0»). Revocation Latency — та же идея разрыва песни по возрасту, формализованная как метрика authority-слоя, применимая не только к рёбрам-маршрутам, но к произвольному claim.

**Как используется:** [`04_EXPERIMENTS.md`](04_EXPERIMENTS.md) §E3 — сравнение `L_R` между raw history / vector RAG / shared graph (без гейта) / staleness-only / full authority protocol. Ожидаемый порядок: full authority protocol даёт наименьшую `L_R` без роста FAR на неinvalidated claims (иначе это не revocation, а просто агрессивное забывание).

## 4. Provenance Amplification Factor (PAF)

```
PAF = authority(claim, после k ретрансляций без нового origin) / authority(claim, при origin)
```

**Прямая операционализация Theorem 1.** Для корректной по Theorem 1 системы:

```
PAF ≤ 1   (не растёт от чистой ретрансляции)
```

Строгий тест: `PAF ≈ 1` (не строго `< 1`, потому что небольшое падение из-за transport delay/staleness — ожидаемо и не является провалом теоремы; рост — является). Измеряется на прогоне [`04_EXPERIMENTS.md`](04_EXPERIMENTS.md) §E1, per-hop, для нескольких архитектур сравнения (наивный shared context, vector memory, наивный graph merge по счётчику источников, source-count trust, полный provenance-aware authority-протокол).

## 5. Effective independent support n_eff

Вспомогательная величина, нужная и для PAF, и для Authority Recall:

```
n_eff(m) = |{независимые origin_ids, поддерживающие m}|
```

в противоположность наивному подсчёту «сколько агентов сейчас утверждают m» (что включает транспортные копии одного origin). Прямое следствие разграничения `origin_ids` / `provenance_parents` (§2 `01_FORMAL_MODEL.md`).

## 6. Как метрики распределяются по экспериментам (сводная таблица)

| Метрика | Основной эксперимент | Theorem, которую операционализирует |
|---|---|---|
| PAF, n_eff | E1 (Social Amplification) | Theorem 1 |
| authority(n_independent) кривая | E2 (Independent Corroboration) | Theorem 3 |
| Revocation Latency, FAR | E3 (Stale Truth) | Theorem 2 |
| AP/AR по ролям | E4 (Role-dependent Knowledge) | receiver-specificity (README §6.4) |
| Spearman/calibration/sign accuracy Û vs τ_true | E5 (Utility Estimator) | §4 `01_FORMAL_MODEL.md` |
| AP/AR ablation-таблица | E6 (Authority Gating Ablation) | — (проверка необходимости каждого гейта) |
| FAR + task success | E9 (Full LLM Society), E10 (Long Horizon) | все три Theorem одновременно, на реалистичном субстрате |

---

**Файл:** `docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/03_METRICS.md`
