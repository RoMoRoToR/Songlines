# Songlines: status on 2026-05-09

Этот файл фиксирует, что уже было сделано по статье, бенчмаркам и коду, на каком этапе проект находится сейчас, и что осталось открытым.

## 1. Где мы находимся сейчас

- Основной paper собран и лежит в:
  - [songlines_symbolic_memory.tex](/Users/taniyashuba/PycharmProjects/Songlines/docs/Formatting_Instructions_For_NeurIPS_2026/songlines_symbolic_memory.tex)
  - [songlines_symbolic_memory.pdf](/Users/taniyashuba/PycharmProjects/Songlines/docs/Formatting_Instructions_For_NeurIPS_2026/songlines_symbolic_memory.pdf)
- Текущий PDF успешно компилируется через `pdflatex`.
- Текущая версия статьи уже переведена из режима “система навигации” в режим “диагностический framework для semantic navigation”.
- Main narrative сейчас держится на трех столпах:
  - `Q/R/M/C` stage decomposition,
  - oracle-stage intervention block,
  - contrast with a stronger learned baseline.

## 2. Что было реализовано по содержанию статьи

### 2.1. Диагностический каркас Q/R/M/C

В статье и коде закреплена четырехстадийная декомпозиция semantic-place retrieval:

- `Q`: query formation
- `R`: retrieval satisfaction
- `M`: target materialization
- `C`: completion after retrieval

Сделано:

- формальная probabilistic factorization;
- identifiability theorem under a conditional stage-Markov assumption;
- разделение между:
  - operational stage rates,
  - nested episode-level starred estimators `Q* / R* / M* / C*`;
- отдельный consistency audit, показывающий, что product nested estimators matches semantic-path completion, а не raw success.

### 2.2. Symbolic-semantic memory stack

Сохранен и описан memory stack, который:

- токенизирует observation в symbolic evidence;
- обновляет place-level semantic memory;
- хранит transition statistics;
- хранит episodic traces;
- поддерживает state-conditioned semantic retrieval;
- materializes retrieved candidate into an actionable target.

В paper это сведено в компактный pipeline-level presentation, а детали вынесены в appendix.

### 2.3. Oracle-stage intervention block

Это один из главных апгрейдов статьи.

Реализовано:

- targeted oracle interventions на двух hardest task families:
  - `goal_region` on `MiniGrid-FourRooms-v0`
  - `hazard_recovery` on `MiniGrid-LavaGapS7-v0`
- 4 режима:
  - `base`
  - `oracle retrieval`
  - `oracle materialization`
  - `oracle controller`
- canonical run protocol:
  - `10 seeds`
  - `8 episodes per seed`

Главный вывод уже встроен в paper:

- `hazard_recovery` действительно retrieval/materialization-limited:
  - `0.39 -> 0.60` under oracle retrieval
  - `0.39 -> 0.59` under oracle materialization
  - only `0.39 -> 0.43` under oracle controller
- `goal_region` не улучшается end-to-end under isolated oracles:
  - success остается `0.08`
  - но semantic-path completion поднимается under retrieval/materialization repair
  - это поддерживает interpretation “downstream of candidate selection”, а не “pure query failure”

### 2.4. Oracle diagnostics beyond topline

Добавлены дополнительные oracle diagnostics:

- oracle activation rates;
- stage deltas under oracle modes;
- semantic-path completion deltas;
- qualitative oracle case studies.

Это закрывает reviewer-risk вида:

- “oracles may not have activated”
- “flat success may be due to missing intervention coverage”

### 2.5. Stronger learned baseline

Paper больше не опирается только на tabular baseline.

Реализован stronger learned baseline:

- `behavior-cloned full-state baseline`
- код:
  - [scripts/learned_external_baseline_minigrid.py](/Users/taniyashuba/PycharmProjects/Songlines/scripts/learned_external_baseline_minigrid.py)

В paper baseline story теперь сформулирована так:

- learned baseline может быть very strong success-only reference;
- но он не expose-ит meaningful intermediate failure states;
- именно поэтому `Q/R/M/C` нужен как diagnostic layer, а не как competing leaderboard metric.

### 2.6. Noise robustness block

Добавлен отдельный robustness block:

- semantic tag dropout;
- false positives in local semantic evidence;
- stage-wise degradation under noise.

Код:

- [scripts/benchmark_semantic_noise_robustness.py](/Users/taniyashuba/PycharmProjects/Songlines/scripts/benchmark_semantic_noise_robustness.py)

В paper и appendix это уже отражено как supplementary evidence, а не как headline contribution.

## 3. Что было сделано по экспериментальной части

### 3.1. Main MiniGrid benchmark

Статья переведена на `canonical 10-seed benchmark`.

Это было сделано, чтобы убрать старую несостыковку:

- раньше main benchmark местами был на `3 seeds`;
- oracle block уже был на `10 seeds`;
- reviewer мог бы считать CI и protocol inconsistent.

Сейчас main benchmark в тексте нормализован к `10 seed` framing.

### 3.2. Cross-layout split

Сохранен и корректно описан cross-layout result для `goal_region`:

- easy layout `Empty-6x6`
- hard layout `FourRooms`

Именно этот split поддерживает вывод, что `goal_region` collapses on the hard layout and that the failure is not relieved by isolated retrieval/materialization replacement.

### 3.3. MiniWorld runtime and transfer block

MiniWorld runtime был поднят и починен.

Сделано:

- устранен forced headless/EGL path на macOS;
- исправлен output-directory bug в `compare_songline_miniworld.py`;
- подтвержден runtime on `MiniWorld-FourRooms-v0`.

Код и runtime support:

- [scripts/songline_miniworld.py](/Users/taniyashuba/PycharmProjects/Songlines/scripts/songline_miniworld.py)
- [scripts/compare_songline_miniworld.py](/Users/taniyashuba/PycharmProjects/Songlines/scripts/compare_songline_miniworld.py)
- [songline_drive/miniworld_support.py](/Users/taniyashuba/PycharmProjects/Songlines/songline_drive/miniworld_support.py)

Важно:

- MiniWorld block был реализован как portability check;
- позже article narrative был переведен в сторону BabyAI portability, а не MiniWorld-first story.
- То есть MiniWorld integration в кодовой базе есть и работает, но в текущей paper narrative он больше не является центральным non-MiniGrid block.

### 3.4. BabyAI portability block

Реализован более полезный reviewer-facing transfer block на BabyAI.

Код:

- [scripts/compare_songline_babyai.py](/Users/taniyashuba/PycharmProjects/Songlines/scripts/compare_songline_babyai.py)

Сделано:

- wrapper, который конвертирует BabyAI mission в semantic intent;
- reuse существующего memory stack;
- reuse существующего query-debug / `Q/R/M/C` logging.

Поддерживающий код:

- [scripts/songline_minigrid.py](/Users/taniyashuba/PycharmProjects/Songlines/scripts/songline_minigrid.py)
- [songline_drive/scene_encoder.py](/Users/taniyashuba/PycharmProjects/Songlines/songline_drive/scene_encoder.py)

Текущий paper-facing BabyAI result:

- environment: `BabyAI-GoToObjMaze-v0`
- methods:
  - `random`
  - `greedy`
  - `babyai_semantic_node_v1`
  - `babyai_semantic_plan_v1`
- topline success:
  - `random = 0.125`
  - `greedy = 0.175`
  - `babyai_semantic_node_v1 = 0.175`
  - `babyai_semantic_plan_v1 = 0.175`
- semantic methods are the only ones with non-zero `Q/R/M/C` traces.

Interpretation in paper:

- этот блок проверяет portability of the diagnostic pipeline;
- он не продается как strong navigation competitiveness result.

## 4. Что было сделано по paper framing

### 4.1. Abstract / intro reframing

Статья была переписана так, чтобы не звучать как:

- “we built a navigation system that gets X success”.

Она теперь звучит как:

- “we built a diagnostic framework for semantic navigation that produces stage-attributable failures and intervention-guiding measurements”.

Это отражено в:

- `Abstract`
- `Introduction`
- `Discussion`
- `Conclusion`

### 4.2. Closed-loop reading of oracle block

Один из ключевых narrative upgrades:

- oracle block теперь подан как limited closed diagnostic loop:
  - diagnose via `Q/R/M/C`
  - intervene via oracle replacement
  - verify attribution counterfactually

Пока это не “retrained v2 agent loop”, а именно oracle-based loop closure.

### 4.3. Failure analysis refinement

Сейчас failure analysis четко разведен:

- `goal_region` = downstream failure after candidate selection;
- `hazard_recovery` = upstream retrieval/materialization bottleneck.

Это важная часть текущей value proposition статьи.

### 4.4. Learned baseline reframing

Learned baseline больше не выглядит как “paper defeated by a stronger policy”.

Он переосмыслен так:

- да, strong opaque policy может выигрывать по success;
- но она не дает interpretable intermediate failure states;
- значит framework нужен именно там, где scalar success is not explanatory.

### 4.5. Limitations section

Limitations были переписаны в явном виде.

Сейчас они честно фиксируют:

- основная evidence still mostly MiniGrid;
- BabyAI is only a portability check;
- observable-query assumption is real;
- loop is closed via oracle, not yet via retrained retrieval module;
- explicit-memory LLM baseline is still missing.

## 5. Что было сделано по структуре paper и appendix

Статья была существенно перестроена, чтобы main text выглядел чище и closer to NeurIPS main-track expectations.

Текущее состояние:

- main text более компактный;
- существенная часть diagnostic details вынесена в appendix;
- appendix содержит:
  - formal proof;
  - operational vs nested estimator definitions;
  - memory/task schema;
  - BabyAI portability block;
  - extended diagnostics;
  - assist on/off analysis;
  - reproducibility commands;
  - asset and license note.

## 6. Что было сделано по figures и tables

### 6.1. Основные paper figures

Собраны и используются article-facing figures:

- [article_stage_heatmap.png](/Users/taniyashuba/PycharmProjects/Songlines/docs/Formatting_Instructions_For_NeurIPS_2026/songlines_symbolic_memory_figures/article_stage_heatmap.png)
- [article_oracle_stage_interventions.png](/Users/taniyashuba/PycharmProjects/Songlines/docs/Formatting_Instructions_For_NeurIPS_2026/songlines_symbolic_memory_figures/article_oracle_stage_interventions.png)
- [article_baseline_comparison.png](/Users/taniyashuba/PycharmProjects/Songlines/docs/Formatting_Instructions_For_NeurIPS_2026/songlines_symbolic_memory_figures/article_baseline_comparison.png)

### 6.2. Appendix/support figures

Подготовлены и сохранены:

- stage consistency
- synthetic validation
- oracle case studies
- funnel/stage-profile panel
- memory scaling
- noise robustness
- assist comparison
- cross-layout
- BabyAI portability figure

См. директорию:

- [songlines_symbolic_memory_figures](/Users/taniyashuba/PycharmProjects/Songlines/docs/Formatting_Instructions_For_NeurIPS_2026/songlines_symbolic_memory_figures)

### 6.3. Cosmetic LaTeX pass

Сделано:

- укорочены method labels в широких таблицах;
- добавлены пояснения с полными method names;
- сжаты широкие таблицы;
- отдельно была ужата `Figure 2` pipeline;
- дополнительно была ужата первая appendix table, чтобы она влезала на страницу.

## 7. Что было сделано по reproducibility и licensing

### 7.1. Reproducibility block

В article appendix перечислены source-of-truth scripts:

- [scripts/benchmark_symbolic_memory_article.py](/Users/taniyashuba/PycharmProjects/Songlines/scripts/benchmark_symbolic_memory_article.py)
- [scripts/analyze_symbolic_memory_article.py](/Users/taniyashuba/PycharmProjects/Songlines/scripts/analyze_symbolic_memory_article.py)
- [scripts/validate_qrmc_factorization.py](/Users/taniyashuba/PycharmProjects/Songlines/scripts/validate_qrmc_factorization.py)
- [scripts/benchmark_oracle_stage_interventions.py](/Users/taniyashuba/PycharmProjects/Songlines/scripts/benchmark_oracle_stage_interventions.py)
- [scripts/benchmark_semantic_noise_robustness.py](/Users/taniyashuba/PycharmProjects/Songlines/scripts/benchmark_semantic_noise_robustness.py)
- [scripts/compare_songline_babyai.py](/Users/taniyashuba/PycharmProjects/Songlines/scripts/compare_songline_babyai.py)

### 7.2. License note

Appendix asset note обновлен:

- MiniGrid cited as MIT-licensed
- BabyAI cited as MIT-licensed
- local benchmark/baseline scripts planned for MIT release at camera-ready

### 7.3. Checklist fixes

Сделаны правки в:

- [checklist.tex](/Users/taniyashuba/PycharmProjects/Songlines/docs/Formatting_Instructions_For_NeurIPS_2026/checklist.tex)

В том числе:

- обновлены justification references;
- обновлены asset/license items;
- приведены в соответствие section references after restructuring.

## 8. Что уже реализовано в коде, а не только в тексте

Реально существуют и используются:

- canonical article benchmark runner;
- article analysis/figure generation;
- oracle-stage benchmark;
- noise robustness benchmark;
- stronger learned baseline script;
- MiniWorld runner and comparison script;
- BabyAI comparison script;
- BabyAI mission-to-intent adaptation;
- MiniWorld macOS runtime support patch.

То есть paper changes не чисто редакторские: они уже поддержаны отдельными скриптами и experimental artifacts.

## 9. Текущий stage проекта

Сейчас проект находится на этапе:

- `paper polishing / submission preparation`

То есть:

- framework formulation уже собрана;
- main experiments уже есть;
- oracle block уже есть;
- stronger baseline уже есть;
- BabyAI portability already exists;
- noise robustness already exists;
- appendix/reproducibility/licensing already partly cleaned up.

Это уже не stage “designing the contribution” и не stage “first experiments”.
Это stage “compress, verify, and decide what is still worth adding before submission”.

## 10. Что остается проблемным или неполностью закрытым

### 10.1. Environment difficulty

Главная содержательная слабость все еще в том, что основной evidence mostly MiniGrid.

BabyAI улучшает story, но пока:

- он подан как portability check,
- а не как second strong benchmark.

### 10.2. Closed loop is still oracle-based

Сейчас paper утверждает actionable diagnosis через oracle-loop closure.

Еще не реализовано:

- реальное targeted retrieval fix,
- retrained v2 system,
- measurement that `Delta R` grows while unrelated stages stay stable.

Это был бы следующий scientifically strongest upgrade.

### 10.3. Missing explicit-memory LLM baseline

Это прямо зафиксировано как missing comparison.

Сейчас есть:

- random
- graph-only
- SPTM-like
- behavior-cloned learned baseline

Но нет:

- MemGPT/Reflexion-style or any explicit-memory LLM-agent baseline

### 10.4. Some easy-task baselines are still too strong

Даже после всех narrative fixes, `water/rest` остаются positive-control tasks on small layouts.
Это не ломает paper, но ограничивает how impressive those toplines look to a hard reviewer.

## 11. Что логично делать дальше

Если продолжать с максимальной scientific return, порядок такой:

1. Реальный `diagnose -> targeted fix -> verify` loop on hazard retrieval.
2. Более сильный harder environment block beyond portability-only BabyAI.
3. Если хватит времени, explicit-memory LLM baseline.
4. Финальный LaTeX cleanup and submission hygiene pass.

Если продолжать с максимальной submission practicality, порядок такой:

1. Еще раз проверить page budget и visual cleanliness PDF.
2. Прогнать final compile twice and verify all refs.
3. Проверить, что checklist, appendix labels, and references are internally consistent.
4. Зафиксировать final artifact list and reproduction commands.

## 12. Короткий итог

На текущий момент реализовано:

- диагностический framework `Q/R/M/C`;
- theoretical factorization + identifiability;
- symbolic-semantic memory instantiation;
- canonical 10-seed MiniGrid benchmark;
- oracle-stage intervention block;
- oracle activation/stage-delta diagnostics;
- qualitative oracle cases;
- stronger behavior-cloned learned baseline;
- semantic noise robustness block;
- MiniWorld runtime support;
- BabyAI portability benchmark;
- paper restructuring for main-track presentation;
- reproducibility and license notes.

Текущее состояние проекта:

- paper уже выглядит как серьезная benchmark/diagnostic submission;
- strongest part of the contribution now is the oracle-supported diagnostic attribution story;
- weakest remaining part is still the environment/generalization side, not the framework definition itself.
