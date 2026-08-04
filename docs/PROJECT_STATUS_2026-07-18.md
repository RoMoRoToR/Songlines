# Songlines — гид по проекту и отчёт (2026-07-18)

Единый документ: (1) гид по структуре, (2) отчёт «что и где сделано»,
(3) кластер ИТМО: подключение и работа. Предыдущие статусы:
`PROJECT_STATUS_2026-05-09.md`, `MEMORY_IMPLEMENTATIONS_REPORT_2026-05-18.md`,
`ROADMAP_LLM_COLLECTIVE_MEMORY_2026-06-12.md`.

---

## 1. Статьи (papers/) — все компилируются, 0 ошибок

| Статья | EN | RU | Суть |
|---|---|---|---|
| `symbolic_memory` | 33 стр. | 30 | исходная одноагентная символическая память |
| `qrmc_measurement_framework` | **49** | 47 | **Paper 1**: Q/R/M/C протокол; VMAS-валидация с CI, ALFWorld+LLM, MAPPO-базлайн, cluster-robust статистика, θ/ε-робастность |
| `qrmc_aaai27` | 10+2 supp | 6 | сжатая AAAI-версия Paper 1 |
| `collective_semantic_memory` | **23** | 23 | **Paper 2**: CSM (merge/trust/staleness), place identity by meaning, категорная теория (gluing-теорема, фазовая диаграмма, Σ-alignment, коалгебры), **«Context is not memory»** (3 оси концепции) |
| `route_warp` | 7 | 8 | **Paper 3**: перенос маршрута vs места, закон разрыва, hazard-стратификация, W7–W9 |
| `semantic_warp` | 10 | 11 | warp по провенансу (W0–W4) |

Сборка: `cd papers/<dir> && pdflatex songlines_*.tex` (×2).

## 2. Карта код → статья → данные

| Код (experiments/) | Статья/раздел | Данные |
|---|---|---|
| `big_experiment/` (cadence, oracle R/M/C, robustness, ε, θ, effect sizes, assumption1) | P1 §3–4, App E/F | `tmp/cluster/*`, `tmp/big_experiment_*` |
| `vmas_portability/` | P1 App F.7 (полная валидация: Spearman −0.999/+0.995, CI искл. 0) | `tmp/cluster/vmas_full` |
| `alfworld_qrmc/` | P1 App «External substrate: ALFWorld» (3B: 21×R_fail/4×M_fail; 7B ждёт ноду) | `tmp/cluster/alfworld_*` |
| `mappo_baseline/` | P1 App H (MAPPO 0.667 succ, Q\*=1.00, P(M\|R)=1.00 — collapse структурен) | `tmp/cluster/mappo*` |
| `llm_collective/` (+`hf_backend.py`) | P1 LLM-блок (step-budget: 7B C-fail, 3B R-fail) | `tmp/cluster/textnav*`, `tmp/llm_steplimit_*` |
| `context_vs_structure/` (**3 оси концепции**) | **P2 App «Context is not memory»** | `tmp/cluster/ctx_vs_struct`, `persist_llm`, `growth_llm`, локальные `tmp/ctx_ollama_bigL` |
| `warp/` (route r0–r3, semantic identity w7–w9, alignment_defect, symbol_alignment) | P2 §6–7, P3 | `tmp/warp/*`, `tmp/cluster/route_*` |
| `collective_semantic_memory/` (CSM, phase diagram) | P2 §5, §7.3 | `tmp/cluster/csm_benchmark`, fig_phase_diagram |
| `place_identity_demo/` | P2 §6 (демо + coord-free matching) | `tmp/ctx_smoke*` |
| `commnet_*` | P1 App H | в папках экспериментов |

## 3. Отчёт: что сделано в июльской кампании

### Теория (Paper 2)
- **Gluing-теорема** (§7.2): коллективная память = colimit диаграммы локальных
  решёток; faithful ⟺ coherent; 2 вычислимых свидетеля.
- **Adjunction-defect** (§6): 0 в 30/30 (translation+SE(2)); fail-closed 30/30.
- **Фазовая диаграмма trust×cadence** (§7.3): существование colimit как
  функция (τ,K); heatmap-фигура.
- **Σ-alignment** (§7.4): инфоморфизмы (Barwise–Seligman); перевод по
  co-occurrence на якорях: acc 1.00, def_Σ≈шуму; naive-рукав ломается.
- **Коалгебры** (Appendix): котипы/деструкторы/Greatest Bisimulation
  (запрос Гусаровой), словарь water-примера.

### Рецензентские раунды (Paper 1) — все must-fix закрыты
cluster-robust effect sizes (81 ячейка), Y≠C\* (в т.ч. abstract),
θ+ε-робастность, cadence-робастность (сдвиг гаснет при изобилии),
seed-30 стабильность, LLM step-budget (C-локализация), semantic frame-free
E.3, номенклатура/дизайн-таблица 35 640. Открыт только Habitat/3D.

### Кластерная кампания (июль 17–18)
- **Wave 1**: 19 CPU-задач (реплики свипов + масштаб seeds ×2.5–5) — всё чисто.
- **Wave 1b/2**: VMAS-full (**оба наклона Claim 1, CI искл. 0**), MAPPO
  (SOTA-базлайн), single-agent 30×4×2, TextNav-HF (2 модели), ALFWorld-3B.
- **Concept-suite** (3 оси «Songlines сохраняет контекст, raw теряет»):
  - Ось 1: raw `count` 0.00 при **любом** L (механизм: «2»/«3» = сумма
    упоминаний); raw fact 0.00 при 800/2000 **при полном приёме**
    (prompt_eval_count=25k); graph 1.00 всюду при ~350Б.
  - Ось 2: replay 0→25КБ и падает; songlines 0.62–1.00 при ~0.3КБ.
  - Ось 3: graph **растёт** 0.40→1.00 с G (alignment запирается); raw — ни
    одного верного ответа (cross-frame identity вне возможностей окна).
- **Инциденты, пойманные до статьи** (все механизмы установлены): eager-L²
  OOM ×2, kwarg-rename TF5, NFS-tmpdir, ReadTimeout, 11GB-нода для 7B,
  отравленные кэши (311+351 файлов вычищено точечно). Правило: ни одна
  клетка не цитируется без установленного механизма отказа.

### Открытое
- alfworld-7B ждёт ноду nike (job в очереди, кэш чист).
- LLM-роадмап стадии 1–5 (PeerLLMAgent, NL-обмен, LLM-консолидация).
- Habitat/ProcTHOR 3D; Σ-alignment full (incommensurability, drift).

---

## 4. Кластер КТ (ИТМО) — подключение и работа

### Доступ (однократная настройка)
```bash
# 1) ключ без пароля для автоматизации
ssh-keygen -t ed25519 -f ~/.ssh/ctlab_auto -N ""
ssh-copy-id -i ~/.ssh/ctlab_auto.pub rzamotaev@ctlab.itmo.ru   # пароль 1 раз

# 2) ~/.ssh/config (уже настроено на этом маке):
Host ctlab
  HostName ctlab.itmo.ru
  User rzamotaev
  IdentityFile ~/.ssh/ctlab_auto
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
Host sphinx
  HostName sphinx
  User rzamotaev
  ProxyJump ctlab
  IdentityFile ~/.ssh/ctlab_auto
  ...

# 3) проверка:  ssh sphinx 'hostname; sinfo | head'
```
⚠️ Пароль от аккаунта светился в переписке — **смени**: `ssh ctlab 'passwd'`.

### Топология
- Вход: `ctlab.itmo.ru` (=horse) → `sphinx` (SLURM-контроллер; на нём НЕ считать).
- CPU (очередь `main`): griffin/kraken (128 ядер), orthrus-1/2, meduza-1/2 (1ТБ RAM).
- GPU (очередь `gpu`): **nike, kali — TITAN RTX 24ГБ×2** (для 7B+ моделей),
  midas (3080Ti 12ГБ), mars (2080Ti+TITAN), laplas/turing (1080Ti 11ГБ ×3).
- Диски: home `/nfs/home/<u>` (20ГБ, общий), **работать в
  `/mnt/tank/scratch/rzamotaev/`** (250ГБ, общий для всех нод).
- Wiki: https://ctlab.itmo.ru/wiki/ ; Telegram-анонсы кластера — см. wiki.

### Наше окружение на кластере
- Проект: `/mnt/tank/scratch/rzamotaev/songlines/` (rsync-зеркало).
- Python: `/mnt/tank/scratch/rzamotaev/miniconda3/bin/python` (CPU-джобы);
  GPU-env: `.../miniconda3/envs/gpu/bin/python` (**torch cu124** — драйвер нод
  = CUDA 12.4, base-torch cu130 не работает!).
- HF-модели: `HF_HOME=/mnt/tank/scratch/rzamotaev/hf_home` (Qwen2.5-3B/7B
  застейджены; на нодах `HF_HUB_OFFLINE=1`).
- ALFWorld: `ALFWORLD_DATA=.../alfworld_data`, конфиг `.../alfworld_config.yaml`.

### Скрипты (cluster/)
```bash
bash cluster/rsync_up.sh          # пуш проекта (исключая tmp/docs/venv; НЕ трёт logs)
ssh sphinx 'bash .../cluster/setup_env.sh'        # CPU-deps (miniconda)
ssh sphinx 'bash .../cluster/setup_env_wave2.sh'  # torch/transformers/alfworld/veca+веса
ssh sphinx 'bash .../cluster/setup_env_gpu.sh'    # gpu-env (cu124)
ssh sphinx 'bash .../cluster/submit_all.sh'       # Wave 1 (19 CPU-задач)
ssh sphinx 'bash .../cluster/submit_wave2.sh'     # Wave 1b+2 (CPU+GPU)
bash cluster/collect_results.sh   # результаты → tmp/cluster_results/
# мониторинг: ssh sphinx 'squeue -u rzamotaev'
```

### Грабли (проверено на себе)
1. **GPU-драйвер = CUDA 12.4** → только torch cu124 (gpu-env), не cu130.
2. **7B fp16 ≈ 15ГБ** → только nike/kali (24ГБ): `#SBATCH -w nike`.
   На 11ГБ нодах — OOM на каждом вызове, ошибки кэшируются!
3. **sm75 без flash-attention**: eager-attention даёт L²-матрицу →
   OOM при >5к токенов. В `hf_backend.py` стоят каскад logits_to_keep,
   sdpa/efficient-kernel и токен-гард `HF_CTX_TOKEN_CAP` (деф. 24k) —
   длинноконтекстные прогоны при необходимости делать локально через Ollama.
4. **`export TMPDIR=/tmp`** в sbatch — textworld/rmtree ломается на NFS (.nfsXXXX).
5. **Кэш LLM-ответов хранит и ошибки** — перед ре-раном чистить:
   `grep -l "LLM_ERROR" .../*.cache/*.txt | xargs rm`; после прогона
   проверять `grep -c LLM_ERROR` по ВСЕМ файлам, не по одному.
6. QOS-лимит ~64 CPU на юзера — очередь дренится сама.
7. rsync с `--delete`: `cluster/logs`, `cluster/jobs` и `tmp/` в исключениях.
