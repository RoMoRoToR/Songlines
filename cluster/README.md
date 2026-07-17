# Развёртывание на кластере КТ (ITMO, SLURM)

Wave 1 — весь CPU-пакет экспериментов (19 задач, numpy/scipy/pandas/matplotlib),
масштабированные seeds. Wave 2 (GPU/LLM: ALFWorld) — отдельно.

## Шаг 0 (один раз): ключевой доступ + смена пароля
```bash
# на Маке: беспарольный ключ для автоматизации
ssh-keygen -t ed25519 -f ~/.ssh/ctlab_auto -N ""
ssh-copy-id -i ~/.ssh/ctlab_auto.pub rzamotaev@ctlab.itmo.ru   # спросит пароль (1 раз)
# добавить в ~/.ssh/config к Host ctlab:  IdentityFile ~/.ssh/ctlab_auto
# ВАЖНО: пароль был в переписке — смени его:  ssh ctlab 'passwd'
```
Home (`/nfs/home`) общий для всех нод — ключ работает и на sphinx.

## Шаг 1: закинуть проект (с Мака)
```bash
bash cluster/rsync_up.sh
```

## Шаг 2: окружение (на sphinx, один раз)
```bash
ssh sphinx 'bash /mnt/tank/scratch/rzamotaev/songlines/cluster/setup_env.sh'
```

## Шаг 3: запустить все эксперименты
```bash
ssh sphinx 'bash /mnt/tank/scratch/rzamotaev/songlines/cluster/submit_all.sh'
# мониторинг: ssh sphinx 'squeue -u rzamotaev'
```

## Шаг 4: забрать результаты (с Мака)
```bash
bash cluster/collect_results.sh   # → tmp/cluster_results/
```

## Состав Wave 1 (queue main, CPU)
| job | что | масштаб |
|---|---|---|
| cadence_full | главный 35,640-run свип (реплика) | 40 seeds, 32 cpu |
| extra_K, scale_N12 | расширенные каденции, N=12 | как в статье |
| cadence_robustness | scarcity/borderline/abundance | **100 seeds** (было 20) |
| eps_sensitivity | ε-порог | **50 seeds** (было 15) |
| oracle_interventions | oracle R/M/C | 15 seeds |
| assumption1_stress | стресс Assumption 1 | 4×10^5 эп. |
| route_r0–r3 | route-warp cliff/rupture/hazard | r1/r3 **50 seeds** |
| w7, w8 | semantic identity + full stack | w8 **50 seeds** |
| alignment_defect | adjunction defect | как в статье |
| semantic_cadence | frame-free Q/R/M/C | **40 seeds** (было 8) |
| symbol_alignment | Σ-alignment | **100 seeds** (было 20) |
| coord_free_matching | coordinate-free matching | **100 seeds** (было 20) |
| phase_diagram | trust×cadence + fig | **50 seeds** (было 12) |
| csm_benchmark | CSM vs peer | как в статье |

Результаты пишутся в `tmp/cluster/` на scratch; логи — `cluster/logs/`.

## Wave 2 (GPU, потом)
ALFWorld + LLM: `-p gpu -w kali --gres=gpu:1`; предварительно застейджить
данные ALFWorld и веса модели в `/mnt/tank/scratch/rzamotaev/` c sphinx.
