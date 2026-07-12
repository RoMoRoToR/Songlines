# commnet_baseline — learned-communication базлайн под Q/R/M/C-контрактом

CommNet-базлайн для мультиагентной секции
[`papers/qrmc_aaai27/`](../../papers/qrmc_aaai27/) и полной версии:
проверка структурного предсказания, что learned-канал с непрерывными
сообщениями даёт конкурентный success при **схлопнутой наблюдаемости
стадий** (вектор связи никогда не бывает информативно пуст → Q
насыщается, R/M сливаются; явного target-lock интерфейса нет).

- `eval_with_qrmc.py` — прогон обученной политики под тем же
  Q/R/M/C-логгером, что и символические архитектуры.
- Числа: success 0.667 vs 0.65 у symbolic peer при K=8; P(M\*|R\*)=1.00
  структурно. Детали — [`RESULTS.md`](RESULTS.md).
- PPO-версия и side-by-side сравнение REINFORCE/PPO —
  `../commnet_ppo_baseline/`.
