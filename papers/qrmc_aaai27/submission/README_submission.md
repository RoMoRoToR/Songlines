# Пакет подачи AAAI-27 — Q/R/M/C

Дедлайны: **abstract 21 июля**, **paper 28 июля**, **supplements 31 июля 2026**.

## Что где

| Файл | Куда загружать |
|---|---|
| `../songlines_qrmc_aaai27.pdf` | Main paper (8 стр.: 6 стр. контента + refs + reproducibility checklist) |
| `abstract.txt` | Поле Abstract в форме подачи (плоский текст, без LaTeX) |
| `TechnicalSupplement_QRMC.pdf` | Supplementary material (полная рукопись, 47 стр.) |
| `code_supplement.zip` | Supplementary material (код + данные, 1.2 МБ / 255 файлов) |

## Чек перед загрузкой

- [x] Анонимность: PDF-метаданные пустые, «Anonymous submission» на титулах,
      в архиве нет имён/e-mail/абсолютных путей (проверено sweep'ом).
- [x] Лимит: технический контент ≤ 7 стр. (Conclusion на p6, References с p6).
- [x] Checklist заполнен (31 ответ) и вкомпилирован в main paper.
- [ ] **Автор**: подтвердить лицензию кода в
      `ReproducibilityChecklist_qrmc_aaai27.tex` (~строка 225, TODO) —
      после подтверждения снять `% DRAFT`-комментарии.
- [ ] **Автор**: решить, прикладывать ли companion-препринты
      (semantic_warp / route_warp / CSM PDF из `papers/`) вторым
      supplementary-файлом — в тексте есть ссылки «companion manuscript».
- [ ] Регистрация подачи и keywords — до 21 июля.

## Как пересобрать пакет

Код-архив: staging в `tmp/aaai27_code_supplement/` (см. его README.md с
картой «секция статьи → код → данные»); при изменениях кода пересоздать:
`cd tmp && zip -rq ../papers/qrmc_aaai27/submission/code_supplement.zip aaai27_code_supplement`.
Technical Supplement — копия `papers/qrmc_measurement_framework/*.pdf`.
