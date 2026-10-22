# Руководство по REST API InsightFace Server

**Языки:** [English](api.md) · [中文](api.zh-CN.md) · [日本語](api.ja.md) · [Deutsch](api.de.md) · [Español](api.es.md) · [Français](api.fr.md) · Русский · [Português](api.pt.md) · [한국어](api.ko.md)

Здесь описаны назначение, входные данные, работа сервера, успешный результат и ошибки каждого публичного API. Установка и первый поиск приведены в [руководстве пользователя](user-guide.ru.md), точная схема запущенной версии — в `/docs` и `/openapi.json`.


Модели идентифицируются по `model_id`; отдельное поле `model_version` больше не возвращается. Существующие Collection сохраняют `embedding_contract_id`.

Для проверки живого лица см. [настройку, установку модели и результаты](#необязательный-addon-проверки-живого-лица). В разделах операций также описано её влияние.

## Общие правила

- Базовый путь `/v1`, JSON в `snake_case`, изображения JPEG/PNG/WebP/BMP как multipart.
- Поставляемый Compose отключает авторизацию для изолированной оценки. Если она включена, всё кроме health требует `Authorization: Bearer <api_key>`; если выключена, заголовок нужно полностью убрать.
- В каждом ответе есть `x-request-id`, а JSON повторяет его как `request_id`.
- confidence/quality/threshold находятся в `0..1`. Similarity — не вероятность, а исходный cosine `[-1,1]`; порог по умолчанию `0.4`, совпадение при `similarity >= threshold`.
- Cursor непрозрачен и возвращается без изменений только в тот же путь, Collection, Person и фильтр.
- Частые статусы: 400 ввод, 401 auth, 404 нет ресурса, 409 конфликт, 413 размер, 422 изображение/лицо, 429 лимит, 503 timeout/модель/индекс.

```bash
BASE_URL=http://127.0.0.1:18097
AUTH_HEADER="Authorization: Bearer ${INSIGHTFACE_API_KEY}"
curl -fsS "${BASE_URL}/v1/health"
```

## Необязательный addon проверки живого лица

В `server/config/server.toml` проверка живого лица по умолчанию отключена: `inference.addons` и `addons.auto_download` равны `[]`. Старые настройки без этих ключей также остаются отключёнными.

**Включение через командную строку, в том числе до первого запуска Server:**

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license --enable-liveness
```

`--enable-liveness` сначала проверяет возможность обновить существующую конфигурацию. Затем устанавливает и проверяет базовый пакет, настроенные для установки дополнения и модель проверки живого лица, после чего добавляет `liveness` в `inference.addons` и `addons.auto_download`, сохраняя остальные элементы, комментарии и настройки. Проверенный кэш используется повторно, но настройка включения сохраняется и при наличии моделей в кэше. Сбой загрузки не изменяет конфигурацию; ошибка её сохранения приводит к явному сообщению и ненулевому коду завершения. Годные файлы в кэше можно использовать при повторной попытке.

Оба сервиса Compose монтируют весь существующий каталог `server/config` с записью в `/etc/insightface`, сохраняя `create_host_path: false`. Поэтому установщик атомарно обновляет конфигурацию хоста без работающего Server. Каталог и `server.toml` должны существовать.

Запускать Server заранее не нужно. При новой установке следующий `up -d` включит проверку; если Server уже работает, выполните `docker compose -f server/deploy/compose.cpu.yml restart server`. Одна команда `up -d` не перечитывает сохранённые настройки. Для CUDA используйте `compose.cuda12.yml`.

Без `--enable-liveness` команда `models install` сохраняет прежнее поведение и не записывает конфигурацию; по умолчанию проверка выключена. `models addons install liveness` только загружает и проверяет дополнение, но не включает его. Также можно включить проверку через **Система → Проверка живого лица**, как описано ниже.

В разделе **Система → Проверка живого лица** загрузите модель и включите её для следующего запуска. После проверки SHA-256 `liveness` добавляется в оба списка; остальные элементы, комментарии и настройки сохраняются. Проверенный файл используется повторно. Для применения **перезапустите Server вручную**. Ошибки допускают повтор; неудачная загрузка не включает проверку.

[Монтирование и права для загрузки через Web](user-guide.ru.md#монтирование-и-права-для-загрузки-через-web).

**Дополнительно: ручная настройка.** Эти параметры заменяют флаг включения или действие Web; установите модель перед перезапуском.

```toml
[inference]
addons = ["liveness"]
liveness_mode = "normal"
liveness_threshold = 0.8
liveness_compare_scope = "both"
liveness_on_registration = false

[addons]
auto_download = ["liveness"]
```

### Установка модели и запуск

`inference.addons` управляет выполнением, а `addons.auto_download` — дополнительной загрузкой при установке базового пакета. С `["liveness"]` addon устанавливается и для уже кешированного пакета. При запуске Server загрузки нет. Установщик и Server читают один файл.

Выполняйте команды из корня репозитория с существующим `server/config/server.toml`. Поставляемый Compose запускает установщик как root, создаёт каталог моделей при необходимости и монтирует `/models` с записью; загрузка дополнения создаёт `addons`. Ручная подготовка UID/GID и прав не нужна. См. [начальную настройку в руководстве пользователя](user-guide.ru.md).

```bash
docker compose -f server/deploy/compose.cpu.yml run --rm models addons install liveness
docker compose -f server/deploy/compose.cpu.yml run --rm models addons verify liveness
docker compose -f server/deploy/compose.cpu.yml up -d --force-recreate
```

Отсутствующая включённая модель останавливает запуск с `addon_model_missing`, неверная — с `addon_model_invalid`. Addon не отключается незаметно.

### Результаты проверки живого лица

| Результат | `status` | `is_live` | `live_score` |
| --- | --- | --- | --- |
| Живое лицо | `ok` | `true` | `[0, 1]` |
| Подделка | `ok` | `false` | `[0, 1]` |
| Неподходящий вход | `input_rejected` | `null` | `null` |

Только недостаточная область исходного изображения вокруг выровненного лица приводит к `input_rejected`. Такой результат дополнительно содержит `liveness.reason` — пояснение для пользователя; у живого лица и подделки поля `reason` нет. FaceAnalysis и API всегда возвращают этот текст на английском; перевод отображается только в Web-интерфейсе. В логике программы используйте `status` и `is_live`, а не текст `reason`. В ранее сохранённых результатах `reason` может отсутствовать; в этом случае клиент может показать общее сообщение об отклонении входного изображения.

```json
{
  "status": "input_rejected",
  "is_live": null,
  "live_score": null,
  "reason": "Insufficient image area around the face for liveness detection. Move the face toward the center, step back from the camera, or use a less tightly cropped image."
}
```

`normal` распознаёт только лица, прошедшие проверку; `observe` сохраняет результат и продолжает распознавание. Без проверки поле `liveness` отсутствует. Три основных поля — `status`, `is_live` и `live_score`: живое лицо и подделка получают `status: ok`, логическое значение и оценку; отклонённый вход — `status: input_rejected` и два значения `null`.

`/v1/detect` возвращает HTTP 200 и для отрицательных результатов. В режиме `normal` embeddings, сравнение и поиск возвращают HTTP 422 `liveness_fake` или `liveness_input_rejected` с `error.details.liveness`; сравнение добавляет `details.side`. Сбой инференса возвращает HTTP 503 `liveness_unavailable`. Сбои выполнения прерывают операцию и в `normal`, и в `observe`; они не преобразуются в `input_rejected`.

При создании Person и добавлении FaceSample проверка по умолчанию пропускается: `[inference].liveness_on_registration=false` не запускает модель и не добавляет `liveness` к новым образцам. При `true` и включённом addon применяется режим `normal`/`observe`; отклонённые изображения содержат `reason` и `liveness`. Проверка качества по `review_mode` и валидация внешних эмбеддингов сохраняются. `review_mode=off` и `external_trusted` не обходят включённую проверку при регистрации. Запрос не может переопределить эту настройку запуска. Ранее сохранённые результаты остаются доступны.

RTSP отличает `liveness_blocked` от `unknown` и использует счётчик `liveness_blocked_faces`. Заблокированные лица не создают события входа человека/неизвестного и сбрасывают подтверждение. При сбое инференса ранее показанные личности очищаются.

`liveness_compare_scope` выбирает стороны `/v1/compare`: `both` (по умолчанию) проверяет обе, `source` — исходное изображение, `target` — целевое. Лицо признаётся живым при `live_score >= liveness_threshold`.

`models addons install liveness` сохраняет опубликованную модель в `/models/addons/liveness.onnx`; на хосте Compose это `server/.models/addons/liveness.onnx`. Ошибки запуска: `addon_model_missing` и `addon_model_invalid`. `/v1/models` и `/v1/system` возвращают активные дополнения в `addons`.

[Настройка и рабочие процессы](user-guide.ru.md#необязательный-addon-проверки-живого-лица).

## Система

### `GET /v1/health`

**Назначение/ввод:** публичная readiness-проверка, без параметров и auth. **Результат:** проверяет запуск и SQLite quick_check; 200 с `status`, `auth_enabled`, `request_id`. **Ошибка:** `503 not_ready`.

### `GET /v1/system`

**Назначение/ввод:** безопасная диагностика, без параметров. **Результат:** 200 с OS/CPU/GPU, Driver, CUDA/cuDNN/ORT, Provider, моделью, DB, mount, счётчиками, поиском, безопасной конфигурацией и параллелизмом; без секретов, изображений и embeddings. **Ошибки:** 401, 503.

### `GET /v1/models`

`addons` сообщает активные дополнения отдельно от базовой модели. Проверьте `liveness` и действующие настройки в `safe_config` ответа системы. Эти интерфейсы только читают данные и не устанавливают модели.

**Назначение/ввод:** проверенные detector/recognizer, Provider и лицензия; без параметров. **Результат:** 200 `models`, `execution_provider`, `license`. **Ошибка:** 401.

Базовые пакеты `raccoon_s` и `raccoon_l` поддерживают CPU и CUDA и устанавливаются инструментом моделей до запуска. Этот API перечисляет используемые компоненты, а не каталог загрузок. Действие Web ниже управляет только проверкой живого лица. Collection привязана к модели распознавания и предобработке: смена пакета не преобразует существующие векторы и может вернуть `409 collection_model_mismatch`. Включение только проверки живого лица не меняет этот контракт.

### `GET /v1/addons/liveness`

**Назначение:** Прочитать состояние установки и настройки следующего запуска без загрузки или изменений. Это API управления, а не отдельный API инференса проверки живого лица.

**Результат:** HTTP 200. `enabled` показывает состояние работающего процесса. `installed` означает, что файл прошёл проверку опубликованного SHA-256, а не что проверка включена. `configured_enabled` читает выбор для следующего запуска из текущего файла; `restart_required` означает отличие от `enabled`. До перезапуска `safe_config` в `/v1/system` продолжает описывать работающий процесс.

`state` принимает `idle` (проверенного файла нет), `downloading` (идёт подготовка), `ready` (проверенный файл доступен) или `error` (ошибка подготовки, файла или настроек). Само по себе `ready` не означает сохранения включённой настройки или завершения перезапуска.

`can_enable` показывает доступность подготовки через Web. Если она недоступна, `unavailable_code` содержит стабильный код причины, а `unavailable_reason` — пояснение; иначе оба равны `null`. `error` равен `null` или содержит `code` и `message`. `model_path` — локальный путь модели; `config_file` — выбранный путь TOML или `null`. Ответ также содержит `request_id`.

Значения `unavailable_code`: `config_file_missing` (файл настроек не выбран), `config_file_not_regular` (не обычный файл), `config_file_mount` (файл смонтирован отдельно), `config_not_writable` (нет прав записи настроек), `addon_directory_not_writable` (нет прав записи в каталог дополнения), `addon_config_invalid` (неверные настройки), `addon_model_invalid` (неверная модель), `server_stopping` (сервер останавливается).

```json
{
  "enabled": false,
  "installed": true,
  "configured_enabled": true,
  "restart_required": true,
  "can_enable": true,
  "unavailable_code": null,
  "unavailable_reason": null,
  "state": "ready",
  "error": null,
  "model_path": "/models/addons/liveness.onnx",
  "config_file": "/etc/insightface/server.toml",
  "request_id": "3ed21e89-4595-4eed-a699-1df42ca62032"
}
```

### `POST /v1/addons/liveness/enable`

**Назначение:** Загрузить модель и сохранить её включение при следующем запуске. Отправить пустой JSON-объект `{}` с `Content-Type: application/json`. URL модели и другие параметры не принимаются.

```bash
curl -sS "${BASE_URL}/v1/addons/liveness" -H "${AUTH_HEADER}"
curl -sS "${BASE_URL}/v1/addons/liveness/enable" -H "${AUTH_HEADER}" \
  -H 'Content-Type: application/json' -d '{}'
```

**Результат:** HTTP 202 возвращает те же поля, что GET, и подтверждает принятие задания, а не включение проверки. Опрашивайте `GET /v1/addons/liveness` до завершения. Повторные запросы используют выполняемое задание; закрытие браузера его не отменяет.

Только после загрузки и проверки SHA-256 в `[inference].addons` и `[addons].auto_download` файла `config_file` добавляется `liveness`. Остальные значения и комментарии сохраняются, проверенный файл используется повторно. При `installed=true`, `configured_enabled=true` и `restart_required=true` перезапустите сервер вручную. Новый процесс вернёт `enabled=true` и `restart_required=false`. Горячей перезагрузки и API переключения базового пакета нет.

**Ошибки:** Ошибки запроса имеют обычный формат: `400 invalid_addon_request`, если тело отличается от `{}`; `401 unauthorized` при ошибке аутентификации; `403 origin_not_allowed` для запрещённого источника браузера; `409 addon_management_unavailable` при неподходящих путях, правах или настройках; `415 json_required`, если тип содержимого не JSON. Браузер должен использовать тот же источник, что сервер, либо явно разрешённый CORS-источник.

Принятое задание может завершиться ошибкой позднее: GET по-прежнему возвращает HTTP 200 с `state=error` и `error.code`. `addon_download_failed` не меняет настройки: проверьте сеть или прокси сервера. При `addon_config_save_failed` исправьте настройки или права каталога; проверенный файл можно использовать повторно. `addon_config_invalid` означает неверный TOML на диске. При `addon_model_invalid` замените или удалите повреждённый файл; он не перезаписывается незаметно. `addon_job_in_progress` означает подготовку другим процессом: подождите и обновите состояние. Повторяйте POST после устранения причины.

## Stateless-операции

### `POST /v1/detect`

Каждое проверенное лицо содержит `liveness.status`, `liveness.is_live` и `liveness.live_score`. Результаты подделки и `input_rejected` также возвращают HTTP 200 без извлечения признаков распознавания. `input_rejected` означает недостаточную область изображения вокруг лица; `liveness.reason` объясняет, как скорректировать изображение. Если `liveness` отсутствует, проверки не было.

**Ввод:** multipart `image` обязателен, `max_faces` 1–100, необязательный `collection_id`. **Работа/результат:** объединяет разрешения, делает общую NMS и сортирует по площади; 200 `faces` с рамками/5 точками/score/quality и `processing_ms`. Нет лица — корректный пустой список. **Ошибки:** 400 старый min_score, 404 Collection, 413, 422 invalid_image, 503.

```bash
curl -sS "${BASE_URL}/v1/detect" -H "${AUTH_HEADER}" -F 'image=@group.jpg' -F 'max_faces=10'
```

### `POST /v1/compare`

`liveness_compare_scope` (`both`, `source`, `target`) выбирает стороны для проверки перед распознаванием. В `normal` отказ возвращает HTTP 422 `liveness_fake` / `liveness_input_rejected`, `error.details.liveness` и `error.details.side`, без сходства. `observe` продолжает сравнение и возвращает результаты у проверенных лиц.

**Ввод:** multipart `source`, `target`, необязательные `threshold` 0–1 и `collection_id`. **Результат:** выбирает одно лицо из каждого изображения; 200 `matched`, cosine `similarity`, фактический threshold, оба face и время. **Ошибки:** 404, 413, 422 invalid_image/face_not_found, 503.

### `POST /v1/embeddings`

При включённой проверке в `normal` подделка или неподходящий ввод возвращает HTTP 422 `liveness_fake` / `liveness_input_rejected` и `error.details.liveness`; эмбеддинг не извлекается. `observe` возвращает эмбеддинг и результат проверки лица.

**Ввод:** multipart `image`, необязательный `collection_id`. **Результат:** 200 с выбранным face, L2-нормированным embedding, моделью и временем. Для обычной регистрации не нужен; в лог не записывается. **Ошибки:** 400 старый face_selection, 404, 413, 422, 503.

## Collections

### `POST /v1/collections`

**Ввод:** JSON `id`, `name`; необязательные description, threshold (0.4), metadata, save_face_crops, `detection`, `search` с profile/capacity/max_faces_per_person/load_policy. **Работа/результат:** фиксирует модель, предобработку и поисковый контракт; 201 с полной `collection`. **Ошибки:** 400 конфигурация, 409 exists, 503 индекс.

```bash
curl -sS "${BASE_URL}/v1/collections" -H "${AUTH_HEADER}" -H 'Content-Type: application/json' -d '{"id":"employees","name":"Employees","threshold":0.4}'
```

### `GET /v1/collections`

**Ввод:** query `limit` 1–100 (50), необязательный cursor. **Результат:** 200 `collections`, nullable `next_cursor`. **Ошибки:** 400 invalid_cursor, 401.

### `GET /v1/collections/{collection_id}`

**Ввод:** ID Collection в пути. **Результат:** 200 `collection`, количества Person/Face и `embedding_contract_id`. **Ошибка:** 404.

### `PATCH /v1/collections/{collection_id}`

**Ввод:** ID; JSON name/description/threshold/metadata/save_face_crops, capacity/max/load поиска и detection. Null, неизвестные поля, модель и search profile менять нельзя. **Результат:** 200 полная Collection; detection действует со следующего запроса. **Ошибки:** 400, 404, 409, 503.

### `DELETE /v1/collections/{collection_id}`

**Ввод:** ID; query `force=false`, true для непустой Collection. **Результат:** 204 без тела. **Ошибки:** 404, 409 collection_not_empty, 503.

## Person и FaceSample

### `POST /v1/collections/{collection_id}/persons`

Создание Person и добавление FaceSamples по умолчанию пропускают проверку (`liveness_on_registration=false`). После включения `normal` отклоняет подделки и неподходящие изображения, а `observe` сохраняет результат и продолжает. Проверка качества следует выбранному `review_mode`. Список отказов отдельно показывает фактический `reason` и результат проверки живого лица.

**Ввод:** Collection; multipart повторяемые `images`, необязательные id/name/external_id, metadata как JSON-строка, `review_mode=off|standard|strict`, `embedding_mode=server|external_trusted`; внешнему режиму нужны векторы и contract ID. **Работа/результат:** проверяет каждое изображение; 201 `person`, принятые `faces`, `rejected_images`; частичный успех разрешён, все отклонены — 422 без Person. **Ошибки:** 400, 404, 409 ID/контракт/ёмкость, 413, 422, 503.

```bash
curl -sS "${BASE_URL}/v1/collections/employees/persons" -H "${AUTH_HEADER}" -F 'id=alice' -F 'review_mode=off' -F 'images=@alice.jpg'
```

### `GET /v1/collections/{collection_id}/persons`

**Ввод:** Collection; query limit/cursor/`search` по ID, имени или external ID. **Результат:** 200 `persons`, `next_cursor`. **Ошибки:** 400 cursor, 404.

### `GET /v1/collections/{collection_id}/persons/{person_id}`

**Ввод:** ID Collection и Person. **Результат:** 200 `person` с face_count. **Ошибка:** 404.

### `PATCH /v1/collections/{collection_id}/persons/{person_id}`

**Ввод:** IDs; JSON name/external_id/metadata-объект. **Результат:** 200 обновлённый Person. **Ошибки:** 400, 404, 409 external_id_exists.

### `DELETE /v1/collections/{collection_id}/persons/{person_id}`

**Ввод:** IDs. **Результат:** удаляет Person, FaceSamples, embeddings и crops, синхронизирует индекс, 204. **Ошибки:** 404, 503.

### `POST /v1/collections/{collection_id}/persons/{person_id}/faces`

Создание Person и добавление FaceSamples по умолчанию пропускают проверку (`liveness_on_registration=false`). После включения `normal` отклоняет подделки и неподходящие изображения, а `observe` сохраняет результат и продолжает. Проверка качества следует выбранному `review_mode`. Список отказов отдельно показывает фактический `reason` и результат проверки живого лица.

**Ввод:** IDs; повторяемые images и те же review/embedding-поля, что при создании Person. **Результат:** 201 `faces`, `rejected_images`, возможен частичный успех. **Ошибки:** ошибки регистрации плюс 404 Person.

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces`

**Ввод:** IDs; query limit 1–100 и cursor. **Результат:** 200 metadata `faces`, `has_crop`, `next_cursor`, без embedding и bytes. **Ошибки:** 400 cursor, 404.

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}/image`

**Ввод:** три ID. **Результат:** если сохранено, 200 `image/jpeg`, crop 112×112, `Cache-Control:no-store`; request ID только в header. **Ошибки:** 401, 404 face/face_image_not_found.

### `DELETE /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}`

**Ввод:** три ID. **Результат:** удаляет embedding/crop/строку индекса, 204. **Ошибки:** 404, 503.

## Поиск

### `POST /v1/collections/{collection_id}/search`

При включённой проверке в `normal` подделка или неподходящий запрос возвращает HTTP 422 `liveness_fake` / `liveness_input_rejected` и `error.details.liveness`; поиск не выполняется. Это отличается от успешного пустого списка совпадений. `observe` продолжает поиск и возвращает результат у лица запроса.

**Ввод:** Collection; multipart `image`, `limit` 1–100 (5), необязательный threshold или значение Collection. **Работа/результат:** сравнивает выбранное лицо со всеми samples, берёт максимум по Person; 200 `searched_face`, отсортированные `matches`, threshold и время. Нет совпадений — пустой список. **Ошибки:** 404, 409 модель, 413, 422 изображение/лицо, 503 индекс/timeout.

```bash
curl -sS "${BASE_URL}/v1/collections/employees/search" -H "${AUTH_HEADER}" -F 'image=@query.jpg' -F 'limit=5'
```

## RTSP-мониторы

Конфигурация Monitor сохраняется в SQLite, а включённая задача восстанавливается после перезапуска сервера. Видеокадры не сохраняются; события находятся только в ограниченном кольцевом буфере памяти.

### `POST /v1/monitors`

**Назначение:** Создать постоянный RTSP Monitor. **Ввод:** JSON с ID, именем, `source`, Collection, `inference_fps` (2), необязательным порогом, буфером/политикой событий и `preview_enabled` (false). **Результат:** 201 с очищенным `monitor`; реквизиты шифруются при хранении. **Ошибки:** 400, 404, 409, 429.

### `GET /v1/monitors`

**Назначение:** Постранично перечислить Monitors. **Ввод:** `limit` 1–100 (50) и неизменённый непрозрачный `cursor` прошлого ответа. **Результат:** 200 с `monitors` и `next_cursor`, без реквизитов доступа. **Ошибки:** 400 `invalid_cursor`, 401.

### `GET /v1/monitors/{monitor_id}`

**Назначение:** Прочитать конфигурацию и сводку выполнения Monitor. **Ввод:** `monitor_id` в пути. **Результат:** 200 с политикой событий, очищенным источником, preview и состоянием. **Ошибки:** 401, 404 `monitor_not_found`.

### `PATCH /v1/monitors/{monitor_id}`

**Назначение:** Частично изменить поля кроме ID и включать/выключать через `enabled`. **Ввод:** частичный JSON; `event_policy` также частичная, null-порог наследует Collection. **Результат:** 200 с полным Monitor; источник/Collection/частота/политика перезапускают задачу. **Ошибки:** 400, 404, 429.

### `DELETE /v1/monitors/{monitor_id}`

**Назначение:** Навсегда удалить Monitor. **Ввод:** `monitor_id` в пути. **Результат:** останавливает декодер, инференс и RTSP, удаляет события из памяти и возвращает 204; Collection остаётся. **Ошибки:** 401, 404.

### `GET /v1/monitors/{monitor_id}/state`

При включённой проверке в `normal` заблокированные лица имеют `status: liveness_blocked` и отдельный результат. Они учитываются в `liveness_blocked_faces`, а не в `unknown_faces`, и не создают события входа. `observe` продолжает распознавание. Неподходящий ввод и подделка отображаются раздельно.

**Назначение:** Опросить текущее состояние из headless-клиента. **Ввод:** ID Monitor. **Результат:** 200 со связью, фактическим FPS, задержкой, пропусками, текущими известными/неизвестными лицами, preview, переподключениями и безопасной ошибкой, без embeddings. **Ошибки:** 401, 404.

### `GET /v1/monitors/{monitor_id}/events`

**Назначение:** Получить временные события входа/выхода/ошибки/восстановления. **Ввод:** `limit` 1–1000 и последний непрозрачный `cursor`. **Результат:** 200 с `events`, `next_cursor`, `truncated`, `stream_reset`; перезапуск удаляет события. **Ошибки:** 400 `invalid_cursor`, 401, 404.

### `GET /v1/monitors/{monitor_id}/preview.mjpeg`

**Назначение:** Открыть необязательный сырой MJPEG preview, по умолчанию выключенный. **Ввод:** ID и обычный Bearer-заголовок; ключ нельзя помещать в URL. **Результат:** долгий `multipart/x-mixed-replace`, кодируемый только при зрителях; рамки клиент берёт из `/state`. **Ошибки:** 401, 404, 409 `preview_disabled`, 503.

## Повтор запросов

GET можно повторять. Перед повтором DELETE проверьте состояние. Если результат создания Person/Face неизвестен из-за сети, прочитайте ID до нового POST. Повторяйте только 429 и временные 503 с ограниченным exponential backoff и jitter; ошибки 4xx требуют исправления запроса.
