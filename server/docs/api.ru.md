# Руководство по REST API InsightFace Server

**Языки:** [English](api.md) · [中文](api.zh-CN.md) · [日本語](api.ja.md) · [Deutsch](api.de.md) · [Español](api.es.md) · [Français](api.fr.md) · Русский · [Português](api.pt.md) · [한국어](api.ko.md)

Здесь описаны назначение, входные данные, работа сервера, успешный результат и ошибки каждого публичного API. Установка и первый поиск приведены в [руководстве пользователя](user-guide.ru.md), точная схема запущенной версии — в `/docs` и `/openapi.json`.

## Общие правила

- Базовый путь `/v1`, JSON в `snake_case`, изображения JPEG/PNG/WebP как multipart.
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

## Система

### `GET /v1/health`

**Назначение/ввод:** публичная readiness-проверка, без параметров и auth. **Результат:** проверяет запуск и SQLite quick_check; 200 с `status`, `auth_enabled`, `request_id`. **Ошибка:** `503 not_ready`.

### `GET /v1/system`

**Назначение/ввод:** безопасная диагностика, без параметров. **Результат:** 200 с OS/CPU/GPU, Driver, CUDA/cuDNN/ORT, Provider, моделью, DB, mount, счётчиками, поиском, безопасной конфигурацией и параллелизмом; без секретов, изображений и embeddings. **Ошибки:** 401, 503.

### `GET /v1/models`

**Назначение/ввод:** проверенные detector/recognizer, Provider и лицензия; без параметров. **Результат:** 200 `models`, `execution_provider`, `license`. **Ошибка:** 401.

## Stateless-операции

### `POST /v1/detect`

**Ввод:** multipart `image` обязателен, `max_faces` 1–100, необязательный `collection_id`. **Работа/результат:** объединяет разрешения, делает общую NMS и сортирует по площади; 200 `faces` с рамками/5 точками/score/quality и `processing_ms`. Нет лица — корректный пустой список. **Ошибки:** 400 старый min_score, 404 Collection, 413, 422 invalid_image, 503.

```bash
curl -sS "${BASE_URL}/v1/detect" -H "${AUTH_HEADER}" -F 'image=@group.jpg' -F 'max_faces=10'
```

### `POST /v1/compare`

**Ввод:** multipart `source`, `target`, необязательные `threshold` 0–1 и `collection_id`. **Результат:** выбирает одно лицо из каждого изображения; 200 `matched`, cosine `similarity`, фактический threshold, оба face и время. **Ошибки:** 404, 413, 422 invalid_image/face_not_found, 503.

### `POST /v1/embeddings`

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

**Ввод:** IDs; повторяемые images и те же review/embedding-поля, что при создании Person. **Результат:** 201 `faces`, `rejected_images`, возможен частичный успех. **Ошибки:** ошибки регистрации плюс 404 Person.

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces`

**Ввод:** IDs; query limit 1–100 и cursor. **Результат:** 200 metadata `faces`, `has_crop`, `next_cursor`, без embedding и bytes. **Ошибки:** 400 cursor, 404.

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}/image`

**Ввод:** три ID. **Результат:** если сохранено, 200 `image/jpeg`, crop 112×112, `Cache-Control:no-store`; request ID только в header. **Ошибки:** 401, 404 face/face_image_not_found.

### `DELETE /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}`

**Ввод:** три ID. **Результат:** удаляет embedding/crop/строку индекса, 204. **Ошибки:** 404, 503.

## Поиск

### `POST /v1/collections/{collection_id}/search`

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

**Назначение:** Опросить текущее состояние из headless-клиента. **Ввод:** ID Monitor. **Результат:** 200 со связью, фактическим FPS, задержкой, пропусками, текущими известными/неизвестными лицами, preview, переподключениями и безопасной ошибкой, без embeddings. **Ошибки:** 401, 404.

### `GET /v1/monitors/{monitor_id}/events`

**Назначение:** Получить временные события входа/выхода/ошибки/восстановления. **Ввод:** `limit` 1–1000 и последний непрозрачный `cursor`. **Результат:** 200 с `events`, `next_cursor`, `truncated`, `stream_reset`; перезапуск удаляет события. **Ошибки:** 400 `invalid_cursor`, 401, 404.

### `GET /v1/monitors/{monitor_id}/preview.mjpeg`

**Назначение:** Открыть необязательный сырой MJPEG preview, по умолчанию выключенный. **Ввод:** ID и обычный Bearer-заголовок; ключ нельзя помещать в URL. **Результат:** долгий `multipart/x-mixed-replace`, кодируемый только при зрителях; рамки клиент берёт из `/state`. **Ошибки:** 401, 404, 409 `preview_disabled`, 503.

## Повтор запросов

GET можно повторять. Перед повтором DELETE проверьте состояние. Если результат создания Person/Face неизвестен из-за сети, прочитайте ID до нового POST. Повторяйте только 429 и временные 503 с ограниченным exponential backoff и jitter; ошибки 4xx требуют исправления запроса.
