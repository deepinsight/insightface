# Guía de uso de la API REST de InsightFace Server

**Idiomas:** [English](api.md) · [中文](api.zh-CN.md) · [日本語](api.ja.md) · [Deutsch](api.de.md) · Español · [Français](api.fr.md) · [Русский](api.ru.md) · [Português](api.pt.md) · [한국어](api.ko.md)

Esta guía explica el propósito, la entrada, el trabajo del servidor, el resultado y los errores de todas las rutas públicas. Consulte instalación y primer uso en la [guía de usuario](user-guide.es.md) y el esquema exacto en ejecución en `/docs` o `/openapi.json`.

## Reglas comunes

- Ruta base `/v1`, JSON `snake_case`, imágenes JPEG/PNG/WebP como multipart.
- El Compose incluido desactiva auth para evaluación aislada. Si se activa, todo salvo health requiere `Authorization: Bearer <api_key>`; si está desactivada, omita el header, no envíe uno vacío.
- Toda respuesta tiene `x-request-id`; JSON repite `request_id`.
- confidence/quality/threshold usan `0..1`. Similarity no es probabilidad: es coseno `[-1,1]`; valor predeterminado `0.4`, coincide si `similarity >= threshold`.
- Un cursor es opaco y solo se reutiliza sin cambios con la misma ruta, Collection, Person y filtro.
- Estados habituales: 400 entrada, 401 auth, 404 ausente, 409 conflicto, 413 tamaño, 422 imagen/rostro, 429 límite, 503 timeout/modelo/índice.

```bash
BASE_URL=http://127.0.0.1:18097
AUTH_HEADER="Authorization: Bearer ${INSIGHTFACE_API_KEY}"
curl -fsS "${BASE_URL}/v1/health"
```

## Sistema

### `GET /v1/health`

**Uso/entrada:** readiness pública, sin parámetros ni autenticación. **Resultado:** comprueba inicio y SQLite quick_check; 200 con `status`, `auth_enabled`, `request_id`. **Error:** `503 not_ready`.

### `GET /v1/system`

**Uso/entrada:** diagnóstico seguro, sin parámetros. **Resultado:** 200 con OS/CPU/GPU, Driver, CUDA/cuDNN/ORT, Provider, modelo, DB, montajes, totales, búsqueda, configuración segura y concurrencia; nunca secretos, imágenes ni embeddings. **Errores:** 401, 503.

### `GET /v1/models`

**Uso/entrada:** modelos detector/recognizer verificados, Provider y licencia; sin parámetros. **Resultado:** 200 `models`, `execution_provider`, `license`. **Error:** 401.

## Operaciones faciales sin estado

### `POST /v1/detect`

**Entrada:** multipart `image` obligatorio, `max_faces` 1–100, `collection_id` opcional. **Proceso/resultado:** fusiona resoluciones, NMS global y orden por área; 200 `faces` con cajas/5 puntos/score/calidad y `processing_ms`. Sin rostro es lista vacía correcta. **Errores:** 400 min_score antiguo, 404 Collection, 413, 422 invalid_image, 503.

```bash
curl -sS "${BASE_URL}/v1/detect" -H "${AUTH_HEADER}" -F 'image=@group.jpg' -F 'max_faces=10'
```

### `POST /v1/compare`

**Entrada:** multipart `source`, `target`, `threshold` opcional 0–1 y `collection_id`. **Resultado:** elige un rostro por imagen; 200 `matched`, coseno `similarity`, threshold efectivo, ambos rostros y tiempo. **Errores:** 404, 413, 422 invalid_image/face_not_found, 503.

### `POST /v1/embeddings`

**Entrada:** multipart `image`, `collection_id` opcional. **Resultado:** 200 con rostro seleccionado, embedding L2, modelo y tiempo. No hace falta para registro normal y el vector no se registra en logs. **Errores:** 400 face_selection antiguo, 404, 413, 422, 503.

## Collections

### `POST /v1/collections`

**Entrada:** JSON `id`, `name`; opcionales description, threshold (0.4), metadata, save_face_crops, `detection` y `search` con profile/capacity/max_faces_per_person/load_policy. **Proceso/resultado:** fija modelo, preprocesamiento y contrato de búsqueda; 201 con `collection` resuelta. **Errores:** 400 configuración, 409 exists, 503 índice.

```bash
curl -sS "${BASE_URL}/v1/collections" -H "${AUTH_HEADER}" -H 'Content-Type: application/json' -d '{"id":"employees","name":"Employees","threshold":0.4}'
```

### `GET /v1/collections`

**Entrada:** query `limit` 1–100 (50), cursor opcional. **Resultado:** 200 `collections`, `next_cursor` nullable. **Errores:** 400 invalid_cursor, 401.

### `GET /v1/collections/{collection_id}`

**Entrada:** ID de Collection en path. **Resultado:** 200 `collection`, conteos Person/Face y `embedding_contract_id`. **Error:** 404.

### `PATCH /v1/collections/{collection_id}`

**Entrada:** path ID; JSON name/description/threshold/metadata/save_face_crops, capacidad/max/load de búsqueda y detection. No admite null, campos desconocidos, cambiar modelo ni search profile. **Resultado:** 200 Collection completa; detection rige desde la siguiente petición. **Errores:** 400, 404, 409, 503.

### `DELETE /v1/collections/{collection_id}`

**Entrada:** path ID; query `force=false`, true para no vacía. **Resultado:** 204 sin body. **Errores:** 404, 409 collection_not_empty, 503.

## Personas y FaceSamples

### `POST /v1/collections/{collection_id}/persons`

**Entrada:** path Collection; multipart `images` repetible, id/name/external_id opcionales, metadata como JSON string, `review_mode=off|standard|strict`, `embedding_mode=server|external_trusted`; modo externo añade vectores y contract ID. **Proceso/resultado:** revisa cada imagen; 201 `person`, `faces` aceptadas y `rejected_images`; admite éxito parcial, todo rechazado devuelve 422 sin crear Person. **Errores:** 400, 404, 409 ID/contrato/capacidad, 413, 422, 503.

```bash
curl -sS "${BASE_URL}/v1/collections/employees/persons" -H "${AUTH_HEADER}" -F 'id=alice' -F 'review_mode=off' -F 'images=@alice.jpg'
```

### `GET /v1/collections/{collection_id}/persons`

**Entrada:** Collection; query limit/cursor/`search` por ID, nombre o external ID. **Resultado:** 200 `persons`, `next_cursor`. **Errores:** 400 cursor, 404.

### `GET /v1/collections/{collection_id}/persons/{person_id}`

**Entrada:** IDs Collection y Person. **Resultado:** 200 `person` con face_count. **Error:** 404.

### `PATCH /v1/collections/{collection_id}/persons/{person_id}`

**Entrada:** IDs; JSON name/external_id/metadata objeto. **Resultado:** 200 Person actualizada. **Errores:** 400, 404, 409 external_id_exists.

### `DELETE /v1/collections/{collection_id}/persons/{person_id}`

**Entrada:** IDs. **Resultado:** elimina Person, FaceSamples, embeddings y crops, sincroniza índice, 204. **Errores:** 404, 503.

### `POST /v1/collections/{collection_id}/persons/{person_id}/faces`

**Entrada:** IDs; images repetibles y mismos campos review/embedding que crear Person. **Resultado:** 201 `faces`, `rejected_images`, éxito parcial. **Errores:** registro más 404 Person.

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces`

**Entrada:** IDs; query limit 1–100 y cursor. **Resultado:** 200 metadata de `faces`, `has_crop`, `next_cursor`, sin embedding ni bytes. **Errores:** 400 cursor, 404.

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}/image`

**Entrada:** tres IDs. **Resultado:** si existe, 200 `image/jpeg`, crop 112×112, `Cache-Control:no-store`; request ID solo en header. **Errores:** 401, 404 face/face_image_not_found.

### `DELETE /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}`

**Entrada:** tres IDs. **Resultado:** elimina embedding/crop/fila del índice, 204. **Errores:** 404, 503.

## Búsqueda

### `POST /v1/collections/{collection_id}/search`

**Entrada:** Collection; multipart `image`, `limit` 1–100 (5), threshold opcional o valor de Collection. **Proceso/resultado:** compara el rostro elegido con todas las muestras y usa el máximo por Person; 200 `searched_face`, `matches` ordenados, threshold y tiempo. Sin match es lista vacía. **Errores:** 404, 409 modelo, 413, 422 imagen/rostro, 503 índice/timeout.

```bash
curl -sS "${BASE_URL}/v1/collections/employees/search" -H "${AUTH_HEADER}" -F 'image=@query.jpg' -F 'limit=5'
```

## Monitores RTSP

La configuración del Monitor persiste en SQLite y una tarea habilitada se restaura al reiniciar el servidor. No se guardan fotogramas; los eventos viven solo en un búfer circular limitado en memoria.

### `POST /v1/monitors`

**Uso:** Crear un Monitor RTSP persistente. **Entrada:** JSON con ID, nombre, `source`, Collection, `inference_fps` (2), umbral opcional, búfer/política de eventos y `preview_enabled` (false). **Resultado:** 201 con `monitor` censurado; las credenciales se cifran al almacenarse. **Errores:** 400, 404, 409, 429.

### `GET /v1/monitors`

**Uso:** Enumerar Monitores con paginación. **Entrada:** `limit` 1–100 (50) y el `cursor` opaco de la respuesta anterior, sin modificarlo. **Resultado:** 200 con `monitors` y `next_cursor`, nunca credenciales. **Errores:** 400 `invalid_cursor`, 401.

### `GET /v1/monitors/{monitor_id}`

**Uso:** Leer la configuración y el resumen de ejecución de un Monitor. **Entrada:** `monitor_id` en la ruta. **Resultado:** 200 con política de eventos, fuente censurada, vista previa y estado. **Errores:** 401, 404 `monitor_not_found`.

### `PATCH /v1/monitors/{monitor_id}`

**Uso:** Actualizar parcialmente campos salvo el ID y arrancar/parar con `enabled`. **Entrada:** JSON parcial; `event_policy` también es parcial y umbral null hereda la Collection. **Resultado:** 200 con el Monitor completo; cambiar fuente/Collection/frecuencia/política reinicia la tarea. **Errores:** 400, 404, 429.

### `DELETE /v1/monitors/{monitor_id}`

**Uso:** Eliminar permanentemente un Monitor. **Entrada:** `monitor_id` en la ruta. **Resultado:** detiene decodificador, inferencia y RTSP, descarta eventos de memoria y devuelve 204; no elimina la Collection. **Errores:** 401, 404.

### `GET /v1/monitors/{monitor_id}/state`

**Uso:** Consultar el estado actual desde clientes sin interfaz. **Entrada:** ID del Monitor. **Resultado:** 200 con conexión, FPS efectivo, latencia, saltos, rostros reconocidos/desconocidos, vista previa, reconexiones y error seguro, sin embeddings. **Errores:** 401, 404.

### `GET /v1/monitors/{monitor_id}/events`

**Uso:** Obtener eventos volátiles de entrada/salida/error/recuperación. **Entrada:** `limit` 1–1000 y el último `cursor` opaco. **Resultado:** 200 con `events`, `next_cursor`, `truncated` y `stream_reset`; el reinicio pierde los eventos. **Errores:** 400 `invalid_cursor`, 401, 404.

### `GET /v1/monitors/{monitor_id}/preview.mjpeg`

**Uso:** Abrir la vista previa MJPEG cruda, desactivada por defecto. **Entrada:** ID y cabecera Bearer normal; nunca la clave en la URL. **Resultado:** `multipart/x-mixed-replace` largo, codificado solo con espectadores; el cliente dibuja cajas desde `/state`. **Errores:** 401, 404, 409 `preview_disabled`, 503.

## Reintentos

GET se puede reintentar. Verifique estado antes de repetir DELETE. Si el resultado de crear Person/Face es incierto por red, consulte el ID antes de repetir POST. Reintente solo 429 y 503 transitorios con backoff exponencial limitado y jitter; corrija los 4xx.
