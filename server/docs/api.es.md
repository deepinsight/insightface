# Guía de uso de la API REST de InsightFace Server

**Idiomas:** [English](api.md) · [中文](api.zh-CN.md) · [日本語](api.ja.md) · [Deutsch](api.de.md) · Español · [Français](api.fr.md) · [Русский](api.ru.md) · [Português](api.pt.md) · [한국어](api.ko.md)

Esta guía explica el propósito, la entrada, el trabajo del servidor, el resultado y los errores de todas las rutas públicas. Consulte instalación y primer uso en la [guía de usuario](user-guide.es.md) y el esquema exacto en ejecución en `/docs` o `/openapi.json`.


Los modelos se identifican mediante `model_id`; las respuestas omiten `model_version`. Las Collections existentes conservan su `embedding_contract_id`.

Para usar la prueba de vida, consulte [configuración, instalación y resultados](#addon-opcional-de-prueba-de-vida). Cada flujo explica también sus efectos.

## Reglas comunes

- Ruta base `/v1`, JSON `snake_case`, imágenes JPEG/PNG/WebP/BMP como multipart.
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

## Addon opcional de prueba de vida

La prueba de vida está desactivada por defecto en `server/config/server.toml`: `inference.addons` y `addons.auto_download` son `[]`. Las configuraciones antiguas sin estas claves siguen desactivadas.

**Activar desde la línea de comandos, incluso antes del primer inicio de Server:**

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license --enable-liveness
```

`--enable-liveness` comprueba primero que la configuración existente pueda actualizarse. Instala y verifica el paquete base, los addons configurados para instalación y la prueba de vida, y después añade `liveness` a `inference.addons` y `addons.auto_download`, conservando otras entradas, comentarios y ajustes. Reutiliza cachés verificados y guarda la activación incluso si el modelo ya estaba en caché. Un fallo de descarga no modifica la configuración; un fallo al guardarla devuelve un error explícito y un código de salida distinto de cero. Los archivos válidos en caché pueden reutilizarse al reintentar.

Ambos servicios Compose montan todo el directorio existente `server/config` con escritura en `/etc/insightface`, con `create_host_path: false`. Así el instalador actualiza atómicamente la configuración del host sin que Server esté ejecutándose. Deben existir el directorio y `server.toml`.

Server no tiene que estar en ejecución. En una instalación nueva, el siguiente `up -d` activa la prueba de vida; si Server ya está ejecutándose, use `docker compose -f server/deploy/compose.cpu.yml restart server`. `up -d` por sí solo no recarga los ajustes guardados. Para CUDA use `compose.cuda12.yml`.

Sin `--enable-liveness`, `models install` mantiene su comportamiento y no escribe configuración; la prueba de vida sigue desactivada por defecto. `models addons install liveness` solo descarga y verifica el addon, sin activarlo. También puede usar **Sistema → Detección de vida** como se explica a continuación.

En **Sistema → Detección de vida**, descargue el modelo y actívelo para el próximo inicio. Tras verificar SHA-256, se añade `liveness` a ambas listas conservando las demás entradas, comentarios y opciones. Se reutiliza una copia ya verificada. **Reinicie el Server manualmente** para aplicar el cambio. Los errores permiten reintentar; una descarga fallida no activa la prueba.

[Montajes y permisos para descargas Web](user-guide.es.md#montajes-y-permisos-para-descargas-web).

**Avanzado: configuración manual.** Estos ajustes son una alternativa al parámetro de activación o a la acción Web; instale el modelo antes de reiniciar.

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

### Instalación del modelo y arranque

`inference.addons` controla el uso y `addons.auto_download` la descarga adicional al instalar un paquete base. Con `["liveness"]` se instala el addon incluso con el paquete base en caché. No hay descargas al iniciar el Server. Instalador y Server leen el mismo archivo.

Ejecute los comandos desde la raíz del repositorio con `server/config/server.toml` presente. El Compose incluido ejecuta el instalador como root, crea el directorio de modelos si falta y monta `/models` con escritura; la descarga del addon crea `addons`. No hace falta preparar UID/GID ni permisos manualmente. Consulte la [configuración inicial de la guía de usuario](user-guide.es.md).

```bash
docker compose -f server/deploy/compose.cpu.yml run --rm models addons install liveness
docker compose -f server/deploy/compose.cpu.yml run --rm models addons verify liveness
docker compose -f server/deploy/compose.cpu.yml up -d --force-recreate
```

Un modelo activado ausente detiene el inicio con `addon_model_missing`; uno inválido produce `addon_model_invalid`. No se desactiva silenciosamente el addon.

### Resultados de la prueba de vida

| Resultado | `status` | `is_live` | `live_score` |
| --- | --- | --- | --- |
| Rostro vivo | `ok` | `true` | `[0, 1]` |
| Suplantación | `ok` | `false` | `[0, 1]` |
| Entrada rechazada | `input_rejected` | `null` | `null` |

Solo una superficie insuficiente de la imagen original alrededor del rostro alineado produce `input_rejected`. Este resultado añade `liveness.reason`, una explicación para el usuario; los resultados de rostro vivo o falsificación omiten `reason`. FaceAnalysis y la API devuelven siempre este texto en inglés; solo la interfaz Web traduce su presentación. Use `status` e `is_live` en la lógica del programa, sin interpretar el texto de `reason`. Los resultados antiguos guardados pueden carecer de `reason`; el cliente puede mostrar entonces un aviso genérico de entrada rechazada.

```json
{
  "status": "input_rejected",
  "is_live": null,
  "live_score": null,
  "reason": "Insufficient image area around the face for liveness detection. Move the face toward the center, step back from the camera, or use a less tightly cropped image."
}
```

`normal` reconoce solo rostros que superan la prueba; `observe` devuelve el resultado y continúa el reconocimiento. Si no se evalúa, se omite `liveness`. Los tres campos principales son `status`, `is_live` y `live_score`: los resultados de rostro vivo o suplantación usan `status: ok`, booleano y puntuación; una entrada rechazada usa `status: input_rejected` y dos valores `null`.

`/v1/detect` devuelve HTTP 200 incluso con resultados negativos. En `normal`, embeddings, comparación y búsqueda devuelven HTTP 422 `liveness_fake` o `liveness_input_rejected` con `error.details.liveness`; comparación añade `details.side`. Un fallo de inferencia devuelve HTTP 503 `liveness_unavailable`. Los fallos de ejecución interrumpen la operación tanto en `normal` como en `observe`; no se convierten en `input_rejected`.

El registro de personas y la adición de FaceSamples omiten la prueba de vida por defecto: `[inference].liveness_on_registration=false` no ejecuta el modelo y omite `liveness` en las muestras nuevas. Con `true` y el addon habilitado se aplica `normal`/`observe`; los rechazos incluyen `reason` y `liveness`. La revisión de calidad según `review_mode` y la validación de embeddings externos siguen activas. `review_mode=off` y `external_trusted` no evitan una prueba de registro habilitada. Las peticiones no pueden modificar esta configuración de inicio. Los resultados previamente guardados permanecen disponibles.

RTSP distingue `liveness_blocked` de `unknown` y usa el contador `liveness_blocked_faces`. Los rostros bloqueados no generan eventos de entrada de personas/desconocidos y reinician la confirmación. Los fallos de inferencia borran las identidades mostradas anteriormente.

`liveness_compare_scope` elige los lados evaluados de `/v1/compare`: `both` (predeterminado) para ambos, `source` para la imagen de origen y `target` para la imagen de destino. Se considera vivo si `live_score >= liveness_threshold`.

`models addons install liveness` guarda el modelo publicado en `/models/addons/liveness.onnx`; en el anfitrión Compose, en `server/.models/addons/liveness.onnx`. Los errores de arranque son `addon_model_missing` y `addon_model_invalid`. `/v1/models` y `/v1/system` muestran los complementos activos en `addons`.

[Configuración y operaciones](user-guide.es.md#addon-opcional-de-prueba-de-vida).

## Sistema

### `GET /v1/health`

**Uso/entrada:** readiness pública, sin parámetros ni autenticación. **Resultado:** comprueba inicio y SQLite quick_check; 200 con `status`, `auth_enabled`, `request_id`. **Error:** `503 not_ready`.

### `GET /v1/system`

**Uso/entrada:** diagnóstico seguro, sin parámetros. **Resultado:** 200 con OS/CPU/GPU, Driver, CUDA/cuDNN/ORT, Provider, modelo, DB, montajes, totales, búsqueda, configuración segura y concurrencia; nunca secretos, imágenes ni embeddings. **Errores:** 401, 503.

### `GET /v1/models`

`addons` muestra los addons activos separados del modelo base. Compruebe `liveness` y los ajustes efectivos de `safe_config` en la respuesta de sistema. Estos endpoints son de solo lectura y no instalan modelos.

**Uso/entrada:** modelos detector/recognizer verificados, Provider y licencia; sin parámetros. **Resultado:** 200 `models`, `execution_provider`, `license`. **Error:** 401.

Los paquetes base `raccoon_s` y `raccoon_l` funcionan en CPU y CUDA y se instalan con la herramienta de modelos antes del arranque. Esta API enumera componentes en ejecución, no un catálogo de descargas. La acción Web siguiente solo administra la prueba de vida. Las Collections están vinculadas al modelo de reconocimiento y al preprocesamiento: cambiar el paquete no convierte los vectores existentes y puede generar `409 collection_model_mismatch`. Activar solo la prueba de vida no cambia ese contrato.

### `GET /v1/addons/liveness`

**Uso:** Consultar la instalación y los ajustes del próximo arranque sin descargar ni modificar nada. Es una API de administración, no una API independiente de inferencia de prueba de vida.

**Resultado:** HTTP 200. `enabled` indica el estado del proceso actual. `installed` indica que el archivo supera el SHA-256 publicado, no que la prueba esté activa. `configured_enabled` lee la selección del próximo arranque del archivo actual; `restart_required` indica que difiere de `enabled`. Hasta reiniciar, `safe_config` de `/v1/system` sigue mostrando la configuración del proceso actual.

`state` es `idle` (sin modelo verificado), `downloading` (preparación en curso), `ready` (modelo verificado disponible) o `error` (fallo de preparación, archivo o configuración). `ready` por sí solo no confirma que se haya guardado la activación ni completado el reinicio.

`can_enable` indica si la preparación Web está disponible. Si no lo está, `unavailable_code` ofrece un código estable del motivo y `unavailable_reason` una explicación; en otro caso ambos son `null`. `error` es `null` o un objeto con `code` y `message`. `model_path` es la ruta local del modelo; `config_file`, la ruta TOML seleccionada o `null`. La respuesta también incluye `request_id`.

Los valores de `unavailable_code` son `config_file_missing` (sin archivo de configuración seleccionado), `config_file_not_regular` (no es un archivo normal), `config_file_mount` (archivo montado individualmente), `config_not_writable` (configuración sin escritura), `addon_directory_not_writable` (directorio del complemento sin escritura), `addon_config_invalid` (configuración inválida), `addon_model_invalid` (modelo inválido) y `server_stopping` (servidor apagándose).

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

**Uso:** Descargar y configurar la prueba de vida para el siguiente arranque. Enviar un objeto JSON vacío `{}` con `Content-Type: application/json`. No se aceptan URL de modelos ni otros parámetros.

```bash
curl -sS "${BASE_URL}/v1/addons/liveness" -H "${AUTH_HEADER}"
curl -sS "${BASE_URL}/v1/addons/liveness/enable" -H "${AUTH_HEADER}" \
  -H 'Content-Type: application/json' -d '{}'
```

**Resultado:** HTTP 202 devuelve los mismos campos que GET y confirma la aceptación del trabajo, no la activación. Consultar `GET /v1/addons/liveness` hasta finalizar. Las solicitudes duplicadas comparten el trabajo activo; cerrar el navegador no lo cancela.

Solo después de descargar y verificar el SHA-256 se añade `liveness` a `[inference].addons` y `[addons].auto_download` en `config_file`, conservando los demás valores y comentarios. Se reutiliza un archivo verificado. Cuando `installed=true`, `configured_enabled=true` y `restart_required=true`, reiniciar manualmente el servidor. El nuevo proceso mostrará `enabled=true` y `restart_required=false`. No hay recarga en caliente ni API para cambiar el paquete base.

**Errores:** Los errores de solicitud usan el formato habitual: `400 invalid_addon_request` si el cuerpo no es `{}`, `401 unauthorized` si falla la autenticación, `403 origin_not_allowed` para un origen del navegador no permitido, `409 addon_management_unavailable` por rutas, permisos o configuración incompatibles, y `415 json_required` si el contenido no es JSON. El navegador debe usar el mismo origen del servidor o uno expresamente permitido por CORS.

Un trabajo aceptado puede fallar después: GET sigue devolviendo HTTP 200 con `state=error` y `error.code`. `addon_download_failed` no modifica la configuración; revisar la red o el proxy del servidor. Ante `addon_config_save_failed`, corregir la configuración o los permisos de directorio; el modelo verificado puede reutilizarse. `addon_config_invalid` indica TOML inválido en disco. `addon_model_invalid` exige sustituir o eliminar el archivo dañado, que nunca se sobrescribe silenciosamente. `addon_job_in_progress` indica otro proceso preparando el modelo: esperar y actualizar. Corregir la causa antes de repetir POST.

## Operaciones faciales sin estado

### `POST /v1/detect`

Cada rostro evaluado incluye `liveness.status`, `liveness.is_live` y `liveness.live_score`. Los resultados de suplantación e `input_rejected` también devuelven HTTP 200, sin extraer características de reconocimiento. `input_rejected` indica una superficie de imagen insuficiente alrededor del rostro; `liveness.reason` explica cómo ajustar la imagen. Si falta `liveness`, no se evaluó.

**Entrada:** multipart `image` obligatorio, `max_faces` 1–100, `collection_id` opcional. **Proceso/resultado:** fusiona resoluciones, NMS global y orden por área; 200 `faces` con cajas/5 puntos/score/calidad y `processing_ms`. Sin rostro es lista vacía correcta. **Errores:** 400 min_score antiguo, 404 Collection, 413, 422 invalid_image, 503.

```bash
curl -sS "${BASE_URL}/v1/detect" -H "${AUTH_HEADER}" -F 'image=@group.jpg' -F 'max_faces=10'
```

### `POST /v1/compare`

`liveness_compare_scope` (`both`, `source`, `target`) elige los lados evaluados antes del reconocimiento. En `normal`, un rechazo devuelve HTTP 422 `liveness_fake` / `liveness_input_rejected`, `error.details.liveness` y `error.details.side`, sin similitud. `observe` continúa y adjunta el resultado a los rostros evaluados.

**Entrada:** multipart `source`, `target`, `threshold` opcional 0–1 y `collection_id`. **Resultado:** elige un rostro por imagen; 200 `matched`, coseno `similarity`, threshold efectivo, ambos rostros y tiempo. **Errores:** 404, 413, 422 invalid_image/face_not_found, 503.

### `POST /v1/embeddings`

Con prueba de vida en `normal`, una suplantación o una entrada no apta devuelve HTTP 422 `liveness_fake` / `liveness_input_rejected` y `error.details.liveness`; no se extrae el embedding. `observe` devuelve el embedding y el resultado de vida del rostro.

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

La creación de Person y la adición de FaceSamples omiten la prueba de vida por defecto (`liveness_on_registration=false`). Si se activa, `normal` rechaza suplantaciones y entradas no aptas; `observe` conserva el resultado y continúa. La revisión de calidad sigue el `review_mode` elegido. Los rechazos muestran el `reason` real y el resultado de vida por separado.

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

La creación de Person y la adición de FaceSamples omiten la prueba de vida por defecto (`liveness_on_registration=false`). Si se activa, `normal` rechaza suplantaciones y entradas no aptas; `observe` conserva el resultado y continúa. La revisión de calidad sigue el `review_mode` elegido. Los rechazos muestran el `reason` real y el resultado de vida por separado.

**Entrada:** IDs; images repetibles y mismos campos review/embedding que crear Person. **Resultado:** 201 `faces`, `rejected_images`, éxito parcial. **Errores:** registro más 404 Person.

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces`

**Entrada:** IDs; query limit 1–100 y cursor. **Resultado:** 200 metadata de `faces`, `has_crop`, `next_cursor`, sin embedding ni bytes. **Errores:** 400 cursor, 404.

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}/image`

**Entrada:** tres IDs. **Resultado:** si existe, 200 `image/jpeg`, crop 112×112, `Cache-Control:no-store`; request ID solo en header. **Errores:** 401, 404 face/face_image_not_found.

### `DELETE /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}`

**Entrada:** tres IDs. **Resultado:** elimina embedding/crop/fila del índice, 204. **Errores:** 404, 503.

## Búsqueda

### `POST /v1/collections/{collection_id}/search`

Con prueba de vida en `normal`, una suplantación o una consulta no apta devuelve HTTP 422 `liveness_fake` / `liveness_input_rejected` y `error.details.liveness`; no se ejecuta la búsqueda. No equivale a una lista vacía de coincidencias correcta. `observe` continúa y devuelve el resultado en el rostro consultado.

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

Con prueba de vida en `normal`, los rostros bloqueados tienen `status: liveness_blocked` y un resultado separado. Cuentan en `liveness_blocked_faces`, no en `unknown_faces`, y no generan eventos de entrada. `observe` continúa el reconocimiento. La interfaz distingue entrada rechazada y suplantación.

**Uso:** Consultar el estado actual desde clientes sin interfaz. **Entrada:** ID del Monitor. **Resultado:** 200 con conexión, FPS efectivo, latencia, saltos, rostros reconocidos/desconocidos, vista previa, reconexiones y error seguro, sin embeddings. **Errores:** 401, 404.

### `GET /v1/monitors/{monitor_id}/events`

**Uso:** Obtener eventos volátiles de entrada/salida/error/recuperación. **Entrada:** `limit` 1–1000 y el último `cursor` opaco. **Resultado:** 200 con `events`, `next_cursor`, `truncated` y `stream_reset`; el reinicio pierde los eventos. **Errores:** 400 `invalid_cursor`, 401, 404.

### `GET /v1/monitors/{monitor_id}/preview.mjpeg`

**Uso:** Abrir la vista previa MJPEG cruda, desactivada por defecto. **Entrada:** ID y cabecera Bearer normal; nunca la clave en la URL. **Resultado:** `multipart/x-mixed-replace` largo, codificado solo con espectadores; el cliente dibuja cajas desde `/state`. **Errores:** 401, 404, 409 `preview_disabled`, 503.

## Reintentos

GET se puede reintentar. Verifique estado antes de repetir DELETE. Si el resultado de crear Person/Face es incierto por red, consulte el ID antes de repetir POST. Reintente solo 429 y 503 transitorios con backoff exponencial limitado y jitter; corrija los 4xx.
