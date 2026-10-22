# InsightFace Server REST API usage guide

**Languages:** English · [中文](api.zh-CN.md) · [日本語](api.ja.md) · [Deutsch](api.de.md) · [Español](api.es.md) · [Français](api.fr.md) · [Русский](api.ru.md) · [Português](api.pt.md) · [한국어](api.ko.md)

This document explains how to call every public endpoint, what each parameter
means, what work the server performs, and what a successful result looks like.
Start with the [step-by-step user guide](user-guide.md) if the
container and model are not running yet. The API is rooted at `/v1`, uses
snake_case JSON fields, and accepts images as `multipart/form-data`. It is not
an AWS Rekognition or CompreFace compatibility contract.

For liveness, see [configuration, model installation and result meanings](#optional-liveness-addon). The workflow sections below also explain how it affects each operation.

## Common behavior

The supplied Compose files disable authentication by default for isolated
evaluation. When an operator enables authentication, every endpoint except
`GET /v1/health` requires:

```http
Authorization: Bearer <api_key>
```

`GET /v1/health` is public so container orchestrators and the Web UI can check
readiness and whether API authentication is enabled. Other `/v1` endpoints
require authentication only when `auth_enabled` is true.

Every response has an `x-request-id` UUID header. JSON response bodies repeat it
as `request_id`. Successful DELETE requests return HTTP 204 with no body.

Detection and quality scores use `0.0..1.0`:

- `detection_score` is detector confidence;
- `quality.score`, `sharpness`, `brightness`, and `pose` are documented local
  quality signals, not AWS metrics;
- `similarity` is the raw cosine value in `[-1.0, 1.0]`, not a probability;
- recognition `threshold` accepts `[0.0, 1.0]` and defaults to `0.4`;
- `threshold` is inclusive: `similarity >= threshold` matches.

Bounding boxes include both forms:

```json
{
  "pixels": {"x": 120, "y": 80, "width": 240, "height": 280},
  "normalized": {"left": 0.12, "top": 0.08, "width": 0.24, "height": 0.28}
}
```

JPEG, PNG, WebP, and BMP are accepted. EXIF orientation is applied before inference.
The default compressed-image limit is 10 MiB, decoded limit is 40 million
pixels, and whole-request limit is 64 MiB.

## Errors

Errors use standard HTTP status codes and one envelope:

```json
{
  "error": {
    "code": "face_not_found",
    "message": "No usable face was detected.",
    "details": {}
  },
  "request_id": "3ed21e89-4595-4eed-a699-1df42ca62032"
}
```

Common status mappings include 400 invalid parameters, 401 missing/invalid API
key, 404 missing resource, 409 resource/model conflict, 413 request or image too
large, 422 invalid image or unusable face, 500 unexpected error, and 503 timeout
or unavailable runtime. HTTP 429 can indicate a configured stream limit; the Server does not include a general request rate limiter.

OpenAPI is available at `/openapi.json`, with a same-origin interactive viewer at
`/docs`.

## First API workflow

Set the server address once. Leave `AUTH_HEADER` empty only when health reports
`auth_enabled: false`:

```bash
BASE_URL=http://127.0.0.1:18097
AUTH_HEADER="Authorization: Bearer ${INSIGHTFACE_API_KEY}"
curl -fsS "${BASE_URL}/v1/health"
```

Then create a Collection, enroll one Person, and search with a different image:

```bash
curl -sS "${BASE_URL}/v1/collections" -H "${AUTH_HEADER}" \
  -H 'Content-Type: application/json' \
  -d '{"id":"employees","name":"Employees","threshold":0.4}'

curl -sS "${BASE_URL}/v1/collections/employees/persons" -H "${AUTH_HEADER}" \
  -F 'id=alice' -F 'name=Alice' -F 'review_mode=off' \
  -F 'images=@alice-enroll.jpg'

curl -sS "${BASE_URL}/v1/collections/employees/search" -H "${AUTH_HEADER}" \
  -F 'image=@alice-query.jpg' -F 'limit=5'
```

Do not send an empty `Authorization` header when authentication is disabled;
omit the header completely. Shell examples below show the header so they work
unchanged in authenticated deployments.

## Client rules and retry safety

- Use a client timeout longer than the configured server request timeout.
- Treat `x-request-id` as the correlation ID and log it without logging images,
  embeddings, RTSP credentials, or API keys.
- Reuse an opaque `next_cursor` only with the same endpoint, Collection, Person,
  and filter values. Never parse or construct it.
- GET requests are safe to retry. Retry DELETE only after checking current
  state. Do not automatically retry Person/FaceSample creation after an
  ambiguous network failure; first query the resource ID supplied by the client.
- `429` and transient `503` may be retried with bounded exponential backoff and
  jitter. Validation `4xx` responses require changing the request.
- Content type matters: Collection/Person patches use JSON; image operations and
  enrollment use multipart; the saved face endpoint returns JPEG; MJPEG is a
  streaming response.

## Optional liveness addon

### Enable and install the model

Liveness is disabled by default in `server/config/server.toml`: both `inference.addons` and `addons.auto_download` are `[]`. Existing configurations that omit these keys also remain disabled.

**Enable from the command line, including before the first Server startup:**

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license --enable-liveness
```

`--enable-liveness` first checks that the existing configuration can be updated. It installs and verifies the base package, configured installation addons, and liveness, then adds `liveness` to both `inference.addons` and `addons.auto_download`, preserving other entries, comments, and settings. Verified caches are reused, but the enable setting is still saved on a cache hit. A failed download leaves the configuration unchanged; a configuration-save failure exits with an error and a nonzero status. Successfully cached files can be reused on retry.

Both Compose services mount the whole existing `server/config` directory writable at `/etc/insightface`, with `create_host_path: false`, so the installer can atomically update the host configuration without a running Server. The directory and `server.toml` must exist.

The Server does not need to be running. On a new deployment, the next `up -d` starts with liveness enabled; if Server is already running, use `docker compose -f server/deploy/compose.cpu.yml restart server`. `up -d` alone does not reload saved settings. For CUDA, use `compose.cuda12.yml`.

Without `--enable-liveness`, `models install` keeps the existing behavior and does not write configuration; the default remains disabled. `models addons install liveness` only downloads/verifies the addon and does not enable it. You can also enable through **System → Liveness** as described below.

In **System → Liveness**, choose **Download and enable after restart**. The Server downloads the published model, verifies its SHA-256, then adds `liveness` to both configuration lists while preserving other entries, comments, and settings. A verified cached file is reused. The current process stays unchanged: **manually restart the Server** to enable liveness. Download or configuration errors are shown with a retry action; a failed download does not enable liveness. A downloaded file alone does not activate it.

**Advanced: configure liveness manually.** The following settings are an alternative to the enabling flag or Web action; install the model before restarting.

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

`inference.addons` controls runtime use; `addons.auto_download` independently controls installation. Setting the latter to `["liveness"]` installs the addon alongside any supported base package, including a cached base package. Server startup never downloads models. Installer and Server read the same configuration file.

Use the current Compose files and existing configuration from the [initial user-guide setup](user-guide.md). Server and installer run as root with one writable model mount; Compose creates the model root, and explicit addon downloads create `addons/`. No host UID/GID or manual permission setup is required.

```bash
docker compose -f server/deploy/compose.cpu.yml run --rm models install buffalo_l --accept-license
# Or install only the addon:
docker compose -f server/deploy/compose.cpu.yml run --rm models addons install liveness
docker compose -f server/deploy/compose.cpu.yml run --rm models addons verify liveness
docker compose -f server/deploy/compose.cpu.yml up -d --force-recreate
```

For CUDA use `compose.cuda12.yml`. The file is stored at
`server/.models/addons/liveness.onnx` on the Compose host and is visible at
`/models/addons/liveness.onnx` in the container. All addon models share this flat
`addons/` directory. The installer verifies the pinned SHA-256 of the
[published model](https://github.com/deepinsight/insightface-model-addons/releases/download/addons/liveness.onnx).
Compose mounts the model root once with write access; addon downloads create its `addons/` subdirectory as needed. The configuration directory is also writable for Web management. Images contain code and dependencies, not pretrained weights. Upgrading with the default disabled configuration needs no addon download.

If an existing deployment enables liveness without installing it, startup fails
with `addon_model_missing`, the required path and an installation command.
A corrupt/unreadable file produces `addon_model_invalid`. Neither case silently
disables the addon. Run the writable installer and restart; if verification
reports corruption, replace the invalid file with the verified published model.
Upgrading code alone with liveness disabled keeps existing collections and
embeddings usable. The additive database migration preserves old rows; liveness
does not change the recognition model digest or `embedding_contract_id`.

### Read liveness results

Each evaluated face exposes these three core fields inside `liveness`:

| Result | `status` | `is_live` | `live_score` |
| --- | --- | --- | --- |
| Live | `ok` | `true` | Number in `[0, 1]` |
| Fake | `ok` | `false` | Number in `[0, 1]` |
| Insufficient image area around the face | `input_rejected` | `null` | `null` |

Only insufficient source-image area around the aligned face produces `input_rejected`. This result adds `liveness.reason`, a human-readable explanation; live and fake results omit `reason`. FaceAnalysis and the API always return this text in English; only the Web UI translates its display. Use `status` and `is_live` for program logic, not the wording of `reason`. Older saved results may lack `reason`; clients can show a generic input-rejected message as a fallback.

```json
{
  "status": "input_rejected",
  "is_live": null,
  "live_score": null,
  "reason": "Insufficient image area around the face for liveness detection. Move the face toward the center, step back from the camera, or use a less tightly cropped image."
}
```

`is_live` uses `live_score >= liveness_threshold`. The field `liveness` is
omitted when disabled, skipped for enrollment, or excluded by compare scope. A missing field therefore
means the face was not evaluated; `is_live: null` means its input was rejected.

- `normal` (default): detect/select the face, evaluate liveness, then extract
  recognition features only if it passes. A failed selected face is not replaced
  with another live face in the background.
- `observe`: evaluate liveness and continue recognition for fake/rejected input.
  Inference failures still return an error.
- `liveness_compare_scope` is `both` (default), `source`, or `target`; it only
  changes which `/v1/compare` image receives liveness evaluation. Requests cannot
  override these startup settings. Enrollment has its own opt-in setting below.

### Detection, recognition and errors

`/v1/detect` returns HTTP 200 with per-face liveness, including negative results,
and never extracts embeddings. In `normal`, `/v1/embeddings`, `/v1/compare` and
Collection search return HTTP 422 with `liveness_fake` or
`liveness_input_rejected`; `error.details.liveness` contains the result, including `reason` for input rejection.
Compare adds `error.details.side` (`source` or `target`). No similarity or match
is returned for a blocked operation. A runtime failure returns HTTP 503
`liveness_unavailable`, rather than a fake classification. Runtime failures stop the operation in both `normal` and `observe`; they are not `input_rejected` results.

### Enrollment defaults

Enrollment skips liveness by default. With
`[inference].liveness_on_registration = false`, creating a Person and adding
FaceSamples do not run the liveness model or include `liveness` in new samples.
Detection, recognition/external-embedding validation, and the selected
`review_mode` still apply. This startup setting cannot be overridden per request.

Set `liveness_on_registration = true` to apply the configured `normal`/`observe`
policy during enrollment when the addon is enabled. In `normal`, rejected images
have `reason: liveness_fake` or `liveness_input_rejected`, plus `liveness`;
partial success is preserved. If none pass when creating a Person, the existing
`registration_failed` error contains `rejected_images`. `review_mode=off` and
`embedding_mode=external_trusted` do not bypass an enabled enrollment check.
In `observe`, enrollment continues and saves the liveness result. Previously
saved snapshots remain visible; samples never evaluated omit the field.

### RTSP and Web UI

RTSP faces blocked in `normal` have outer `status: liveness_blocked` and no
identity. They count toward `liveness_blocked_faces`, not `unknown_faces`, do not
emit person/unknown-enter events, and restart identity confirmation. Runtime
liveness errors clear stale displayed identities. `observe` continues matching.
The Web UI displays liveness and distinct rejection states; `/v1/models` and
`/v1/system` report enabled addons separately from the base model files.

## System

### `GET /v1/health`

**Use:** Public readiness and container health check. **Parameters:** none.

**Expected work and result:** It returns 200 with `status: "ready"` when startup has
completed and SQLite `quick_check` succeeds, otherwise 503. The public
`auth_enabled` boolean lets clients decide whether to present API-key controls;
it does not expose the configured key or its hash.

```bash
curl -sS "${BASE_URL}/v1/health"
```

```json
{"status":"ready","auth_enabled":false,"request_id":"..."}
```

**Common errors:** `503 not_ready` while models, database, or search indexes are
not ready. Health deliberately does not require authentication.

### `GET /v1/system`

**Use:** Operator diagnostics. **Parameters:** none.

**Expected work and result:** Returns safe diagnostics: server/OS/architecture/CPU, GPU and Compute Capability
when present, NVIDIA driver, CUDA/cuDNN/ONNX Runtime, actual providers, model
summary, database and path status, aggregate counts, API-key state, safe limits,
and recent structured error summaries. It does not return API keys, images, or
embeddings. `safe_config.detection` reports the immutable system detection
profile and `safe_config.max_detected_faces` reports its safety cap. The API has
no runtime mutation endpoint for the system profile.
`safe_config.inference_max_concurrency` reports the resolved process-wide model
budget (CPU default 4, CUDA default 8). Runtime diagnostics additionally expose
active, waiting and peak model work. Detect, Compare, Embeddings, enrollment,
Search query extraction and RTSP recognition share this budget.

```bash
curl -sS "${BASE_URL}/v1/system" -H "${AUTH_HEADER}"
```

**Success:** HTTP 200 with runtime, model, database, path, aggregate count,
search backend, safe configuration, authentication-state, and recent-error
objects. Secrets, images, crops, and embeddings are excluded.

**Common errors:** `401 unauthorized`; `503 request_timeout` if diagnostics
cannot finish before the request deadline.

Collection creation copies this system profile unless the request supplies
overrides. Collection profiles are stored in SQLite, may be patched, and are
used by Collection-bound enrollment and Search operations. Existing embeddings
are not automatically re-extracted after a profile change.

### `GET /v1/models`

The `addons` list reports active addons separately from base model components. Check it for `liveness`; the System response also exposes the effective liveness settings in `safe_config`. These read-only endpoints do not install models.

**Use:** Read the verified model bundle and actual provider. **Parameters:** none.

**Expected work and result:** HTTP 200 returns `models`,
`execution_provider`, and the verified license summary. It does not return model
bytes or the private signing key.

```bash
curl -sS "${BASE_URL}/v1/models" -H "${AUTH_HEADER}"
```

**Common errors:** `401 unauthorized`.

Supported base packages include `raccoon_s` and `raccoon_l` on CPU and CUDA; install them with the model tool before startup. This endpoint lists the running model components, not downloadable packages. The Web action below manages only liveness. Collections remain bound to their recognition model and preprocessing contract: changing a base package does not convert existing embeddings and can produce `409 collection_model_mismatch`. Enabling liveness alone does not change that contract.

### `GET /v1/addons/liveness`

**Use:** Read installation and next-startup settings without downloading or changing anything. This is a management endpoint, not a standalone liveness inference API.

**Result:** HTTP 200. `enabled` is the running process’s state; `installed` means the file passed the published SHA-256 check and does not mean liveness is enabled. `configured_enabled` reads the current configuration file for the next startup; `restart_required` means it differs from `enabled`. `safe_config` from `/v1/system` continues to describe the running process until restart.

`state` is `idle` (no verified model), `downloading` (preparation in progress), `ready` (verified model available), or `error` (preparation, file, or configuration error). In particular, `ready` alone does not mean configuration was enabled or a restart completed.

`can_enable` reports whether Web preparation is available. When unavailable, `unavailable_code` is a stable reason code and `unavailable_reason` is explanatory text; both are otherwise `null`. `error` is `null` or an object with `code` and `message`. `model_path` is the local model path; `config_file` is the selected TOML path or `null`. Every response also contains `request_id`.

`unavailable_code` values are `config_file_missing` (no config selected), `config_file_not_regular` (not a regular file), `config_file_mount` (single-file mount), `config_not_writable` (config or its directory is not writable), `addon_directory_not_writable` (addon storage is not writable), `addon_config_invalid` (invalid config), `addon_model_invalid` (invalid model), and `server_stopping` (shutdown in progress).

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

**Use:** Download and configure liveness for the next startup. Send an empty JSON object `{}` with `Content-Type: application/json`; no model URL or other parameters are accepted.

```bash
curl -sS "${BASE_URL}/v1/addons/liveness" -H "${AUTH_HEADER}"
curl -sS "${BASE_URL}/v1/addons/liveness/enable" -H "${AUTH_HEADER}" \
  -H 'Content-Type: application/json' -d '{}'
```

**Result:** HTTP 202 returns the same status fields as GET and means the job was accepted, not that liveness is active. Poll `GET /v1/addons/liveness` until preparation finishes. Duplicate requests share an active job; closing the browser does not cancel it.

Only after download and SHA-256 verification does the job add `liveness` to `[inference].addons` and `[addons].auto_download` in `config_file`, preserving unrelated values and comments. A verified cached file is reused. Once `installed=true`, `configured_enabled=true` and `restart_required=true`, manually restart the Server. The new process reports `enabled=true` and `restart_required=false`. There is no hot reload or base-model switching API.

**Errors:** Request errors use the normal error envelope: `400 invalid_addon_request` for a body other than `{}`, `401 unauthorized` for missing or invalid credentials when authentication is enabled, `403 origin_not_allowed` for a disallowed browser origin, `409 addon_management_unavailable` for unavailable paths/permissions/configuration, and `415 json_required` for a non-JSON content type. Browser origins must match the Server or an explicitly allowed CORS origin.

An accepted job can fail later: GET still returns HTTP 200, with `state=error` and `error.code`. `addon_download_failed` leaves configuration unchanged; check the Server’s network/proxy and retry. `addon_config_save_failed` requires fixing configuration or directory permissions; a verified download remains reusable. `addon_config_invalid` means the on-disk TOML is invalid. `addon_model_invalid` requires restoring or removing the invalid cached file; it is never silently overwritten. `addon_job_in_progress` means another process holds the preparation lock; wait and refresh. Correct the cause before repeating POST.

## Stateless face operations

### `POST /v1/detect`

With liveness enabled, each evaluated face includes `liveness.status`, `liveness.is_live` and `liveness.live_score`. Detect returns HTTP 200 for fake and `input_rejected` results as well; it does not extract recognition features. `input_rejected` means there is insufficient image area around the face; `liveness.reason` explains how to adjust the image. An omitted `liveness` means it was not evaluated.

**Use:** Detect all usable faces without writing data. Send
`multipart/form-data` with:

- `image` (required);
- `max_faces` (optional, 1–100);
- `collection_id` (optional; uses that Collection's detection profile instead
  of the system profile).

Faces are ordered by descending area. No face is a successful response with
`faces: []`.

```bash
curl -sS http://localhost:18097/v1/detect \
  -H "Authorization: Bearer ${INSIGHTFACE_API_KEY}" \
  -F 'image=@group.jpg' \
  -F 'max_faces=10' \
  -F 'collection_id=employees'
```

**Success:** HTTP 200 with `faces`, `processing_ms`, and `request_id`. Each face
contains pixel/normalized bounding boxes, five landmarks, detector confidence,
and quality signals. Embeddings are not returned or persisted.

**Common errors:** `400 request_detection_override_not_supported` for the
deprecated `min_score`; `404` for an unknown `collection_id`; `413` for size
limits; `422 invalid_image`; `503 request_timeout`.

### `POST /v1/compare`

Liveness runs before recognition on the sides selected by `liveness_compare_scope` (`both`, `source` or `target`). In `normal`, a blocked side returns HTTP 422 `liveness_fake` or `liveness_input_rejected`, with `error.details.liveness` and `error.details.side`; there is no similarity result. `observe` continues the comparison and includes the liveness result on each evaluated face.

**Use:** Compare one selected face from each of two images without persistence.
Send `multipart/form-data` with:

- `source` and `target` (required);
- `threshold` (optional, `0.0..1.0`; server default `0.4`);
- `collection_id` (optional detection-profile source).

The active profile's single-face strategy selects one face in each image. A
missing face in either image returns `422 face_not_found`.

```bash
curl -sS http://localhost:18097/v1/compare \
  -H "Authorization: Bearer ${INSIGHTFACE_API_KEY}" \
  -F 'source=@source.jpg' \
  -F 'target=@target.jpg' \
  -F 'threshold=0.4'
```

**Success:** HTTP 200 with `matched`, raw cosine `similarity`, the effective
`threshold`, selected source/target face summaries, `processing_ms`, and
`request_id`.

**Common errors:** `404` for an unknown Collection; `413` for size limits;
`422 invalid_image` or `face_not_found`; `503 request_timeout`.

### `POST /v1/embeddings`

With liveness enabled in `normal`, fake or unsuitable input returns HTTP 422 `liveness_fake` or `liveness_input_rejected` and `error.details.liveness`; no embedding is extracted. `observe` returns the embedding together with the face liveness result.

**Use:** Extract the selected face embedding for a trusted integration. Send
`multipart/form-data` with:

- `image` (required);
- `collection_id` (optional detection-profile source).

Returns the profile-selected face with its embedding and model information. This
authenticated endpoint is intentionally not used by normal registration/search
flows. Embeddings are sensitive biometric templates and are not logged.

```bash
curl -sS "${BASE_URL}/v1/embeddings" -H "${AUTH_HEADER}" \
  -F 'image=@portrait.jpg' -F 'collection_id=employees'
```

**Success:** HTTP 200 with one item in `faces`, its L2-normalized embedding,
`model`, `processing_ms`, and `request_id`.

**Common errors:** `400 request_detection_override_not_supported` for the
deprecated `face_selection`; `404` unknown Collection; `413`; `422
invalid_image` or `face_not_found`; `503 request_timeout`.

## Collections

### `POST /v1/collections`

**Use:** Create an isolated identity database and pin its model, detection, and
search contract. Send `application/json`:

```json
{
  "id": "employees",
  "name": "Company Employees",
  "description": "Employee face collection",
  "threshold": 0.4,
  "save_face_crops": false,
  "detection": {
    "input_sizes": [[96, 96], [512, 512]],
    "threshold": 0.5,
    "nms_threshold": 0.4,
    "single_face_selection": "largest"
  },
  "search": {
    "profile": "fp32_v1",
    "capacity_rows": 100000,
    "max_faces_per_person": 20,
    "load_policy": "lazy"
  },
  "metadata": {"site": "shanghai"}
}
```

`id` is `_default`, or 1–64 characters beginning with a letter or digit and
otherwise containing letters, digits, `.`, `_`, and `-`. `name` is required. If
omitted, `threshold` uses `INSIGHTFACE_DEFAULT_THRESHOLD`.

`search.profile` accepts exactly `fp32_v1`, `fp16_v1`, `bf16_v1`,
`int8_x736_v1`, or `int8_x1000_v1`; there is no implicit rerank profile. The
recommended/default INT8 scale is 736, while the overall Collection default
remains FP32. The other defaults are 100,000 rows, 20 FaceSamples per Person,
and lazy loading. `_default` uses eager
loading when no load policy is provided. The resolved values are persisted and
returned as `search_profile`, `capacity_rows`, `max_faces_per_person`, and
`load_policy` on Collection resources.

The CPU native backend supports FP32, BF16, and INT8; FP16 is CUDA-only. The
CUDA backend supports all five profiles. A persisted profile unsupported by the
active backend fails explicitly and is never silently converted to another
profile or execution provider. CUDA BF16 additionally requires an SM80-or-newer
device. See the [User Guide](user-guide.md) for profile and capacity semantics
and the complete support matrix.

Creation binds model ID, bundle digest, embedding dimension, and
preprocessing version. Those fields cannot be patched. Every Collection
response also exposes a stable opaque `embedding_contract_id` derived from
those pinned fields. Clients must copy, not construct, that ID when using
trusted external enrollment.

`detection` is optional and may partially override the system profile copied at
creation. `single_face_selection` accepts `largest` and `center_largest`.
`center_largest` maximizes one pixel-space score:
`area - 2.0 * ((face_cx - image_cx)^2 + (face_cy - image_cy)^2)`.
Detection confidence does not participate in this selection score.
Collection responses return the resolved `detection` object and a monotonically
increasing `detection_revision`.

`save_face_crops` is optional and defaults to the deployment's
`INSIGHTFACE_SAVE_FACE_CROPS` value, which is `false` unless explicitly changed.
The resolved boolean is persisted on the Collection and does not follow later
environment changes. When enabled, accepted 112×112 bounding-box crops—not original
uploads—are encoded as JPEG and stored as BLOBs in SQLite. This can materially
increase database and backup size.

```bash
curl -sS "${BASE_URL}/v1/collections" -H "${AUTH_HEADER}" \
  -H 'Content-Type: application/json' \
  -d '{"id":"employees","name":"Employees","threshold":0.4}'
```

**Success:** HTTP 201 with the resolved `collection`, including immutable model
binding, detection profile, search settings, counts, and timestamps.

**Common errors:** `400 invalid_detection_profile`,
`unsupported_search_profile`, or `search_capacity_too_large`; `409
collection_exists`; `503 search_index_unavailable`.

### `GET /v1/collections`

**Use:** List Collections. **Query parameters:** `limit` integer 1–100, default
50; `cursor` optional opaque continuation token.

```bash
curl -sS "${BASE_URL}/v1/collections?limit=50" -H "${AUTH_HEADER}"
```

**Success:** HTTP 200 with `collections` and nullable `next_cursor`. Pass the
returned cursor unchanged to the same endpoint. **Common errors:** `400
invalid_cursor`; `401 unauthorized`.

### `GET /v1/collections/{collection_id}`

**Use:** Read one Collection. **Path parameter:** `collection_id`, the stable
Collection ID.

```bash
curl -sS "${BASE_URL}/v1/collections/employees" -H "${AUTH_HEADER}"
```

**Success:** HTTP 200 with `collection`, current `person_count`, `face_count`,
and `embedding_contract_id`. **Common errors:** `404 resource_not_found`; `409
collection_model_mismatch` when model-bound use is attempted under an
incompatible active bundle.

### `PATCH /v1/collections/{collection_id}`

**Use:** Update mutable Collection policy. **Path parameter:** `collection_id`.
The `application/json` body may update `name`, `description`, `threshold`, `metadata`, and
`save_face_crops`. Changing crop storage affects later registration requests:
existing crops are neither backfilled nor deleted, and an in-flight registration
may finish using the value it already read. A nested
`search` object may update `capacity_rows`, `max_faces_per_person`, and
`load_policy`; incompatible reductions are rejected. `search_profile` is
immutable through this endpoint because changing it requires an index rebuild.
The nested `detection` object may update any detection-profile field. An
in-flight request keeps its original immutable snapshot; later requests see the
new revision. No existing FaceSample is reprocessed.
Unknown fields and explicit nulls are rejected.

```bash
curl -sS -X PATCH "${BASE_URL}/v1/collections/employees" \
  -H "${AUTH_HEADER}" -H 'Content-Type: application/json' \
  -d '{"threshold":0.45,"detection":{"single_face_selection":"center_largest"}}'
```

**Success:** HTTP 200 with the complete updated `collection`; later requests see
the new policy. **Common errors:** `400` for null/unknown/invalid values; `404`;
`409` for incompatible capacity reductions or model contract; `503` if the
index cannot apply the mutation.

### `DELETE /v1/collections/{collection_id}`

**Use:** Delete a Collection. **Path parameter:** `collection_id`. **Query
parameter:** `force` boolean, default `false`.

An empty Collection is deleted. A non-empty Collection returns
`409 collection_not_empty`; repeat with `force=true` only when deleting all
People and FaceSamples is intended.

```bash
curl -sS -X DELETE "${BASE_URL}/v1/collections/employees?force=true" \
  -H "${AUTH_HEADER}"
```

**Success:** HTTP 204 with no body. **Common errors:** `404`; `409
collection_not_empty`; `503` if the index cannot complete deletion.

Registration that would exceed `capacity_rows` returns
`409 collection_capacity_exceeded` without committing the extra FaceSample.
Registration that would exceed `max_faces_per_person` returns
`409 person_face_limit_exceeded`.

## People and FaceSamples

### `POST /v1/collections/{collection_id}/persons`

Liveness is skipped by default for new Persons and added FaceSamples (`liveness_on_registration=false`). When the administrator enables it, `normal` rejects fake or unsuitable input; `observe` keeps the result and continues. Enrollment review still applies. Rejection lists show the actual `reason` and any liveness result separately.

**Use:** Create a Person and enroll one or more FaceSamples in one request.
**Path parameter:** `collection_id`. Send `multipart/form-data` fields:

- `images` (required and repeatable; default maximum 20);
- `id` (optional; a UUID is generated when omitted);
- `name`, `external_id` (optional);
- `metadata` (optional JSON object encoded as a multipart string, default `{}`);
- `review_mode` (`off`, `standard`, or `strict`; default `off`).
- `embedding_mode` (`server` or `external_trusted`; default `server`);
- `external_embeddings` (required only for `external_trusted`; JSON array with
  exactly one feature vector per `images` part);
- `embedding_contract_id` (required only for `external_trusted`; copy the
  current Collection value exactly).

```bash
curl -sS http://localhost:18097/v1/collections/employees/persons \
  -H "Authorization: Bearer ${INSIGHTFACE_API_KEY}" \
  -F 'id=employee-001' \
  -F 'name=Alice' \
  -F 'external_id=HR-1001' \
  -F 'metadata={"department":"sales"}' \
  -F 'review_mode=standard' \
  -F 'images=@alice1.jpg' \
  -F 'images=@alice2.jpg'
```

Every mode requires a valid image containing at least one detected face and a
finite, correctly sized, L2-normalized embedding. `off` uses the Collection's
single-face strategy and skips configurable quality thresholds. `standard` and `strict`
require exactly one face; `standard` additionally applies the configured
minimum face size, detector score, quality, and pose rules. `strict` applies
`standard`, then requires the candidate's maximum similarity to its Person's
existing samples to be strictly greater than its maximum similarity to every
other Person. Similarity is evaluated with the Collection's pinned search
profile; a tie is rejected.

When a Person has no existing FaceSample, the first standard-quality candidate
bootstraps that Person and skips the similarity comparison. Later candidates in
the same multipart request use earlier accepted candidates as class-in
references. A batch can partially succeed:

With `embedding_mode=server`, the service aligns each accepted face, extracts
its recognition feature, and L2-normalizes it. With
`embedding_mode=external_trusted`, the image is still decoded, detected, and
subjected to the same `review_mode` rules, but the recognition model is not
run. In `off`, the trusted caller asserts that vector `i` belongs to the largest
face in image part `i`. There is no automatic fallback to server extraction;
image and vector counts must match. Invalid or rejected images do not cause
their supplied vectors to be enrolled.

The external vector must contain finite numeric values, be nonzero, match the
Collection dimension and `embedding_contract_id`, and have L2 norm within
`1.0 ± 0.0002`. A vector outside that tolerance is rejected for that image as
`invalid_external_embedding`; it is not silently repaired or replaced. A
passing vector is normalized once more after FP32 conversion to remove small
floating-point drift before SQLite commit.
`strict` review performs class-in/class-out comparisons using that final
external vector. A trusted caller is solely responsible for ensuring that the
vector was extracted from the paired image with the declared pipeline; the
server deliberately does not re-extract a feature to prove the association.

```json
{
  "person": {"id": "employee-001", "face_count": 1},
  "faces": [{"id": "a-face-uuid", "quality": {"score": 0.91}}],
  "rejected_images": [
    {"index": 1, "filename": "alice2.jpg", "reason": "multiple_faces"}
  ],
  "request_id": "a-uuid"
}
```

Current rejection reasons include `invalid_image`, `image_too_large`,
`face_not_found`, `multiple_faces`, `face_too_small`, `low_detection_score`,
`low_quality`, `extreme_pose`, `invalid_embedding`, and
`identity_similarity_conflict`. A strict similarity rejection also reports
`same_person_similarity`, `other_person_similarity`, `other_person_id`, and
`matched_face_id`. If no image is accepted, the request returns
`422 registration_failed` and no Person is created.

**Success:** HTTP 201 with `person`, accepted `faces`, `rejected_images`, and
`request_id`. Partial success is still HTTP 201.

**Common errors:** `400` invalid ID/metadata or too many images; `404`
Collection; `409` Person/external-ID, embedding-contract, capacity, or
per-Person limit conflict; `413`; `422 registration_failed`; `503
search_index_unavailable`. If a 503 contains `write_committed: true`, do not
blindly retry—read the Person first.

### `GET /v1/collections/{collection_id}/persons`

**Use:** List or filter People. **Path parameter:** `collection_id`. **Query
parameters:** `limit` 1–100, default 50; opaque `cursor`; optional `search` up to
200 characters matching Person ID, name, or external ID.

```bash
curl -sS "${BASE_URL}/v1/collections/employees/persons?limit=50&search=alice" \
  -H "${AUTH_HEADER}"
```

**Success:** HTTP 200 with `persons` and nullable `next_cursor`. **Common
errors:** `400 invalid_cursor`; `404` Collection.

### `GET /v1/collections/{collection_id}/persons/{person_id}`

**Use:** Read one Person. **Path parameters:** `collection_id` and `person_id`.

```bash
curl -sS "${BASE_URL}/v1/collections/employees/persons/alice" \
  -H "${AUTH_HEADER}"
```

**Success:** HTTP 200 with `person`, including current `face_count` and
timestamps. **Common errors:** `404` Collection or Person.

### `PATCH /v1/collections/{collection_id}/persons/{person_id}`

**Use:** Update Person display data. **Path parameters:** `collection_id` and
`person_id`. The JSON body accepts `name`, `external_id`, and object `metadata`;
unknown fields are rejected and `metadata` cannot be null.

```bash
curl -sS -X PATCH "${BASE_URL}/v1/collections/employees/persons/alice" \
  -H "${AUTH_HEADER}" -H 'Content-Type: application/json' \
  -d '{"name":"Alice Chen","metadata":{"department":"sales"}}'
```

**Success:** HTTP 200 with the complete updated `person`. **Common errors:**
`400` invalid body; `404`; `409 external_id_exists`.

### `DELETE /v1/collections/{collection_id}/persons/{person_id}`

**Use:** Delete one Person, all FaceSamples, embeddings, and optional crops.
**Path parameters:** `collection_id` and `person_id`.

```bash
curl -sS -X DELETE "${BASE_URL}/v1/collections/employees/persons/alice" \
  -H "${AUTH_HEADER}"
```

**Success:** HTTP 204 with no body; later searches cannot return the deleted
Person. **Common errors:** `404`; `503 search_index_unavailable`.

### `POST /v1/collections/{collection_id}/persons/{person_id}/faces`

Liveness is skipped by default for new Persons and added FaceSamples (`liveness_on_registration=false`). When the administrator enables it, `normal` rejects fake or unsuitable input; `observe` keeps the result and continues. Enrollment review still applies. Rejection lists show the actual `reason` and any liveness result separately.

**Use:** Add FaceSamples to an existing Person. **Path parameters:**
`collection_id` and `person_id`. Add repeatable multipart `images`.
`review_mode` has the
same values, default, registration rules, and partial-success response described
above. `embedding_mode`, `external_embeddings`, and `embedding_contract_id`
also have exactly the same semantics as Person creation.

See the [User Guide](user-guide.md) for the complete
request example and trust-boundary guidance.

```bash
curl -sS "${BASE_URL}/v1/collections/employees/persons/alice/faces" \
  -H "${AUTH_HEADER}" -F 'review_mode=standard' \
  -F 'images=@alice-2.jpg' -F 'images=@alice-3.webp'
```

**Success:** HTTP 201 with accepted `faces` and `rejected_images`; partial
success is allowed. **Common errors:** the same registration, capacity,
contract, size, quality, and index errors as Person creation, plus `404` Person.

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces`

**Use:** Page through FaceSample metadata. **Path parameters:** `collection_id`
and `person_id`. **Query parameters:** `limit` 1–100, default 50, and opaque
`cursor`. Stored embeddings
and crop bytes are not returned. Each item has `has_crop: true` only when a
stored crop is available.

```bash
curl -sS "${BASE_URL}/v1/collections/employees/persons/alice/faces?limit=50" \
  -H "${AUTH_HEADER}"
```

**Success:** HTTP 200 with `faces` and nullable `next_cursor`. **Common errors:**
`400 invalid_cursor`; `404` Collection or Person.

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}/image`

**Use:** Download an optional saved crop for administration. **Path parameters:**
`collection_id`, `person_id`, and `face_id`. Returns the stored 112×112
bounding-box crop as `image/jpeg`. The endpoint requires the same
Bearer authentication as other non-health API calls and returns
`Cache-Control: no-store`. Clients should not treat it as an original image. If
the FaceSample exists but has no stored crop, the server returns a not-found
error rather than synthesizing or reconstructing an image.

```bash
curl -sS http://localhost:18097/v1/collections/employees/persons/employee-001/faces/face-uuid/image \
  -H "Authorization: Bearer ${INSIGHTFACE_API_KEY}" \
  -o face-crop.jpg
```

**Success:** HTTP 200 JPEG with `Cache-Control: no-store`; this response has no
JSON `request_id`, so use the `x-request-id` header. **Common errors:** `404`
FaceSample or `face_image_not_found`; `401 unauthorized`.

### `DELETE /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}`

**Use:** Delete one FaceSample, embedding, and optional crop. **Path parameters:**
`collection_id`, `person_id`, and `face_id`.

```bash
curl -sS -X DELETE "${BASE_URL}/v1/collections/employees/persons/alice/faces/face-uuid" \
  -H "${AUTH_HEADER}"
```

**Success:** HTTP 204 with no body; the row is removed from the active index
before success. **Common errors:** `404`; `503 search_index_unavailable`.

Successful face and Person deletions synchronously update the active search
generation. A later search in the same process cannot return the deleted row.

## Search

### `POST /v1/collections/{collection_id}/search`

With liveness enabled in `normal`, fake or unsuitable query input returns HTTP 422 `liveness_fake` or `liveness_input_rejected` and `error.details.liveness`; the search does not run. This differs from a successful empty match list. `observe` continues searching and returns liveness on the query face.

**Use:** Search a Collection using the selected face in one query image.
**Path parameter:** `collection_id`. Multipart fields:

- `image` (required);
- `limit` (optional, 1–100, default 5);
- `threshold` (optional `0.0..1.0`; defaults to the Collection threshold).

The Collection profile selects the input face, which is searched against every FaceSample. Each
Person receives its highest FaceSample score. Only People at or above threshold
are returned, in descending score order. No match is `matches: []`; no usable
query face is `422 face_not_found`.

```bash
curl -sS http://localhost:18097/v1/collections/employees/search \
  -H "Authorization: Bearer ${INSIGHTFACE_API_KEY}" \
  -F 'image=@unknown.jpg' \
  -F 'limit=5'
```

Example match:

```json
{
  "person": {
    "id": "employee-001",
    "name": "Alice",
    "external_id": "HR-1001",
    "metadata": {"department": "sales"}
  },
  "similarity": 0.8642,
  "matched_face_id": "a-face-uuid"
}
```

**Success:** HTTP 200 with `searched_face`, ordered `matches`, effective
`threshold`, `processing_ms`, and `request_id`. No match is `matches: []`.

**Common errors:** `404` Collection; `409 collection_model_mismatch`; `413`;
`422 invalid_image` or `face_not_found`; `503 search_index_unavailable` or
`request_timeout`.

## RTSP Monitors

A Monitor is a persistent server-side RTSP recognition task. Its configuration is
stored in SQLite and an enabled Monitor resumes when the Server restarts. Video
frames are never saved. Recent events live only in a bounded memory buffer and
are lost on restart. The decoder retains only the newest frame, so slow inference
reduces the effective rate instead of building a delayed frame queue.

### `POST /v1/monitors`

**Use:** Create and optionally start a persistent Monitor. **Body:** Send
`application/json` with the configuration below. `source.url` accepts only
`rtsp://` or `rtsps://`; credentials are AES-GCM encrypted under `/data` and the
API only returns a redacted source.

```json
{
  "id": "front-gate",
  "name": "Front gate",
  "description": "Main entrance",
  "enabled": true,
  "source": {"type": "rtsp", "url": "rtsp://viewer:secret@camera.example/live"},
  "collection_id": "employees",
  "inference_fps": 2.0,
  "match_threshold": null,
  "event_buffer_size": 1000,
  "event_policy": {
    "confirm_frames": 3,
    "absence_timeout_seconds": 3.0,
    "cooldown_seconds": 10.0,
    "emit_unknown": true
  },
  "preview_enabled": false
}
```

`match_threshold: null` inherits the Collection threshold.
`event_buffer_size` is 10–10000. Web preview is deliberately off by default;
recognition and event collection do not require a viewer.

```bash
curl -sS "${BASE_URL}/v1/monitors" -H "${AUTH_HEADER}" \
  -H 'Content-Type: application/json' -d @monitor.json
```

**Success:** HTTP 201 with `monitor`, its redacted source, effective defaults, and
runtime summary. **Common errors:** `400 invalid_request`; `404` Collection;
`409 monitor_exists`; `429 monitor_limit_exceeded`.

### `GET /v1/monitors`

**Use:** List persistent Monitor configurations and compact runtime summaries.
**Query:** `limit` is 1–100 (default 50); `cursor` is an opaque value returned as
`next_cursor` and must not be parsed or modified by clients.

```bash
curl -sS "${BASE_URL}/v1/monitors?limit=50" -H "${AUTH_HEADER}"
```

**Success:** HTTP 200 with ordered `monitors` and nullable `next_cursor`.
**Common errors:** `400 invalid_cursor` for an invalid, altered, or wrong-scope
cursor; `401 unauthorized` when authentication is enabled.

### `GET /v1/monitors/{monitor_id}`

**Use:** Read one persisted Monitor configuration and its latest runtime summary.
**Path:** `monitor_id` is the caller-selected ID used during creation. The
returned RTSP URL omits user information and query values.

```bash
curl -sS "${BASE_URL}/v1/monitors/front-gate" -H "${AUTH_HEADER}"
```

**Success:** HTTP 200 with `monitor`, including `event_policy`,
`preview_enabled`, timestamps, and `runtime`. **Common errors:** `404
monitor_not_found`; `401 unauthorized`.

### `PATCH /v1/monitors/{monitor_id}`

**Use:** Partially update a Monitor; its `id` cannot change. **Body:** Supply one
or more mutable fields from create. `event_policy` itself is partial. Send a new
`source` only when rotating the RTSP URL or credentials. Set
`match_threshold` to `null` to return to the Collection default.

```bash
curl -sS -X PATCH "${BASE_URL}/v1/monitors/front-gate" \
  -H "${AUTH_HEADER}" -H 'Content-Type: application/json' \
  -d '{"inference_fps":1.5,"event_policy":{"confirm_frames":5}}'
```

Changing source, Collection, cadence, threshold, or event policy restarts that
Monitor task. Set `enabled` to `false` or `true` to stop or start it. Name,
description, preview, and buffer-size changes apply without a task restart.

**Success:** HTTP 200 with the complete updated `monitor`. **Common errors:**
`400 invalid_request`; `404` Monitor or Collection; `429
monitor_limit_exceeded`.

### `DELETE /v1/monitors/{monitor_id}`

**Use:** Permanently remove one Monitor configuration. **Path:** `monitor_id`.
This stops its decoder and inference threads, releases the RTSP connection, and
discards its in-memory state and events. It does not delete its Collection.

```bash
curl -sS -X DELETE "${BASE_URL}/v1/monitors/front-gate" \
  -H "${AUTH_HEADER}"
```

**Success:** HTTP 204 with no body. **Common errors:** `404 monitor_not_found`;
`401 unauthorized`.

### `GET /v1/monitors/{monitor_id}/state`

With liveness enabled in `normal`, blocked faces show `status: liveness_blocked` and their separate liveness result. They increment `liveness_blocked_faces`, not `unknown_faces`, and do not trigger person/unknown entry events. `observe` continues recognition. Input rejection is displayed separately from a fake result.

**Use:** Poll the live state needed by headless clients or the Web UI.
**Result fields:** `status`, `connected`, source dimensions/FPS, configured and
actual inference rate, processing time, skipped frames, current recognized and
unknown faces, preview viewers, reconnect count, and last safe error. Embeddings
and source credentials are never included.

```bash
curl -sS "${BASE_URL}/v1/monitors/front-gate/state" -H "${AUTH_HEADER}"
```

**Success:** HTTP 200 with `state`; a disabled Monitor normally reports
`stopped`. **Common errors:** `404 monitor_not_found`; `401 unauthorized`.

### `GET /v1/monitors/{monitor_id}/events`

**Use:** Pull recent enter, exit, error, and recovery events without maintaining
a long-lived connection. **Query:** `limit` is 1–1000 (default 100); pass the
previous `next_cursor` on the next poll. The cursor is an opaque signed string
containing an internal stream epoch and sequence.

The first call without a cursor returns the newest events up to `limit`.
Subsequent calls return later events. `truncated: true` means the client fell
behind the bounded ring; `stream_reset: true` means the task restarted and the
old cursor belongs to another epoch. Events are not durable and are lost when
the process restarts.

```bash
curl -sS "${BASE_URL}/v1/monitors/front-gate/events?limit=100" \
  -H "${AUTH_HEADER}"
```

**Success:** HTTP 200 with `events`, `next_cursor`, `has_more`, `truncated`, and
`stream_reset`. **Common errors:** `400 invalid_cursor`; `404
monitor_not_found`; `401 unauthorized`.

### `GET /v1/monitors/{monitor_id}/preview.mjpeg`

**Use:** Open an optional raw MJPEG preview. **Authentication:** Send the same
Bearer header as every other API call; never put the API key in a URL. The
endpoint returns unannotated `multipart/x-mixed-replace` JPEG frames. A client
uses `/state` to draw boxes and labels.

JPEG encoding runs lazily only while `preview_enabled` is true and at least one
viewer is connected. Closing the preview does not stop recognition. Clients
should reconnect with bounded backoff after transport loss.

**Success:** HTTP 200 as a long-lived binary stream, not JSON. **Common errors:**
`409 preview_disabled`; `503 stream_unavailable`; `404 monitor_not_found`; `401
unauthorized`.


Model, model-component, Collection and FaceSample responses omit `model_version`; `model_id` identifies a package or component according to the response context. Existing Collections retain their `embedding_contract_id`; new Collections use a contract without model-version metadata. Trusted external callers should use the contract ID returned by their target Collection.
