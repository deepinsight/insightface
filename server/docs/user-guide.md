# InsightFace Server user guide

**Languages:** English · [中文](user-guide.zh-CN.md) · [日本語](user-guide.ja.md) · [Deutsch](user-guide.de.md) · [Español](user-guide.es.md) · [Français](user-guide.fr.md) · [Русский](user-guide.ru.md) · [Português](user-guide.pt.md) · [한국어](user-guide.ko.md)

This is the step-by-step operating guide for first-time users. It starts with an
empty checkout and ends with a working Collection, enrolled Person, and search
result. The same operations are available through the Web UI, `/v1` API, and
Python SDK. For every HTTP field and response, open the
[API usage guide](api.md).

For liveness, see [configuration, model installation and result meanings](#optional-liveness-addon). The workflow sections below also explain how it affects each operation.

## Start here: from zero to a working server

You need a Linux x86_64 host with Docker Engine and Docker Compose. A CUDA
deployment additionally needs a supported NVIDIA driver and NVIDIA Container
Toolkit. Do not install host CUDA, cuDNN, ONNX Runtime, Python, or OpenCV.

CPU example:

Run from the repository root with `server/config/server.toml` present. Server and the model installer run as root (`0:0`). Compose creates `server/.models` if missing and mounts it once at `/models` with write access; `addons/` is created when an addon download is requested. No UID/GID exports or manual addon-directory or permission setup is needed.

```bash
docker compose -f server/deploy/compose.cpu.yml pull
docker compose -f server/deploy/compose.cpu.yml run --rm models install buffalo_l
docker compose -f server/deploy/compose.cpu.yml up -d
curl -fsS http://127.0.0.1:18097/v1/health
```

Optionally install and configure liveness by using this model-install command instead:

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license --enable-liveness
```

The Server does not need to be running. On a new deployment, the next `up -d` starts with liveness enabled; if Server is already running, use `docker compose -f server/deploy/compose.cpu.yml restart server`. `up -d` alone does not reload saved settings. For CUDA, use `compose.cuda12.yml`.

For NVIDIA GPU, replace `compose.cpu.yml` with `compose.cuda12.yml` and use port
`18098`. The model installer shows the model license before download. Public
InsightFace pretrained models are restricted to non-commercial research unless
you have a separate commercial license.

The bundled Compose files default to `auth_enabled=false` for isolated evaluation.
No API key field is required in that mode, and the Web UI hides its key control.
Before exposing the service to other users or networks, enable authentication
before startup:

```bash
export INSIGHTFACE_AUTH_ENABLED=true
export INSIGHTFACE_API_KEY='replace-with-a-long-random-secret'
docker compose -f server/deploy/compose.cpu.yml up -d
```

Open `http://SERVER:18097/` for CPU or `http://SERVER:18098/` for CUDA. Complete
the first workflow in this order: check **Dashboard**, create a Collection,
register one Person with at least one clear image, then use **Search** with a
different image of that Person. A successful no-match is an empty list; it is
not a server failure. Stop with `docker compose ... down` without `-v`; adding
`-v` permanently removes the named data volume.

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

System distinguishes verified installation (`installed`), current execution (`enabled`), saved next-start configuration (`configured_enabled`) and pending restart (`restart_required`). Downloading or saving does not change running inference. To disable, save `inference.addons=[]` and `addons.auto_download=[]` in the same file and manually restart. The Web action leaves the enrollment setting unchanged; its default remains `liveness_on_registration=false`.

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
embeddings usable. Database migrations preserve existing samples and embeddings; liveness
does not change the recognition model digest or `embedding_contract_id`.

### Web download permissions

The supplied Compose files use a single writable `/models` mount, with
`create_host_path: true` for both Server and the model installer. They run as
root (`0:0`) with Docker's default capabilities; no `cap_drop: [ALL]` override
is applied. The container root filesystem stays read-only, and
`no-new-privileges` remains enabled. This simplifies shared model access without
host UID/GID or `chmod 777` setup. Root can modify files in the writable mounts;
newly downloaded files may be owned by root on the host.

Both services mount the whole existing `server/config` directory writable at
`/etc/insightface` so the Web action and `--enable-liveness` can atomically save
`server.toml`. Keep this directory and file present; Compose does not create the
configuration source. For custom deployments, use the actual model/configuration
paths and equivalent writable directory mounts in both services. Read-only custom
mounts can serve existing models, but cannot support Web downloads or
configuration saves.

Use `compose.cuda12.yml` for CUDA. An existing model file alone does not enable
liveness. After a successful Web action, run
`docker compose -f server/deploy/compose.cpu.yml restart server` to apply the saved
settings. Recreate containers after changing mounts, user, capabilities, or proxy
environment variables. Set `HTTP_PROXY`, `HTTPS_PROXY` and `NO_PROXY` before
creation if downloads need a proxy; Compose passes them to Server and model tool.
Use a proxy LAN address reachable from the container; its `127.0.0.1` does not refer to the Mac.
The action uses existing API-key authentication. Without authentication, users
who can reach the API can also prepare liveness. The action uses the fixed published model; it accepts no custom download URL and does not switch base models.

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

In the Web UI, both new-person enrollment and adding samples display the actual
rejection `reason` first, with any liveness result on a separate line. For
example, `low_quality` can appear alongside a passed liveness result; passing
liveness does not bypass enrollment quality checks.

### RTSP and Web UI

RTSP faces blocked in `normal` have outer `status: liveness_blocked` and no
identity. They count toward `liveness_blocked_faces`, not `unknown_faces`, do not
emit person/unknown-enter events, and restart identity confirmation. Runtime
liveness errors clear stale displayed identities. `observe` continues matching.
The Web UI displays liveness and distinct rejection states; `/v1/models` and
`/v1/system` report enabled addons separately from the base model files.

## 1. Sign in and check readiness

Open `http://SERVER:18097/` for CPU or `http://SERVER:18098/` for CUDA 12. If authentication is enabled, choose **Configure API key**, paste the key supplied by the operator, and select **Use for this tab**. The browser keeps it only in memory; reloading or closing the tab clears it.

Check **Dashboard** or **System** before enrolling data. The service, database, model and provider must be ready. A CUDA deployment must report `CUDAExecutionProvider`; it never silently falls back to CPU.

The Dashboard always shows **Liveness enabled** or **Liveness disabled** beneath the model name. System displays the installed model, current runtime state and any pending restart separately.

## 2. Create a Collection

Open **Collections**, choose **New collection**, and set:

- a stable ID such as `employees`;
- a display name and optional metadata;
- the default cosine threshold, initially `0.4`;
- a search profile supported by the current host;
- capacity and maximum FaceSamples per Person;
- detector input sizes, detector/NMS thresholds, and a single-face strategy;
- optional 112×112 `bounding-box crop` JPEG storage—not an aligned recognition
  input—disabled by default.

A Collection is pinned to the active model identity, digest, embedding dimension and preprocessing version. Its detection profile starts as a copy of the system profile and may be changed later; each update affects the next request and increments `detection_revision`, but does not reprocess existing FaceSamples. `largest` prefers area. `center_largest` maximizes `area - 2.0 × squared pixel distance from the face-box center to the image center`; detection confidence is not part of this score.

## 3. Register a Person

Open **People**, select a Collection, then **Register person**. Provide an optional stable person ID, name, external ID and JSON metadata. Drop one or more JPEG, PNG, WebP, or BMP images.

Enrollment review modes are:

- `off`: use the Collection single-face strategy; multiple faces are allowed;
- `standard`: require one usable face and apply size, detection, sharpness, brightness and pose checks;
- `strict`: apply standard checks and require the sample's best within-person similarity to exceed its best outside-person similarity.

Batch enrollment supports partial success. Review each rejected image and its
reason before retrying; the service does not retain rejected originals. When
crop storage is enabled, only a `bounding-box crop` resized to 112×112 is
stored—not the original upload or the aligned recognition input.

Trusted systems may send a precomputed, L2-normalized embedding using `external_trusted`. An image is still required for detection and quality review, but the server does not re-extract the embedding. The embedding contract must exactly match the Collection.

Liveness is skipped by default for new Persons and added FaceSamples (`liveness_on_registration=false`). When the administrator enables it, `normal` rejects fake or unsuitable input; `observe` keeps the result and continues. Enrollment quality review follows the selected `review_mode`. Rejection lists show the actual `reason` and any liveness result separately.

## 4. Detect and compare

Use **Detect** to upload one image and inspect boxes, five landmarks, confidence and heuristic quality. No face is a successful result with an empty list.

Use **Compare** to upload source and target images. Select the system or a Collection detection profile; its strategy chooses one usable face in each image. The result contains raw cosine `similarity`, the selected `threshold`, and `matched`. Similarity is not a probability. If either image has no usable face, the API returns `422 face_not_found`.

With liveness enabled, each evaluated face includes `liveness.status`, `liveness.is_live` and `liveness.live_score`. Detect returns HTTP 200 for fake and `input_rejected` results as well; it does not extract recognition features. `input_rejected` means there is insufficient image area around the face; `liveness.reason` explains how to adjust the image. An omitted `liveness` means it was not evaluated.

Liveness runs before recognition on the sides selected by `liveness_compare_scope` (`both`, `source` or `target`). In `normal`, a blocked side returns HTTP 422 `liveness_fake` or `liveness_input_rejected`, with `error.details.liveness` and `error.details.side`; there is no similarity result. `observe` continues the comparison and includes the liveness result on each evaluated face.

## 5. Search a Collection

Open **Search**, select the Collection, upload a query image, set a result limit and optionally override the threshold. The Collection detection profile chooses the query face. Results are sorted by similarity; a Person's score is the maximum score among that Person's FaceSamples. No match is a successful empty list.

Newly accepted FaceSamples are committed to SQLite and then added to the in-memory index before the successful response is returned. Deletions update both stores. On restart the index is rebuilt from SQLite, which remains authoritative.

With liveness enabled in `normal`, fake or unsuitable query input returns HTTP 422 `liveness_fake` or `liveness_input_rejected` and `error.details.liveness`; the search does not run. This differs from a successful empty match list. `observe` continues searching and returns liveness on the query face.

## 6. RTSP camera monitoring

Open **Camera monitoring** and choose **New Monitor**. Give the task an ID and
name, enter an `rtsp://` or `rtsps://` source, select a Collection, and choose the
inference rate and optional match threshold. The event settings control how many
consecutive observations confirm a face, when absence creates an exit event, the
duplicate-event cooldown, and how many recent events stay in memory.

**Web video preview is off by default.** Enable it only when an operator needs a
visual check. Recognition and events continue without a preview. When enabled,
the server sends raw JPEG frames and the Web UI draws green boxes for enrolled
people and amber boxes for detected but unenrolled faces using `/state` results.

The Monitor runs on the server independently of the browser. Closing the page
does not stop it, and enabled Monitors resume after a Server restart. Use
**Start/Stop** to change `enabled`, **Edit** to rotate the RTSP source or tune its
settings, and **Delete** to remove the task. The decoder keeps only the newest
frame; if processing exceeds the requested interval, stale frames are skipped
instead of queued.

Monitor configuration is stored in SQLite. RTSP credentials are encrypted in
`/data` and are never returned by the API. Video frames are not saved. Recent
enter/exit/error/recovery events exist only in a bounded in-memory ring and are
lost on restart. Use HTTPS when the UI/API crosses an untrusted network and
restrict Monitor administration to trusted operators.

With liveness enabled in `normal`, blocked faces show `status: liveness_blocked` and their separate liveness result. They increment `liveness_blocked_faces`, not `unknown_faces`, and do not trigger person/unknown entry events. `observe` continues recognition. Input rejection is displayed separately from a fake result.

## 7. Update and delete data

Collections and Persons can be edited from their lists. Deleting a FaceSample removes its embedding and optional crop. Deleting a non-empty Collection requires explicit force confirmation. Back up `/data` before bulk or destructive maintenance.

## 8. API and Python SDK

The developer OpenAPI schema explorer is at `/docs`; task-oriented API instructions are in this Help manual. Every API response carries `x-request-id`; include it when reporting a problem.

```python
from insightface_server import Client

client = Client("http://localhost:18097", api_key="your-key")
client.create_collection(collection_id="employees", name="Employees", threshold=0.4)
client.add_person("employees", person_id="alice", images=["alice-1.jpg", "alice-2.jpg"])
matches = client.search("employees", "query.jpg", limit=5)
```

## 9. Data, backup and security

- Persist `/data`, the writable model root, and the configuration directory. The container root filesystem stays read-only; API authentication still controls Web model-management access.
- Back up the SQLite database and configured crop storage together while writes are stopped or by using a SQLite-safe snapshot method.
- API keys are stored as hashes. Supplying a different `INSIGHTFACE_API_KEY` on a later start intentionally rotates the active key for that data volume.
- Do not log images, embeddings or keys. Keep broad CORS disabled unless required.
- Model files are not included in the image. The **System** page reads the
  active package's `MODEL.LICENSE` and displays its actual grant. If that file
  is absent, the Server reports the model as non-commercial by default. An
  existing but invalid, mismatched, inactive, or expired signed license still
  prevents startup. Commercial use requires a separate license; visit
  <https://www.insightface.ai>.

## 10. Troubleshooting

`401 unauthorized` means the tab has no current key or the key was rotated. `409 collection_model_mismatch` means the Collection was created with a different model contract. `422 face_not_found` means no usable face was selected. A CUDA startup failure is intentional when the Driver, GPU, model session, provider or warm-up validation fails. Check **System**, container logs and the response `request_id`.

## 11. Models and model licenses

The images do not contain models. The one-shot `models` service installs a
package into `server/.models`, while normal Server startup remains offline:

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models verify buffalo_l
```

Supported public packages are `buffalo_l` (`det_10g.onnx` +
`w600k_r50.onnx`), `buffalo_m` (`det_2.5g.onnx` + `w600k_r50.onnx`),
`buffalo_s` and `buffalo_sc` (`det_500m.onnx` + `w600k_mbf.onnx`),
`antelopev2` (`scrfd_10g_bnkps.onnx` + `glintr100.onnx`), `raccoon_s`
(`det_10g_wo.onnx` + `w600k_mbf.onnx`), and `raccoon_l`
(`det_10g_wo.onnx` + `w600k_r50.onnx`). Server installs only detection and
recognition from each package; a Raccoon verifier is not installed or loaded.
Installation creates
`manifest.json` and signed `MODEL.LICENSE`. Without `--accept-license`, an
interactive terminal asks for confirmation before downloading; noninteractive
commands require the flag and otherwise exit without downloading. `models verify` validates
the package identity, signed license, validity dates, and current authorization;
unlike runtime display fallback, this explicit verification command requires a
signed license file.
Catalog archives come from the dedicated
[`model-zoo` GitHub Release](https://github.com/deepinsight/insightface/releases/tag/model-zoo).

Public InsightFace pretrained models are for non-commercial research unless a
separate commercial license has been issued. A private model can use the same
manifest and offline signed license format. The license identifies `model_id`;
it is a compliance credential, not DRM or a model-file checksum.

## 12. Startup-only configuration

The common startup file is `server/config/server.toml`. Compose mounts its containing directory writable at `/etc/insightface`; the file is `/etc/insightface/server.toml`. Restart the container to apply a saved change. Defaults are:

```toml
[inference]
max_concurrency = "auto" # CPU 4, CUDA 8
addons = []
liveness_mode = "normal"
liveness_threshold = 0.8
liveness_compare_scope = "both"
liveness_on_registration = false

[addons]
auto_download = []

[detection]
input_sizes = [[96, 96], [512, 512]]
threshold = 0.50
nms_threshold = 0.40
single_face_selection = "largest"
max_detected_faces = 100

[web]
disabled = false
```

Dynamic SCRFD runs every configured resolution, maps candidates back to the
source image, merges all candidates, and performs one global NMS. Settings are
read once for inference. The Web liveness action saves next-start settings; it
does not hot-reload the running process. New Collections copy the system
detection profile, after which their profile can be updated independently for
the next request. Stateless Detect and Embeddings use the system profile;
Compare can use the system profile or a selected Collection; enrollment and
search use their Collection.

Set `[web].disabled=true` for API-only operation. `/v1` and `/openapi.json`
remain available, while `/`, `/docs`, guides, and frontend assets are not
registered.

## 13. Exact-search profiles and capacity

The System response advertises only profiles available on the current CPU/GPU.
A Collection fixes one profile when created; a search request cannot change it.

| Profile | Stored representation | Typical availability |
| --- | --- | --- |
| `fp32_v1` | FP32 | CPU and CUDA |
| `fp16_v1` | FP16 | CUDA |
| `bf16_v1` | BF16 | supported CPU or SM80+ CUDA |
| `int8_x736_v1` | INT8, scale 736 | CPU and CUDA; recommended INT8 |
| `int8_x1000_v1` | INT8, scale 1000 | compatibility profile |

All are flat exhaustive searches over every live FaceSample; low-precision
profiles approximate FP32 scores but are not ANN indexes. INT8 dot products
accumulate into INT32. Public similarities and thresholds remain raw cosine.

`capacity_rows` reserves the maximum live rows for that Collection and avoids
routine growth pauses. Approximate vector storage for 512 dimensions is
2,048 bytes per FP32 row, 1,024 per FP16/BF16 row, and 512 per INT8 row,
before IDs and workspaces. Set capacity from an actual memory budget. The
default is `100000`; the deployment guardrail defaults to `10000000`.
`max_faces_per_person` defaults to `20` and limits sample count, not the number
of people.

## 14. CUDA support and fail-fast verification

The CUDA image contains CUDA Runtime 12.9.1, cuDNN 9.24.0, Python 3.11, and
`onnxruntime-gpu==1.27.0`. The host needs only Driver, Docker Engine, NVIDIA
Container Toolkit, and a compatible GPU.

- Turing, Ampere, Ada, and Hopper: Driver R535 or newer.
- Blackwell and RTX 50 series: Driver 570.26 or newer.
- New deployments should prefer a stable R580 or newer driver.

Architecture compatibility is not a claim that every GPU SKU is formally
certified. On every CUDA start the Server checks GPU model, Compute Capability,
Driver, actual CUDA/cuDNN/ORT versions, the presence of
`CUDAExecutionProvider`, real detector and recognizer Sessions, and real
warm-up inference. It audits provider placement and terminates instead of
silently falling back to CPU. Confirm the result on **System** before use.

## 15. Build, upgrade, backup, and recovery

You can build directly from a complete local source directory, including
uncommitted changes or a directory without `.git`. Git commits and pushes
are not prerequisites for building.

```bash
make -C server build-cpu
make -C server build-cuda12
```

After the tests pass, publish the same image that was tested. Committing or
organizing the same source afterwards does not require rebuilding. Changes to
files included in the image, such as code, frontend assets, or bundled user
help, require another build and validation.

Then add `--pull never` to Compose model/install and `up` commands to use the
local image. Builds use pinned base images and locked dependencies, but require
network access for those inputs. The public tags are
`0.3.1-cpu`/`0.3.1-cuda12`; moving `cpu`/`cuda12` tags point to the latest
stable variant, and there is deliberately no `latest` tag.

Before upgrading, stop writes and create a SQLite-safe snapshot of `/data`
plus any crop storage. Keep `/models` and its license files. Start the new
container against a copy first, check migrations and `/v1/health`, then verify
the model contract and a known search. Use `docker compose down` without `-v`;
`docker compose down -v` deletes the named data volume.

### Upgrade to 0.3.1

Version 0.3.1 simplifies Docker deployment: both services run as root, use
Docker's default capabilities, and share one writable model mount. Compose
creates the model root if missing; explicit addon downloads create `addons/`.
No host UID/GID or shared-group preparation is required.

Since 0.3.0, Server supports `raccoon_s` and `raccoon_l` with their model manifests,
optional liveness, Web addon installation, and BMP input. Server loads Raccoon's
detector and recognizer, not its verifier. These features and API response
contracts are unchanged in 0.3.1.

**1.** Update the Server source and Compose files to the 0.3.1 version while keeping
your `server/config/server.toml` settings and deployment overrides. Preserve
the existing model path, `/data` volume name, crop storage, ports, and API key
settings. For custom Compose files, update both the `server` and `models`
service images to `0.3.1-cpu` or `0.3.1-cuda12` as appropriate. Apply the same
Compose files, overrides, and project name you normally use to the commands
below.

Updating image tags alone is insufficient. Update custom Compose files and
overrides to set both services to `user: "0:0"`, remove `cap_drop: [ALL]` and
legacy UID/GID/group settings, mount `/models` once with write access and
`create_host_path: true`, and remove the separate `/models/addons` mount. Keep
the container root filesystem read-only and retain `no-new-privileges`. Preserve
the existing configuration directory and file: both services need the whole
directory writable for atomic saves. Replace the installer's old read-only
single-file configuration mount with the directory mount.
Existing model files and addon caches remain in place; no recursive permission
reset or addon-directory preparation is required for the standard deployment.

**2.** Pull the new images and recreate the Server container. From the repository
root, choose the commands for your existing deployment:

CPU:

```bash
docker compose -f server/deploy/compose.cpu.yml pull server models
docker compose -f server/deploy/compose.cpu.yml up -d --no-build --force-recreate server
curl -fsS http://127.0.0.1:18097/v1/health
```

CUDA:

```bash
docker compose -f server/deploy/compose.cuda12.yml pull server models
docker compose -f server/deploy/compose.cuda12.yml up -d --no-build --force-recreate server
curl -fsS http://127.0.0.1:18098/v1/health
```

If you build locally, build the 0.3.1 images first and use
`up -d --no-build --pull never --force-recreate server` instead of pulling.
`docker compose restart` alone does not switch to a new image or apply mount
changes.

**3.** Startup applies database migrations automatically. Wait for `/v1/health` to
report `ready` and version `0.3.1`, then check **System** for the expected
model and execution provider. Confirm that your existing Collections and
people are present and try a known search. Keeping the same model and
embedding contract preserves samples, embeddings, and Collection contract
IDs; no re-enrollment is needed.

**Liveness is optional after upgrading.** The shipped configuration and older
configurations without addon keys both leave it disabled, so upgrading alone
requires no liveness download. Server startup never downloads models. To enable
it, follow [the liveness setup](#optional-liveness-addon): prepare the
[Web download mounts and permissions](#web-download-permissions), choose
**System → Liveness → Download and enable after restart**, wait for successful
installation and configuration saving, then manually restart Server. The
defaults are `normal`, threshold `0.8`, and `liveness_on_registration=false`.
The model stays at `<models_dir>/addons/liveness.onnx`.

**Using Raccoon is a separate model change.** Upgrading Server keeps your
current model package. To adopt `raccoon_s` or `raccoon_l`, install the selected
package in a separate model directory using the
[model installation instructions](#11-models-and-model-licenses), then configure
a deployment to use it. Collections must match the new model's embedding
contract; create compatible Collections and re-enroll, or perform a separate
data migration. The Web UI does not switch base model packages.

**API and SDK compatibility since 0.3.0:** Model, Collection, and FaceSample results no
longer include `model_version`; model identity uses `model_id`, and Collection
compatibility uses `embedding_contract_id`. Update clients that require the
removed field and use SDK `0.3.1` when upgrading the supplied Python client.
When liveness is evaluated, `liveness` contains the core fields `status`, `is_live`, and
`live_score`, plus `reason` only for `input_rejected`; it is omitted when not evaluated. See the
[liveness response and error rules](#detection-recognition-and-errors) before
enabling it for recognition requests.

For network exposure, terminate HTTPS at a trusted reverse proxy, allow only
required origins rather than broad CORS, apply edge rate/body/time limits, and
protect the data volume and backups as biometric data. The Server has one
undifferentiated API key in phase one and is not a multi-tenant authorization
system.


Model packages are identified by name, such as `buffalo_l`, with no separate `model_version`. Server upgrades that keep the same recognition model and embedding contract preserve existing Collection contract IDs, samples and embeddings; re-enrollment is unnecessary. Changing recognition models is a separate migration, and an incompatible Collection returns `collection_model_mismatch` for enrollment and search. New Collections use an embedding contract without a model-version field.
