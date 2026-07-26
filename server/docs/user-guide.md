# InsightFace Server user guide

**Languages:** English · [中文](user-guide.zh-CN.md) · [日本語](user-guide.ja.md) · [Deutsch](user-guide.de.md) · [Español](user-guide.es.md) · [Français](user-guide.fr.md) · [Русский](user-guide.ru.md) · [Português](user-guide.pt.md) · [한국어](user-guide.ko.md)

This is the step-by-step operating guide for first-time users. It starts with an
empty checkout and ends with a working Collection, enrolled Person, and search
result. The same operations are available through the Web UI, `/v1` API, and
Python SDK. For every HTTP field and response, open the
[API usage guide](api.md).

## Start here: from zero to a working server

You need a Linux x86_64 host with Docker Engine and Docker Compose. A CUDA
deployment additionally needs a supported NVIDIA driver and NVIDIA Container
Toolkit. Do not install host CUDA, cuDNN, ONNX Runtime, Python, or OpenCV.

CPU example:

```bash
mkdir -p server/.models
docker compose -f server/deploy/compose.cpu.yml pull
docker compose -f server/deploy/compose.cpu.yml run --rm models install buffalo_l
docker compose -f server/deploy/compose.cpu.yml up -d
curl -fsS http://127.0.0.1:18097/v1/health
```

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

## 1. Sign in and check readiness

Open `http://SERVER:18097/` for CPU or `http://SERVER:18098/` for CUDA 12. If authentication is enabled, choose **Configure API key**, paste the key supplied by the operator, and select **Use for this tab**. The browser keeps it only in memory; reloading or closing the tab clears it.

Check **Dashboard** or **System** before enrolling data. The service, database, model and provider must be ready. A CUDA deployment must report `CUDAExecutionProvider`; it never silently falls back to CPU.

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

Open **People**, select a Collection, then **Register person**. Provide an optional stable person ID, name, external ID and JSON metadata. Drop one or more JPEG, PNG, or WebP images.

Enrollment review modes are:

- `off`: use the Collection single-face strategy; multiple faces are allowed;
- `standard`: require one usable face and apply size, detection, sharpness, brightness and pose checks;
- `strict`: apply standard checks and require the sample's best within-person similarity to exceed its best outside-person similarity.

Batch enrollment supports partial success. Review each rejected image and its
reason before retrying; the service does not retain rejected originals. When
crop storage is enabled, only a `bounding-box crop` resized to 112×112 is
stored—not the original upload or the aligned recognition input.

Trusted systems may send a precomputed, L2-normalized embedding using `external_trusted`. An image is still required for detection and quality review, but the server does not re-extract the embedding. The embedding contract must exactly match the Collection.

## 4. Detect and compare

Use **Detect** to upload one image and inspect boxes, five landmarks, confidence and heuristic quality. No face is a successful result with an empty list.

Use **Compare** to upload source and target images. Select the system or a Collection detection profile; its strategy chooses one usable face in each image. The result contains raw cosine `similarity`, the selected `threshold`, and `matched`. Similarity is not a probability. If either image has no usable face, the API returns `422 face_not_found`.

## 5. Search a Collection

Open **Search**, select the Collection, upload a query image, set a result limit and optionally override the threshold. The Collection detection profile chooses the query face. Results are sorted by similarity; a Person's score is the maximum score among that Person's FaceSamples. No match is a successful empty list.

Newly accepted FaceSamples are committed to SQLite and then added to the in-memory index before the successful response is returned. Deletions update both stores. On restart the index is rebuilt from SQLite, which remains authoritative.

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

- Persist `/data`; mount `/models` read-only.
- Back up the SQLite database and configured crop storage together while writes are stopped or by using a SQLite-safe snapshot method.
- API keys are stored as hashes. Supplying a different `INSIGHTFACE_API_KEY` on a later start intentionally rotates the active key for that data volume.
- Do not log images, embeddings or keys. Keep broad CORS disabled unless required.
- Model files are not included in the image. InsightFace-provided open-source
  pretrained models, including `buffalo_l`, are for non-commercial research use
  only. Commercial use requires a separate license; visit
  <https://www.insightface.ai>. The **System** page displays the same notice.

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
`buffalo_sc` (`det_500m.onnx` + `w600k_mbf.onnx`), and `antelopev2`
(`scrfd_10g_bnkps.onnx` + `glintr100.onnx`). Installation creates
`manifest.json` and signed `MODEL.LICENSE`. Without `--accept-license`, the
tool prints the terms and exits without downloading. `models verify` validates
the package identity, signed license, validity dates, and current authorization.

Public InsightFace pretrained models are for non-commercial research unless a
separate commercial license has been issued. A private model can use the same
manifest and offline signed license format. The license identifies `model_id`;
it is a compliance credential, not DRM or a model-file checksum.

## 12. Startup-only configuration

The common startup file is `server/config/server.toml`. Compose mounts it
read-only at `/etc/insightface/server.toml`; edit it before startup and restart
the container to apply a change. Defaults are:

```toml
[inference]
max_concurrency = "auto" # CPU 4, CUDA 8

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
read once; there is no runtime settings API. New Collections copy the system
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

Users may build both images from the complete repository checkout:

```bash
make -C server build-cpu
make -C server build-cuda12
```

Then add `--pull never` to Compose model/install and `up` commands to use the
local image. Builds use pinned base images and locked dependencies, but require
network access for those inputs. The public tags are
`0.2.0-cpu`/`0.2.0-cuda12`; moving `cpu`/`cuda12` tags point to the latest
stable variant, and there is deliberately no `latest` tag.

Before upgrading, stop writes and create a SQLite-safe snapshot of `/data`
plus any crop storage. Keep `/models` and its license files. Start the new
container against a copy first, check migrations and `/v1/health`, then verify
the model contract and a known search. Use `docker compose down` without `-v`;
`docker compose down -v` deletes the named data volume.

For network exposure, terminate HTTPS at a trusted reverse proxy, allow only
required origins rather than broad CORS, apply edge rate/body/time limits, and
protect the data volume and backups as biometric data. The Server has one
undifferentiated API key in phase one and is not a multi-tenant authorization
system.
