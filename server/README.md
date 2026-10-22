# InsightFace Server

**Languages:** English · [中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [Русский](README.ru.md) · [Português](README.pt.md) · [한국어](README.ko.md)

> **One GPU. 50M+ face vectors. High-speed INT8-quantized search with no material accuracy loss.**

**A self-hosted face-recognition server with a Web UI, a straightforward REST
API, SQLite, and local CPU or NVIDIA GPU inference in one container.**

```text
upload an image -> detect, check liveness, compare, enroll, or search
```

> **Model license:** Public InsightFace pretrained models are generally
> available for non-commercial research only. Commercial use requires separate
> authorization from [InsightFace](https://www.insightface.ai).

InsightFace Server is a simpler, privacy-focused alternative to AWS Rekognition
for common face-recognition workflows on infrastructure you control. Images,
embeddings, models, and indexes can remain inside your network. It is **not** an
AWS-compatible replacement and does not implement SigV4, IAM, Region, or AWS
resource semantics.

Current release: **0.3.1**, Linux x86_64.

| Runtime | Image |
| --- | --- |
| CPU | `ghcr.io/deepinsight/insightface-server:0.3.1-cpu` |
| NVIDIA GPU | `ghcr.io/deepinsight/insightface-server:0.3.1-cuda12` |

The moving `cpu` and `cuda12` tags identify the latest stable release in each
runtime family. There is no ambiguous `latest` tag. See
[Maintainer Guide — English](docs/maintainer-guide.md) for the release policy.

**0.3.1 deployment update:** Server and the model installer now run as root,
with one writable model mount and automatic addon-directory creation on download.
No host UID/GID or shared-group setup is needed. Update Compose as well as the
image tag; keep existing models, configuration, and data. Since 0.3.0, Server also
supports Raccoon manifests, optional liveness, and BMP input; API/SDK results omit
`model_version`. See [upgrade steps](docs/user-guide.md#upgrade-to-031).

![InsightFace Server dashboard in English](docs/images/customer/dashboard-en.jpg)

## Highlights

- SCRFD face detection, five landmarks, alignment, ArcFace embeddings, L2
  normalization, raw cosine similarity, and exact 1:N Person search.
- Multi-resolution detection with one merged NMS and `largest` or
  `center_largest` single-face selection.
- `Collection -> Person -> FaceSample` storage with model-bound Collections,
  multi-image enrollment, partial success, metadata, and explicit rejection
  reasons.
- Enrollment `review_mode`: `off`, `standard`, or `strict`; optional
  `external_trusted` precomputed embeddings.
- Exact GPU search with FP32, FP16, BF16, and INT8 vector storage.
- Multilingual Web UI for Dashboard, Collections, People, Detect, Compare,
  Search, RTSP monitoring, System diagnostics, and Help.
- 31 snake_case REST operations under `/v1`, including authenticated
  `/v1/embeddings`, plus a typed lightweight Python SDK.
- Persistent server-side RTSP Monitors with bounded in-memory events,
  independent clients, and optional `preview.mjpeg`; closing the browser does
  not stop monitoring.
- SQLite as the durable source of truth, disposable in-memory exact indexes,
  one writable model mount, a writable configuration directory, persistent `/data`, migrations, health checks, and
  strict CUDA startup validation without silent CPU fallback.
- JPEG, PNG, WebP, and BMP input; original uploads are not retained by default.

Liveness is disabled by default: `inference.addons = []` and `addons.auto_download = []` in `server/config/server.toml`. **System → Liveness** downloads and verifies the model, then saves activation for the next manual restart; a verified cached model is reused. Startup never downloads models. An enabled addon missing its model stops startup with an installation hint. Each evaluated face returns the core fields `status`, `is_live`, and `live_score`; only `input_rejected` adds a human-readable `reason`. The default mode is `normal`; registration skips liveness by default (`liveness_on_registration = false`). See [configuration, Web permissions and upgrade instructions](docs/user-guide.md#optional-liveness-addon).

### RTX 5090 GPU search performance

On one NVIDIA GeForce RTX 5090 (32,607 MiB), the native CUDA exact-flat index
stored up to **58.9M 512-dimensional image vectors in INT8**.

| GPU data type | Maximum image vectors | 10M Top-5 p50 | 10M serial QPS |
| --- | ---: | ---: | ---: |
| FP32 | 15.8M | 12.84 ms | 77.85 |
| FP16 | 30.7M | 6.83 ms | 146.32 |
| BF16 | 30.7M | 6.83 ms | 146.33 |
| INT8 | **58.9M** | **3.84 ms** | **260.81** |

INT8 delivered 3.73x the measured capacity and 3.35x the 10M Top-5 throughput
of FP32. These are GPU-only measurements on the same RTX 5090 with Driver
580.105.08 and CUDA 12.9. Capacity is an isolated native-index ceiling without
loaded ONNX models or Server workload. Speed uses exactly 10M image vectors,
exhaustive GPU-resident Top-5, one query in flight, 10 warm-ups, and 100
measured queries. The index is exact within each stored representation;
quantization can still change scores relative to FP32. A production deployment
must reserve VRAM for models, requests, concurrency, index rebuilds, and
allocator headroom.

### ICCV21-MFR multi-racial MR-ALL accuracy

We evaluated the native search profiles on the multi-racial (MR) test set from
[ICCV21-MFR](../challenges/iccv21-mfr/) using its MR-ALL all-pairs 1:1 protocol
at FAR `1e-6`. Every profile used the same L2-normalized 512-dimensional
`buffalo_l` embeddings extracted once through the Server API; only the stored
vector and search-compute representation changed.

| Search profile | MR-ALL at FAR 1e-6 | Cosine threshold | Difference from FP32 |
| --- | ---: | ---: | ---: |
| FP32 | 91.249107% | 0.407787 | — |
| FP16 | 91.249197% | 0.407787 | +0.000090 pp |
| BF16 | 91.248502% | 0.407787 | -0.000605 pp |
| **INT8** | **91.248005%** | **0.407739** | **-0.001102 pp** |

**INT8 has no material accuracy loss on this benchmark:** at the
challenge-style two-decimal reporting precision, FP32 and INT8 both score
**91.25% MR-ALL**, while the unrounded difference is only 0.0011 percentage
points. This preserves the 3.73x measured capacity and 3.35x 10M Top-5
throughput advantages shown above. The comparison measures vector
storage/search precision, not INT8 model inference.

![Collection management in English](docs/images/customer/collections-en.jpg)

![RTSP Monitor in English; private address redacted](docs/images/customer/monitoring-en.jpg)

## Quick start

Requirements:

- Linux x86_64 with Docker Engine and Docker Compose;
- for CUDA: a supported NVIDIA GPU, NVIDIA Driver, and NVIDIA Container
  Toolkit.

The host does not need Python, OpenCV, ONNX Runtime, CUDA Toolkit, or cuDNN.
Public images do not contain models, customer data, API keys, or production
configuration.

From a complete InsightFace repository checkout, install a model into
`server/.models`:

Run from the repository root with `server/config/server.toml` present. Server and the model installer run as root (`0:0`). Compose creates `server/.models` if missing and mounts it once at `/models` with write access; `addons/` is created when an addon download is requested. No UID/GID exports or manual addon-directory or permission setup is needed.

```bash
docker compose -f server/deploy/compose.cpu.yml pull
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license
```

Optionally install and configure liveness by using this model-install command instead:

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license --enable-liveness
```

The Server does not need to be running. On a new deployment, the next `up -d` starts with liveness enabled; if Server is already running, use `docker compose -f server/deploy/compose.cpu.yml restart server`. `up -d` alone does not reload saved settings. For CUDA, use `compose.cuda12.yml`.

The model tool supports all seven FaceAnalysis packages: `buffalo_l`,
`buffalo_m`, `buffalo_s`, `buffalo_sc`, `antelopev2`, `raccoon_s`, and
`raccoon_l`. It installs only each package's detector and recognizer, writes
`manifest.json` and the signed `MODEL.LICENSE`, and ignores Raccoon's
PrivateFrame-only verifier. Use `models verify` to check the installed package.
Model terms remain separate from the Server source license. Catalog archives
are downloaded from the dedicated
[`model-zoo` GitHub Release](https://github.com/deepinsight/insightface/releases/tag/model-zoo).

Start CPU:

```bash
docker compose -f server/deploy/compose.cpu.yml up -d
curl -fsS http://127.0.0.1:18097/v1/health
```

Start CUDA 12 instead:

```bash
docker compose -f server/deploy/compose.cuda12.yml pull
docker compose -f server/deploy/compose.cuda12.yml \
  run --rm models install buffalo_l --accept-license
docker compose -f server/deploy/compose.cuda12.yml up -d
curl -fsS http://127.0.0.1:18098/v1/health
```

Open `http://SERVER:18097/` for CPU or `http://SERVER:18098/` for CUDA. Create a
Collection, register a Person with one or more photos, and search with another
photo. Stop with `docker compose ... down` without `-v` to retain the database
volume.

The supplied Compose files set authentication to `false` for isolated
evaluation. Before exposing the service to other users or networks:

```bash
export INSIGHTFACE_AUTH_ENABLED=true
export INSIGHTFACE_API_KEY='replace-with-a-long-random-secret'
docker compose -f server/deploy/compose.cpu.yml up -d
```

See the [beginner user guide](docs/user-guide.md) for the complete first-run
workflow.

## Build from source

You can build directly from a complete local source directory, including
uncommitted changes or a directory without `.git`. Git commits and pushes
are not prerequisites for building.

The Dockerfiles copy `server/` and selected inference modules from
`python-package/insightface/`, so the complete source directory is the build context.

CPU:

```bash
make -C server build-cpu
docker compose -f server/deploy/compose.cpu.yml \
  run --rm --pull never models install buffalo_l --accept-license
docker compose -f server/deploy/compose.cpu.yml \
  up -d --no-build --pull never
```

CUDA 12:

```bash
make -C server build-cuda12
docker compose -f server/deploy/compose.cuda12.yml \
  run --rm --pull never models install buffalo_l --accept-license
docker compose -f server/deploy/compose.cuda12.yml \
  up -d --no-build --pull never
```

`--pull never` ensures Compose uses the locally built image. The build still
downloads pinned base images and dependencies; model installation separately
downloads the accepted model package.

## Core behavior

- Similarity is raw cosine, not probability. Thresholds use `0.0..1.0` and
  default to `0.4`.
- A Collection pins the model and embedding contract. A mismatch remains
  visible but enrollment/search returns `collection_model_mismatch`.
- The startup detection profile is copied into new Collections; Collection
  profiles can then be updated independently for subsequent requests.
- Optional face storage saves a 112x112 bounding-box JPEG crop, not the original
  upload or the aligned recognition input, and is off by default.
- SQLite commits are authoritative. Index mutations complete before successful
  enrollment/deletion responses return, and indexes rebuild from SQLite after
  restart.
- Responses carry `x-request-id`; list APIs use opaque signed cursors.

Exact fields, defaults, lifecycle rules, and failure behavior live in the
linked documentation below.

## API and SDK

Main API groups:

- system: `/v1/health`, `/v1/system`, `/v1/models`;
- stateless faces: `/v1/detect`, `/v1/compare`, `/v1/embeddings`;
- Collection, Person, and FaceSample CRUD;
- Collection Person search;
- RTSP Monitor configuration, state, events, and preview.

Use the [complete REST API guide](docs/api.md) for every parameter, response,
error, and example. Interactive OpenAPI remains available at `/docs`.

Python:

```python
from insightface_server import Client

with Client("http://localhost:18097", api_key=None) as client:
    faces = client.detect("photo.jpg")
    matches = client.search("employees", "unknown.jpg", limit=5)
```

See the [User Guide](docs/user-guide.md) for installation, inputs, methods, and
complete SDK workflows.

## Security

Face images and embeddings are biometric data. For network deployment, enable
authentication, terminate HTTPS at a trusted reverse proxy, restrict Docker and
volume access, keep broad CORS disabled, and define backup, retention, deletion,
consent, and incident-response policies. Do not log images, embeddings, RTSP
credentials, or API keys.

The Server does not provide built-in TLS, user accounts, RBAC, cloud IAM, or a
legal-compliance layer. Deployment and security guidance is in the
[User Guide](docs/user-guide.md).

## Phase-one scope

This release does not implement AWS/CompreFace compatibility, CUDA 11, Jetson,
ARM64, Windows containers, TensorRT, Kubernetes, distributed Workers, persistent
Monitor events or recording/NVR, deepfake detection, or demographic
attributes.

## Documentation

- [User Guide](docs/user-guide.md) — complete installation, configuration,
  model, Web UI, SDK, GPU, security, backup, and troubleshooting guidance.
- [REST API Guide](docs/api.md) — every public endpoint, field, behavior,
  result, error, pagination rule, and example.
- [Maintainer Guide — English](docs/maintainer-guide.md) — architecture,
  search internals, testing, contribution rules, and container releases.

The localized User Guide and API Guide are the exact Markdown sources rendered
by the Web UI Help page; only their presentation differs.

## License

See the single [LICENSING.md](LICENSING.md) entry point. In short, the Server
source and Python SDK are MIT licensed; that declaration does not cover model
files, model weights, datasets, or third-party components. Public InsightFace
pretrained models are generally limited to non-commercial research unless
separately authorized. Commercial licensing information:
<https://www.insightface.ai>.
