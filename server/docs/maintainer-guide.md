# InsightFace Server maintainer guide

> **Maintainer reference — English only.**
>
> This document covers source architecture, implementation contracts, testing,
> and release work. Operators and API consumers should use the localized
> [User Guide](user-guide.md) and [REST API Guide](api.md). No information
> required for normal installation or use is intentionally kept only here.

## 1. Scope and repository boundaries

The phase-one product is one process and one container:

```text
Web UI
REST API
SQLite
local ONNX Runtime inference
mutable in-memory exact search indexes
server-side RTSP Monitor tasks
```

Primary code lives under `server/`. It may import selected existing modules from
`python-package/insightface/`, but Server work must not modify algorithm,
training, or package behavior unless a separate upstream change explicitly
requires it.

The supported Server runtime is built from the complete source tree; a `.git`
directory is not required. Its Docker image places both `server/backend` and
`python-package` on `PYTHONPATH`. The
backend-only wheel produced during release checks is not a standalone inference
distribution; the independently installable client is the SDK wheel under
`server/sdk/python`.

Do not commit model binaries, signed customer licenses, issuer private keys,
real face images, customer data, generated databases, or production
configuration. The model root is one writable `/models` mount. The Web liveness
installer creates `addons/` on an explicit download and writes the configuration directory;
it does not switch base model packages. `/data` is mutable, persistent runtime
state.

## 2. Source architecture

```text
server/
├── backend/insightface_server/
│   ├── api/          request and response schemas, authentication
│   ├── inference/    provider-independent pipeline and ONNX implementation
│   ├── licensing/    compatibility facade for shared model-license handling
│   ├── models/       manifest, packages, embedding contract
│   ├── search/       Python facade, native ABI, reference implementation
│   ├── services/     application workflows and RTSP tasks
│   └── storage/      migrations, repository, secrets, crop storage
├── native/           C/C++ CPU and CUDA exact-search implementations
├── frontend/         dependency-free Web UI and OpenAPI viewer
├── sdk/python/       lightweight typed synchronous client
├── migrations/       ordered SQLite migrations
├── docker/           pinned CPU and CUDA image definitions
├── deploy/           public Compose definitions
├── config/           startup-only TOML
├── docs/             localized user/API guides and this guide
├── scripts/          manifest, snapshot, smoke, and opt-in validation tools
└── tests/            unit, API, SDK, UI, native, real-model, Docker tests
```

FastAPI owns process lifecycle. Storage and model readiness complete before
readiness becomes healthy. The Web UI calls only public `/v1` operations.
SQLite is authoritative; native indexes are disposable projections.

## 3. Inference lifecycle and concurrency

Each process creates exactly one detector ONNX Runtime Session and one
recognizer Session. Sessions are reused concurrently; a request must never
construct a model Session. Request-specific detector policy and dynamic input
state must remain local to the call.

The process-wide inference limiter is shared by Detect, Compare, Embeddings,
enrollment, Search query extraction, and RTSP frames. `max_concurrency="auto"`
resolves to 4 on CPU and 8 on CUDA. FastAPI requests are asynchronous, while
blocking decode and inference work runs in worker threads. The limiter bounds
model pipelines rather than serializing the whole HTTP request.

ONNX Runtime Sessions support concurrent `run` calls. Do not add a global
inference lock to work around request-local mutable state; eliminate or isolate
that state. Locks remain appropriate for:

- SQLite writes and migrations;
- one Collection's index mutation/rebuild/revision barrier;
- Monitor lifecycle and bounded event-ring mutation;
- cache publication where duplicate construction would violate an invariant.

Dynamic SCRFD evaluates every configured input resolution, maps candidates back
to source-image coordinates, concatenates them, and applies exactly one global
NMS. It does not NMS each resolution independently and merge the winners.
Single-face selection supports:

```text
largest
center_largest
```

`center_largest` maximizes:

```text
area - 2.0 * squared_distance(face_box_center, image_center)
```

Confidence is not part of that selection score.

Startup performs real detector warm-up at every configured resolution and a
real recognizer warm-up, validates the embedding dimension, and publishes the
actual runtime summary through `/v1/system` and `/v1/models`.

## 4. CUDA strict-provider contract

The CUDA image fixes Ubuntu 22.04, Python 3.11, CUDA Runtime 12.9.1, cuDNN
9.24.0, and the Microsoft CUDA 12 `onnxruntime-gpu==1.27.0` wheel. Never infer
an actual runtime version from a tag alone.

Startup must fail unless all of these succeed:

1. Linux x86_64 and version-pin validation;
2. GPU, Compute Capability, and Driver discovery;
3. `CUDAExecutionProvider` availability and primary Session placement;
4. actual loaded CUDA and cuDNN library version inspection;
5. manifest and model-license policy validation (a missing file defaults to
   non-commercial, while an existing invalid license fails);
6. real detector and recognizer Session creation;
7. real warm-up through both graphs;
8. recognition output-dimension verification;
9. ORT profile evidence for CUDA kernels;
10. strict rejection of unexpected CPU/non-CUDA model compute.

Dynamic graphs may place bounded integer shape-metadata operations on
`CPUExecutionProvider`. The strict audit may allow only explicit reviewed
operators with bounded metadata output. Never add convolution, recognition, or
a broad operator class to silence an audit failure.

Architecture compatibility, actual validation, and Community Tested reports
are distinct labels. A hardware claim requires a dated record containing the
build origin, local image ID and any published registry digest, GPU/Driver,
actual CUDA/cuDNN/ORT, model
identities/digests, provider lists, strict audit, functional flow, and
consistency result.

## 5. Model bundle and offline license design

A managed public model bundle contains detector and recognizer ONNX files,
`manifest.json`, and a signed `MODEL.LICENSE`. A custom runtime bundle may omit
the license file; it is then explicitly reported as non-commercial by default.
An existing invalid, mismatched, inactive, or expired signed license still
fails startup. Normal startup never downloads models. The `models` Compose tool
supports install and strict verification for public packages; the manifest
helper supports controlled private bundles. Catalog downloads share the
`MODEL_ZOO_RELEASE_BASE_URL` for the dedicated
[`model-zoo` GitHub Release](https://github.com/deepinsight/insightface/releases/tag/model-zoo);
packages are identified by name and have no separate release/version metadata.

The strict V1 manifest is the runtime truth for its declared bundle metadata.
The extensible V2 manifest instead names tasks and their input contracts and
does not define a separate model-version field. A V2 task may declare an
optional ONNX SHA-256; the Server verifies declared detector/recognizer digests
and always calculates their actual digests. Older manifests may contain
`model_version`; loading ignores it. New manifests, API responses, SDK result
types, and database rows omit that field. Together, the manifest and runtime
inspection determine:

- detector and recognizer file names;
- task and model identities;
- input size and dynamic-input behavior;
- embedding dimension;
- normalization and preprocessing versions;
- verified-when-declared and runtime-calculated artifact SHA-256 used for
  diagnostics and Collection compatibility.

The signed model license is an offline compliance credential, not DRM. It binds
authorization to `model_id`, issuer, use, and optional validity interval. It
does not bind the exact artifact digest so an authorized customer may perform
approved FP16 or other format conversion. Verification uses the bundled
InsightFace Ed25519 public key. The issuer private key stays under an ignored
private issuer directory and never enters a container or commit.

The five Buffalo/Antelope public package licenses start at
`2021-09-22T00:00:00Z`; the two Raccoon licenses start at
`2026-08-29T00:00:00Z`. They have no end date and state non-commercial use. A
future private model can share a detector with a public bundle but needs a
license whose `model_id` matches the private recognizer identity. A
not-yet-effective or expired commercial license must fail startup clearly.

Changing detector or recognizer can change detection, alignment, preprocessing,
or embedding semantics. The computed bundle contract therefore pins every
Collection. Do not bypass `collection_model_mismatch`; phase one intentionally
requires explicit rebuild or migration.

### 5.1 Optional liveness and Web preparation

The shipped configuration and omitted addon keys both default to disabled:
`inference.addons = []` and `addons.auto_download = []`. Enabling liveness does
not change the recognition model digest or the existing Collection contract.
The model catalog pins the public URL, size and SHA-256; addons remain flat at
`<models_dir>/addons/liveness.onnx`, independent of the base model bundle.
`raccoon_s` and `raccoon_l` are supported detector/recognizer packages. The
Server does not load their verifier model or provide Web base-package switching.

`GET /v1/addons/liveness` distinguishes current execution (`enabled`), a
verified file (`installed`), next-start configuration (`configured_enabled`),
and `restart_required`. `POST /v1/addons/liveness/enable` accepts an empty JSON
object and returns 202 while an owned background task downloads and verifies
the catalog artifact, then updates the two addon lists in the same
`server.toml`. It preserves other settings and comments, validates the new
TOML, and commits via a temporary file and atomic replacement. The job shares
an advisory download lock with the CLI and a stable config lock with other
processes. Duplicate requests join the running task; failures expose stable
codes, and retries reuse verified artifacts. Closing a browser does not cancel
the download. Shutdown prevents a later configuration commit.

The running engine never reloads this configuration: an operator restarts it
to apply the change. Startup stays offline and fails with an actionable
`addon_model_missing` or `addon_model_invalid` error for a selected unavailable
addon. The Web action follows existing API authentication, requires JSON, and
checks browser Origin. The app has no separate administrator role. Read-only
deployments remain supported; status explains why preparation is unavailable.
Use the stable `unavailable_code` and error codes for UI localization;
`unavailable_reason` and backend messages are diagnostic text, not translation
keys. Render authored Markdown without running interface translation over it.
Since 0.3.1, CPU and CUDA images default to root (`0:0`), and both Compose
services use that identity with Docker's default capabilities. Do not retain a
`cap_drop: [ALL]` override: root needs the standard file-access capabilities for
existing host-owned model and data directories. No privileged mode is used.
The root filesystem remains read-only, with `no-new-privileges` and bounded
`/tmp`. Root can modify files in the writable mounts; newly downloaded files may
be root-owned on the host.

Each service has one writable `/models` bind with `create_host_path: true`.
There is no nested addon bind, host UID/GID export, shared group, or manual addon
preparation. The installer creates `addons/` only for an explicit addon download
(including an explicitly configured `addons.auto_download` during model install).
Normal startup remains offline, and disabled liveness requires no addon directory.
Both services mount the existing whole configuration directory writable at
`/etc/insightface` for atomic replacement. The installer no longer uses a read-only
single-file configuration mount.
Configuration sources must exist and are not created by Compose. Compose forwards
download proxies to both services, while the Docker health probe uses an explicit proxy-free opener for localhost regardless
of `HTTP_PROXY`, `HTTPS_PROXY`, or `NO_PROXY`. Web permissions and proxy setup are in the
[user guide](user-guide.md#web-download-permissions).

`models install <package> --enable-liveness` provides first-install activation
without a running Server. It checks configuration writability before model work,
installs and verifies the base package, configured installation addons, and
liveness, then atomically appends `liveness` to `inference.addons` and
`addons.auto_download`. Other entries, comments, and settings are preserved.
Cache hits still apply activation. Download failure leaves configuration
unchanged; save failure reports an error and exits nonzero, with valid caches
available for retry. Without the flag, installation does not write configuration;
`models addons install liveness` remains download-only. The next first startup
reads the saved configuration; an already running Server requires an explicit
restart, because `up -d` alone does not reload startup settings.

Every evaluated face has the core fields `status`, `is_live`, and `live_score`.
Fake is `status=ok` with `is_live=false`. Only insufficient source-image area
around the aligned face produces `status=input_rejected`, with both values null
and an additional human-readable `reason`. Live and fake results omit `reason`.
FaceAnalysis and the API always return this English explanation:

> Insufficient image area around the face for liveness detection. Move the face toward the center, step back from the camera, or use a less tightly cropped image.

Use `status` and `is_live` for program logic; `reason` is not an enumeration code.
Only the Web UI translates the display, with a generic fallback for older saved
results without `reason`. An omitted result means no evaluation. In `normal`,
failed/rejected liveness stops recognition; `observe` continues. Detection
still returns all detected faces and their liveness results with HTTP 200.
Recognition operations return the documented 422 error; inference failures
return 503 `liveness_unavailable` and stop the operation in both modes. Invalid
landmarks raise `ValueError` and alignment failures raise `RuntimeError` in
FaceAnalysis; Server maps these to 503, not `input_rejected` or fake.
Registration skips liveness by default;
`liveness_on_registration=true` applies the configured mode, and enrollment
review or external embeddings do not bypass it. RTSP keeps blocked faces
separate from unknown identities.

## 6. Embedding and score contract

The recognizer consumes the five-point aligned face. Optional persisted crops
are a separate 112×112 resized bounding-box crop; they are not the aligned
recognition tensor.

Canonical durable embeddings are FP32 and L2-normalized. Query embeddings are
also normalized. Public search and threshold decisions use:

```text
similarity = dot(normalized_database_embedding, normalized_query_embedding)
```

Similarity is raw cosine in `[-1, 1]`, not probability. Public thresholds are
restricted to `[0, 1]`, default to `0.4`, and match inclusively with
`similarity >= threshold`.

`external_trusted` intentionally skips recognition extraction. Every feature
still requires a same-index image for decode, detection, selection, quality,
metadata, and optional crop behavior. Validate image/vector count, exact
Collection `embedding_contract_id`, dimension, finite values, nonzero norm, and
L2 norm within `1.0 ± 0.0002`. Passing values are converted to canonical FP32
and normalized once for floating-point drift.

The service cannot prove that an external vector belongs to its supplied image.
This mode places the caller inside the biometric trust boundary. Never log the
vector, silently replace it with a server feature, or invoke the recognizer
after an `external_trusted` validation failure.

Review modes:

- `off`: select using the Collection strategy; multiple faces are allowed;
- `standard`: exactly one usable face plus configured size, detector,
  sharpness, brightness, aggregate quality, and pose checks;
- `strict`: standard plus best within-Person similarity greater than best
  outside-Person similarity.

Quality rules are documented heuristics, not AWS-equivalent quality scores.

## 7. Exact search implementation

Phase one uses one exact mutable in-process index per active Collection. It does
not use FAISS and does not perform approximate candidate selection. A Person's
score is the maximum similarity among all of that Person's live FaceSamples.
Ties are deterministic.

Profiles:

| Profile | Stored vector | CPU | CUDA | Notes |
| --- | --- | --- | --- | --- |
| `fp32_v1` | FP32 | yes | yes | flat FP32 inner product |
| `fp16_v1` | FP16 | no | yes | low-precision approximation |
| `bf16_v1` | BF16 | supported CPU | SM80+ | low-precision approximation |
| `int8_x736_v1` | INT8 | yes | yes | recommended INT8 |
| `int8_x1000_v1` | INT8 | yes | yes | compatibility contract |

Production images load the native CPU or CUDA shared library and expose only
its capability mask. Python then rejects an unavailable profile before
Collection creation/load; it does not attempt to load an incompatible symbol
and continue.

INT8 encoding is deterministic:

```text
q = clamp(round_half_away_from_zero(x * S), -128, 127)
score_internal = int32_dot(q_database, q_query) / S²
```

`S` is 736 or 1000 as encoded in the immutable profile name. Accumulation is
INT32. Exact row and Person ordering uses the unclipped internal score; only the
public response is clamped to cosine range. There is no rerank stage.

All profiles exhaustively score every live FaceSample, so “exact” describes
candidate coverage. Low-precision arithmetic can still produce ordering
differences relative to FP32.

## 8. Capacity, mutation, and restart invariants

`capacity_rows` is both initial reservation and the Server-level live-row limit.
The default is 100,000, guarded by a deployment maximum of 10,000,000. Vector
bytes alone for 512 dimensions are:

```text
FP32: 2,048 bytes/row
FP16 or BF16: 1,024 bytes/row
INT8: 512 bytes/row
```

IDs, liveness metadata, group mappings, score buffers, Top-K workspaces, and
allocator overhead are additional. CUDA reserves row/group metadata and
grouped Top-K workspaces with Collection capacity to avoid first-query growth.

SQLite is authoritative. An accepted add/delete holds the Collection index
lock and follows this durable barrier:

1. write FaceSample change, pending search change, and next
   `search_revision` in one SQLite transaction;
2. commit SQLite;
3. apply the mutation to the active native generation;
4. verify native applied revision equals SQLite revision;
5. acknowledge covered pending changes;
6. return success.

A successful add is therefore immediately searchable; a successful deletion is
immediately absent. If native mutation fails after the commit, discard and
rebuild the generation. If rebuild also fails, return
`503 search_index_unavailable` with `write_committed=true` and committed
revision.

No index binary is persisted. At startup, eager Collections rebuild before
readiness; lazy Collections rebuild on first search. Rebuild streams SQLite in
batches, verifies the revision did not change, validates row counts, and
atomically publishes the generation.

CPU deletion reuses a slot. CUDA deletion creates a tombstone; a later add may
trigger a deterministic rebuild from live SQLite rows to reuse physical
capacity. The process lock beside SQLite prevents two application workers from
serving unsafe independent indexes over one data directory.

CUDA grouped Top-K stays device-resident and evaluates every live row, reduces
to the best FaceSample per Person, and returns only final records over PCIe.
CPU uses the same score/order contract with host result handling.

## 9. SQLite, cursors, secrets, and crops

Add a new ordered migration instead of modifying an applied schema. Migrations
run transactionally. SQLite uses foreign keys, WAL, a busy timeout, and an
in-process write lock.

Opaque list and event cursors must remain signed and scoped. Resource list
cursors use `/data/cursor.key` and must not reveal offsets or SQL. Monitor event
cursors contain an opaque epoch/sequence contract and report truncation or
stream reset without promising durability.

The startup API key is stored only as a random-salted scrypt digest. A different
`INSIGHTFACE_API_KEY` on a later start atomically deactivates the previous
credential and activates the new digest. Phase one deliberately has no runtime
multi-key, scope, role, or revocation API.

RTSP credentials are encrypted under a data-volume key. API responses always
redact user information and query secrets. Never include credentials in preview
URLs, logs, errors, or events.

`save_face_crops` is resolved per Collection and defaults false. Accepted
registrations may store one 112×112 bounding-box JPEG BLOB. Original uploads,
aligned recognition inputs, rejected images, and RTSP frames are never stored.
Logical deletion is not forensic erasure from WAL, snapshots, or backups.

## 10. RTSP Monitor runtime

Monitor configuration persists in SQLite. Each enabled Monitor owns a server
task, decoder lifecycle, latest-frame slot, schedule, state snapshot, and
bounded in-memory event ring. It shares the process inference budget and model
Sessions; it does not create separate ONNX Sessions.

The decoder keeps only the newest frame. If inference takes longer than the
requested interval, stale frames are counted as skipped instead of queued.
Multiple clients can independently poll one Monitor's state and recent events;
they see the same server state but maintain their own cursors.

Events are intentionally non-durable. If no client polls, old events fall out
of the bounded ring. A process/task restart creates a new stream epoch.

`preview_enabled` defaults false. Recognition and events do not depend on a
preview client. JPEG encoding begins lazily only while preview is enabled and
at least one viewer is connected. `/preview.mjpeg` carries raw frames; clients
draw labels from `/state`. Closing every browser must not stop the Monitor.

Lifecycle updates:

- source, Collection, cadence, threshold, or event-policy changes restart the
  task;
- name, description, preview, and buffer-size changes do not require restart;
- `enabled=false` stops runtime but preserves configuration;
- DELETE stops runtime and removes configuration and volatile state.

## 11. Security invariants

The Server processes sensitive biometric data. Preserve these design rules:

- no image, embedding, API key, RTSP credential, or multipart-body logging;
- default-deny CORS, exact trusted origins only;
- image-byte, decoded-pixel, request-body, image-count, and request-time limits;
- one writable `/models` mount, on-demand addon creation, a writable configuration
  directory for atomic startup-setting saves, and persistent `/data`;
- root UID/GID `0:0` with Docker's default capabilities, a read-only root
  filesystem, `no-new-privileges`, bounded `/tmp`, and no privileged mode;
- no remote model download during normal startup;
- authenticated and non-cacheable crop/embedding access;
- self-hosted UI with CSP and no CDN, analytics, remote font, or third-party JS;
- plain HTTP only inside the container; deployment owns TLS termination,
  ingress/egress restriction, and rate limiting.

Do not present face recognition as the sole control for a high-impact decision.
Thresholds and quality settings require deployment-specific validation.

## 12. Local development

Use Python 3.11:

```bash
python3.11 -m venv server/.venv
. server/.venv/bin/activate
python -m pip install -r server/requirements.dev.lock
```

CPU and CUDA lock files remain separate because only one ONNX Runtime package
may be installed in each image. Frontend tests use Node's built-in test runner
and have no package-install step.

Unified commands from the repository root:

```bash
make -C server lint
make -C server test
make -C server test-api
make -C server test-sdk
make -C server test-frontend
make -C server test-native-cpu
make -C server build-cpu
make -C server build-cuda12
make -C server run-cpu
make -C server run-cuda12
make -C server test-cpu
make -C server test-cuda12
make -C server test-consistency
make -C server smoke-test
```

Public tests use mock inference and synthetic data; they must not require
models, commercial assets, real faces, GPUs, cameras, or external network
services. Published images force ONNX mode and reject mock inference.

## 13. Native and real-model validation

`make -C server test-native-cpu` builds the native library and checks the C ABI,
score semantics, grouped Person Top-K, capacity add/delete/reuse, CPU
FP32/BF16/INT8, and explicit unsupported FP16 behavior. The library avoids
`-march=native`; runtime dispatch selects a compatible optimized kernel.

Real-model tests are opt-in and use ignored `/models` plus private authorized
images. Record at least:

- detection count and area ordering;
- embedding dimension and L2 norm;
- CPU/GPU cosine tolerance;
- 1:N Person Top-K order;
- decisions near thresholds;
- actual Session providers and CUDA strict audit;
- model/runtime/hardware identity.

Docker validation records the build origin and result, local image ID, any
published registry digest, build-time verifier, startup output, actual
`/v1/system`, CRUD/enrollment/search, restart persistence,
strict CUDA audit, and CPU/GPU consistency. Never derive a validation claim
only from Dockerfile text or a base-image tag.

RTSP E2E additionally checks credential encryption/redaction, task restoration,
bounded event cursor behavior, no-backlog scheduling, viewer-independent
execution, reconnects, optional preview, and client-side overlays.

## 14. Public API documentation change gate

A public API addition, modification, rename, deprecation, or removal is not
complete until the same change includes:

1. FastAPI/Pydantic behavior and OpenAPI metadata;
2. all nine localized `docs/api*.md` operation sections, including purpose,
   authentication, parameters, defaults/ranges/enums, server behavior, success,
   errors, side effects, pagination/retry guidance, and an example;
3. all affected localized `docs/user-guide*.md` workflows;
4. Python SDK and Web UI behavior when exposed there;
5. API, SDK, UI, and documentation contract tests;
6. reviewed `make -C server update-api-docs` snapshot diff;
7. compatibility or migration notes in the affected guides.

`tests/api/test_documentation_contract.py` compares the runtime public
method/path set with every localized API Guide and compares live OpenAPI with
`docs/openapi.snapshot.json`. Never resolve a failure by updating only the
snapshot.

README files are GitHub overview/quick-start pages and are not copied into
images. User Guides and API Guides are the single user-facing Markdown sources
for both GitHub and Web UI. This file is the single English-only maintainer
reference. `/docs` and `/openapi.json` remain the live machine-oriented schema.

## 15. Container versioning and release

InsightFace Server has no repository-hosted CI or release pipeline. The
repository owner performs validation, image publication, and stable-channel
promotion manually from trusted Linux hosts. This keeps registry credentials
and private model/test assets outside the repository.

All public variants share:

```text
ghcr.io/deepinsight/insightface-server
```

Each stable version has two immutable tags built from the same source inputs:

```text
<major>.<minor>.<patch>-cpu
<major>.<minor>.<patch>-cuda12
```

`cpu` and `cuda12` are moving stable channels. Never publish `latest`, never
overwrite an immutable version tag, and never move either stable channel until
both versioned variants have passed validation.

This workflow publishes Server images to GHCR. It does not require Git commits,
Server Git tags, or a Server GitHub Release. Model release assets are maintained
separately by the repository owner.

### 15.1 Prepare and validate the source

Run the version and structural consistency checks against the local source tree:

```bash
make -C server release-preflight
# Backward-compatible precheck entry point, with the same checks:
make -C server release-precheck
```

Both entry points run the same 39 checks for synchronized backend, SDK,
documentation, Compose, Docker, and license metadata. Uncommitted changes, a
missing `.git` directory, or unavailable Git commands produce no warning or
failure in these checks.
The script's `--relaxed` flag remains accepted for compatibility; it uses the
same rules and labels its report `mode: precheck`.

Reports use `schema_version: 2`. Optional `git` metadata contains
`head_revision` (a string or `null`) and `dirty` (a boolean or `null`).
`source_revision` is populated only when a valid HEAD and a clean tree are
known; otherwise it is `null`. This describes the source at precheck time and
does not identify an image built earlier. A HEAD value from a dirty tree must
not be presented as the exact source of that build.

Run the quality checks before qualifying the images:

```bash
make -C server lint
make -C server test
make -C server test-api
make -C server test-sdk
make -C server test-frontend
make -C server test-native-cpu
```

Build both Python distributions in isolated output directories and run metadata
validation with the maintainer's packaging environment:

```bash
python3.11 -m build --sdist --wheel --outdir /tmp/ifs-server-dist server
python3.11 -m build --sdist --wheel --outdir /tmp/ifs-sdk-dist server/sdk/python
python3.11 -m twine check /tmp/ifs-server-dist/* /tmp/ifs-sdk-dist/*
```

### 15.2 Build and qualify both images

Build both variants from the same source inputs, including any intended local
edits. Keep files used by the build unchanged between the CPU and CUDA builds.
The source tree can be a working checkout or a complete copy without Git
metadata. A retained source copy, archive, or content digest can help later
traceability, but no particular snapshot mechanism is required.

```bash
export RELEASE_VERSION=0.3.1
export RELEASE_IMAGE=ghcr.io/deepinsight/insightface-server

make -C server build-cpu \
  SERVER_VERSION="$RELEASE_VERSION" \
  CPU_IMAGE="$RELEASE_IMAGE:$RELEASE_VERSION-cpu"
make -C server build-cuda12 \
  SERVER_VERSION="$RELEASE_VERSION" \
  CUDA_IMAGE="$RELEASE_IMAGE:$RELEASE_VERSION-cuda12"
```

Record the resulting local image IDs and associate the validation results with
those images. The following commands assume both images are on the current
Docker host; if separate build hosts are used, retain each host's image ID and
the corresponding validation record.

```bash
CPU_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$RELEASE_IMAGE:$RELEASE_VERSION-cpu")" || exit 1
CUDA_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$RELEASE_IMAGE:$RELEASE_VERSION-cuda12")" || exit 1
export CPU_IMAGE_ID CUDA_IMAGE_ID
printf 'CPU image ID: %s\nCUDA image ID: %s\n' "$CPU_IMAGE_ID" "$CUDA_IMAGE_ID"
```

Qualify the supplied Compose files with absent model and addon directories, an
existing configuration directory, and a fresh data volume. Verify automatic
model-root creation, base-model installation and cache reuse, startup with
liveness disabled and no addon directory, explicit CLI and Web addon download,
configuration saving, manual-restart activation, and repeated cache reuse. Test
`models install <package> --enable-liveness` before the first Server startup,
including cached models, preserved configuration comments/entries, unavailable
configuration failing before downloads, download failure leaving configuration
unchanged, and configuration-save failure returning nonzero. Confirm plain
installation and `models addons install liveness` do not write configuration.
Confirm root identity, Docker's default capabilities, one writable model bind,
and the retained read-only root filesystem and `no-new-privileges`.

Also qualify an upgrade using isolated copies of the previous configuration,
model files, and data volume. Apply the new Compose settings and all custom
overrides, rather than changing image tags alone: remove old UID/GID, shared-group,
`cap_drop: [ALL]`, read-only model-root, nested-addon, and installer read-only
single-file configuration settings. Both services need the whole existing
configuration directory mounted writable. Keep the actual
paths and volume name. Check existing Collections, samples, and search results
after recreation and restart. Keep backup and rollback instructions aligned with
the [User Guide](user-guide.md#upgrade-to-031).

Run the CPU image on a Linux x86-64 host and the CUDA image on a compatible
NVIDIA host with the verified model directory mounted writable. For both
variants, check startup, `/v1/health`, `/v1/system`, CRUD/enrollment/search,
restart persistence, and a private authorized image containing exactly one
usable face:

```bash
python3.11 server/scripts/smoke_test.py \
  --base-url http://127.0.0.1:8080 \
  --image /absolute/path/to/release-test-image.jpg
```

The CUDA result must report `CUDAExecutionProvider` and pass strict-provider
startup checks. Compare CPU and CUDA results for the same image and retain only
redacted validation notes; never commit the image, embedding, credentials, RTSP
URLs, or raw biometric output.

### 15.3 Retain the validated images and build record

Keep the validated local images for publication and record their image IDs,
build command, version, and test results. A source snapshot or digest captured
at build time is optional supporting evidence. Git HEAD may also be recorded
when available, together with whether the tree had local changes. Neither a
dirty HEAD nor a preflight report created later proves the exact build inputs.

Publish these same validated images. Committing or organizing Git history
afterward does not require a rebuild when the build inputs have not changed.
README files are not copied into Server images, so README-only edits also do
not require a container rebuild. User/API Guides, this maintainer guide, guide
images, and frontend assets are included in the images. Changes to those files
or to any other image inputs require rebuilding the affected images and
validating them again before publication. Do not rebuild an already validated
image just to attach a later Git commit to the release.

When transferring an image between Docker hosts, preserve it with
`docker save`/`docker load` or an existing registry digest and verify that the
image ID is unchanged. A local image ID and a registry manifest digest identify
different objects; record both instead of comparing them directly.

### 15.4 Publish manually to GHCR

Authenticate using a maintainer token with package-write permission, supplied
through standard input:

```bash
read -r -p "GHCR user: " GHCR_USER
read -r -s -p "GHCR token: " GHCR_TOKEN
printf '%s' "$GHCR_TOKEN" |
  docker login ghcr.io --username "$GHCR_USER" --password-stdin
unset GHCR_TOKEN
```

Before pushing, run both commands below. Each must fail specifically because
the manifest is not found. A returned manifest means the immutable tag already
exists; an authentication, DNS, timeout, or other operational error is also a
hard stop. Never infer “not found” from an arbitrary command failure.

```bash
for VARIANT in cpu cuda12; do
  if docker buildx imagetools inspect \
    "$RELEASE_IMAGE:$RELEASE_VERSION-$VARIANT"; then
    echo "immutable tag already exists; stop" >&2
    exit 1
  fi
  read -r -p "Confirm $VARIANT failed only with manifest-not-found [yes]: " CONFIRM
  test "$CONFIRM" = yes || exit 1
done
```

After confirming both immutable tags are absent, check that the local tags
still point to the images that passed validation, then push them. Restore
`CPU_IMAGE_ID` and `CUDA_IMAGE_ID` from the build record if this is a new shell
or publication host; do not replace them with IDs from unverified images.

```bash
test "$(docker image inspect --format '{{.Id}}' "$RELEASE_IMAGE:$RELEASE_VERSION-cpu")" = "$CPU_IMAGE_ID" || exit 1
test "$(docker image inspect --format '{{.Id}}' "$RELEASE_IMAGE:$RELEASE_VERSION-cuda12")" = "$CUDA_IMAGE_ID" || exit 1
docker push "$RELEASE_IMAGE:$RELEASE_VERSION-cpu" || exit 1
docker push "$RELEASE_IMAGE:$RELEASE_VERSION-cuda12" || exit 1
```

Resolve and validate the two remote digests. These digest references, not the
tag names, are the promotion inputs:

```bash
export CPU_DIGEST="$(
  docker buildx imagetools inspect \
    "$RELEASE_IMAGE:$RELEASE_VERSION-cpu" \
    --format '{{json .Manifest}}' |
  python3.11 -c 'import json,sys; print(json.load(sys.stdin)["digest"])'
)"
export CUDA_DIGEST="$(
  docker buildx imagetools inspect \
    "$RELEASE_IMAGE:$RELEASE_VERSION-cuda12" \
    --format '{{json .Manifest}}' |
  python3.11 -c 'import json,sys; print(json.load(sys.stdin)["digest"])'
)"
case "$CPU_DIGEST:$CUDA_DIGEST" in
  sha256:*:sha256:*) ;;
  *) echo "invalid release digest" >&2; exit 1 ;;
esac
```

Read the published images back by those exact digests and verify that their
local image IDs match the validated images:

```bash
docker pull "$RELEASE_IMAGE@$CPU_DIGEST" || exit 1
test "$(docker image inspect --format '{{.Id}}' "$RELEASE_IMAGE@$CPU_DIGEST")" = "$CPU_IMAGE_ID" || exit 1
docker pull "$RELEASE_IMAGE@$CUDA_DIGEST" || exit 1
test "$(docker image inspect --format '{{.Id}}' "$RELEASE_IMAGE@$CUDA_DIGEST")" = "$CUDA_IMAGE_ID" || exit 1
```

Record any existing `cpu` and `cuda12` stable-channel digests before changing
them. Only after both versioned digests are readable and verified, move the two
stable channels to those exact digest references:

```bash
docker buildx imagetools create \
  --prefer-index=false \
  --tag "$RELEASE_IMAGE:cpu" "$RELEASE_IMAGE@$CPU_DIGEST" || exit 1
docker buildx imagetools create \
  --prefer-index=false \
  --tag "$RELEASE_IMAGE:cuda12" "$RELEASE_IMAGE@$CUDA_DIGEST" || exit 1
```

Read both stable tags back and require their digests to match the recorded
versioned digests:

```bash
test "$(
  docker buildx imagetools inspect "$RELEASE_IMAGE:cpu" \
    --format '{{json .Manifest}}' |
  python3.11 -c 'import json,sys; print(json.load(sys.stdin)["digest"])'
)" = "$CPU_DIGEST" || exit 1
test "$(
  docker buildx imagetools inspect "$RELEASE_IMAGE:cuda12" \
    --format '{{json .Manifest}}' |
  python3.11 -c 'import json,sys; print(json.load(sys.stdin)["digest"])'
)" = "$CUDA_DIGEST" || exit 1
docker logout ghcr.io
```

Registry updates to `cpu` and `cuda12` are not atomic. Record their previous
digests before changing them. If promotion is interrupted, stop, inspect all
four tags, and manually restore both stable channels to the recorded digests or
finish the matching pair; never rebuild or overwrite a versioned tag.

## 16. Contribution checklist

- Preserve unrelated work in a dirty tree.
- Add migrations; never rewrite an applied schema.
- Keep model Sessions singleton and request policy isolated.
- Keep SQLite authoritative and index mutations behind the revision barrier.
- Keep API, SDK, Web UI, all localized guides, and OpenAPI synchronized.
- Test error and retry behavior, not only the success path.
- Retain actual hardware/model evidence for compatibility claims.
- Do not weaken provider, license, digest, capacity, or input checks to make an
  incompatible environment appear healthy.
- Never push images, code, releases, or customer artifacts unless explicitly
  authorized.


### Model identity and existing Collections

Models are identified by `model_id` (the package name). The runtime bundle digest,
embedding dimension and preprocessing contract still protect Collection
compatibility. Model weights, inference and embeddings are unchanged by removing
the separate model-version field.

Migration `0009_model_identity.sql` persists each existing Collection's original
`ifsemb-v1-sha256:` identifier before dropping the old model-version columns.
Historical external FaceSample contract IDs stay unchanged, so existing trusted
clients can continue using their recorded identifiers. New Collections use an
`ifsemb-v2-sha256:` identifier over `[model_id, model_digest, embedding_dimension,
preprocessing_version]`. These prefixes version the contract format, not models.
Take a database backup before upgrading; older Server images cannot read the
migrated schema without restoring that backup.
