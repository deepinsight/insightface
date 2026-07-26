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

Do not commit model binaries, signed customer licenses, issuer private keys,
real face images, customer data, generated databases, or production
configuration. `/models` is read-only runtime input; `/data` is mutable,
persistent runtime state.

## 2. Source architecture

```text
server/
├── backend/insightface_server/
│   ├── api/          request and response schemas, authentication
│   ├── inference/    provider-independent pipeline and ONNX implementation
│   ├── licensing/    offline model-license verification and trusted keys
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
5. manifest and signed model-license verification;
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
source revision, image digest, GPU/Driver, actual CUDA/cuDNN/ORT, model
identities/digests, provider lists, strict audit, functional flow, and
consistency result.

## 5. Model bundle and offline license design

A model bundle contains detector and recognizer ONNX files, `manifest.json`,
and `MODEL.LICENSE`. Normal startup never downloads models. The `models`
Compose tool supports install and verify for public packages; the manifest
helper supports controlled private bundles.

The manifest is the runtime truth for:

- detector and recognizer file names;
- task and model identities;
- model version;
- input size and dynamic-input behavior;
- embedding dimension;
- normalization and preprocessing versions;
- artifact SHA-256 used for diagnostics and Collection compatibility.

The signed model license is an offline compliance credential, not DRM. It binds
authorization to `model_id`, issuer, use, and optional validity interval. It
does not bind the exact artifact digest so an authorized customer may perform
approved FP16 or other format conversion. Verification uses the bundled
InsightFace Ed25519 public key. The issuer private key stays under an ignored
private issuer directory and never enters a container or commit.

The four public package licenses start at `2021-09-22T00:00:00Z`, have no end
date, and state non-commercial use. A future private model can share a detector
with a public bundle but needs a license whose `model_id` matches the private
recognizer identity. A not-yet-effective or expired commercial license must
fail startup clearly.

Changing detector or recognizer can change detection, alignment, preprocessing,
or embedding semantics. The computed bundle contract therefore pins every
Collection. Do not bypass `collection_model_mismatch`; phase one intentionally
requires explicit rebuild or migration.

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
- `/models` read-only and `/data` the only durable writable area;
- non-root UID/GID 10001, read-only root filesystem, dropped capabilities,
  `no-new-privileges`, bounded `/tmp`;
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

Docker validation records build result/digest, build-time verifier, startup
output, actual `/v1/system`, CRUD/enrollment/search, restart persistence,
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

Each stable version has two immutable tags built from one source revision:

```text
<major>.<minor>.<patch>-cpu
<major>.<minor>.<patch>-cuda12
```

`cpu` and `cuda12` are moving stable channels. Never publish `latest`, never
overwrite an immutable version tag, and never move either stable channel until
both versioned variants have passed validation.

### 15.1 Prepare and validate the source

Use the relaxed precheck while the tree contains local edits:

```bash
make -C server release-precheck
```

Before publication, commit all release source and use a clean worktree. The
strict check verifies synchronized backend, SDK, documentation, Compose, Docker,
license metadata, Git revision, and release-tag state:

```bash
make -C server release-preflight
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

Build both immutable version tags from the exact committed revision:

```bash
export RELEASE_VERSION=0.2.0
export RELEASE_IMAGE=ghcr.io/deepinsight/insightface-server
export RELEASE_SHA="$(git rev-parse HEAD)"

make -C server build-cpu \
  SERVER_VERSION="$RELEASE_VERSION" \
  CPU_IMAGE="$RELEASE_IMAGE:$RELEASE_VERSION-cpu"
make -C server build-cuda12 \
  SERVER_VERSION="$RELEASE_VERSION" \
  CUDA_IMAGE="$RELEASE_IMAGE:$RELEASE_VERSION-cuda12"
```

Run the CPU image on a Linux x86-64 host and the CUDA image on a compatible
NVIDIA host with the verified model directory mounted read-only. For both
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

### 15.3 Lock the source identity

Fetch tags before any registry write and stop on a tag conflict. Reuse an
existing release tag only when it is an annotated, validly signed tag for the
exact release commit; otherwise create and verify it. Pushing the same
source-bound tag is safe to retry, while a conflicting remote tag is rejected:

```bash
git fetch origin --tags || exit 1
if git show-ref --verify --quiet "refs/tags/server-v$RELEASE_VERSION"; then
  if test "$(git rev-list -n 1 "server-v$RELEASE_VERSION")" != "$RELEASE_SHA"; then
    echo "release tag points to another revision" >&2
    exit 1
  fi
else
  git tag -s "server-v$RELEASE_VERSION" "$RELEASE_SHA" \
    -m "InsightFace Server $RELEASE_VERSION" || exit 1
fi
git tag -v "server-v$RELEASE_VERSION" || exit 1
git push origin \
  "refs/tags/server-v$RELEASE_VERSION:refs/tags/server-v$RELEASE_VERSION" ||
  exit 1
```

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

Only after confirming both immutable tags are absent, push both validated
images:

```bash
docker push "$RELEASE_IMAGE:$RELEASE_VERSION-cpu"
docker push "$RELEASE_IMAGE:$RELEASE_VERSION-cuda12"
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

Record any existing `cpu` and `cuda12` stable-channel digests before changing
them. Only after both versioned digests are readable and verified, move the two
stable channels to those exact digest references:

```bash
docker buildx imagetools create \
  --prefer-index=false \
  --tag "$RELEASE_IMAGE:cpu" "$RELEASE_IMAGE@$CPU_DIGEST"
docker buildx imagetools create \
  --prefer-index=false \
  --tag "$RELEASE_IMAGE:cuda12" "$RELEASE_IMAGE@$CUDA_DIGEST"
```

Read both stable tags back and require their digests to match the recorded
versioned digests:

```bash
test "$(
  docker buildx imagetools inspect "$RELEASE_IMAGE:cpu" \
    --format '{{json .Manifest}}' |
  python3.11 -c 'import json,sys; print(json.load(sys.stdin)["digest"])'
)" = "$CPU_DIGEST"
test "$(
  docker buildx imagetools inspect "$RELEASE_IMAGE:cuda12" \
    --format '{{json .Manifest}}' |
  python3.11 -c 'import json,sys; print(json.load(sys.stdin)["digest"])'
)" = "$CUDA_DIGEST"
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
