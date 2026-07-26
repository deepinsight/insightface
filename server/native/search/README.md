# InsightFace Server native exact search

This directory contains the product-facing native search ABI used by the
Server. It is derived from the standalone benchmark implementation, but does
not contain benchmark datasets, extracted embeddings, or result files.

## ABI and score contract

Both `libifs_search_cpu` and `libifs_search_cuda` export the same C ABI from
`include/ifs_search.h`. ABI v2 is fixed to 512-dimensional input. Every input
vector and query must be finite, FP32, and L2-normalized before it crosses the
ABI boundary.

All returned scores use **raw cosine** semantics. Because the input is
normalized, FP32 inner product equals cosine similarity. FP16 and BF16 return
their low-precision approximation. INT8 uses the per-index scale encoded by
the profile (`S=736` or legacy-compatible `S=1000`):

```text
q = clamp(round_half_away_from_zero(x * S), -128, 127)
score_internal = int32_dot(q_database, q_query) / (S * S)
similarity = clamp(score_internal, -1, 1)
```

CPU, CUDA, and the NumPy reference deliberately perform the multiply and
half-away-from-zero rounding with FP32 semantics. The native index uses the
unclipped internal score for exact ordering and grouped best-Face selection,
then clamps scores returned by the ABI to the public cosine range. The unscaled
INT32 accumulator is never exposed by the production ABI.

Profiles:

- `FP32_V1`: CPU and CUDA
- `FP16_V1`: CUDA only; CPU returns `IFS_SEARCH_UNSUPPORTED`
- `BF16_V1`: CPU and CUDA
- `INT8_X736_V1`: CPU and CUDA; recommended INT8 profile
- `INT8_X1000_V1`: CPU and CUDA

The two INT8 scales are separate per-index contracts and can coexist in one
process. Profile code 3 remains x1000; x736 appends code 4. Existing x1000
Collections are never silently reinterpreted as x736.

Unsupported profiles and CUDA failures are fail-closed. There is no dtype or
CPU fallback inside either native library.

The CUDA capability mask is device-specific. In particular, BF16 is exposed
only on Ampere/SM80 or newer; Turing still supports FP32, FP16, and INT8. Index
creation rechecks the selected device and rejects unsupported profiles.

## Capacity and deletion

`reserve_rows` reserves contiguous storage at creation. `max_rows` is the hard
logical/physical row limit checked before add. Configure them to the same value
for a production Collection. CUDA reserves vector/group metadata, score and
delete workspaces, and face/group candidate buffers for exact Top-K up to 100,
so a within-capacity query does not trigger a first-query workspace allocation.
Automatic growth replaces all of those buffers together and preserves the
stable group metadata.

CPU deletion reuses freed slots. CUDA deletion uses tombstones, so
`physical_rows` continues increasing after delete/add cycles. The Server must
rebuild a CUDA collection generation before tombstones consume its capacity.

## Person Top-K

ABI v2 accepts a stable numeric group/Person ID for every FaceSample and
implements strict grouped Top-K. Every live face score participates, each
Person retains its highest-scoring face (then the lowest vector ID for an exact
score tie), and Persons are sorted by score descending and group ID ascending.
It never uses a fixed face-level oversample, so it is exact for the Server's
`max FaceSample score` Person policy.

The CUDA implementation keeps row-to-group metadata on the device. A two-pass
GPU reduction first finds each Person's best score and then its deterministic
best FaceSample; the existing multi-stage exact Top-K reduction selects at most
100 Persons. Only those final K `(group_id, vector_id, score)` records cross
PCIe. Incremental add, tombstone delete, group reactivation, explicit reserve,
and automatic growth all preserve the device group metadata.

CUDA sets `IFS_SEARCH_CAP_GROUPED_DEVICE_RESIDENT` together with
`IFS_SEARCH_CAP_GROUPED_PERSON_TOPK`, and does not set the host-reference flag.
CPU continues to set `IFS_SEARCH_CAP_GROUPED_HOST_REFERENCE`: its exact grouped
implementation still materializes face scores in host memory. Both paths have
identical ordering and raw-cosine score semantics.

## Build and smoke test

CPU:

```bash
cmake -S server/native/search -B build/native-search \
  -DIFS_SEARCH_BUILD_CPU=ON \
  -DIFS_SEARCH_BUILD_CUDA=OFF \
  -DIFS_SEARCH_BUILD_TESTS=ON
cmake --build build/native-search -j
ctest --test-dir build/native-search --output-on-failure
```

CUDA 12.9:

```bash
cmake -S server/native/search -B build/native-search-cuda \
  -DIFS_SEARCH_BUILD_CPU=OFF \
  -DIFS_SEARCH_BUILD_CUDA=ON \
  -DIFS_SEARCH_BUILD_TESTS=ON
cmake --build build/native-search-cuda -j
ctest --test-dir build/native-search-cuda --output-on-failure
```

The default CUDA list contains SASS for 75, 80, 86, 89, 90, 100, 103, and 120,
plus PTX for the newest target. It is intentionally explicit rather than
`native`; release builds must contain the architectures claimed by the Server
compatibility document. Hardware certification remains release-gating work.
