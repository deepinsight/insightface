# InspireFace for HarmonyOS

This HAR module adds a complete ArkTS API on top of the existing InspireFace C/C++ core. The Android JNI and Java APIs are unchanged.

The HarmonyOS API covers resource validation and lifecycle, sessions, mutable image streams, image bitmaps, face tracking and owned result snapshots, landmarks, feature extraction/comparison and alignment, the full face pipeline, FeatureHub CRUD/search/persistence, component diagnostics, hardware configuration, logging, and resource diagnostics. Low-level C allocation and copy functions are consolidated into ArkTS objects with deterministic `close()` methods.

All 117 public C API symbols are tracked by a build-time parity manifest. The HarmonyOS bridge currently exposes 81 consolidated Node-API methods; adding a C API without a mapped native export, type declaration, and ArkTS route fails the HarmonyOS build.

## Build

Install or point to an OpenHarmony Native SDK and run from the repository root:

```bash
OHOS_NATIVE_HOME=/path/to/native-sdk/native ./command/build_harmonyos_napi.sh
```

The staged module is written to:

```text
build/inspireface-harmonyos-napi-arm64-v8a/install/HarmonyOS/har
```

Import that directory as a HAR module in DevEco Studio, or package it with the project's normal Hvigor workflow. The native module is already placed at `src/main/libs/arm64-v8a/libinspireface_napi.so`, and its declarations are under `src/main/cpp/types/libinspireface_napi`.

## Ownership and threading

- `ImageStream` copies the input bytes, so a camera buffer may be reused immediately after construction.
- `Session.close()`, `ImageStream.close()`, and `ImageBitmap.close()` are idempotent; call them deterministically instead of waiting for garbage collection.
- `Session.track()` returns an owned face-result snapshot. Release it with `Session.releaseFaceResult()` after feature extraction and pipeline processing are complete.
- Calls using the same session are serialized in native code. Separate sessions may be used by separate ArkTS workers, but JavaScript wrapper objects must not be transferred between workers.
- The API is synchronous and does not create application task queues. Applications retain control over worker and scheduling policy.

## Pipeline and FeatureHub

Create a session with the required feature mask, track a frame, and pass the owned result to the pipeline:

```ts
const features = Feature.LIVENESS | Feature.MASK_DETECT | Feature.QUALITY |
  Feature.INTERACTION | Feature.FACE_ATTRIBUTE | Feature.FACE_EMOTION;
const session = InspireFace.createSession({ featureMask: features, maxFaces: 5 });
const image = InspireFace.createImageStream(bytes, width, height, ImageFormat.RGBA);
const faces = session.track(image);
try {
  const pipeline = session.processPipeline(image, faces);
  const firstLiveness = pipeline.rgbLiveness[0];
} finally {
  session.releaseFaceResult(faces);
  image.close();
  session.close();
}
```

FeatureHub uses `bigint` IDs so the complete signed 64-bit C API ID range remains lossless:

```ts
FeatureHub.enable({
  primaryKeyMode: PrimaryKeyMode.MANUAL_INPUT,
  searchMode: SearchMode.EXHAUSTIVE,
  searchThreshold: 0.48
});
FeatureHub.insert(feature, 1001n);
const match = FeatureHub.search(feature);
FeatureHub.disable();
```

The standard HarmonyOS build does not include CUDA, CoreML, OpenCV image I/O/GUI, RGA, or RKNN. Their parity methods remain present but report `UNSUPPORTED` where the underlying C API provides a status. Raw-buffer image input, MNN inference, bitmap memory operations, tracking, pipeline processing, and FeatureHub are available.

See `examples/basic.ets` for the smallest tracking flow.
