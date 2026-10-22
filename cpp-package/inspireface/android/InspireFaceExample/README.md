# InspireFace Android Example

A CameraX-based InspireFace Android SDK (1.2.0) example. The launcher is a square-grid
feature menu with a global model selector (`Pikachu` / `Megatron`). The selected model is
loaded when a feature page opens and is shown in a small label on every feature page.

## Try the Android app

<p>
  <a href="http://fir.tunm.top/pro/pz7b3dgv">
    <img src="docs/images/inspireface-android-example-app-download.png" width="220" alt="Download the InspireFace Android example">
  </a>
</p>

<p>
  <strong>Download the Android example and try it now.</strong><br>
  Scan the QR code or <strong><a href="http://fir.tunm.top/pro/pz7b3dgv">Download App</a></strong>.
</p>

The current menu contains eight dedicated pages:

- **Silent liveness (RGB anti-spoofing)**: streams a per-frame liveness score for the current face, averages it over a sliding window, and labels the face real/spoof against a threshold.
- **Action liveness (cooperative)**: generates a random challenge sequence (blink / head shake / mouth open / head raise), prompts each action in turn, and detects completion with per-action timeouts and face-loss failure handling.
- **Pose recognition**: displays whatever action you perform — the latest action in large text, with a fading row of smaller history entries below (at most 6 shown; rising-edge debounced, so a held pose is recorded once).
- **Face 1:1**: selects two local images, detects and numbers every face, selects face 1 by default, and lets you tap any A/B face box to immediately rerun the comparison. The circular gauge shows the converted similarity percentage and the SDK-recommended threshold verdict.
- **Face management**: searches, adds, renames, replaces, and deletes identities in the currently selected model library. Enrollment supports either gallery multi-face selection or an automatic camera flow that tracks face 1, waits for 1 stable second, then fills a 2-second red/yellow/green ring. Motion immediately resets and hides the ring.
- **Face recognition**: switches between Photo input and Video stream tabs. Photo mode detects and numbers faces, searches a single face immediately, and reruns the search when a numbered face is tapped. Its collapsed-by-default settings panel sits below the photo picker; detection input px, maximum face count, and minimum face px rebuild the Session and persist locally. Video mode reuses the CameraX tracking pipeline, tracks only face 0, searches after roughly one stable second, renders the result at the bottom, and supports front/rear cameras.
- **Face tracking**: detects and numbers every image face, then displays the selected face's SDK-native 106 dense landmarks. Medium and large faces are annotated directly; genuinely small faces use a bottom-right 148dp magnifier cropped to 2.4× the detected box. The Video tracking tab renders track-ID colors, four-corner boxes and 106 points with OpenGL.
- **Face attributes**: analyzes a selected image face for mask state, age bracket, image quality, expression state, ethnicity, gender and left/right eye state. Tapping another numbered face updates the result immediately, and small faces reuse the expanded bottom-right crop magnifier.

Debug aids on each camera page:

- **Euler angles** switch: live Yaw / Pitch / Roll readout for the tracked face (~10 Hz; note that in the 1.2.0 JNI only `angles[0]` is trustworthy, so with multiple faces the first face is shown).
- **Landmarks** rendering is currently hidden; `LandmarkGlView` remains available as an OpenGL ES 2.0 overlay for a future menu entry or debug switch.

A **Flip camera** chip switches between the front and back lens at runtime. No SDK-side
changes are needed for that: every frame is pre-rotated by its own `rotationDegrees`
before being handed to InspireFace as an upright `CAMERA_ROTATION_0` buffer, so the new
lens's sensor orientation is absorbed per frame — only the display mirroring flips, and
the mode state machine restarts.

Language: the first launch defaults to **English** regardless of the system language. The
language chip switches between English and Chinese, and Android 13+ also exposes both in
the system per-app language settings. The selected language persists across launches.

## Architecture

```
CameraX ImageAnalysis (YUV_420_888, 640x480, KEEP_ONLY_LATEST)
   └─ UprightFaceCameraAnalyzer (single-threaded analysis executor)
        ├─ Nv21Converter.convert()        YUV_420_888 → tight NV21 (VU interleave probed once, then bulk-copied)
        ├─ Nv21Converter.rotateUpright()  Java-side rotation to upright
        ├─ CreateImageStreamFromByteBuffer(nv21, CAMERA_ROTATION_0)
        ├─ ExecuteFaceTrack               LIGHT_TRACK mode
        ├─ FaceAnalyzer / EnrollmentFaceAnalyzer
        │    └─ mode pipeline or first-face stability state machine
        └─ ReleaseImageStream             released within the same frame
```

- `HomeActivity` — square-grid feature menu and global model selection
- `FaceCompareActivity` — local image decoding, face feature extraction and 1:1 comparison
- `FaceManagementActivity` — CRUD UI for model-isolated identities, crops and FeatureHub data
- `FaceRecognitionActivity` / `StillImageSessionSettings` — photo multi-face selection, model-scoped 1:N search and persisted Session parameters
- `view/RecognitionFaceAnalyzer` — video face-0 stability gate, feature extraction and model-library search
- `FaceDetectionActivity` / `widget/FaceLandmarkOverlayView` — image multi-face detection, 106-point overlay and small-face magnifier
- `FaceAttributeActivity` / `face/FaceAttributeProcessor` — selectable still-image mask, quality, demographic and interaction attributes
- `view/FaceCaptureActivity` / `EnrollmentFaceAnalyzer` — first-face stable camera enrollment and automatic capture
- `view/CameraPreviewController` — reusable CameraX preview, 4:3 analysis, lens fallback and front/rear switching
- `view/UprightFaceCameraAnalyzer` — shared YUV→upright NV21, face tracking and native stream/session lifecycle
- `face/FaceImageProcessor` / `face/FaceCropUtils` / `widget/FaceImageOverlayView` — shared multi-face extraction, expanded crop and tappable numbered boxes
- `face/FaceRepository` — model-scoped persistent FeatureHub, crop files and metadata
- `view/LivenessActivity` — shared CameraX screen and the silent-liveness entry
- `view/ActionLivenessActivity` / `PoseActivity` — dedicated routes that select their fixed controller mode
- `view/FaceAnalyzer` — liveness-mode pipeline, performance stats and debug readouts
- `view/LivenessController` — the state machines for all three modes (tunables live at the top of this class)
- `view/Nv21Converter` — fast YUV→NV21 conversion + NV21 rotation
- `view/FaceOverlayView` — face bracket overlay (center-crop mapping + front mirror)
- `view/LandmarkGlView` — OpenGL landmark overlay
- `view/FaceEngine` — model-aware GlobalLaunch/GlobalTerminate and session creation
- `FaceModelPrefs` / `LocalePrefs` / `App` — persisted global model and per-app language

## Model-isolated face storage

Pikachu and Megatron never share face features, crop images, metadata, or ID sequences.
The app stores them under separate app-private paths:

```text
files/face_hub/Pikachu/features.db
files/face_hub/Pikachu/crops/
shared_prefs/face_records_Pikachu.xml

files/face_hub/Megatron/features.db
files/face_hub/Megatron/crops/
shared_prefs/face_records_Megatron.xml
```

FeatureHub uses manual primary keys and persistent storage. Switching the global model
therefore opens a different native database as well as a different crop/metadata set.

## Key design decisions (verified against SDK source)

1. **Pre-rotate NV21 on the Java side and always pass `CAMERA_ROTATION_0`.**
   The SDK (≤1.2.3) crops RGB-liveness input using "the rotated upright full frame + an
   un-rotated face rect", so passing 90/270 rotation constants misplaces the crop and
   corrupts silent-liveness scores. Pre-rotating (~1–2 ms at 640×480) sidesteps that
   entirely, and every SDK output coordinate lands directly in display orientation, so
   the overlays only need the front-camera mirror. Also note the SDK's rotation
   constants are the *opposite* of Android's `rotationDegrees` (Android 90 → SDK
   ROTATION_270); pre-rotation avoids that trap too.

2. **Action liveness has three hard prerequisites** (miss one and actions never fire):
   - `DETECT_MODE_LIGHT_TRACK` (other modes rebuild tracked faces every frame, so the
     temporal action window never accumulates);
   - `enableInteractionLiveness` at session creation;
   - `enableFaceQuality` at session creation (loads the pose model — without it yaw/pitch
     stay 0 and shake/head-raise can never trigger).

3. **Action flag semantics** (SDK-internal 10-frame sliding window + rules):
   blink is a one-call pulse (the window resets after it); shake latches while both yaw
   extremes sit in the rolling window (~10 calls); mouth-open/head-raise are
   level-triggered. The controller therefore uses **edge gates**: each challenge step
   must observe the flag at 0 before a 1 counts, and the SDK's `normal` flag (warm-up
   indicator, also raised for ~9 calls after every blink-induced reset) quarantines the
   placeholder zeros so a pose held through a natural blink is neither double-counted
   (pose mode) nor accepted as fresh (action mode).

4. **`CreateImageStreamFromByteBuffer` does not copy** — the native stream aliases the
   byte[] until `ReleaseImageStream`. Safe pattern: create → track → pipeline → release
   within one frame, never overwriting the byte[] before release (this implementation
   reuses two persistent buffers on a single thread, which satisfies that naturally).

5. **Silent liveness**: single-frame score with the author-encoded 0.88 decision
   boundary; every pipeline call converts the full frame internally (a known SDK hot
   spot), so the pipeline runs every 2nd frame with an 8-sample sliding average — same
   accuracy, half the cost.

6. **`MultipleFaceData.angles[i]` is only valid for `i == 0`** (a 1.2.0 JNI bug writes
   face[0]'s angles into every slot); this app only reads it in single-face flows.

## Tunables

At the top of `LivenessController`:

| Parameter | Default | Meaning |
|---|---|---|
| `RGB_LIVENESS_THRESHOLD` | 0.88 | silent-liveness real/spoof boundary |
| `SCORE_WINDOW` | 8 | sliding average window for scores |
| `SILENT_PIPELINE_INTERVAL` | 2 | run the anti-spoofing pipeline every N frames |
| `ACTIONS_PER_RUN` | 3 | challenge actions per round |
| `ACTION_TIMEOUT_MS` | 8000 | per-action timeout |
| `MIN_FACE_WIDTH_RATIO` | 0.18 | minimum face width as a fraction of frame width |
| `POSE_HISTORY_MAX` | 6 | pose-mode entries shown (1 large + 5 history) |

## Requirements

- JDK 17 (required by AGP 8.6.1; Android Studio's embedded JDK works)
- Android Studio Ladybug+ — the Gradle 8.7 wrapper is committed, no local Gradle needed
- Network access to `google()`, `mavenCentral()` and `jitpack.io` on first sync
  (the InspireFace SDK and its bundled model packs resolve from JitPack)
- An ARM Android device running Android 7.0 / API 24 or newer. The app compiles and targets
  Android 15 / API 35; Android has no declared upper install limit.
- The SDK ships arm64-v8a / armeabi-v7a only, so x86/x86_64 emulators and Intel-only
  ChromeOS devices cannot run the native face engine. The arm64 native libraries and the
  compatibility bridge are built/aligned for Android 15's 16 KB page-size devices.

`local.properties` is intentionally not committed; Android Studio regenerates it, or set
`ANDROID_HOME` for command-line builds.

## Running

The first feature launch is slower while the bundled model packs are unpacked from assets.

```bash
./gradlew :app:installDebug
```

This project now covers liveness, pose, 1:1 comparison, face management, and FeatureHub
1:N photo search. For other SDK capabilities, continue with the upstream
[InspireFace](https://github.com/HyperInspire/InspireFace) Android example.
