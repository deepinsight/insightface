package com.example.inspireface_example.view;

import android.graphics.RectF;
import android.os.SystemClock;

import androidx.annotation.Nullable;

import com.insightface.sdk.inspireface.InspireFace;
import com.insightface.sdk.inspireface.base.FaceEulerAngle;
import com.insightface.sdk.inspireface.base.ImageStream;
import com.insightface.sdk.inspireface.base.MultipleFaceData;
import com.insightface.sdk.inspireface.base.Point2f;
import com.insightface.sdk.inspireface.base.Session;

/**
 * Per-frame pipeline: YUV_420_888 → NV21 → rotate upright → InspireFace track → mode
 * logic → UI.
 *
 * Frames are handed to the SDK already upright with CAMERA_ROTATION_0 (see
 * {@link Nv21Converter} for why), so every SDK coordinate is directly in preview
 * orientation. Everything runs on the single-threaded analysis executor, which also
 * keeps all session calls serialized. The ImageProxy is closed as soon as the NV21 copy
 * exists so CameraX can refill the buffer while inference runs, and the stream is
 * created, used and released within this frame because the SDK aliases the byte[]
 * rather than copying it.
 */
final class FaceAnalyzer extends UprightFaceCameraAnalyzer {

    interface Listener {
        /** Called on the analysis thread. */
        void onUiState(LivenessController.UiState state);

        /** Called on the analysis thread, throttled. */
        void onPerf(double fps, long latencyMs);

        /**
         * Called on the analysis thread while the euler readout is enabled, throttled.
         * {@code angle} is null when no face is tracked.
         */
        void onEulerAngles(FaceEulerAngle angle);

        /** Called once on the analysis thread if the session cannot be created. */
        void onSessionError();
    }

    private static final long PERF_REPORT_INTERVAL_MS = 500;
    private static final long EULER_REPORT_INTERVAL_MS = 100;

    private static final int LANDMARK_FLOATS = 106 * 2;

    private final LivenessController controller;
    private final FaceOverlayView overlay;
    private final LandmarkGlView landmarkView;
    private final Listener listener;
    /** Front camera previews are displayed mirrored, back camera ones are not. */
    private volatile boolean mirrored;
    private float[] landmarkScratch = new float[LANDMARK_FLOATS];

    private volatile boolean eulerEnabled;
    private volatile boolean landmarksEnabled;
    private long lastEulerReport;

    // Perf tracking
    private long windowStart;
    private int windowFrames;
    private double fps;
    private double emaLatencyMs;
    private long lastPerfReport;

    FaceAnalyzer(LivenessController controller, FaceOverlayView overlay,
                 LandmarkGlView landmarkView, Listener listener, boolean mirrored) {
        this.controller = controller;
        this.overlay = overlay;
        this.landmarkView = landmarkView;
        this.listener = listener;
        this.mirrored = mirrored;
    }

    @Override
    protected Session createSession() {
        return FaceEngine.createPreviewSession();
    }

    @Override
    protected void onFaces(Session session, ImageStream stream,
                           @Nullable MultipleFaceData faces, byte[] uprightNv21,
                           int uprightWidth, int uprightHeight, long frameStart) {
        if (faces == null) {
            return;
        }
        LivenessController.UiState state =
                controller.onFrame(session, stream, faces, uprightWidth, uprightHeight);
        overlay.submit(buildOverlayFrame(faces, uprightWidth, uprightHeight, state.boxColor));
        listener.onUiState(state);
        reportEulerAngles(faces);
        renderLandmarks(faces, uprightWidth, uprightHeight);
        trackPerf(frameStart);
    }

    @Override
    protected void onSessionError() {
        listener.onSessionError();
    }

    /** Safe to call from any thread. */
    void setEulerEnabled(boolean enabled) {
        eulerEnabled = enabled;
    }

    /**
     * Safe to call from any thread; flip together with the camera lens. Rotation needs
     * no per-lens handling — every frame is pre-rotated by its own rotationDegrees — so
     * mirroring is the only display difference between the lenses.
     */
    void setMirrored(boolean mirrored) {
        this.mirrored = mirrored;
    }

    /** Safe to call from any thread. */
    void setLandmarksEnabled(boolean enabled) {
        landmarksEnabled = enabled;
    }

    /** Dense 106-point landmarks for every tracked face, batched into one GL submission. */
    private void renderLandmarks(MultipleFaceData faces, int uprightWidth, int uprightHeight) {
        if (!landmarksEnabled) {
            return;
        }
        if (faces.detectedNum == 0) {
            landmarkView.clearPoints();
            return;
        }
        int needed = faces.detectedNum * LANDMARK_FLOATS;
        if (landmarkScratch.length < needed) {
            landmarkScratch = new float[needed];
        }
        int n = 0;
        for (int i = 0; i < faces.detectedNum; i++) {
            Point2f[] landmarks = InspireFace.GetFaceDenseLandmarkFromFaceToken(faces.tokens[i]);
            if (landmarks == null) {
                continue;
            }
            for (Point2f p : landmarks) {
                landmarkScratch[n++] = p.x;
                landmarkScratch[n++] = p.y;
            }
        }
        landmarkView.submitPoints(landmarkScratch, n / 2, uprightWidth, uprightHeight, mirrored);
    }

    private void reportEulerAngles(MultipleFaceData faces) {
        if (!eulerEnabled) {
            return;
        }
        long now = SystemClock.elapsedRealtime();
        if (now - lastEulerReport < EULER_REPORT_INTERVAL_MS) {
            return;
        }
        lastEulerReport = now;
        // Only angles[0] is trustworthy — the 1.2.0 JNI writes face[0]'s angles into
        // every slot, so per-face readout beyond the first face is impossible anyway.
        listener.onEulerAngles(faces.detectedNum > 0 ? faces.angles[0] : null);
    }

    /** The stream is already upright, so SDK rects map to the preview directly. */
    private FaceOverlayView.Frame buildOverlayFrame(MultipleFaceData faces,
                                                    int uprightWidth, int uprightHeight, int color) {
        RectF[] rects = new RectF[faces.detectedNum];
        for (int i = 0; i < faces.detectedNum; i++) {
            rects[i] = new RectF(faces.rects[i].x, faces.rects[i].y,
                    faces.rects[i].x + faces.rects[i].width,
                    faces.rects[i].y + faces.rects[i].height);
        }
        return new FaceOverlayView.Frame(
                uprightWidth, uprightHeight, mirrored, rects, color);
    }

    private void trackPerf(long start) {
        long now = SystemClock.elapsedRealtime();
        long cost = now - start;
        emaLatencyMs = emaLatencyMs == 0 ? cost : emaLatencyMs * 0.85 + cost * 0.15;
        windowFrames++;
        if (windowStart == 0) {
            windowStart = now;
        } else if (now - windowStart >= 1000) {
            fps = windowFrames * 1000.0 / (now - windowStart);
            windowFrames = 0;
            windowStart = now;
        }
        if (now - lastPerfReport >= PERF_REPORT_INTERVAL_MS && fps > 0) {
            lastPerfReport = now;
            listener.onPerf(fps, Math.round(emaLatencyMs));
        }
    }
}
