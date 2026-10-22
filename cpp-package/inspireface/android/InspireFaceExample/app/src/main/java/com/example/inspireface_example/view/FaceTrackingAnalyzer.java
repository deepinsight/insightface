package com.example.inspireface_example.view;

import android.graphics.Color;
import android.os.SystemClock;
import android.util.SparseIntArray;

import androidx.annotation.Nullable;

import com.insightface.sdk.inspireface.InspireFace;
import com.insightface.sdk.inspireface.base.FaceRect;
import com.insightface.sdk.inspireface.base.ImageStream;
import com.insightface.sdk.inspireface.base.MultipleFaceData;
import com.insightface.sdk.inspireface.base.Point2f;
import com.insightface.sdk.inspireface.base.Session;

import java.util.Arrays;

/** Runs configurable multi-face tracking and submits one batched GL frame per camera frame. */
public final class FaceTrackingAnalyzer extends UprightFaceCameraAnalyzer {

    public interface Listener {
        void onSessionReady();

        /** Throttled callback on the camera analysis executor. */
        void onStats(double fps, long latencyMs);

        void onSessionError();
    }

    private static final long REPORT_INTERVAL_MS = 300L;
    private static final int[] TRACK_COLORS = {
            0xFF00E5A0, 0xFFFFC400, 0xFF40C4FF, 0xFFFF6E9C, 0xFFC6A5FF,
            0xFFFF8A65, 0xFF76FF03, 0xFF18FFFF, 0xFFFFD180, 0xFFEA80FC
    };

    private final FaceTrackingGlView overlay;
    private final int detectMode;
    private final int detectPixelLevel;
    private final int maxFaces;
    private final int minimumFacePixelSize;
    private final Listener listener;
    private final float[] pointVertices = new float[
            FaceTrackingGlView.MAX_FACES * FaceTrackingGlView.LANDMARKS_PER_FACE
                    * FaceTrackingGlView.FLOATS_PER_VERTEX];
    private final float[] boxVertices = new float[
            FaceTrackingGlView.MAX_FACES * FaceTrackingGlView.BOX_VERTICES_PER_FACE
                    * FaceTrackingGlView.FLOATS_PER_VERTEX];
    private final SparseIntArray colorSlotByTrackId = new SparseIntArray();
    private final int[] currentTrackIds = new int[FaceTrackingGlView.MAX_FACES];
    private final boolean[] usedColorSlots = new boolean[TRACK_COLORS.length];

    private volatile boolean mirrored = true;
    private long fpsWindowStart;
    private int fpsWindowFrames;
    private double fps;
    private double latencyEmaMs;
    private long lastReport;

    public FaceTrackingAnalyzer(FaceTrackingGlView overlay,
                                int detectMode, int detectPixelLevel,
                                int maxFaces, int minimumFacePixelSize,
                                Listener listener) {
        this.overlay = overlay;
        this.detectMode = detectMode;
        this.detectPixelLevel = detectPixelLevel;
        this.maxFaces = maxFaces;
        this.minimumFacePixelSize = minimumFacePixelSize;
        this.listener = listener;
    }

    @Override
    protected Session createSession() {
        return FaceEngine.createFaceTrackingSession(
                detectMode, detectPixelLevel, maxFaces, minimumFacePixelSize);
    }

    @Override
    protected void onSessionReady() {
        listener.onSessionReady();
    }

    @Override
    protected void onFaces(Session session, ImageStream stream,
                           @Nullable MultipleFaceData faces, byte[] uprightNv21,
                           int uprightWidth, int uprightHeight, long frameStart) {
        int detected = faces == null ? 0
                : Math.min(faces.detectedNum, FaceTrackingGlView.MAX_FACES);
        if (detected <= 0) {
            overlay.clearTracking();
            reportStats(frameStart);
            return;
        }

        int pointCount = 0;
        int boxCount = 0;
        updateTrackColors(faces, detected);
        for (int i = 0; i < detected; i++) {
            int color = TRACK_COLORS[colorSlotByTrackId.get(currentTrackIds[i])];
            Point2f[] landmarks = faces.tokens != null && i < faces.tokens.length
                    && faces.tokens[i] != null
                    ? InspireFace.GetFaceDenseLandmarkFromFaceToken(faces.tokens[i]) : null;
            if (faces.rects != null && i < faces.rects.length && faces.rects[i] != null) {
                boxCount = appendBrackets(
                        boxVertices, boxCount, faces.rects[i], color);
            }
            if (landmarks == null || landmarks.length == 0) {
                continue;
            }
            for (Point2f point : landmarks) {
                if (point == null || pointCount >= FaceTrackingGlView.MAX_FACES
                        * FaceTrackingGlView.LANDMARKS_PER_FACE) {
                    continue;
                }
                appendVertex(pointVertices, pointCount++, point.x, point.y, color);
            }
        }

        overlay.submit(pointVertices, pointCount, boxVertices, boxCount,
                uprightWidth, uprightHeight, mirrored);
        reportStats(frameStart);
    }

    @Override
    protected void onSessionError() {
        overlay.clearTracking();
        listener.onSessionError();
    }

    public void setMirrored(boolean mirrored) {
        this.mirrored = mirrored;
        overlay.clearTracking();
    }

    public void clearTracking() {
        overlay.clearTracking();
    }

    private void reportStats(long frameStart) {
        long now = SystemClock.elapsedRealtime();
        long latency = Math.max(0L, now - frameStart);
        latencyEmaMs = latencyEmaMs == 0.0
                ? latency : latencyEmaMs * 0.85 + latency * 0.15;
        fpsWindowFrames++;
        if (fpsWindowStart == 0L) {
            fpsWindowStart = now;
        } else if (now - fpsWindowStart >= 1_000L) {
            fps = fpsWindowFrames * 1_000.0 / (now - fpsWindowStart);
            fpsWindowFrames = 0;
            fpsWindowStart = now;
        }
        if (now - lastReport >= REPORT_INTERVAL_MS) {
            lastReport = now;
            listener.onStats(fps, Math.round(latencyEmaMs));
        }
    }

    /** Keeps active IDs on distinct palette slots while preserving each surviving ID's color. */
    private void updateTrackColors(MultipleFaceData faces, int detected) {
        for (int i = 0; i < detected; i++) {
            int sdkTrackId = faces.trackIds != null && i < faces.trackIds.length
                    ? faces.trackIds[i] : -1;
            // A missing SDK ID still needs a unique key for this frame.
            currentTrackIds[i] = sdkTrackId >= 0 ? sdkTrackId : Integer.MIN_VALUE + i;
        }
        for (int i = colorSlotByTrackId.size() - 1; i >= 0; i--) {
            if (!containsTrackId(currentTrackIds, detected, colorSlotByTrackId.keyAt(i))) {
                colorSlotByTrackId.removeAt(i);
            }
        }
        Arrays.fill(usedColorSlots, false);
        for (int i = 0; i < colorSlotByTrackId.size(); i++) {
            usedColorSlots[colorSlotByTrackId.valueAt(i)] = true;
        }
        for (int i = 0; i < detected; i++) {
            int trackId = currentTrackIds[i];
            if (colorSlotByTrackId.indexOfKey(trackId) >= 0) {
                continue;
            }
            int slot = firstFreeColorSlot();
            colorSlotByTrackId.put(trackId, slot);
            usedColorSlots[slot] = true;
        }
    }

    private int firstFreeColorSlot() {
        for (int i = 0; i < usedColorSlots.length; i++) {
            if (!usedColorSlots[i]) {
                return i;
            }
        }
        return 0;
    }

    private static boolean containsTrackId(int[] ids, int count, int wanted) {
        for (int i = 0; i < count; i++) {
            if (ids[i] == wanted) {
                return true;
            }
        }
        return false;
    }

    private static int appendBrackets(float[] target, int vertexCount,
                                      FaceRect rect, int color) {
        float left = rect.x;
        float top = rect.y;
        float right = rect.x + rect.width;
        float bottom = rect.y + rect.height;
        float length = Math.min(rect.width, rect.height) * 0.22f;

        vertexCount = appendLine(target, vertexCount,
                left, top + length, left, top, color);
        vertexCount = appendLine(target, vertexCount,
                left, top, left + length, top, color);
        vertexCount = appendLine(target, vertexCount,
                right - length, top, right, top, color);
        vertexCount = appendLine(target, vertexCount,
                right, top, right, top + length, color);
        vertexCount = appendLine(target, vertexCount,
                left, bottom - length, left, bottom, color);
        vertexCount = appendLine(target, vertexCount,
                left, bottom, left + length, bottom, color);
        vertexCount = appendLine(target, vertexCount,
                right - length, bottom, right, bottom, color);
        vertexCount = appendLine(target, vertexCount,
                right, bottom, right, bottom - length, color);
        return vertexCount;
    }

    private static int appendLine(float[] target, int vertexCount,
                                  float x0, float y0, float x1, float y1, int color) {
        appendVertex(target, vertexCount++, x0, y0, color);
        appendVertex(target, vertexCount++, x1, y1, color);
        return vertexCount;
    }

    private static void appendVertex(float[] target, int vertexIndex,
                                     float x, float y, int color) {
        int offset = vertexIndex * FaceTrackingGlView.FLOATS_PER_VERTEX;
        target[offset] = x;
        target[offset + 1] = y;
        target[offset + 2] = Color.red(color) / 255f;
        target[offset + 3] = Color.green(color) / 255f;
        target[offset + 4] = Color.blue(color) / 255f;
        target[offset + 5] = Color.alpha(color) / 255f;
    }
}
