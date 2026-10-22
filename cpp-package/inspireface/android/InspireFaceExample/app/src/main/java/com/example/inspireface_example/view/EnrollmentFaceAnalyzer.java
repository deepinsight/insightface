package com.example.inspireface_example.view;

import android.content.Context;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.YuvImage;
import android.os.SystemClock;

import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;

import com.example.inspireface_example.R;
import com.insightface.sdk.inspireface.base.FaceRect;
import com.insightface.sdk.inspireface.base.ImageStream;
import com.insightface.sdk.inspireface.base.MultipleFaceData;
import com.insightface.sdk.inspireface.base.Session;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

/** Tracks SDK face 0 and captures it after one warm-up second plus two stable seconds. */
final class EnrollmentFaceAnalyzer extends UprightFaceCameraAnalyzer {

    enum Stage { NO_FACE, MOVE_CLOSER, HOLD_STILL, CAPTURING, COMPLETE }

    interface Listener {
        /** Called on the analysis thread. Progress is meaningful only for CAPTURING. */
        void onState(Stage stage, float progress);

        /** Called once with an app-private temporary JPEG. */
        void onCaptured(String path);

        void onCaptureError();

        void onSessionError();
    }

    private static final long STATE_REPORT_INTERVAL_MS = 80L;
    private static final float MIN_FACE_WIDTH_RATIO = 0.18f;

    private final FaceOverlayView overlay;
    private final Listener listener;
    private final File captureDirectory;
    private final FaceStabilityGate stabilityGate = new FaceStabilityGate();
    private final int accentColor;
    private final int redColor;
    private final int yellowColor;
    private final int greenColor;
    private long lastStateReport;
    private Stage lastStage;
    private int lastProgressBucket = -1;
    private volatile boolean mirrored = true;
    private volatile boolean resetRequested;
    private boolean captured;

    EnrollmentFaceAnalyzer(Context context, FaceOverlayView overlay,
                           File captureDirectory, Listener listener) {
        this.overlay = overlay;
        this.captureDirectory = captureDirectory;
        this.listener = listener;
        accentColor = ContextCompat.getColor(context, R.color.liveness_accent);
        redColor = ContextCompat.getColor(context, R.color.liveness_fail);
        yellowColor = ContextCompat.getColor(context, R.color.liveness_warn);
        greenColor = accentColor;
    }

    @Override
    protected boolean shouldSkipFrame() {
        return captured;
    }

    @Override
    protected void beforeFrame() {
        if (resetRequested) {
            resetRequested = false;
            clearStability();
        }
    }

    @Override
    protected Session createSession() {
        return FaceEngine.createTrackingSession();
    }

    @Override
    protected void onFaces(Session session, ImageStream stream,
                           @Nullable MultipleFaceData faces, byte[] upright,
                           int uprightWidth, int uprightHeight, long frameStart) {
        if (faces == null || faces.detectedNum == 0) {
            clearStability();
            overlay.submit(null);
            reportState(Stage.NO_FACE, 0f);
            return;
        }
        FaceRect first = faces.rects[0];
        RectF face = new RectF(first.x, first.y,
                first.x + first.width, first.y + first.height);
        int trackId = faces.trackIds != null && faces.trackIds.length > 0
                ? faces.trackIds[0] : 0;
        if (face.width() < uprightWidth * MIN_FACE_WIDTH_RATIO) {
            clearStability();
            submitFrame(face, uprightWidth, uprightHeight,
                    -1f, redColor, redColor);
            reportState(Stage.MOVE_CLOSER, 0f);
            return;
        }

        float progress = stabilityGate.update(trackId,
                face.left, face.top, face.right, face.bottom,
                SystemClock.elapsedRealtime());
        if (progress < 0f) {
            submitFrame(face, uprightWidth, uprightHeight,
                    -1f, accentColor, accentColor);
            reportState(Stage.HOLD_STILL, 0f);
            return;
        }

        int progressColor = progress >= 1f ? greenColor
                : progress >= 0.5f ? yellowColor : redColor;
        submitFrame(face, uprightWidth, uprightHeight,
                progress, progressColor, progressColor);
        reportState(progress >= 1f ? Stage.COMPLETE : Stage.CAPTURING, progress);
        if (progress >= 1f) {
            File capture = writeCapture(upright, uprightWidth, uprightHeight, face);
            if (capture == null) {
                clearStability();
                listener.onCaptureError();
            } else {
                captured = true;
                listener.onCaptured(capture.getAbsolutePath());
            }
        }
    }

    @Override
    protected void onSessionError() {
        listener.onSessionError();
    }

    void setMirrored(boolean mirrored) {
        this.mirrored = mirrored;
    }

    void resetTracking() {
        resetRequested = true;
        overlay.submit(null);
    }

    private void clearStability() {
        stabilityGate.reset();
    }

    private void submitFrame(RectF face, int imageWidth, int imageHeight,
                             float progress, int boxColor, int progressColor) {
        overlay.submit(new FaceOverlayView.Frame(imageWidth, imageHeight, mirrored,
                new RectF[]{new RectF(face)}, boxColor, progress, progressColor));
    }

    private void reportState(Stage stage, float progress) {
        long now = SystemClock.elapsedRealtime();
        int bucket = Math.round(progress * 100f);
        if (stage != lastStage || bucket != lastProgressBucket
                || now - lastStateReport >= STATE_REPORT_INTERVAL_MS) {
            lastStage = stage;
            lastProgressBucket = bucket;
            lastStateReport = now;
            listener.onState(stage, progress);
        }
    }

    private File writeCapture(byte[] nv21, int width, int height, RectF face) {
        if ((!captureDirectory.exists() && !captureDirectory.mkdirs())
                || !captureDirectory.isDirectory()) {
            return null;
        }
        File destination;
        try {
            destination = File.createTempFile("face_", ".jpg", captureDirectory);
        } catch (IOException e) {
            return null;
        }
        Rect crop = paddedEvenCrop(face, width, height);
        try (FileOutputStream output = new FileOutputStream(destination)) {
            boolean compressed = new YuvImage(
                    nv21, ImageFormat.NV21, width, height, null)
                    .compressToJpeg(crop, 94, output);
            if (!compressed) {
                destination.delete();
                return null;
            }
            output.flush();
            output.getFD().sync();
            return destination;
        } catch (IOException | RuntimeException e) {
            destination.delete();
            return null;
        }
    }

    private static Rect paddedEvenCrop(RectF face, int width, int height) {
        float paddingX = face.width() * 0.45f;
        float paddingY = face.height() * 0.55f;
        int left = Math.max(0, ((int) Math.floor(face.left - paddingX)) & ~1);
        int top = Math.max(0, ((int) Math.floor(face.top - paddingY)) & ~1);
        int right = Math.min(width, ((int) Math.ceil(face.right + paddingX) + 1) & ~1);
        int bottom = Math.min(height, ((int) Math.ceil(face.bottom + paddingY) + 1) & ~1);
        if (right <= left + 2 || bottom <= top + 2) {
            return new Rect(0, 0, width & ~1, height & ~1);
        }
        return new Rect(left, top, right, bottom);
    }
}
