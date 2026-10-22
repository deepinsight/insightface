package com.example.inspireface_example.view;

import android.content.Context;
import android.graphics.RectF;
import android.os.SystemClock;

import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;

import com.example.inspireface_example.R;
import com.example.inspireface_example.face.FaceRecord;
import com.example.inspireface_example.face.FaceRepository;
import com.insightface.sdk.inspireface.InspireFace;
import com.insightface.sdk.inspireface.base.FaceFeature;
import com.insightface.sdk.inspireface.base.FaceRect;
import com.insightface.sdk.inspireface.base.ImageStream;
import com.insightface.sdk.inspireface.base.MultipleFaceData;
import com.insightface.sdk.inspireface.base.Session;

/** Tracks only SDK face 0 and searches it after roughly one stable second. */
public final class RecognitionFaceAnalyzer extends UprightFaceCameraAnalyzer {

    public enum State {
        NO_FACE, MOVE_CLOSER, HOLD_STILL, SEARCHING, MATCHED, NO_MATCH, EMPTY_LIBRARY
    }

    public interface Listener {
        /** Called on the camera analysis executor. */
        void onState(State state, @Nullable FaceRecord record,
                     float confidence, float threshold);

        void onSessionError();
    }

    private static final long STABLE_RECOGNITION_MS = 1_000L;
    private static final float MIN_FACE_WIDTH_RATIO = 0.12f;

    private final FaceOverlayView overlay;
    private final FaceRepository repository;
    private final boolean libraryEmpty;
    private final Listener listener;
    private final FaceStabilityGate stabilityGate =
            new FaceStabilityGate(STABLE_RECOGNITION_MS, 1L);
    private final int waitingColor;
    private final int matchColor;
    private final int noMatchColor;
    private final int warningColor;

    private volatile boolean mirrored = true;
    private volatile boolean resetRequested;
    private boolean recognizedStableRun;
    private State lastState;
    private int currentBoxColor;

    public RecognitionFaceAnalyzer(Context context, FaceOverlayView overlay,
                                   FaceRepository repository, boolean libraryEmpty,
                                   Listener listener) {
        this.overlay = overlay;
        this.repository = repository;
        this.libraryEmpty = libraryEmpty;
        this.listener = listener;
        waitingColor = ContextCompat.getColor(context, R.color.liveness_accent);
        matchColor = waitingColor;
        noMatchColor = ContextCompat.getColor(context, R.color.liveness_fail);
        warningColor = ContextCompat.getColor(context, R.color.liveness_warn);
        currentBoxColor = waitingColor;
    }

    @Override
    protected void beforeFrame() {
        if (resetRequested) {
            resetRequested = false;
            resetRecognition();
        }
    }

    @Override
    protected Session createSession() {
        return FaceEngine.createVideoRecognitionSession();
    }

    @Override
    protected void onFaces(Session session, ImageStream stream,
                           @Nullable MultipleFaceData faces, byte[] uprightNv21,
                           int uprightWidth, int uprightHeight, long frameStart) {
        if (faces == null || faces.detectedNum == 0) {
            resetRecognition();
            overlay.submit(null);
            report(State.NO_FACE, null, Float.NaN, Float.NaN);
            return;
        }

        FaceRect first = faces.rects[0];
        RectF face = new RectF(first.x, first.y,
                first.x + first.width, first.y + first.height);
        if (face.width() < uprightWidth * MIN_FACE_WIDTH_RATIO) {
            resetRecognition();
            currentBoxColor = warningColor;
            submitFace(face, uprightWidth, uprightHeight);
            report(State.MOVE_CLOSER, null, Float.NaN, Float.NaN);
            return;
        }

        int trackId = faces.trackIds != null && faces.trackIds.length > 0
                ? faces.trackIds[0] : 0;
        float stable = stabilityGate.update(trackId,
                face.left, face.top, face.right, face.bottom,
                SystemClock.elapsedRealtime());
        if (stable < 0f) {
            recognizedStableRun = false;
            currentBoxColor = waitingColor;
            submitFace(face, uprightWidth, uprightHeight);
            report(State.HOLD_STILL, null, Float.NaN, Float.NaN);
            return;
        }

        if (!recognizedStableRun) {
            recognizedStableRun = true;
            if (libraryEmpty) {
                currentBoxColor = warningColor;
                submitFace(face, uprightWidth, uprightHeight);
                report(State.EMPTY_LIBRARY, null, Float.NaN, Float.NaN);
            } else {
                recognize(session, stream, faces, face, uprightWidth, uprightHeight);
            }
        } else {
            submitFace(face, uprightWidth, uprightHeight);
        }
    }

    private void recognize(Session session, ImageStream stream, MultipleFaceData faces,
                           RectF face, int imageWidth, int imageHeight) {
        currentBoxColor = warningColor;
        submitFace(face, imageWidth, imageHeight);
        report(State.SEARCHING, null, Float.NaN, Float.NaN);
        FaceFeature feature = InspireFace.ExtractFaceFeature(
                session, stream, faces.tokens[0]);
        FaceRepository.SearchResult result = repository.search(feature);
        if (result.matched && result.record != null) {
            currentBoxColor = matchColor;
            report(State.MATCHED, result.record, result.confidence, result.threshold);
        } else {
            currentBoxColor = noMatchColor;
            report(State.NO_MATCH, null, result.confidence, result.threshold);
        }
        submitFace(face, imageWidth, imageHeight);
    }

    @Override
    protected void onSessionError() {
        listener.onSessionError();
    }

    public void setMirrored(boolean mirrored) {
        this.mirrored = mirrored;
    }

    public void resetTracking() {
        resetRequested = true;
        overlay.submit(null);
    }

    private void resetRecognition() {
        stabilityGate.reset();
        recognizedStableRun = false;
        currentBoxColor = waitingColor;
    }

    private void submitFace(RectF face, int imageWidth, int imageHeight) {
        overlay.submit(new FaceOverlayView.Frame(
                imageWidth, imageHeight, mirrored,
                new RectF[]{new RectF(face)}, currentBoxColor));
    }

    private void report(State state, @Nullable FaceRecord record,
                        float confidence, float threshold) {
        if (state == lastState && state != State.MATCHED && state != State.NO_MATCH) {
            return;
        }
        lastState = state;
        listener.onState(state, record, confidence, threshold);
    }
}
