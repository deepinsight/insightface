package com.insightface.sdk.inspireface;

import com.insightface.sdk.inspireface.base.FaceCaptureConfig;
import com.insightface.sdk.inspireface.base.FaceCaptureProgress;
import com.insightface.sdk.inspireface.base.FaceCaptureResult;
import com.insightface.sdk.inspireface.base.FaceCaptureResultBuffer;
import com.insightface.sdk.inspireface.base.ImageStream;
import com.insightface.sdk.inspireface.base.Session;

/** Synchronous face-capture policy. It owns no camera, worker thread, or executor. */
public final class FaceCapture implements AutoCloseable {
    public static final long FILTER_NONE = 0L;
    public static final long FILTER_FACE_COUNT = 1L << 0;
    public static final long FILTER_FACE_SIZE = 1L << 1;
    public static final long FILTER_FACE_POSITION = 1L << 2;
    public static final long FILTER_FACE_BOUNDARY = 1L << 3;
    public static final long FILTER_STABILITY = 1L << 4;
    public static final long FILTER_POSE = 1L << 5;
    public static final long FILTER_QUALITY = 1L << 6;
    public static final long FILTER_SHARPNESS = 1L << 7;
    public static final long FILTER_BRIGHTNESS = 1L << 8;
    public static final long FILTER_TRACK_COUNT = 1L << 9;

    public static final long REJECT_NONE = 0L;
    public static final long REJECT_NO_FACE = 1L << 0;
    public static final long REJECT_MULTIPLE_FACES = 1L << 1;
    public static final long REJECT_FACE_TOO_SMALL = 1L << 2;
    public static final long REJECT_FACE_TOO_LARGE = 1L << 3;
    public static final long REJECT_FACE_OFF_CENTER = 1L << 4;
    public static final long REJECT_FACE_OUT_OF_BOUNDS = 1L << 5;
    public static final long REJECT_UNSTABLE = 1L << 6;
    public static final long REJECT_POSE = 1L << 7;
    public static final long REJECT_QUALITY = 1L << 8;
    public static final long REJECT_SHARPNESS = 1L << 9;
    public static final long REJECT_BRIGHTNESS = 1L << 10;
    public static final long REJECT_TRACK_COUNT_TOO_LOW = 1L << 11;

    public static final int STATE_IDLE = 0;
    public static final int STATE_STABILIZING = 1;
    public static final int STATE_COLLECTING = 2;
    public static final int STATE_READY = 3;
    public static final int STATE_FINISHED = 4;
    public static final int STATE_TRACK_LOST = 5;

    static {
        System.loadLibrary("InspireFace");
    }

    private long nativeHandle;

    private FaceCapture() {}

    public static FaceCaptureConfig defaultConfig() {
        FaceCaptureConfig config = new FaceCaptureConfig();
        throwIfError(nativeGetDefaultConfig(config), "Get default face capture config");
        return config;
    }

    public static FaceCapture create(Session session, FaceCaptureConfig config) {
        if (session == null || session.handle == 0L) {
            throw new IllegalArgumentException("Session must be open");
        }
        if (config == null) {
            config = defaultConfig();
        }
        config.validate();
        FaceCapture capture = new FaceCapture();
        throwIfError(nativeCreate(session.handle, config, capture), "Create face capture session");
        return capture;
    }

    public synchronized FaceCaptureProgress update(ImageStream stream, long frameId,
                                                   long timestampMs) {
        validateUpdate(stream, frameId, timestampMs);
        FaceCaptureProgress progress = new FaceCaptureProgress();
        throwIfError(nativeUpdate(requireHandle(), stream.handle, frameId, timestampMs, progress),
                "Update face capture session");
        return progress;
    }

    /** The snapshot must describe this frame; its tracker count is immutable. */
    public synchronized FaceCaptureProgress update(ImageStream stream,
                                                   FaceDetectionSnapshot snapshot,
                                                   long frameId, long timestampMs) {
        validateUpdate(stream, frameId, timestampMs);
        if (snapshot == null) {
            throw new IllegalArgumentException("FaceDetectionSnapshot must not be null");
        }
        synchronized (snapshot) {
            FaceCaptureProgress progress = new FaceCaptureProgress();
            throwIfError(nativeUpdateWithSnapshot(requireHandle(), stream.handle,
                            snapshot.requireHandle(), frameId, timestampMs, progress),
                    "Update face capture session with snapshot");
            return progress;
        }
    }

    public synchronized FaceCaptureResult[] getResults() {
        FaceCaptureResultBuffer output = new FaceCaptureResultBuffer();
        throwIfError(nativeGetResults(requireHandle(), output), "Get face capture results");
        return output.results;
    }

    public synchronized FaceCaptureProgress finish() {
        FaceCaptureProgress progress = new FaceCaptureProgress();
        throwIfError(nativeFinish(requireHandle(), progress), "Finish face capture session");
        return progress;
    }

    public synchronized void reset() {
        throwIfError(nativeReset(requireHandle()), "Reset face capture session");
    }

    public synchronized boolean isClosed() {
        return nativeHandle == 0L;
    }

    @Override
    public synchronized void close() {
        if (nativeHandle == 0L) {
            return;
        }
        throwIfError(nativeRelease(nativeHandle), "Release face capture session");
        nativeHandle = 0L;
    }

    private long requireHandle() {
        if (nativeHandle == 0L) {
            throw new IllegalStateException("FaceCapture is closed");
        }
        return nativeHandle;
    }

    private static void validateUpdate(ImageStream stream, long frameId, long timestampMs) {
        if (stream == null || stream.handle == 0L) {
            throw new IllegalArgumentException("ImageStream must be open");
        }
        if (frameId < 0L || timestampMs < 0L) {
            throw new IllegalArgumentException("frameId and timestampMs must be non-negative");
        }
    }

    static void throwIfError(int status, String operation) {
        if (status != 0) {
            throw new FaceCaptureException(operation, status);
        }
    }

    private static native int nativeGetDefaultConfig(FaceCaptureConfig config);
    private static native int nativeCreate(long sessionHandle, FaceCaptureConfig config,
                                           FaceCapture output);
    private static native int nativeUpdate(long captureHandle, long streamHandle,
                                           long frameId, long timestampMs,
                                           FaceCaptureProgress output);
    private static native int nativeUpdateWithSnapshot(long captureHandle, long streamHandle,
                                                       long snapshotHandle, long frameId,
                                                       long timestampMs,
                                                       FaceCaptureProgress output);
    private static native int nativeGetResults(long captureHandle,
                                               FaceCaptureResultBuffer output);
    private static native int nativeFinish(long captureHandle, FaceCaptureProgress output);
    private static native int nativeReset(long captureHandle);
    private static native int nativeRelease(long captureHandle);
}
