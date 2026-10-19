package com.insightface.sdk.inspireface;

import com.insightface.sdk.inspireface.base.ImageStream;
import com.insightface.sdk.inspireface.base.Session;

/** Owned detection snapshot for reusing one tracking result in capture assessment. */
public final class FaceDetectionSnapshot implements AutoCloseable {
    static {
        System.loadLibrary("InspireFace");
    }

    private long nativeHandle;

    private FaceDetectionSnapshot() {}

    public static FaceDetectionSnapshot create(Session session, ImageStream stream) {
        if (session == null || session.handle == 0L || stream == null || stream.handle == 0L) {
            throw new IllegalArgumentException("Session and ImageStream must be open");
        }
        FaceDetectionSnapshot snapshot = new FaceDetectionSnapshot();
        int status = nativeCreate(session.handle, stream.handle, snapshot);
        FaceCapture.throwIfError(status, "Create face detection snapshot");
        return snapshot;
    }

    synchronized long requireHandle() {
        if (nativeHandle == 0L) {
            throw new IllegalStateException("FaceDetectionSnapshot is closed");
        }
        return nativeHandle;
    }

    public synchronized boolean isClosed() {
        return nativeHandle == 0L;
    }

    @Override
    public synchronized void close() {
        if (nativeHandle == 0L) {
            return;
        }
        int status = nativeRelease(nativeHandle);
        FaceCapture.throwIfError(status, "Release face detection snapshot");
        nativeHandle = 0L;
    }

    private static native int nativeCreate(long sessionHandle, long streamHandle,
                                           FaceDetectionSnapshot output);
    private static native int nativeRelease(long snapshotHandle);
}
