package com.example.inspireface_example.view;

import android.os.SystemClock;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;

import com.insightface.sdk.inspireface.InspireFace;
import com.insightface.sdk.inspireface.base.ImageStream;
import com.insightface.sdk.inspireface.base.MultipleFaceData;
import com.insightface.sdk.inspireface.base.Session;

/**
 * Shared camera-to-InspireFace pipeline. Subclasses receive an upright NV21 frame,
 * tracked faces and a live stream while this class owns conversion and native lifetimes.
 */
public abstract class UprightFaceCameraAnalyzer implements ImageAnalysis.Analyzer {

    private final Nv21Converter converter = new Nv21Converter();
    private Session session;
    private boolean sessionFailed;
    private volatile boolean released;

    @Override
    public final void analyze(@NonNull ImageProxy image) {
        if (released || sessionFailed || shouldSkipFrame()) {
            image.close();
            return;
        }
        beforeFrame();
        if (session == null) {
            session = createSession();
            if (session == null) {
                sessionFailed = true;
                image.close();
                onSessionError();
                return;
            }
            onSessionReady();
        }

        long frameStart = SystemClock.elapsedRealtime();
        int width = image.getWidth();
        int height = image.getHeight();
        int rotationDegrees = image.getImageInfo().getRotationDegrees();
        byte[] nv21;
        try {
            nv21 = converter.convert(image);
        } finally {
            image.close();
        }
        byte[] upright = converter.rotateUpright(nv21, width, height, rotationDegrees);
        boolean swapped = rotationDegrees == 90 || rotationDegrees == 270;
        int uprightWidth = swapped ? height : width;
        int uprightHeight = swapped ? width : height;

        ImageStream stream = InspireFace.CreateImageStreamFromByteBuffer(
                upright, uprightWidth, uprightHeight,
                InspireFace.STREAM_YUV_NV21, InspireFace.CAMERA_ROTATION_0);
        if (stream == null) {
            return;
        }
        try {
            MultipleFaceData faces = InspireFace.ExecuteFaceTrack(session, stream);
            onFaces(session, stream, faces, upright,
                    uprightWidth, uprightHeight, frameStart);
        } finally {
            InspireFace.ReleaseImageStream(stream);
        }
    }

    /** Called before lazily creating the SDK session, on the analysis executor. */
    protected void beforeFrame() {
    }

    /** Lets a completed state cheaply close subsequent camera frames. */
    protected boolean shouldSkipFrame() {
        return false;
    }

    protected abstract Session createSession();

    /** Called once after lazy Session creation succeeds, on the analysis executor. */
    protected void onSessionReady() {
    }

    protected abstract void onFaces(Session session, ImageStream stream,
                                    @Nullable MultipleFaceData faces, byte[] uprightNv21,
                                    int uprightWidth, int uprightHeight, long frameStart);

    protected abstract void onSessionError();

    /** Must be queued on the same executor used by CameraX analysis. */
    public final void release() {
        released = true;
        if (session != null) {
            FaceEngine.releaseSession(session);
            session = null;
        }
    }
}
