package com.example.inspireface_example.view;

import android.util.Log;

import androidx.annotation.Nullable;

import com.insightface.sdk.inspireface.base.Session;

/**
 * Small compatibility bridge for the landmark flag omitted by InspireFace Android 1.2.0's
 * CreateSession JNI wrapper. Session release and all processing still use the official API.
 */
final class NativeSessionBridge {

    private static final String TAG = "NativeSessionBridge";
    private static final int ENABLE_DETECT_MODE_LANDMARK = 0x00000200;
    private static final boolean AVAILABLE;

    static {
        boolean available;
        try {
            System.loadLibrary("inspireface_session_bridge");
            available = true;
        } catch (UnsatisfiedLinkError error) {
            Log.e(TAG, "Could not load Session compatibility bridge", error);
            available = false;
        }
        AVAILABLE = available;
    }

    private NativeSessionBridge() {
    }

    @Nullable
    static Session createLandmarkSession(int detectMode, int maxFaces,
                                         int detectPixelLevel, int trackByDetectFps) {
        if (!AVAILABLE) {
            return null;
        }
        long handle = nativeCreateLandmarkSession(
                ENABLE_DETECT_MODE_LANDMARK, detectMode,
                maxFaces, detectPixelLevel, trackByDetectFps);
        if (handle == 0L) {
            return null;
        }
        Session session = new Session();
        session.handle = handle;
        return session;
    }

    private static native long nativeCreateLandmarkSession(
            int customOptions, int detectMode, int maxFaces,
            int detectPixelLevel, int trackByDetectFps);
}
