package com.example.inspireface_example;

import android.content.Context;
import android.content.SharedPreferences;

/** Cached still-image detection settings shared by photo-based SDK demos. */
final class StillImageSessionSettings {

    static final int DEFAULT_INPUT_PX = DetectorDefaults.INPUT_PX;
    static final int DEFAULT_MAX_FACES = 10;
    static final int DEFAULT_MIN_FACE_PX = 24;

    static final class Values {
        final int inputPx;
        final int maxFaces;
        final int minFacePx;

        Values(int inputPx, int maxFaces, int minFacePx) {
            this.inputPx = validInputPx(inputPx);
            this.maxFaces = validMaxFaces(maxFaces);
            this.minFacePx = validMinFacePx(minFacePx);
        }
    }

    // Keep the original preference name so existing recognition settings migrate intact.
    private static final String PREFS = "face_recognition_session";
    private static final String KEY_INPUT_PX = "input_px";
    private static final String KEY_MAX_FACES = "max_faces";
    private static final String KEY_MIN_FACE_PX = "min_face_px";

    private StillImageSessionSettings() {
    }

    static Values defaults() {
        return new Values(DEFAULT_INPUT_PX, DEFAULT_MAX_FACES, DEFAULT_MIN_FACE_PX);
    }

    static Values load(Context context) {
        SharedPreferences preferences = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        return new Values(
                preferences.getInt(KEY_INPUT_PX, DEFAULT_INPUT_PX),
                preferences.getInt(KEY_MAX_FACES, DEFAULT_MAX_FACES),
                preferences.getInt(KEY_MIN_FACE_PX, DEFAULT_MIN_FACE_PX));
    }

    static void save(Context context, Values values) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .edit()
                .putInt(KEY_INPUT_PX, values.inputPx)
                .putInt(KEY_MAX_FACES, values.maxFaces)
                .putInt(KEY_MIN_FACE_PX, values.minFacePx)
                .apply();
    }

    private static int validInputPx(int value) {
        return value == 320 || value == 1280 ? value : DEFAULT_INPUT_PX;
    }

    private static int validMaxFaces(int value) {
        return value == 1 || value == 3 || value == 5 ? value : DEFAULT_MAX_FACES;
    }

    private static int validMinFacePx(int value) {
        return value == 48 || value == 64 || value == 128
                ? value : DEFAULT_MIN_FACE_PX;
    }
}
