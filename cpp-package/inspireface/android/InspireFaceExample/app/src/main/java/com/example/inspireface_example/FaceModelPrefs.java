package com.example.inspireface_example;

import android.content.Context;

/** Process-wide model selection shared by the home screen and all feature screens. */
public final class FaceModelPrefs {

    public enum Model {
        PIKACHU("Pikachu"),
        MEGATRON("Megatron");

        private final String sdkName;

        Model(String sdkName) {
            this.sdkName = sdkName;
        }

        public String sdkName() {
            return sdkName;
        }

        static Model fromStored(String value) {
            for (Model model : values()) {
                if (model.sdkName.equals(value)) {
                    return model;
                }
            }
            return MEGATRON;
        }
    }

    private static final String PREFS = "settings";
    private static final String KEY_MODEL = "face_model";

    private FaceModelPrefs() {
    }

    /** Megatron remains the default to preserve the project's previous behavior. */
    public static Model get(Context context) {
        String value = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .getString(KEY_MODEL, Model.MEGATRON.sdkName());
        return Model.fromStored(value);
    }

    public static void set(Context context, Model model) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .edit().putString(KEY_MODEL, model.sdkName()).apply();
    }
}
