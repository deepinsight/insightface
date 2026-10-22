package com.example.inspireface_example.face;

import android.graphics.Bitmap;
import android.graphics.RectF;

import androidx.annotation.Nullable;

/** Shared expanded square crop used by selectable-face magnifiers. */
public final class FaceCropUtils {

    public static final class SquareCrop {
        public final Bitmap bitmap;
        public final int left;
        public final int top;
        public final int size;

        SquareCrop(Bitmap bitmap, int left, int top, int size) {
            this.bitmap = bitmap;
            this.left = left;
            this.top = top;
            this.size = size;
        }
    }

    private FaceCropUtils() {
    }

    @Nullable
    public static SquareCrop createSquare(Bitmap source, RectF face, float scale) {
        if (source == null || source.isRecycled() || face == null
                || source.getWidth() < 2 || source.getHeight() < 2) {
            return null;
        }
        int cropSize = Math.max(2, Math.round(
                Math.max(face.width(), face.height()) * scale));
        cropSize = Math.min(cropSize, Math.min(source.getWidth(), source.getHeight()));
        int left = Math.round(face.centerX() - cropSize / 2f);
        int top = Math.round(face.centerY() - cropSize / 2f);
        left = Math.max(0, Math.min(left, source.getWidth() - cropSize));
        top = Math.max(0, Math.min(top, source.getHeight() - cropSize));
        try {
            Bitmap crop = Bitmap.createBitmap(source, left, top, cropSize, cropSize);
            return crop == source ? null : new SquareCrop(crop, left, top, cropSize);
        } catch (IllegalArgumentException ignored) {
            return null;
        }
    }
}
