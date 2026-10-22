package com.example.inspireface_example.face;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;

import androidx.exifinterface.media.ExifInterface;

import java.io.IOException;
import java.io.InputStream;

/** Decodes a content Uri with bounded memory and applies its EXIF orientation. */
public final class ImageBitmapLoader {

    private ImageBitmapLoader() {
    }

    public static Bitmap decode(Context context, Uri uri, int maxDimension) throws IOException {
        int orientation = readOrientation(context, uri);
        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        try (InputStream input = openInput(context, uri)) {
            BitmapFactory.decodeStream(input, null, bounds);
        }

        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        options.inSampleSize = calculateSampleSize(
                bounds.outWidth, bounds.outHeight, maxDimension);
        Bitmap decoded;
        try (InputStream input = openInput(context, uri)) {
            decoded = BitmapFactory.decodeStream(input, null, options);
        }
        if (decoded == null) {
            throw new IOException("Unable to decode selected image");
        }
        return applyExifOrientation(decoded, orientation);
    }

    private static int readOrientation(Context context, Uri uri) {
        try (InputStream input = openInput(context, uri)) {
            return new ExifInterface(input).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
        } catch (IOException | RuntimeException ignored) {
            return ExifInterface.ORIENTATION_NORMAL;
        }
    }

    private static InputStream openInput(Context context, Uri uri) throws IOException {
        InputStream input = context.getContentResolver().openInputStream(uri);
        if (input == null) {
            throw new IOException("Unable to open selected image");
        }
        return input;
    }

    private static int calculateSampleSize(int width, int height, int maxDimension) {
        int longest = Math.max(width, height);
        int sample = 1;
        while (longest > 0 && longest / sample > maxDimension) {
            sample *= 2;
        }
        return sample;
    }

    private static Bitmap applyExifOrientation(Bitmap bitmap, int orientation) {
        Matrix matrix = new Matrix();
        switch (orientation) {
            case ExifInterface.ORIENTATION_FLIP_HORIZONTAL:
                matrix.setScale(-1f, 1f);
                break;
            case ExifInterface.ORIENTATION_ROTATE_180:
                matrix.setRotate(180f);
                break;
            case ExifInterface.ORIENTATION_FLIP_VERTICAL:
                matrix.setRotate(180f);
                matrix.postScale(-1f, 1f);
                break;
            case ExifInterface.ORIENTATION_TRANSPOSE:
                matrix.setRotate(90f);
                matrix.postScale(-1f, 1f);
                break;
            case ExifInterface.ORIENTATION_ROTATE_90:
                matrix.setRotate(90f);
                break;
            case ExifInterface.ORIENTATION_TRANSVERSE:
                matrix.setRotate(-90f);
                matrix.postScale(-1f, 1f);
                break;
            case ExifInterface.ORIENTATION_ROTATE_270:
                matrix.setRotate(-90f);
                break;
            case ExifInterface.ORIENTATION_NORMAL:
            default:
                return bitmap;
        }
        Bitmap oriented = Bitmap.createBitmap(
                bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        if (oriented != bitmap) {
            bitmap.recycle();
        }
        return oriented;
    }
}
