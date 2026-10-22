package com.example.inspireface_example.face;

import android.graphics.Bitmap;
import android.graphics.RectF;

import com.insightface.sdk.inspireface.InspireFace;
import com.insightface.sdk.inspireface.base.FaceFeature;
import com.insightface.sdk.inspireface.base.FaceRect;
import com.insightface.sdk.inspireface.base.ImageStream;
import com.insightface.sdk.inspireface.base.MultipleFaceData;
import com.insightface.sdk.inspireface.base.Point2f;
import com.insightface.sdk.inspireface.base.Session;

/** Detects all faces once and materializes optional features, crops, or dense landmarks. */
public final class FaceImageProcessor {

    public enum Status { READY, NO_FACE, PROCESS_FAILED }

    public static final class Candidate {
        public final RectF rect;
        public final FaceFeature feature;
        public final Bitmap crop;
        public final Point2f[] denseLandmarks;

        Candidate(RectF rect, FaceFeature feature, Bitmap crop,
                  Point2f[] denseLandmarks) {
            this.rect = rect;
            this.feature = feature;
            this.crop = crop;
            this.denseLandmarks = denseLandmarks;
        }
    }

    public static final class Result {
        public final Status status;
        public final Candidate[] candidates;

        Result(Status status, Candidate[] candidates) {
            this.status = status;
            this.candidates = candidates;
        }

        public RectF[] faceRects() {
            RectF[] rects = new RectF[candidates.length];
            for (int i = 0; i < candidates.length; i++) {
                rects[i] = new RectF(candidates[i].rect);
            }
            return rects;
        }

    }

    private FaceImageProcessor() {
    }

    /** All calls must remain on the thread that owns {@code session}. */
    public static Result detect(Session session, Bitmap bitmap, boolean createCrops) {
        return detect(session, bitmap, createCrops, true, false);
    }

    /** Detection-only variant that materializes the SDK's native 106-point landmarks. */
    public static Result detectWithLandmarks(Session session, Bitmap bitmap) {
        return detect(session, bitmap, false, false, true);
    }

    private static Result detect(Session session, Bitmap bitmap, boolean createCrops,
                                 boolean extractFeatures, boolean extractLandmarks) {
        ImageStream stream = InspireFace.CreateImageStreamFromBitmap(
                bitmap, InspireFace.CAMERA_ROTATION_0);
        if (stream == null) {
            return new Result(Status.PROCESS_FAILED, new Candidate[0]);
        }
        try {
            MultipleFaceData faces = InspireFace.ExecuteFaceTrack(session, stream);
            if (faces == null) {
                return new Result(Status.PROCESS_FAILED, new Candidate[0]);
            }
            if (faces.detectedNum == 0) {
                return new Result(Status.NO_FACE, new Candidate[0]);
            }
            Candidate[] candidates = new Candidate[faces.detectedNum];
            for (int i = 0; i < faces.detectedNum; i++) {
                FaceRect face = faces.rects[i];
                RectF rect = new RectF(face.x, face.y,
                        face.x + face.width, face.y + face.height);
                FaceFeature feature = extractFeatures
                        ? InspireFace.ExtractFaceFeature(session, stream, faces.tokens[i])
                        : null;
                Point2f[] denseLandmarks = extractLandmarks
                        ? InspireFace.GetFaceDenseLandmarkFromFaceToken(faces.tokens[i])
                        : null;
                Bitmap crop = null;
                if (createCrops) {
                    crop = InspireFace.GetFaceAlignmentImage(session, stream, faces.tokens[i]);
                    if (crop == null) {
                        crop = cropFace(bitmap, rect);
                    } else if (crop == bitmap) {
                        // Keep crop ownership independent from the source image. The editor
                        // recycles source and crop bitmaps on separate lifecycle paths.
                        crop = independentCopy(bitmap);
                    }
                }
                candidates[i] = new Candidate(rect, feature, crop, denseLandmarks);
            }
            return new Result(Status.READY, candidates);
        } finally {
            InspireFace.ReleaseImageStream(stream);
        }
    }

    public static void recycleCrops(Candidate[] candidates, int exceptIndex) {
        if (candidates == null) {
            return;
        }
        Bitmap keptCrop = exceptIndex >= 0 && exceptIndex < candidates.length
                ? candidates[exceptIndex].crop : null;
        for (int i = 0; i < candidates.length; i++) {
            Bitmap crop = candidates[i].crop;
            if (i != exceptIndex && crop != null && crop != keptCrop && !crop.isRecycled()) {
                crop.recycle();
            }
        }
    }

    private static Bitmap cropFace(Bitmap bitmap, RectF face) {
        float paddingX = face.width() * 0.18f;
        float paddingY = face.height() * 0.18f;
        int left = Math.max(0, Math.round(face.left - paddingX));
        int top = Math.max(0, Math.round(face.top - paddingY));
        int right = Math.min(bitmap.getWidth(), Math.round(face.right + paddingX));
        int bottom = Math.min(bitmap.getHeight(), Math.round(face.bottom + paddingY));
        if (right <= left || bottom <= top) {
            return null;
        }
        Bitmap crop = Bitmap.createBitmap(bitmap, left, top, right - left, bottom - top);
        return crop == bitmap ? independentCopy(bitmap) : crop;
    }

    private static Bitmap independentCopy(Bitmap source) {
        return source.copy(Bitmap.Config.ARGB_8888, false);
    }
}
