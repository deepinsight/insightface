package com.example.inspireface_example.face;

import android.graphics.Bitmap;
import android.graphics.RectF;

import com.insightface.sdk.inspireface.InspireFace;
import com.insightface.sdk.inspireface.base.CustomParameter;
import com.insightface.sdk.inspireface.base.FaceAttributeResult;
import com.insightface.sdk.inspireface.base.FaceInteractionState;
import com.insightface.sdk.inspireface.base.FaceInteractionsActions;
import com.insightface.sdk.inspireface.base.FaceMaskConfidence;
import com.insightface.sdk.inspireface.base.FaceQualityConfidence;
import com.insightface.sdk.inspireface.base.FaceRect;
import com.insightface.sdk.inspireface.base.ImageStream;
import com.insightface.sdk.inspireface.base.MultipleFaceData;
import com.insightface.sdk.inspireface.base.Session;

/** Runs still-image face detection and the SDK attribute pipeline in one stream lifetime. */
public final class FaceAttributeProcessor {

    private static final int INTERACTION_PIPELINE_CALLS = 10;

    public enum Status { READY, NO_FACE, PROCESS_FAILED }

    public static final class Attribute {
        public final float maskConfidence;
        public final int ageBracket;
        public final float qualityScore;
        public final int jawOpen;
        public final int race;
        public final int gender;
        public final float leftEyeConfidence;
        public final float rightEyeConfidence;

        Attribute(float maskConfidence, int ageBracket, float qualityScore,
                  int jawOpen, int race, int gender,
                  float leftEyeConfidence, float rightEyeConfidence) {
            this.maskConfidence = maskConfidence;
            this.ageBracket = ageBracket;
            this.qualityScore = qualityScore;
            this.jawOpen = jawOpen;
            this.race = race;
            this.gender = gender;
            this.leftEyeConfidence = leftEyeConfidence;
            this.rightEyeConfidence = rightEyeConfidence;
        }
    }

    public static final class Result {
        public final Status status;
        public final FaceImageProcessor.Candidate[] candidates;
        public final Attribute[] attributes;

        Result(Status status, FaceImageProcessor.Candidate[] candidates,
               Attribute[] attributes) {
            this.status = status;
            this.candidates = candidates;
            this.attributes = attributes;
        }

        public RectF[] faceRects() {
            RectF[] rects = new RectF[candidates.length];
            for (int i = 0; i < candidates.length; i++) {
                rects[i] = new RectF(candidates[i].rect);
            }
            return rects;
        }
    }

    private FaceAttributeProcessor() {
    }

    public static Result analyze(Session session, Bitmap bitmap) {
        ImageStream stream = InspireFace.CreateImageStreamFromBitmap(
                bitmap, InspireFace.CAMERA_ROTATION_0);
        if (stream == null) {
            return empty(Status.PROCESS_FAILED);
        }
        try {
            MultipleFaceData faces = InspireFace.ExecuteFaceTrack(session, stream);
            if (faces == null) {
                return empty(Status.PROCESS_FAILED);
            }
            if (faces.detectedNum <= 0) {
                return empty(Status.NO_FACE);
            }

            FaceImageProcessor.Candidate[] candidates = candidates(faces);
            CustomParameter pipeline = InspireFace.CreateCustomParameter()
                    .enableMaskDetect(true)
                    .enableFaceQuality(true)
                    .enableFaceAttribute(true)
                    .enableInteractionLiveness(true);
            if (!InspireFace.MultipleFacePipelineProcess(
                    session, stream, faces, pipeline)) {
                return new Result(Status.PROCESS_FAILED, candidates, new Attribute[0]);
            }

            FaceMaskConfidence masks = InspireFace.GetFaceMaskConfidence(session);
            FaceQualityConfidence qualities = InspireFace.GetFaceQualityConfidence(session);
            FaceAttributeResult faceAttributes = InspireFace.GetFaceAttributeResult(session);

            // The interaction module has a short fresh-track warm-up. Reusing the same
            // still frame lets it produce deterministic eye and jaw state without changing
            // the already cached mask, quality and demographic results above.
            CustomParameter interaction = InspireFace.CreateCustomParameter()
                    .enableInteractionLiveness(true);
            for (int i = 1; i < INTERACTION_PIPELINE_CALLS; i++) {
                InspireFace.MultipleFacePipelineProcess(
                        session, stream, faces, interaction);
            }
            FaceInteractionState eyeStates =
                    InspireFace.GetFaceInteractionStateResult(session);
            FaceInteractionsActions actions =
                    InspireFace.GetFaceInteractionActionsResult(session);

            Attribute[] attributes = new Attribute[candidates.length];
            for (int i = 0; i < attributes.length; i++) {
                attributes[i] = new Attribute(
                        floatAt(masks == null ? null : masks.confidence, i),
                        intAt(faceAttributes == null ? null : faceAttributes.ageBracket, i),
                        floatAt(qualities == null ? null : qualities.confidence, i),
                        intAt(actions == null ? null : actions.jawOpen, i),
                        intAt(faceAttributes == null ? null : faceAttributes.race, i),
                        intAt(faceAttributes == null ? null : faceAttributes.gender, i),
                        floatAt(eyeStates == null ? null
                                : eyeStates.leftEyeStatusConfidence, i),
                        floatAt(eyeStates == null ? null
                                : eyeStates.rightEyeStatusConfidence, i));
            }
            return new Result(Status.READY, candidates, attributes);
        } finally {
            InspireFace.ReleaseImageStream(stream);
        }
    }

    private static FaceImageProcessor.Candidate[] candidates(MultipleFaceData faces) {
        FaceImageProcessor.Candidate[] candidates =
                new FaceImageProcessor.Candidate[faces.detectedNum];
        for (int i = 0; i < faces.detectedNum; i++) {
            FaceRect face = faces.rects[i];
            RectF rect = new RectF(face.x, face.y,
                    face.x + face.width, face.y + face.height);
            candidates[i] = new FaceImageProcessor.Candidate(
                    rect, null, null, null);
        }
        return candidates;
    }

    private static Result empty(Status status) {
        return new Result(status, new FaceImageProcessor.Candidate[0], new Attribute[0]);
    }

    private static float floatAt(float[] values, int index) {
        return values != null && index >= 0 && index < values.length
                ? values[index] : Float.NaN;
    }

    private static int intAt(int[] values, int index) {
        return values != null && index >= 0 && index < values.length ? values[index] : -1;
    }
}
