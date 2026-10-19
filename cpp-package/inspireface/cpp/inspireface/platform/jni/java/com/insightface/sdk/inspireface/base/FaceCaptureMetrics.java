package com.insightface.sdk.inspireface.base;

/** Metrics computed only for filters enabled in the capture configuration. */
public final class FaceCaptureMetrics {
    public long availableFilters;
    public float faceWidthRatio = -1.0F;
    public float centerOffsetX = -1.0F;
    public float centerOffsetY = -1.0F;
    public float stabilityScore = -1.0F;
    public float poseScore = -1.0F;
    public float qualityScore = -1.0F;
    public float sharpnessScore = -1.0F;
    public float brightnessScore = -1.0F;

    public FaceCaptureMetrics() {}
}
