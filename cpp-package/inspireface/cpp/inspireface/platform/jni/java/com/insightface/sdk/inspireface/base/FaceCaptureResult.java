package com.insightface.sdk.inspireface.base;

/** Independently copied Java view of one selected capture candidate. */
public final class FaceCaptureResult {
    public long frameId;
    public long timestampMs;
    public int trackId = -1;
    public int trackCount;
    public float score;
    public FaceRect rect;
    public float roll;
    public float yaw;
    public float pitch;
    public byte[] token;
    public FaceCaptureMetrics metrics = new FaceCaptureMetrics();

    public FaceCaptureResult() {}
}
