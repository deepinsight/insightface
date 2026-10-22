package com.insightface.sdk.inspireface.base;

/** Result of assessing one frame in a face-capture session. */
public final class FaceCaptureProgress {
    public int state;
    public int candidateCount;
    public long frameId;
    public long timestampMs;
    public int trackId = -1;
    public int trackCount;
    public long evaluatedFilters;
    public long rejectReasons;
    public float progress;
    public float currentScore;
    public FaceCaptureMetrics metrics = new FaceCaptureMetrics();

    public FaceCaptureProgress() {}
}
