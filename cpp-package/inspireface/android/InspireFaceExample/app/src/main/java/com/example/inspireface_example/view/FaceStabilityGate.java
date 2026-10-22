package com.example.inspireface_example.view;

/** Pure-Java first-face stability timer, separated so reset/timing behavior is testable. */
final class FaceStabilityGate {

    static final long WARMUP_MS = 1_000L;
    static final long PROGRESS_MS = 2_000L;
    private static final float FRAME_CENTER_TOLERANCE = 0.045f;
    private static final float FRAME_SIZE_TOLERANCE = 0.07f;
    private static final float ANCHOR_CENTER_TOLERANCE = 0.11f;
    private static final float ANCHOR_SIZE_TOLERANCE = 0.13f;

    private final long warmupMs;
    private final long progressMs;

    private boolean tracking;
    private int trackId;
    private long stableSince;
    private float anchorX;
    private float anchorY;
    private float anchorWidth;
    private float anchorHeight;
    private float previousX;
    private float previousY;
    private float previousWidth;
    private float previousHeight;

    FaceStabilityGate() {
        this(WARMUP_MS, PROGRESS_MS);
    }

    FaceStabilityGate(long warmupMs, long progressMs) {
        this.warmupMs = Math.max(0L, warmupMs);
        this.progressMs = Math.max(1L, progressMs);
    }

    /** Returns -1 during the configured warm-up, otherwise progress in [0, 1]. */
    float update(int id, float left, float top, float right, float bottom, long now) {
        float width = Math.max(1f, right - left);
        float height = Math.max(1f, bottom - top);
        float centerX = (left + right) * 0.5f;
        float centerY = (top + bottom) * 0.5f;
        if (!tracking || trackId != id || now < stableSince
                || !isStable(centerX, centerY, width, height)) {
            begin(id, centerX, centerY, width, height, now);
            return -1f;
        }
        previousX = centerX;
        previousY = centerY;
        previousWidth = width;
        previousHeight = height;
        long elapsed = now - stableSince;
        if (elapsed < warmupMs) {
            return -1f;
        }
        return Math.min(1f, (elapsed - warmupMs) / (float) progressMs);
    }

    void reset() {
        tracking = false;
        stableSince = 0L;
    }

    private void begin(int id, float centerX, float centerY,
                       float width, float height, long now) {
        tracking = true;
        trackId = id;
        stableSince = now;
        anchorX = previousX = centerX;
        anchorY = previousY = centerY;
        anchorWidth = previousWidth = width;
        anchorHeight = previousHeight = height;
    }

    private boolean isStable(float centerX, float centerY, float width, float height) {
        return centerDelta(centerX, centerY,
                previousX, previousY, previousWidth, previousHeight)
                <= FRAME_CENTER_TOLERANCE
                && sizeDelta(width, height, previousWidth, previousHeight)
                <= FRAME_SIZE_TOLERANCE
                && centerDelta(centerX, centerY,
                anchorX, anchorY, anchorWidth, anchorHeight)
                <= ANCHOR_CENTER_TOLERANCE
                && sizeDelta(width, height, anchorWidth, anchorHeight)
                <= ANCHOR_SIZE_TOLERANCE;
    }

    private static float centerDelta(float x, float y, float referenceX, float referenceY,
                                     float referenceWidth, float referenceHeight) {
        float scale = Math.max(1f, Math.max(referenceWidth, referenceHeight));
        return (float) Math.hypot(x - referenceX, y - referenceY) / scale;
    }

    private static float sizeDelta(float width, float height,
                                   float referenceWidth, float referenceHeight) {
        return Math.max(Math.abs(width - referenceWidth) / Math.max(1f, referenceWidth),
                Math.abs(height - referenceHeight) / Math.max(1f, referenceHeight));
    }
}
