package com.insightface.sdk.inspireface.base;

/** JNI output holder used to preserve native error codes during result copies. */
public final class FaceCaptureResultBuffer {
    public FaceCaptureResult[] results = new FaceCaptureResult[0];

    public FaceCaptureResultBuffer() {}
}
