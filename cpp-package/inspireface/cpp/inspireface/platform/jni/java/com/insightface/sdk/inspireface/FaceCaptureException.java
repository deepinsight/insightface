package com.insightface.sdk.inspireface;

/** Native face-capture failure with its original InspireFace error code. */
public final class FaceCaptureException extends RuntimeException {
    private final int errorCode;

    public FaceCaptureException(String operation, int errorCode) {
        super(operation + " failed with InspireFace error " + errorCode);
        this.errorCode = errorCode;
    }

    public int getErrorCode() {
        return errorCode;
    }
}
