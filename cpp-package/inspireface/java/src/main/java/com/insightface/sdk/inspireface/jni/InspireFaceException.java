package com.insightface.sdk.inspireface.jni;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

/** Optional exception facade; Native methods themselves preserve the C status codes. */
public final class InspireFaceException extends RuntimeException {
    private static final long serialVersionUID = 1L;
    private final long code;
    public InspireFaceException(long code) {
        super(message(code));
        this.code = code;
    }
    public long getCode() { return code; }
    public static void check(long status) {
        if (status != NativeConstants.HSUCCEED) throw new InspireFaceException(status);
    }
    private static String message(long code) {
        int[] size = new int[1];
        if (Native.HFGetErrorMessage(code, null, 0, size) == 0 && size[0] > 0) {
            ByteBuffer buffer = ByteBuffer.allocateDirect(size[0]);
            if (Native.HFGetErrorMessage(code, buffer, size[0], size) == 0) {
                byte[] text = new byte[size[0] - 1];
                buffer.get(text);
                return "InspireFace (" + code + "): " + new String(text, StandardCharsets.UTF_8);
            }
        }
        return "InspireFace error " + code;
    }
}
