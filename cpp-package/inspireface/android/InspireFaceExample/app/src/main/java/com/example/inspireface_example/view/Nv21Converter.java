package com.example.inspireface_example.view;

import androidx.camera.core.ImageProxy;

import java.nio.ByteBuffer;

/**
 * Converts CameraX YUV_420_888 frames to tightly packed NV21, plus an NV21 rotation step
 * so frames can be handed to InspireFace already upright with CAMERA_ROTATION_0.
 *
 * Rotating on the Java side is deliberate: the 1.2.0 SDK's RGB anti-spoofing crop mixes
 * rotated and unrotated coordinate spaces when a non-zero rotation constant is used, which
 * silently degrades silent-liveness scores. An upright buffer avoids that entirely and
 * makes every SDK output coordinate match the preview orientation.
 *
 * On virtually all camera HALs the U/V planes of YUV_420_888 alias one interleaved
 * VU buffer (i.e. the memory already is NV21). That is detected once on the first
 * frame; afterwards the chroma plane is moved with a single bulk copy. All buffers are
 * reused across frames, so the converter allocates only on size changes.
 *
 * Not thread-safe: use one instance per analysis thread.
 */
final class Nv21Converter {

    private static final int UNKNOWN = -1;
    private static final int INTERLEAVED = 1;
    private static final int PLANAR = 0;

    private byte[] out;
    private byte[] rotated;
    private int uvLayout = UNKNOWN;

    /**
     * Returns the frame as NV21. The returned array is owned by the converter and
     * overwritten by the next call.
     */
    byte[] convert(ImageProxy image) {
        int width = image.getWidth();
        int height = image.getHeight();
        int ySize = width * height;
        int total = ySize + ySize / 2;
        if (out == null || out.length != total) {
            out = new byte[total];
            uvLayout = UNKNOWN;
        }

        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        copyLuma(planes[0], width, height, out);

        ImageProxy.PlaneProxy uPlane = planes[1];
        ImageProxy.PlaneProxy vPlane = planes[2];
        if (uvLayout == UNKNOWN) {
            uvLayout = isVuInterleaved(uPlane, vPlane, width, height) ? INTERLEAVED : PLANAR;
        }
        if (uvLayout == INTERLEAVED) {
            copyChromaInterleaved(uPlane, vPlane, ySize, out);
        } else {
            copyChromaPlanar(uPlane, vPlane, width, height, ySize, out);
        }
        return out;
    }

    /**
     * Rotates a tight NV21 buffer clockwise by the given degrees so it appears upright.
     * Returns {@code src} itself for 0°; otherwise a reused internal buffer.
     */
    byte[] rotateUpright(byte[] src, int width, int height, int rotationDegrees) {
        if (rotationDegrees == 0) {
            return src;
        }
        if (rotated == null || rotated.length != src.length) {
            rotated = new byte[src.length];
        }
        int ySize = width * height;
        int i = 0;
        switch (rotationDegrees) {
            case 90:
                for (int x = 0; x < width; x++) {
                    for (int y = height - 1; y >= 0; y--) {
                        rotated[i++] = src[y * width + x];
                    }
                }
                for (int x = 0; x < width; x += 2) {
                    for (int y = height / 2 - 1; y >= 0; y--) {
                        int s = ySize + y * width + x;
                        rotated[i++] = src[s];
                        rotated[i++] = src[s + 1];
                    }
                }
                break;
            case 180:
                for (int p = ySize - 1; p >= 0; p--) {
                    rotated[i++] = src[p];
                }
                for (int p = src.length - 2; p >= ySize; p -= 2) {
                    rotated[i++] = src[p];
                    rotated[i++] = src[p + 1];
                }
                break;
            case 270:
            default:
                for (int x = width - 1; x >= 0; x--) {
                    for (int y = 0; y < height; y++) {
                        rotated[i++] = src[y * width + x];
                    }
                }
                for (int x = width - 2; x >= 0; x -= 2) {
                    for (int y = 0; y < height / 2; y++) {
                        int s = ySize + y * width + x;
                        rotated[i++] = src[s];
                        rotated[i++] = src[s + 1];
                    }
                }
                break;
        }
        return rotated;
    }

    private static void copyLuma(ImageProxy.PlaneProxy yPlane, int width, int height, byte[] out) {
        ByteBuffer buf = yPlane.getBuffer();
        int base = buf.position();
        int rowStride = yPlane.getRowStride();
        if (rowStride == width) {
            buf.get(out, 0, width * height);
        } else {
            for (int row = 0; row < height; row++) {
                buf.position(base + row * rowStride);
                buf.get(out, row * width, width);
            }
        }
        buf.position(base);
    }

    /** The V buffer already views VUVU…: one bulk copy plus the trailing U byte it cannot see. */
    private static void copyChromaInterleaved(ImageProxy.PlaneProxy uPlane, ImageProxy.PlaneProxy vPlane,
                                              int ySize, byte[] out) {
        ByteBuffer vBuf = vPlane.getBuffer();
        int vPos = vBuf.position();
        int vuSize = ySize / 2;
        int n = Math.min(vBuf.remaining(), vuSize);
        vBuf.get(out, ySize, n);
        vBuf.position(vPos);
        if (n < vuSize) {
            ByteBuffer uBuf = uPlane.getBuffer();
            out[ySize + vuSize - 1] = uBuf.get(uBuf.limit() - 1);
        }
    }

    private static void copyChromaPlanar(ImageProxy.PlaneProxy uPlane, ImageProxy.PlaneProxy vPlane,
                                         int width, int height, int ySize, byte[] out) {
        ByteBuffer uBuf = uPlane.getBuffer();
        ByteBuffer vBuf = vPlane.getBuffer();
        int uBase = uBuf.position();
        int vBase = vBuf.position();
        int uRowStride = uPlane.getRowStride();
        int vRowStride = vPlane.getRowStride();
        int uPixStride = uPlane.getPixelStride();
        int vPixStride = vPlane.getPixelStride();
        int pos = ySize;
        for (int row = 0; row < height / 2; row++) {
            int uRow = uBase + row * uRowStride;
            int vRow = vBase + row * vRowStride;
            for (int col = 0; col < width / 2; col++) {
                out[pos++] = vBuf.get(vRow + col * vPixStride);
                out[pos++] = uBuf.get(uRow + col * uPixStride);
            }
        }
    }

    /**
     * True when the U/V planes alias a single interleaved VU buffer with no row
     * padding, i.e. shifting V by one byte yields exactly the U plane.
     */
    private static boolean isVuInterleaved(ImageProxy.PlaneProxy uPlane, ImageProxy.PlaneProxy vPlane,
                                           int width, int height) {
        if (uPlane.getPixelStride() != 2 || vPlane.getPixelStride() != 2) {
            return false;
        }
        ByteBuffer uBuf = uPlane.getBuffer();
        ByteBuffer vBuf = vPlane.getBuffer();
        int vPos = vBuf.position();
        int uLimit = uBuf.limit();
        vBuf.position(vPos + 1);
        uBuf.limit(uLimit - 1);
        boolean interleaved = vBuf.remaining() == (width * height / 2 - 2)
                && vBuf.compareTo(uBuf) == 0;
        vBuf.position(vPos);
        uBuf.limit(uLimit);
        return interleaved;
    }
}
