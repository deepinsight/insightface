package com.example.inspireface_example.view;

import android.content.Context;
import android.graphics.PixelFormat;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.util.AttributeSet;

import androidx.annotation.Nullable;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/**
 * Transparent OpenGL overlay for multi-face tracking. Colored boxes and dense landmarks are
 * uploaded in two batches, then rendered on GLSurfaceView's own thread only when a new tracking
 * result arrives. Every vertex carries its track-ID color, keeping UI work off the main thread.
 */
public final class FaceTrackingGlView extends GLSurfaceView {

    /** x, y, red, green, blue, alpha. */
    static final int FLOATS_PER_VERTEX = 6;
    static final int MAX_FACES = 10;
    static final int LANDMARKS_PER_FACE = 106;
    static final int BOX_VERTICES_PER_FACE = 16;

    private final TrackingRenderer renderer;

    public FaceTrackingGlView(Context context) {
        this(context, null);
    }

    public FaceTrackingGlView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        float density = getResources().getDisplayMetrics().density;
        renderer = new TrackingRenderer(4.5f * density, 2.2f * density);
        setEGLContextClientVersion(2);
        setEGLConfigChooser(8, 8, 8, 8, 0, 0);
        getHolder().setFormat(PixelFormat.TRANSLUCENT);
        setZOrderMediaOverlay(true);
        setRenderer(renderer);
        setRenderMode(RENDERMODE_WHEN_DIRTY);
        setImportantForAccessibility(IMPORTANT_FOR_ACCESSIBILITY_NO);
    }

    /** Safe from the CameraX analysis thread; data is copied before returning. */
    void submit(float[] pointVertices, int pointCount,
                float[] boxVertices, int boxVertexCount,
                int imageWidth, int imageHeight, boolean mirrored) {
        renderer.setFrame(pointVertices, pointCount, boxVertices, boxVertexCount,
                imageWidth, imageHeight, mirrored);
        requestRender();
    }

    /** Safe from any thread. */
    public void clearTracking() {
        renderer.setFrame(null, 0, null, 0, 0, 0, false);
        requestRender();
    }

    private static final class TrackingRenderer implements Renderer {

        private static final int MAX_POINT_VERTICES = MAX_FACES * LANDMARKS_PER_FACE;
        private static final int MAX_BOX_VERTICES = MAX_FACES * BOX_VERTICES_PER_FACE;
        private static final int STRIDE_BYTES = FLOATS_PER_VERTEX * 4;

        private static final String VERTEX_SHADER = ""
                + "attribute vec2 aPos;\n"
                + "attribute vec4 aColor;\n"
                + "uniform vec4 uXform;\n"
                + "uniform float uPointSize;\n"
                + "varying vec4 vColor;\n"
                + "void main() {\n"
                + "  gl_Position = vec4(aPos.x * uXform.x + uXform.y,\n"
                + "                     aPos.y * uXform.z + uXform.w, 0.0, 1.0);\n"
                + "  gl_PointSize = uPointSize;\n"
                + "  vColor = aColor;\n"
                + "}\n";

        private static final String FRAGMENT_SHADER = ""
                + "precision mediump float;\n"
                + "varying vec4 vColor;\n"
                + "uniform float uPointPass;\n"
                + "void main() {\n"
                + "  float a = vColor.a;\n"
                + "  if (uPointPass > 0.5) {\n"
                + "    float dist = length(gl_PointCoord - vec2(0.5));\n"
                + "    a *= 1.0 - smoothstep(0.35, 0.5, dist);\n"
                + "    if (a <= 0.01) discard;\n"
                + "  }\n"
                + "  gl_FragColor = vec4(vColor.rgb * a, a);\n"
                + "}\n";

        private final Object lock = new Object();
        private final float[] stagedPoints =
                new float[MAX_POINT_VERTICES * FLOATS_PER_VERTEX];
        private final float[] stagedBoxes =
                new float[MAX_BOX_VERTICES * FLOATS_PER_VERTEX];
        private int stagedPointCount;
        private int stagedBoxCount;
        private int imageWidth;
        private int imageHeight;
        private boolean mirrored;

        private final FloatBuffer pointBuffer = allocate(MAX_POINT_VERTICES);
        private final FloatBuffer boxBuffer = allocate(MAX_BOX_VERTICES);
        private final float pointSizePx;
        private final float boxWidthPx;

        private int program;
        private int aPosLoc;
        private int aColorLoc;
        private int uXformLoc;
        private int uPointSizeLoc;
        private int uPointPassLoc;
        private int viewWidth;
        private int viewHeight;

        TrackingRenderer(float pointSizePx, float boxWidthPx) {
            this.pointSizePx = pointSizePx;
            this.boxWidthPx = boxWidthPx;
        }

        void setFrame(float[] points, int pointCount,
                      float[] boxes, int boxCount,
                      int imageWidth, int imageHeight, boolean mirrored) {
            synchronized (lock) {
                stagedPointCount = points == null ? 0
                        : Math.min(pointCount, MAX_POINT_VERTICES);
                stagedBoxCount = boxes == null ? 0
                        : Math.min(boxCount, MAX_BOX_VERTICES);
                if (stagedPointCount > 0) {
                    System.arraycopy(points, 0, stagedPoints, 0,
                            stagedPointCount * FLOATS_PER_VERTEX);
                }
                if (stagedBoxCount > 0) {
                    System.arraycopy(boxes, 0, stagedBoxes, 0,
                            stagedBoxCount * FLOATS_PER_VERTEX);
                }
                this.imageWidth = imageWidth;
                this.imageHeight = imageHeight;
                this.mirrored = mirrored;
            }
        }

        @Override
        public void onSurfaceCreated(GL10 gl, EGLConfig config) {
            synchronized (lock) {
                stagedPointCount = 0;
                stagedBoxCount = 0;
            }
            program = GLES20.glCreateProgram();
            GLES20.glAttachShader(program,
                    compileShader(GLES20.GL_VERTEX_SHADER, VERTEX_SHADER));
            GLES20.glAttachShader(program,
                    compileShader(GLES20.GL_FRAGMENT_SHADER, FRAGMENT_SHADER));
            GLES20.glLinkProgram(program);
            aPosLoc = GLES20.glGetAttribLocation(program, "aPos");
            aColorLoc = GLES20.glGetAttribLocation(program, "aColor");
            uXformLoc = GLES20.glGetUniformLocation(program, "uXform");
            uPointSizeLoc = GLES20.glGetUniformLocation(program, "uPointSize");
            uPointPassLoc = GLES20.glGetUniformLocation(program, "uPointPass");
            GLES20.glClearColor(0f, 0f, 0f, 0f);
            GLES20.glEnable(GLES20.GL_BLEND);
            GLES20.glBlendFunc(GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA);
        }

        @Override
        public void onSurfaceChanged(GL10 gl, int width, int height) {
            GLES20.glViewport(0, 0, width, height);
            viewWidth = width;
            viewHeight = height;
        }

        @Override
        public void onDrawFrame(GL10 gl) {
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);

            int pointCount;
            int boxCount;
            float ax;
            float bx;
            float ay;
            float by;
            synchronized (lock) {
                pointCount = stagedPointCount;
                boxCount = stagedBoxCount;
                if ((pointCount == 0 && boxCount == 0)
                        || imageWidth <= 0 || imageHeight <= 0
                        || viewWidth <= 0 || viewHeight <= 0) {
                    return;
                }
                put(pointBuffer, stagedPoints, pointCount);
                put(boxBuffer, stagedBoxes, boxCount);

                // PreviewView defaults to FILL_CENTER; mirror only the front-camera x axis.
                float scale = Math.max((float) viewWidth / imageWidth,
                        (float) viewHeight / imageHeight);
                float dx = (viewWidth - imageWidth * scale) / 2f;
                float dy = (viewHeight - imageHeight * scale) / 2f;
                float mx = mirrored ? -scale : scale;
                float cx = mirrored ? imageWidth * scale + dx : dx;
                ax = 2f * mx / viewWidth;
                bx = 2f * cx / viewWidth - 1f;
                ay = -2f * scale / viewHeight;
                by = 1f - 2f * dy / viewHeight;
            }

            GLES20.glUseProgram(program);
            GLES20.glUniform4f(uXformLoc, ax, bx, ay, by);
            GLES20.glUniform1f(uPointSizeLoc, pointSizePx);
            GLES20.glEnableVertexAttribArray(aPosLoc);
            GLES20.glEnableVertexAttribArray(aColorLoc);

            if (boxCount > 0) {
                bind(boxBuffer);
                GLES20.glUniform1f(uPointPassLoc, 0f);
                GLES20.glLineWidth(boxWidthPx);
                GLES20.glDrawArrays(GLES20.GL_LINES, 0, boxCount);
            }
            if (pointCount > 0) {
                bind(pointBuffer);
                GLES20.glUniform1f(uPointPassLoc, 1f);
                GLES20.glDrawArrays(GLES20.GL_POINTS, 0, pointCount);
            }

            GLES20.glDisableVertexAttribArray(aColorLoc);
            GLES20.glDisableVertexAttribArray(aPosLoc);
        }

        private void bind(FloatBuffer buffer) {
            buffer.position(0);
            GLES20.glVertexAttribPointer(aPosLoc, 2, GLES20.GL_FLOAT,
                    false, STRIDE_BYTES, buffer);
            buffer.position(2);
            GLES20.glVertexAttribPointer(aColorLoc, 4, GLES20.GL_FLOAT,
                    false, STRIDE_BYTES, buffer);
        }

        private static FloatBuffer allocate(int vertexCount) {
            return ByteBuffer.allocateDirect(
                            vertexCount * FLOATS_PER_VERTEX * 4)
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer();
        }

        private static void put(FloatBuffer buffer, float[] source, int vertexCount) {
            buffer.position(0);
            buffer.put(source, 0, vertexCount * FLOATS_PER_VERTEX);
            buffer.position(0);
        }

        private static int compileShader(int type, String source) {
            int shader = GLES20.glCreateShader(type);
            GLES20.glShaderSource(shader, source);
            GLES20.glCompileShader(shader);
            return shader;
        }
    }
}
