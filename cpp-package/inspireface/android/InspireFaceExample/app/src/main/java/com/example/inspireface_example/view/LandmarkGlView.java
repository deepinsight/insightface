package com.example.inspireface_example.view;

import android.content.Context;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.util.AttributeSet;

import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;

import com.example.inspireface_example.R;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/**
 * Transparent GL overlay that renders the dense facial landmarks as one GL_POINTS draw
 * call: a single ~1 KB vertex upload per frame, the image→screen mapping folded into a
 * vertex-shader uniform, and round anti-aliased sprites from gl_PointCoord. Renders only
 * when a new frame of points arrives (RENDERMODE_WHEN_DIRTY); hidden, the surface is
 * destroyed and costs nothing.
 *
 * Points are supplied in the upright image space the SDK reports (same space as the face
 * rects) and mapped with the same FILL_CENTER + front-mirror convention as
 * {@link FaceOverlayView}.
 */
public class LandmarkGlView extends GLSurfaceView {

    private final PointsRenderer renderer;

    public LandmarkGlView(Context context) {
        this(context, null);
    }

    public LandmarkGlView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        renderer = new PointsRenderer(
                4f * getResources().getDisplayMetrics().density,
                ContextCompat.getColor(context, R.color.liveness_accent));
        setEGLContextClientVersion(2);
        setEGLConfigChooser(8, 8, 8, 8, 0, 0);
        getHolder().setFormat(PixelFormat.TRANSLUCENT);
        setZOrderMediaOverlay(true); // above the camera surface, below regular views
        setRenderer(renderer);
        setRenderMode(RENDERMODE_WHEN_DIRTY);
    }

    /**
     * Safe to call from any thread. {@code xy} holds {@code count} interleaved (x, y)
     * pairs in upright image space; the array is copied before returning.
     */
    void submitPoints(float[] xy, int count, int imageWidth, int imageHeight, boolean mirrored) {
        renderer.setPoints(xy, count, imageWidth, imageHeight, mirrored);
        requestRender();
    }

    /** Safe to call from any thread. */
    void clearPoints() {
        renderer.setPoints(null, 0, 0, 0, false);
        requestRender();
    }

    private static final class PointsRenderer implements Renderer {

        private static final int MAX_POINTS = 512;

        private static final String VERTEX_SHADER = ""
                + "attribute vec2 aPos;\n"
                + "uniform vec4 uXform;\n" // (scaleX, offsetX, scaleY, offsetY) image → NDC
                + "uniform float uPointSize;\n"
                + "void main() {\n"
                + "  gl_Position = vec4(aPos.x * uXform.x + uXform.y,\n"
                + "                     aPos.y * uXform.z + uXform.w, 0.0, 1.0);\n"
                + "  gl_PointSize = uPointSize;\n"
                + "}\n";

        // Premultiplied output: the compositor treats a TRANSLUCENT surface as
        // premultiplied, so straight alpha would leave dst alpha = a² and draw an
        // over-bright halo on the anti-aliased fringe.
        private static final String FRAGMENT_SHADER = ""
                + "precision mediump float;\n"
                + "uniform vec4 uColor;\n"
                + "void main() {\n"
                + "  float dist = length(gl_PointCoord - vec2(0.5));\n"
                + "  float a = uColor.a * (1.0 - smoothstep(0.35, 0.5, dist));\n"
                + "  if (a <= 0.01) discard;\n"
                + "  gl_FragColor = vec4(uColor.rgb * a, a);\n"
                + "}\n";

        private final Object lock = new Object();
        private final float[] staging = new float[MAX_POINTS * 2];
        private int stagingCount;
        private int imageWidth;
        private int imageHeight;
        private boolean mirrored;

        private final FloatBuffer vertexBuffer;
        private final float pointSizePx;
        private final float[] color = new float[4];

        private int program;
        private int aPosLoc;
        private int uXformLoc;
        private int uPointSizeLoc;
        private int uColorLoc;
        private int viewWidth;
        private int viewHeight;

        PointsRenderer(float pointSizePx, int argbColor) {
            this.pointSizePx = pointSizePx;
            color[0] = Color.red(argbColor) / 255f;
            color[1] = Color.green(argbColor) / 255f;
            color[2] = Color.blue(argbColor) / 255f;
            color[3] = Color.alpha(argbColor) / 255f;
            vertexBuffer = ByteBuffer.allocateDirect(MAX_POINTS * 2 * 4)
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer();
        }

        void setPoints(float[] xy, int count, int imageWidth, int imageHeight, boolean mirrored) {
            synchronized (lock) {
                stagingCount = xy == null ? 0 : Math.min(count, MAX_POINTS);
                if (stagingCount > 0) {
                    System.arraycopy(xy, 0, staging, 0, stagingCount * 2);
                }
                this.imageWidth = imageWidth;
                this.imageHeight = imageHeight;
                this.mirrored = mirrored;
            }
        }

        @Override
        public void onSurfaceCreated(GL10 gl, EGLConfig config) {
            // A recreated surface must start empty: GLSurfaceView always draws one frame
            // after creation, which would otherwise flash the pre-destruction points.
            synchronized (lock) {
                stagingCount = 0;
            }
            // The context is recreated whenever the view is shown again — rebuild everything.
            program = GLES20.glCreateProgram();
            GLES20.glAttachShader(program, compileShader(GLES20.GL_VERTEX_SHADER, VERTEX_SHADER));
            GLES20.glAttachShader(program, compileShader(GLES20.GL_FRAGMENT_SHADER, FRAGMENT_SHADER));
            GLES20.glLinkProgram(program);
            aPosLoc = GLES20.glGetAttribLocation(program, "aPos");
            uXformLoc = GLES20.glGetUniformLocation(program, "uXform");
            uPointSizeLoc = GLES20.glGetUniformLocation(program, "uPointSize");
            uColorLoc = GLES20.glGetUniformLocation(program, "uColor");
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

            int count;
            float ax;
            float bx;
            float ay;
            float by;
            synchronized (lock) {
                count = stagingCount;
                if (count == 0 || imageWidth <= 0 || imageHeight <= 0
                        || viewWidth <= 0 || viewHeight <= 0) {
                    return;
                }
                vertexBuffer.position(0);
                vertexBuffer.put(staging, 0, count * 2);
                vertexBuffer.position(0);
                // Same FILL_CENTER mapping as FaceOverlayView, folded into x' = ax*x + bx.
                float scale = Math.max((float) viewWidth / imageWidth, (float) viewHeight / imageHeight);
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
            GLES20.glEnableVertexAttribArray(aPosLoc);
            GLES20.glVertexAttribPointer(aPosLoc, 2, GLES20.GL_FLOAT, false, 0, vertexBuffer);
            GLES20.glUniform4f(uXformLoc, ax, bx, ay, by);
            GLES20.glUniform1f(uPointSizeLoc, pointSizePx);
            GLES20.glUniform4f(uColorLoc, color[0], color[1], color[2], color[3]);
            GLES20.glDrawArrays(GLES20.GL_POINTS, 0, count);
            GLES20.glDisableVertexAttribArray(aPosLoc);
        }

        private static int compileShader(int type, String source) {
            int shader = GLES20.glCreateShader(type);
            GLES20.glShaderSource(shader, source);
            GLES20.glCompileShader(shader);
            return shader;
        }
    }
}
