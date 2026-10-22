package com.example.inspireface_example.view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;

import com.example.inspireface_example.R;

/**
 * Draws face bounding brackets on top of the camera preview.
 *
 * Face rectangles are supplied in the upright image coordinate space (the space the
 * detector worked in, after rotation). The view maps them to screen space assuming the
 * preview uses FILL_CENTER (center-crop), and mirrors horizontally for the front camera.
 */
public class FaceOverlayView extends View {

    /** Immutable per-frame snapshot handed over from the analysis thread. */
    public static final class Frame {
        final int imageWidth;
        final int imageHeight;
        final boolean mirrored;
        final RectF[] rects;
        final int color;
        final float progress;
        final int progressColor;

        public Frame(int imageWidth, int imageHeight, boolean mirrored,
                     RectF[] rects, int color) {
            this(imageWidth, imageHeight, mirrored, rects, color, -1f, color);
        }

        public Frame(int imageWidth, int imageHeight, boolean mirrored,
                     RectF[] rects, int color, float progress, int progressColor) {
            this.imageWidth = imageWidth;
            this.imageHeight = imageHeight;
            this.mirrored = mirrored;
            this.rects = rects;
            this.color = color;
            this.progress = progress;
            this.progressColor = progressColor;
        }
    }

    private final Paint boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint progressTrackPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint progressPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final RectF mapped = new RectF();
    private final RectF progressBounds = new RectF();
    private volatile Frame frame;

    public FaceOverlayView(Context context) {
        this(context, null);
    }

    public FaceOverlayView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(dp(3));
        boxPaint.setStrokeCap(Paint.Cap.ROUND);
        boxPaint.setColor(ContextCompat.getColor(context, R.color.liveness_accent));
        progressTrackPaint.setStyle(Paint.Style.STROKE);
        progressTrackPaint.setStrokeWidth(dp(7));
        progressTrackPaint.setColor(0x66000000);
        progressPaint.setStyle(Paint.Style.STROKE);
        progressPaint.setStrokeWidth(dp(7));
        progressPaint.setStrokeCap(Paint.Cap.ROUND);
    }

    /** Safe to call from any thread. Pass {@code null} rects to clear. */
    public void submit(@Nullable Frame f) {
        frame = f;
        postInvalidateOnAnimation();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        Frame f = frame;
        if (f == null || f.rects == null || f.rects.length == 0
                || f.imageWidth <= 0 || f.imageHeight <= 0) {
            return;
        }
        float vw = getWidth();
        float vh = getHeight();
        // FILL_CENTER: uniform scale that covers the view, centered.
        float scale = Math.max(vw / f.imageWidth, vh / f.imageHeight);
        float dx = (vw - f.imageWidth * scale) / 2f;
        float dy = (vh - f.imageHeight * scale) / 2f;

        boxPaint.setColor(f.color);
        for (int i = 0; i < f.rects.length; i++) {
            RectF r = f.rects[i];
            float left = r.left;
            float right = r.right;
            if (f.mirrored) {
                float l = f.imageWidth - right;
                right = f.imageWidth - left;
                left = l;
            }
            mapped.set(left * scale + dx, r.top * scale + dy,
                    right * scale + dx, r.bottom * scale + dy);
            drawBrackets(canvas, mapped);
            if (i == 0 && f.progress >= 0f) {
                drawProgressRing(canvas, mapped, f.progress, f.progressColor);
            }
        }
    }

    private void drawProgressRing(Canvas canvas, RectF face, float progress, int color) {
        float size = Math.max(face.width(), face.height()) + dp(28);
        float cx = face.centerX();
        float cy = face.centerY();
        progressBounds.set(cx - size / 2f, cy - size / 2f,
                cx + size / 2f, cy + size / 2f);
        canvas.drawOval(progressBounds, progressTrackPaint);
        progressPaint.setColor(color);
        float clamped = Math.max(0f, Math.min(1f, progress));
        canvas.drawArc(progressBounds, -90f, Math.max(1f, clamped * 360f),
                false, progressPaint);
    }

    /** Corner brackets read better over video than a full box. */
    private void drawBrackets(Canvas canvas, RectF r) {
        float len = Math.min(r.width(), r.height()) * 0.22f;
        float radius = dp(6);
        // Top-left
        canvas.drawLine(r.left, r.top + len, r.left, r.top + radius, boxPaint);
        canvas.drawLine(r.left + radius, r.top, r.left + len, r.top, boxPaint);
        canvas.drawArc(r.left, r.top, r.left + 2 * radius, r.top + 2 * radius, 180, 90, false, boxPaint);
        // Top-right
        canvas.drawLine(r.right - len, r.top, r.right - radius, r.top, boxPaint);
        canvas.drawLine(r.right, r.top + radius, r.right, r.top + len, boxPaint);
        canvas.drawArc(r.right - 2 * radius, r.top, r.right, r.top + 2 * radius, 270, 90, false, boxPaint);
        // Bottom-left
        canvas.drawLine(r.left, r.bottom - len, r.left, r.bottom - radius, boxPaint);
        canvas.drawLine(r.left + radius, r.bottom, r.left + len, r.bottom, boxPaint);
        canvas.drawArc(r.left, r.bottom - 2 * radius, r.left + 2 * radius, r.bottom, 90, 90, false, boxPaint);
        // Bottom-right
        canvas.drawLine(r.right - len, r.bottom, r.right - radius, r.bottom, boxPaint);
        canvas.drawLine(r.right, r.bottom - radius, r.right, r.bottom - len, boxPaint);
        canvas.drawArc(r.right - 2 * radius, r.bottom - 2 * radius, r.right, r.bottom, 0, 90, false, boxPaint);
    }

    private float dp(float v) {
        return v * getResources().getDisplayMetrics().density;
    }
}
