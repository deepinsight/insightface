package com.example.inspireface_example.widget;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;

import com.example.inspireface_example.R;
import com.insightface.sdk.inspireface.base.Point2f;

/** Draws one selected face's dense landmarks over a centerCrop ImageView. */
public final class FaceLandmarkOverlayView extends View {

    private final Paint outlinePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint pointPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private float[] points = new float[0];
    private int pointCount;
    private int imageWidth;
    private int imageHeight;

    public FaceLandmarkOverlayView(Context context) {
        this(context, null);
    }

    public FaceLandmarkOverlayView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        outlinePaint.setStyle(Paint.Style.FILL);
        outlinePaint.setColor(0xCC000000);
        pointPaint.setStyle(Paint.Style.FILL);
        pointPaint.setColor(ContextCompat.getColor(context, R.color.liveness_accent));
        setImportantForAccessibility(IMPORTANT_FOR_ACCESSIBILITY_NO);
    }

    public void showPoints(int imageWidth, int imageHeight, @Nullable Point2f[] landmarks) {
        showPoints(imageWidth, imageHeight, landmarks, 0f, 0f);
    }

    public void showPoints(int imageWidth, int imageHeight,
                           @Nullable Point2f[] landmarks,
                           float offsetX, float offsetY) {
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        if (landmarks == null || landmarks.length == 0) {
            points = new float[0];
            pointCount = 0;
        } else {
            points = new float[landmarks.length * 2];
            pointCount = 0;
            for (Point2f landmark : landmarks) {
                if (landmark == null) {
                    continue;
                }
                points[pointCount * 2] = landmark.x - offsetX;
                points[pointCount * 2 + 1] = landmark.y - offsetY;
                pointCount++;
            }
        }
        invalidate();
    }

    public void clearPoints() {
        points = new float[0];
        pointCount = 0;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (pointCount == 0 || imageWidth <= 0 || imageHeight <= 0
                || getWidth() <= 0 || getHeight() <= 0) {
            return;
        }
        float scale = Math.max((float) getWidth() / imageWidth,
                (float) getHeight() / imageHeight);
        float dx = (getWidth() - imageWidth * scale) / 2f;
        float dy = (getHeight() - imageHeight * scale) / 2f;
        float outlineRadius = dp(2.8f);
        float pointRadius = dp(1.6f);
        for (int i = 0; i < pointCount; i++) {
            float x = points[i * 2] * scale + dx;
            float y = points[i * 2 + 1] * scale + dy;
            canvas.drawCircle(x, y, outlineRadius, outlinePaint);
            canvas.drawCircle(x, y, pointRadius, pointPaint);
        }
    }

    private float dp(float value) {
        return value * getResources().getDisplayMetrics().density;
    }
}
