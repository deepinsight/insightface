package com.example.inspireface_example.widget;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;

import com.example.inspireface_example.R;

/**
 * Maps SDK face rectangles over a centerCrop ImageView and lets the user select one.
 * The selected face is green; other detected faces remain visible in white.
 */
public final class FaceImageOverlayView extends View {

    public interface OnFaceSelectedListener {
        void onFaceSelected(int index);
    }

    private final Paint outlinePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint facePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint badgePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint badgeTextPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final RectF mappedRect = new RectF();
    private final int accentColor;
    private int imageWidth;
    private int imageHeight;
    private RectF[] faceRects = new RectF[0];
    private int selectedIndex = -1;
    private int pressedIndex = -1;
    private OnFaceSelectedListener listener;

    public FaceImageOverlayView(Context context) {
        this(context, null);
    }

    public FaceImageOverlayView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        outlinePaint.setStyle(Paint.Style.STROKE);
        outlinePaint.setStrokeWidth(dp(5f));
        outlinePaint.setColor(0xB3000000);

        accentColor = ContextCompat.getColor(context, R.color.liveness_accent);
        facePaint.setStyle(Paint.Style.STROKE);
        facePaint.setStrokeCap(Paint.Cap.ROUND);

        badgePaint.setStyle(Paint.Style.FILL);
        badgeTextPaint.setTextAlign(Paint.Align.CENTER);
        badgeTextPaint.setFakeBoldText(true);
        badgeTextPaint.setTextSize(dp(11f));
        setImportantForAccessibility(IMPORTANT_FOR_ACCESSIBILITY_NO);
    }

    public void showFace(int imageWidth, int imageHeight, @Nullable RectF faceRect) {
        showFaces(imageWidth, imageHeight,
                faceRect == null ? null : new RectF[]{faceRect}, faceRect == null ? -1 : 0);
    }

    public void showFaces(int imageWidth, int imageHeight,
                          @Nullable RectF[] faces, int selectedIndex) {
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        if (faces == null || faces.length == 0) {
            faceRects = new RectF[0];
            this.selectedIndex = -1;
        } else {
            faceRects = new RectF[faces.length];
            for (int i = 0; i < faces.length; i++) {
                faceRects[i] = new RectF(faces[i]);
            }
            this.selectedIndex = selectedIndex >= 0 && selectedIndex < faces.length
                    ? selectedIndex : 0;
        }
        invalidate();
    }

    public void setSelectedIndex(int index) {
        if (index >= 0 && index < faceRects.length && selectedIndex != index) {
            selectedIndex = index;
            invalidate();
        }
    }

    public void setOnFaceSelectedListener(@Nullable OnFaceSelectedListener listener) {
        this.listener = listener;
    }

    public void clearFace() {
        faceRects = new RectF[0];
        selectedIndex = -1;
        pressedIndex = -1;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (faceRects.length == 0 || imageWidth <= 0 || imageHeight <= 0
                || getWidth() <= 0 || getHeight() <= 0) {
            return;
        }
        for (int i = 0; i < faceRects.length; i++) {
            mapRect(faceRects[i], mappedRect);
            boolean selected = i == selectedIndex;
            drawBrackets(canvas, mappedRect, outlinePaint);
            facePaint.setStrokeWidth(dp(selected ? 3f : 2f));
            facePaint.setColor(selected ? accentColor : 0xE6FFFFFF);
            drawBrackets(canvas, mappedRect, facePaint);
            drawBadge(canvas, mappedRect, i, selected);
        }
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        switch (event.getActionMasked()) {
            case MotionEvent.ACTION_DOWN:
                pressedIndex = hitTest(event.getX(), event.getY());
                return pressedIndex >= 0;
            case MotionEvent.ACTION_UP:
                int releasedIndex = hitTest(event.getX(), event.getY());
                if (pressedIndex >= 0 && releasedIndex == pressedIndex) {
                    selectedIndex = releasedIndex;
                    invalidate();
                    performClick();
                    if (listener != null) {
                        listener.onFaceSelected(releasedIndex);
                    }
                }
                pressedIndex = -1;
                return true;
            case MotionEvent.ACTION_CANCEL:
                pressedIndex = -1;
                return true;
            default:
                return pressedIndex >= 0;
        }
    }

    @Override
    public boolean performClick() {
        super.performClick();
        return true;
    }

    private int hitTest(float x, float y) {
        int hit = -1;
        float smallestArea = Float.MAX_VALUE;
        for (int i = 0; i < faceRects.length; i++) {
            mapRect(faceRects[i], mappedRect);
            RectF touchTarget = new RectF(mappedRect);
            touchTarget.inset(-dp(8f), -dp(8f));
            float area = mappedRect.width() * mappedRect.height();
            if (touchTarget.contains(x, y) && area < smallestArea) {
                smallestArea = area;
                hit = i;
            }
        }
        return hit;
    }

    private void mapRect(RectF source, RectF out) {
        // Must match android:scaleType="centerCrop" on the image directly underneath.
        float scale = Math.max((float) getWidth() / imageWidth,
                (float) getHeight() / imageHeight);
        float dx = (getWidth() - imageWidth * scale) / 2f;
        float dy = (getHeight() - imageHeight * scale) / 2f;
        out.set(source.left * scale + dx, source.top * scale + dy,
                source.right * scale + dx, source.bottom * scale + dy);
    }

    private void drawBrackets(Canvas canvas, RectF face, Paint paint) {
        float len = Math.min(face.width(), face.height()) * 0.22f;
        float radius = Math.min(dp(6f), len * 0.45f);
        canvas.drawLine(face.left, face.top + len,
                face.left, face.top + radius, paint);
        canvas.drawLine(face.left + radius, face.top,
                face.left + len, face.top, paint);
        canvas.drawArc(face.left, face.top, face.left + 2f * radius,
                face.top + 2f * radius, 180f, 90f, false, paint);

        canvas.drawLine(face.right - len, face.top,
                face.right - radius, face.top, paint);
        canvas.drawLine(face.right, face.top + radius,
                face.right, face.top + len, paint);
        canvas.drawArc(face.right - 2f * radius, face.top, face.right,
                face.top + 2f * radius, 270f, 90f, false, paint);

        canvas.drawLine(face.left, face.bottom - len,
                face.left, face.bottom - radius, paint);
        canvas.drawLine(face.left + radius, face.bottom,
                face.left + len, face.bottom, paint);
        canvas.drawArc(face.left, face.bottom - 2f * radius,
                face.left + 2f * radius, face.bottom,
                90f, 90f, false, paint);

        canvas.drawLine(face.right - len, face.bottom,
                face.right - radius, face.bottom, paint);
        canvas.drawLine(face.right, face.bottom - radius,
                face.right, face.bottom - len, paint);
        canvas.drawArc(face.right - 2f * radius, face.bottom - 2f * radius,
                face.right, face.bottom, 0f, 90f, false, paint);
    }

    private void drawBadge(Canvas canvas, RectF face, int index, boolean selected) {
        float badgeRadius = dp(10f);
        float cx = face.left + badgeRadius;
        float cy = face.top + badgeRadius;
        cx = Math.max(badgeRadius + dp(2f), cx);
        cy = Math.max(badgeRadius + dp(2f), cy);
        badgePaint.setColor(selected ? accentColor : 0xE6FFFFFF);
        badgeTextPaint.setColor(0xFF07110F);
        canvas.drawCircle(cx, cy, badgeRadius, badgePaint);
        Paint.FontMetrics fm = badgeTextPaint.getFontMetrics();
        float baseline = cy - (fm.ascent + fm.descent) / 2f;
        canvas.drawText(String.valueOf(index + 1), cx, baseline, badgeTextPaint);
    }

    private float dp(float value) {
        return value * getResources().getDisplayMetrics().density;
    }
}
