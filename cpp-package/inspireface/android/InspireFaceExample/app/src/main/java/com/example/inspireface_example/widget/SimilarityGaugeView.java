package com.example.inspireface_example.widget;

import android.animation.ValueAnimator;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.util.AttributeSet;
import android.util.TypedValue;
import android.view.View;

import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;

import com.example.inspireface_example.R;

import java.util.Locale;

/** Circular similarity result with an animated progress ring and centered verdict. */
public final class SimilarityGaugeView extends View {

    private final Paint trackPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint progressPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint valuePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint labelPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final RectF arcBounds = new RectF();

    private final int accentColor;
    private final int failColor;
    private float displayedPercent;
    private boolean hasResult;
    private boolean matched;
    private String label = "";
    private ValueAnimator animator;

    public SimilarityGaugeView(Context context) {
        this(context, null);
    }

    public SimilarityGaugeView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        accentColor = ContextCompat.getColor(context, R.color.liveness_accent);
        failColor = ContextCompat.getColor(context, R.color.liveness_fail);

        trackPaint.setStyle(Paint.Style.STROKE);
        trackPaint.setStrokeCap(Paint.Cap.ROUND);
        trackPaint.setColor(ContextCompat.getColor(context, R.color.compare_ring_track));

        progressPaint.setStyle(Paint.Style.STROKE);
        progressPaint.setStrokeCap(Paint.Cap.ROUND);

        valuePaint.setTextAlign(Paint.Align.CENTER);
        valuePaint.setTypeface(Typeface.create(Typeface.DEFAULT, Typeface.BOLD));
        valuePaint.setColor(ContextCompat.getColor(context, R.color.white));

        labelPaint.setTextAlign(Paint.Align.CENTER);
        labelPaint.setColor(ContextCompat.getColor(context, R.color.home_text_secondary));
    }

    public void showMessage(String message) {
        cancelAnimator();
        hasResult = false;
        displayedPercent = 0f;
        label = message;
        setContentDescription(message);
        invalidate();
    }

    public void showResult(float percent, boolean isMatched, String verdict) {
        cancelAnimator();
        hasResult = true;
        matched = isMatched;
        label = verdict;
        float target = Math.max(0f, Math.min(100f, percent));
        animator = ValueAnimator.ofFloat(0f, target);
        animator.setDuration(550L);
        animator.addUpdateListener(animation -> {
            displayedPercent = (float) animation.getAnimatedValue();
            invalidate();
        });
        animator.start();
        setContentDescription(String.format(Locale.US, "%.1f%%, %s", target, verdict));
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        float stroke = dp(11f);
        trackPaint.setStrokeWidth(stroke);
        progressPaint.setStrokeWidth(stroke);
        progressPaint.setColor(matched ? accentColor : failColor);

        float cx = getWidth() / 2f;
        float cy = getHeight() / 2f;
        float radius = Math.min(getWidth(), getHeight()) / 2f - stroke - dp(4f);
        arcBounds.set(cx - radius, cy - radius, cx + radius, cy + radius);
        canvas.drawArc(arcBounds, -90f, 360f, false, trackPaint);
        if (hasResult && displayedPercent > 0f) {
            canvas.drawArc(arcBounds, -90f, displayedPercent * 3.6f, false, progressPaint);
        }

        valuePaint.setTextSize(sp(31f));
        labelPaint.setTextSize(sp(13f));
        String value = hasResult ? String.format(Locale.US, "%.1f%%", displayedPercent) : "—";
        canvas.drawText(value, cx, cy + dp(3f), valuePaint);
        canvas.drawText(label, cx, cy + dp(30f), labelPaint);
    }

    @Override
    protected void onDetachedFromWindow() {
        cancelAnimator();
        super.onDetachedFromWindow();
    }

    private void cancelAnimator() {
        if (animator != null) {
            animator.cancel();
            animator = null;
        }
    }

    private float dp(float value) {
        return value * getResources().getDisplayMetrics().density;
    }

    private float sp(float value) {
        return TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, value,
                getResources().getDisplayMetrics());
    }
}
