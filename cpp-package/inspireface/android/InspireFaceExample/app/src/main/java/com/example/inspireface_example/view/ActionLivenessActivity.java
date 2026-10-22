package com.example.inspireface_example.view;

import com.example.inspireface_example.R;

/** Dedicated cooperative action-liveness screen. */
public final class ActionLivenessActivity extends LivenessActivity {

    @Override
    protected LivenessController.Mode initialMode() {
        return LivenessController.Mode.ACTION;
    }

    @Override
    protected int pageTitleRes() {
        return R.string.mode_action;
    }
}
