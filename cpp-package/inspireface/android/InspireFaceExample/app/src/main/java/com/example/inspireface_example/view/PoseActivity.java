package com.example.inspireface_example.view;

import com.example.inspireface_example.R;

/** Dedicated pose/action-recognition screen. */
public final class PoseActivity extends LivenessActivity {

    @Override
    protected LivenessController.Mode initialMode() {
        return LivenessController.Mode.POSE;
    }

    @Override
    protected int pageTitleRes() {
        return R.string.mode_pose;
    }
}
