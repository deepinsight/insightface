package com.example.inspireface_example;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;

import com.example.inspireface_example.view.ActionLivenessActivity;
import com.example.inspireface_example.view.LivenessActivity;
import com.example.inspireface_example.view.PoseActivity;
import com.google.android.material.button.MaterialButtonToggleGroup;

/** Launcher page: model selection plus a square-grid menu for the available demos. */
public class HomeActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        WindowCompat.setDecorFitsSystemWindows(getWindow(), false);
        WindowCompat.getInsetsController(getWindow(), getWindow().getDecorView())
                .setAppearanceLightNavigationBars(false);
        setContentView(R.layout.activity_home);
        applyWindowInsets();

        MaterialButtonToggleGroup modelToggle = findViewById(R.id.modelToggle);
        FaceModelPrefs.Model selected = FaceModelPrefs.get(this);
        modelToggle.check(selected == FaceModelPrefs.Model.PIKACHU
                ? R.id.btnModelPikachu : R.id.btnModelMegatron);
        modelToggle.addOnButtonCheckedListener((group, checkedId, isChecked) -> {
            if (!isChecked) {
                return;
            }
            FaceModelPrefs.set(this, checkedId == R.id.btnModelPikachu
                    ? FaceModelPrefs.Model.PIKACHU : FaceModelPrefs.Model.MEGATRON);
        });

        findViewById(R.id.cardSilent).setOnClickListener(
                v -> openFeature(LivenessActivity.class));
        findViewById(R.id.cardAction).setOnClickListener(
                v -> openFeature(ActionLivenessActivity.class));
        findViewById(R.id.cardPose).setOnClickListener(
                v -> openFeature(PoseActivity.class));
        findViewById(R.id.cardCompare).setOnClickListener(
                v -> openFeature(FaceCompareActivity.class));
        findViewById(R.id.cardManagement).setOnClickListener(
                v -> openFeature(FaceManagementActivity.class));
        findViewById(R.id.cardRecognition).setOnClickListener(
                v -> openFeature(FaceRecognitionActivity.class));
        findViewById(R.id.cardDetection).setOnClickListener(
                v -> openFeature(FaceDetectionActivity.class));
        findViewById(R.id.cardAttribute).setOnClickListener(
                v -> openFeature(FaceAttributeActivity.class));
        findViewById(R.id.langSwitch).setOnClickListener(v -> LocalePrefs.toggle(this));
    }

    private void openFeature(Class<?> activityClass) {
        startActivity(new Intent(this, activityClass));
    }

    private void applyWindowInsets() {
        View root = findViewById(R.id.homeRoot);
        int horizontal = root.getPaddingLeft();
        int top = root.getPaddingTop();
        int bottom = root.getPaddingBottom();
        ViewCompat.setOnApplyWindowInsetsListener(root, (v, insets) -> {
            Insets bars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(horizontal, top + bars.top, horizontal, bottom + bars.bottom);
            return insets;
        });
    }
}
