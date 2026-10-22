package com.example.inspireface_example.view;

import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.view.PreviewView;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;

import com.example.inspireface_example.FaceModelPrefs;
import com.example.inspireface_example.LocalePrefs;
import com.example.inspireface_example.R;
import com.example.inspireface_example.permission.CameraPermissionCoordinator;
import com.google.android.material.card.MaterialCardView;
import com.google.android.material.progressindicator.LinearProgressIndicator;
import com.google.android.material.switchmaterial.SwitchMaterial;
import com.insightface.sdk.inspireface.base.FaceEulerAngle;

import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Shared camera screen used by the three dedicated feature activities. Subclasses choose
 * one fixed controller mode while this class owns the common CameraX and overlay lifecycle.
 */
public class LivenessActivity extends AppCompatActivity implements FaceAnalyzer.Listener {

    private PreviewView previewView;
    private FaceOverlayView overlayView;
    private LandmarkGlView landmarkView;
    private TextView promptTitle;
    private TextView promptSub;
    private TextView perfText;
    private TextView eulerText;
    private SwitchMaterial switchEuler;
    private LinearProgressIndicator promptProgress;
    private View btnRestart;

    private final ExecutorService analysisExecutor = Executors.newSingleThreadExecutor();
    private LivenessController controller;
    private FaceAnalyzer analyzer;
    private LivenessController.UiState lastState;
    private CameraPreviewController cameraController;
    private CameraPermissionCoordinator cameraPermission;
    private boolean engineStartingOrReady;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        cameraPermission = new CameraPermissionCoordinator(this,
                new CameraPermissionCoordinator.Listener() {
                    @Override
                    public void onCameraPermissionGranted() {
                        startEngine();
                    }

                    @Override
                    public void onCameraPermissionBlocked(boolean requiresSettings) {
                        showCameraPermissionBlocked(requiresSettings);
                    }
                });
        WindowCompat.setDecorFitsSystemWindows(getWindow(), false);
        setContentView(R.layout.activity_liveness);

        previewView = findViewById(R.id.previewView);
        overlayView = findViewById(R.id.faceOverlay);
        landmarkView = findViewById(R.id.landmarkGlView);
        promptTitle = findViewById(R.id.promptTitle);
        promptSub = findViewById(R.id.promptSub);
        perfText = findViewById(R.id.perfText);
        eulerText = findViewById(R.id.eulerText);
        switchEuler = findViewById(R.id.switchEuler);
        promptProgress = findViewById(R.id.promptProgress);
        btnRestart = findViewById(R.id.btnRestart);
        cameraPermission.bindRecoveryButton(
                findViewById(R.id.btnCameraPermissionAction));

        ((TextView) findViewById(R.id.pageTitle)).setText(pageTitleRes());
        ((TextView) findViewById(R.id.currentModel)).setText(
                getString(R.string.current_model, FaceModelPrefs.get(this).sdkName()));
        findViewById(R.id.btnBack).setOnClickListener(v -> finish());

        switchEuler.setOnCheckedChangeListener((button, checked) -> {
            if (analyzer != null) {
                analyzer.setEulerEnabled(checked);
            }
            eulerText.setText(R.string.euler_no_face);
            eulerText.setVisibility(checked ? View.VISIBLE : View.GONE);
        });
        applyWindowInsets();

        controller = new LivenessController(this);
        controller.setMode(initialMode());
        btnRestart.setOnClickListener(v -> controller.restart());
        // In-app language toggle (bottom-right): English by default, Chinese on demand.
        findViewById(R.id.langSwitch).setOnClickListener(v -> LocalePrefs.toggle(this));
        findViewById(R.id.btnFlipCamera).setOnClickListener(v -> flipCamera());

        cameraPermission.requestAccess();
    }

    private void applyWindowInsets() {
        View topBar = findViewById(R.id.topBar);
        MaterialCardView promptCard = findViewById(R.id.promptCard);
        int cardBaseMargin = (int) (24 * getResources().getDisplayMetrics().density);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.livenessRoot), (v, insets) -> {
            Insets bars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            topBar.setPadding(topBar.getPaddingLeft(), bars.top,
                    topBar.getPaddingRight(), topBar.getPaddingBottom());
            ViewGroup.MarginLayoutParams lp =
                    (ViewGroup.MarginLayoutParams) promptCard.getLayoutParams();
            lp.bottomMargin = cardBaseMargin + bars.bottom;
            promptCard.setLayoutParams(lp);
            return insets;
        });
    }

    /** GlobalLaunch copies model assets on first run — keep it off the main thread. */
    private void startEngine() {
        if (engineStartingOrReady) {
            return;
        }
        engineStartingOrReady = true;
        promptTitle.setText(R.string.msg_initializing);
        promptSub.setVisibility(View.GONE);
        analysisExecutor.execute(() -> {
            boolean ok = FaceEngine.ensureLaunched(this);
            runOnUiThread(() -> {
                if (isDestroyed() || isFinishing()) {
                    return; // model copy can outlive the activity — don't bind a dead lifecycle
                }
                if (ok) {
                    bindCamera();
                } else {
                    engineStartingOrReady = false;
                    promptTitle.setText(R.string.msg_engine_failed);
                }
            });
        });
    }

    private void showCameraPermissionBlocked(boolean requiresSettings) {
        promptTitle.setText(R.string.msg_permission_required);
        promptSub.setText(requiresSettings
                ? R.string.camera_permission_settings_hint
                : R.string.camera_permission_retry_hint);
        promptSub.setVisibility(View.VISIBLE);
    }

    private void bindCamera() {
        analyzer = new FaceAnalyzer(controller, overlayView, landmarkView, this, true);
        analyzer.setEulerEnabled(switchEuler.isChecked());
        analyzer.setLandmarksEnabled(false);
        cameraController = new CameraPreviewController(
                this, this, previewView, analysisExecutor, analyzer,
                new CameraPreviewController.Listener() {
                    @Override
                    public void onCameraReady(boolean frontCamera) {
                        analyzer.setMirrored(frontCamera);
                    }

                    @Override
                    public void onLensChanged(boolean frontCamera) {
                        analyzer.setMirrored(frontCamera);
                        controller.restart();
                        overlayView.submit(null);
                        landmarkView.clearPoints();
                    }

                    @Override
                    public void onCameraError(int messageRes) {
                        promptTitle.setText(messageRes);
                    }
                });
        cameraController.start();
    }

    /**
     * Switches between the front and back lens. The SDK needs no per-lens handling:
     * every frame is pre-rotated by its own rotationDegrees before
     * CreateImageStreamFromByteBuffer (always CAMERA_ROTATION_0), so the new lens's
     * different sensor orientation is absorbed per frame. Only the display mirroring
     * flips, and the mode state restarts so stale tracking can't leak across lenses.
     */
    private void flipCamera() {
        if (cameraController == null || !cameraController.flipCamera()) {
            Toast.makeText(this, R.string.msg_camera_unavailable, Toast.LENGTH_SHORT).show();
        }
    }

    /** Silent liveness is the behavior of the original activity and the first home tile. */
    protected LivenessController.Mode initialMode() {
        return LivenessController.Mode.SILENT;
    }

    protected int pageTitleRes() {
        return R.string.mode_silent;
    }

    // ------------------------------------------------------------------
    // FaceAnalyzer.Listener (analysis thread)
    // ------------------------------------------------------------------

    @Override
    public void onUiState(LivenessController.UiState state) {
        runOnUiThread(() -> {
            if (state.sameContent(lastState)) {
                return;
            }
            lastState = state;
            promptTitle.setText(state.title);
            if (state.subtitle != null) {
                promptSub.setText(state.subtitle);
                promptSub.setVisibility(View.VISIBLE);
            } else {
                promptSub.setVisibility(View.GONE);
            }
            if (state.progress >= 0) {
                promptProgress.setProgress(state.progress);
                promptProgress.setVisibility(View.VISIBLE);
            } else {
                promptProgress.setVisibility(View.GONE);
            }
            btnRestart.setVisibility(state.showRestart ? View.VISIBLE : View.GONE);
        });
    }

    @Override
    public void onPerf(double fps, long latencyMs) {
        runOnUiThread(() -> {
            perfText.setVisibility(View.VISIBLE);
            perfText.setText(String.format(Locale.US,
                    getString(R.string.perf_format), fps, latencyMs));
        });
    }

    @Override
    public void onEulerAngles(FaceEulerAngle angle) {
        runOnUiThread(() -> {
            if (!switchEuler.isChecked()) {
                return;
            }
            if (angle == null) {
                eulerText.setText(R.string.euler_no_face);
            } else {
                eulerText.setText(String.format(Locale.US,
                        getString(R.string.euler_format), angle.yaw, angle.pitch, angle.roll));
            }
        });
    }

    @Override
    public void onSessionError() {
        runOnUiThread(() -> promptTitle.setText(R.string.msg_engine_failed));
    }

    // GL lifecycle follows visibility (start/stop), not focus (resume/pause): in
    // multi-window the activity can be paused but visible with the camera and analysis
    // still running — the landmark overlay must keep rendering there.
    @Override
    protected void onResume() {
        super.onResume();
        cameraPermission.onResume();
    }

    @Override
    protected void onStart() {
        super.onStart();
        landmarkView.onResume();
    }

    @Override
    protected void onStop() {
        landmarkView.onPause();
        super.onStop();
    }

    @Override
    protected void onDestroy() {
        cameraPermission.close();
        if (cameraController != null) {
            cameraController.stop();
        }
        if (analyzer != null) {
            FaceAnalyzer toRelease = analyzer;
            analysisExecutor.execute(toRelease::release);
        }
        analysisExecutor.shutdown();
        super.onDestroy();
    }
}
