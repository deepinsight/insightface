package com.example.inspireface_example.view;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
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
import com.example.inspireface_example.R;
import com.example.inspireface_example.permission.CameraPermissionCoordinator;
import com.google.android.material.card.MaterialCardView;

import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Camera enrollment flow with first-face tracking and a reset-on-motion stability ring. */
public final class FaceCaptureActivity extends AppCompatActivity
        implements EnrollmentFaceAnalyzer.Listener {

    public static final String EXTRA_CAPTURE_PATH = "capture_path";
    private static final long COMPLETE_DISPLAY_MS = 350L;

    private final ExecutorService analysisExecutor = Executors.newSingleThreadExecutor();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    private PreviewView previewView;
    private FaceOverlayView overlayView;
    private TextView promptTitle;
    private TextView promptSubtitle;
    private View flipButton;
    private CameraPreviewController cameraController;
    private EnrollmentFaceAnalyzer analyzer;
    private boolean resultScheduled;
    private boolean resultReturned;
    private String pendingCapturePath;
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
        setContentView(R.layout.activity_face_capture);

        previewView = findViewById(R.id.capturePreview);
        overlayView = findViewById(R.id.captureFaceOverlay);
        promptTitle = findViewById(R.id.capturePromptTitle);
        promptSubtitle = findViewById(R.id.capturePromptSubtitle);
        flipButton = findViewById(R.id.btnFlipCaptureCamera);
        cameraPermission.bindRecoveryButton(
                findViewById(R.id.btnCameraPermissionAction));
        ((TextView) findViewById(R.id.currentModel)).setText(
                getString(R.string.current_model, FaceModelPrefs.get(this).sdkName()));
        findViewById(R.id.btnBack).setOnClickListener(v -> finish());
        flipButton.setOnClickListener(v -> flipCamera());
        applyWindowInsets();

        cameraPermission.requestAccess();
    }

    private void startEngine() {
        if (engineStartingOrReady) {
            return;
        }
        engineStartingOrReady = true;
        promptTitle.setText(R.string.msg_initializing);
        promptSubtitle.setText(R.string.capture_first_face_hint);
        analysisExecutor.execute(() -> {
            boolean ready = FaceEngine.ensureLaunched(this);
            runOnUiThread(() -> {
                if (isFinishing() || isDestroyed()) {
                    return;
                }
                if (ready) {
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
        promptSubtitle.setText(requiresSettings
                ? R.string.camera_permission_settings_hint
                : R.string.camera_permission_retry_hint);
    }

    private void bindCamera() {
        File captureDirectory = new File(getCacheDir(), "face_capture");
        analyzer = new EnrollmentFaceAnalyzer(
                this, overlayView, captureDirectory, this);
        cameraController = new CameraPreviewController(
                this, this, previewView, analysisExecutor, analyzer,
                new CameraPreviewController.Listener() {
                    @Override
                    public void onCameraReady(boolean frontCamera) {
                        analyzer.setMirrored(frontCamera);
                        showState(EnrollmentFaceAnalyzer.Stage.NO_FACE, 0f);
                    }

                    @Override
                    public void onLensChanged(boolean frontCamera) {
                        analyzer.setMirrored(frontCamera);
                        analyzer.resetTracking();
                        showState(EnrollmentFaceAnalyzer.Stage.NO_FACE, 0f);
                    }

                    @Override
                    public void onCameraError(int messageRes) {
                        promptTitle.setText(messageRes);
                    }
                });
        cameraController.start();
    }

    private void flipCamera() {
        if (resultScheduled) {
            return;
        }
        if (cameraController == null || !cameraController.flipCamera()) {
            Toast.makeText(this, R.string.msg_camera_unavailable, Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    public void onState(EnrollmentFaceAnalyzer.Stage stage, float progress) {
        runOnUiThread(() -> showState(stage, progress));
    }

    private void showState(EnrollmentFaceAnalyzer.Stage stage, float progress) {
        if (isFinishing() || isDestroyed()) {
            return;
        }
        switch (stage) {
            case MOVE_CLOSER:
                promptTitle.setText(R.string.capture_move_closer);
                promptSubtitle.setText(R.string.capture_first_face_hint);
                break;
            case HOLD_STILL:
                promptTitle.setText(R.string.capture_hold_still);
                promptSubtitle.setText(R.string.capture_warmup_hint);
                break;
            case CAPTURING:
                promptTitle.setText(R.string.capture_keep_still);
                promptSubtitle.setText(getString(
                        R.string.capture_progress, Math.round(progress * 100f)));
                break;
            case COMPLETE:
                promptTitle.setText(R.string.capture_complete);
                promptSubtitle.setText(R.string.capture_complete_hint);
                break;
            case NO_FACE:
            default:
                promptTitle.setText(R.string.capture_no_face);
                promptSubtitle.setText(R.string.capture_first_face_hint);
                break;
        }
    }

    @Override
    public void onCaptured(String path) {
        runOnUiThread(() -> {
            if (isFinishing() || isDestroyed()) {
                new File(path).delete();
                return;
            }
            resultScheduled = true;
            pendingCapturePath = path;
            flipButton.setEnabled(false);
            showState(EnrollmentFaceAnalyzer.Stage.COMPLETE, 1f);
            mainHandler.postDelayed(() -> {
                if (isFinishing() || isDestroyed()) {
                    new File(path).delete();
                    return;
                }
                resultReturned = true;
                setResult(RESULT_OK,
                        new Intent().putExtra(EXTRA_CAPTURE_PATH, path));
                finish();
            }, COMPLETE_DISPLAY_MS);
        });
    }

    @Override
    public void onCaptureError() {
        runOnUiThread(() -> {
            promptTitle.setText(R.string.capture_failed);
            promptSubtitle.setText(R.string.capture_retry_hint);
        });
    }

    @Override
    public void onSessionError() {
        runOnUiThread(() -> promptTitle.setText(R.string.msg_engine_failed));
    }

    private void applyWindowInsets() {
        View topBar = findViewById(R.id.captureTopBar);
        MaterialCardView promptCard = findViewById(R.id.capturePromptCard);
        int baseBottom = (int) (24 * getResources().getDisplayMetrics().density);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.captureRoot), (v, insets) -> {
            Insets bars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            topBar.setPadding(topBar.getPaddingLeft(), bars.top,
                    topBar.getPaddingRight(), topBar.getPaddingBottom());
            ViewGroup.MarginLayoutParams params =
                    (ViewGroup.MarginLayoutParams) promptCard.getLayoutParams();
            params.bottomMargin = baseBottom + bars.bottom;
            promptCard.setLayoutParams(params);
            return insets;
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        cameraPermission.onResume();
    }

    @Override
    protected void onDestroy() {
        cameraPermission.close();
        mainHandler.removeCallbacksAndMessages(null);
        if (!resultReturned && pendingCapturePath != null) {
            new File(pendingCapturePath).delete();
        }
        if (cameraController != null) {
            cameraController.stop();
        }
        if (analyzer != null) {
            EnrollmentFaceAnalyzer toRelease = analyzer;
            analysisExecutor.execute(toRelease::release);
        }
        analysisExecutor.shutdown();
        super.onDestroy();
    }
}
