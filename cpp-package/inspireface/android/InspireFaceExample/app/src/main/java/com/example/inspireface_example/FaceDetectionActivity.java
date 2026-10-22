package com.example.inspireface_example;

import android.graphics.Bitmap;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.ColorRes;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;

import com.example.inspireface_example.face.FaceImageProcessor;
import com.example.inspireface_example.face.FaceCropUtils;
import com.example.inspireface_example.face.ImageBitmapLoader;
import com.example.inspireface_example.permission.CameraPermissionCoordinator;
import com.example.inspireface_example.view.CameraPreviewController;
import com.example.inspireface_example.view.FaceEngine;
import com.example.inspireface_example.view.FaceTrackingAnalyzer;
import com.example.inspireface_example.view.FaceTrackingGlView;
import com.example.inspireface_example.widget.FaceImageOverlayView;
import com.example.inspireface_example.widget.FaceLandmarkOverlayView;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.chip.ChipGroup;
import com.google.android.material.switchmaterial.SwitchMaterial;
import com.google.android.material.tabs.TabLayout;
import com.insightface.sdk.inspireface.InspireFace;
import com.insightface.sdk.inspireface.base.Session;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Still-image first step of the face detection and tracking demo. */
public final class FaceDetectionActivity extends AppCompatActivity {

    private static final int MAX_IMAGE_DIMENSION = 2048;
    /** Only genuinely small faces need the landmark magnifier. */
    private static final float DIRECT_LANDMARK_FACE_AREA_RATIO = 0.05f;
    private static final float LANDMARK_MAGNIFIER_CROP_SCALE = 2.4f;

    private final ExecutorService sdkExecutor = Executors.newSingleThreadExecutor();
    private final ExecutorService trackingExecutor = Executors.newSingleThreadExecutor();

    private Session session;
    private StillImageSessionSettings.Values savedSettings;
    private FaceTrackingSessionSettings.Values trackingSettings;
    private volatile boolean destroyed;
    private volatile int imageVersion;
    private boolean sessionBusy = true;
    private boolean engineReady;

    private ChipGroup inputPxGroup;
    private ChipGroup maxFacesGroup;
    private ChipGroup minFaceGroup;
    private MaterialButton applySessionButton;
    private MaterialButton resetSessionButton;
    private MaterialButton choosePhotoButton;
    private TextView sessionStatus;
    private View sessionSettingsContent;
    private TextView sessionSettingsToggleText;
    private SwitchMaterial landmarkSwitch;
    private ImageView imageView;
    private View imagePlaceholder;
    private TextView imageStatus;
    private FaceImageOverlayView faceOverlay;
    private FaceLandmarkOverlayView sourceLandmarkOverlay;
    private View magnifierCard;
    private ImageView magnifierImage;
    private FaceLandmarkOverlayView magnifierLandmarkOverlay;
    private View magnifierLandmarkBadge;
    private TextView landmarkDisplayHint;

    private PreviewView trackingPreview;
    private FaceTrackingGlView trackingGlOverlay;
    private TextView trackingStatus;
    private TextView trackingSessionStatus;
    private TextView trackingSettingsToggleText;
    private View trackingSettingsContent;
    private ChipGroup trackingModeGroup;
    private ChipGroup trackingInputPxGroup;
    private ChipGroup trackingMaxFacesGroup;
    private ChipGroup trackingMinFaceGroup;
    private MaterialButton resetTrackingButton;
    private View trackingLoadingIndicator;
    private View flipTrackingButton;
    private CameraPreviewController trackingCameraController;
    private FaceTrackingAnalyzer trackingAnalyzer;
    private boolean videoSelected;
    private boolean trackingStarting;
    private CameraPermissionCoordinator cameraPermission;
    private boolean trackingFrontCamera = true;
    private boolean suppressTrackingSettingChanges;
    private int trackingGeneration;

    private Bitmap imageBitmap;
    private Bitmap magnifierBitmap;
    private FaceImageProcessor.Candidate[] candidates;
    private int selectedIndex = -1;

    private final ActivityResultLauncher<String> photoPicker =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null && !sessionBusy && session != null) {
                    loadPhoto(uri);
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        cameraPermission = new CameraPermissionCoordinator(this,
                new CameraPermissionCoordinator.Listener() {
                    @Override
                    public void onCameraPermissionGranted() {
                        startVideoTracking();
                    }

                    @Override
                    public void onCameraPermissionBlocked(boolean requiresSettings) {
                        if (videoSelected) {
                            setTrackingLoading(false);
                            showTrackingMessage(requiresSettings
                                            ? R.string.camera_permission_settings_hint
                                            : R.string.camera_permission_retry_hint,
                                    R.color.liveness_fail);
                        }
                    }
                });
        WindowCompat.setDecorFitsSystemWindows(getWindow(), false);
        WindowCompat.getInsetsController(getWindow(), getWindow().getDecorView())
                .setAppearanceLightNavigationBars(false);
        setContentView(R.layout.activity_face_detection);
        applyWindowInsets();

        savedSettings = StillImageSessionSettings.load(this);
        trackingSettings = FaceTrackingSessionSettings.load(this);
        bindViews();
        cameraPermission.bindRecoveryButton(
                findViewById(R.id.btnCameraPermissionAction));
        ((TextView) findViewById(R.id.currentModel)).setText(getString(
                R.string.current_model, FaceModelPrefs.get(this).sdkName()));
        findViewById(R.id.btnBack).setOnClickListener(v -> finish());
        choosePhotoButton.setOnClickListener(v -> photoPicker.launch("image/*"));
        findViewById(R.id.detectionImageCard).setOnClickListener(v -> {
            if (!sessionBusy && session != null) {
                photoPicker.launch("image/*");
            }
        });
        faceOverlay.setOnFaceSelectedListener(this::selectFace);
        landmarkSwitch.setOnCheckedChangeListener((button, checked) ->
                renderSelectedLandmarks());
        findViewById(R.id.sessionSettingsHeader).setOnClickListener(v ->
                setSettingsExpanded(sessionSettingsContent.getVisibility() != View.VISIBLE));
        applySessionButton.setOnClickListener(v -> rebuildSession(selectedSettings()));
        resetSessionButton.setOnClickListener(v -> {
            StillImageSessionSettings.Values defaults = StillImageSessionSettings.defaults();
            selectParameters(defaults);
            rebuildSession(defaults);
        });
        flipTrackingButton.setOnClickListener(v -> flipTrackingCamera());
        findViewById(R.id.trackingSettingsHeader).setOnClickListener(v ->
                setTrackingSettingsExpanded(
                        trackingSettingsContent.getVisibility() != View.VISIBLE));
        resetTrackingButton.setOnClickListener(v -> {
            FaceTrackingSessionSettings.Values defaults =
                    FaceTrackingSessionSettings.defaults();
            selectTrackingParameters(defaults);
            applyTrackingSettings(defaults);
        });
        configureTabs();
        selectParameters(savedSettings);
        selectTrackingParameters(trackingSettings);
        configureTrackingAutoApply();
        setSettingsExpanded(false);
        setTrackingSettingsExpanded(false);
        setControlsEnabled(false);
        setTrackingControlsEnabled(false);
        sessionStatus.setText(R.string.recognition_session_rebuilding);
        showTrackingSummary(trackingSettings);
        sdkExecutor.execute(this::initializeDetection);
    }

    private void bindViews() {
        inputPxGroup = findViewById(R.id.inputPxGroup);
        maxFacesGroup = findViewById(R.id.maxFacesGroup);
        minFaceGroup = findViewById(R.id.minFaceGroup);
        applySessionButton = findViewById(R.id.btnApplySession);
        resetSessionButton = findViewById(R.id.btnResetSession);
        choosePhotoButton = findViewById(R.id.btnChooseDetectionPhoto);
        sessionStatus = findViewById(R.id.sessionStatus);
        sessionSettingsContent = findViewById(R.id.sessionSettingsContent);
        sessionSettingsToggleText = findViewById(R.id.sessionSettingsToggleText);
        landmarkSwitch = findViewById(R.id.switchDenseLandmarks);
        imageView = findViewById(R.id.detectionImage);
        imagePlaceholder = findViewById(R.id.detectionImagePlaceholder);
        imageStatus = findViewById(R.id.detectionImageStatus);
        faceOverlay = findViewById(R.id.detectionFaceOverlay);
        sourceLandmarkOverlay = findViewById(R.id.detectionLandmarkOverlay);
        magnifierCard = findViewById(R.id.landmarkMagnifierCard);
        magnifierImage = findViewById(R.id.landmarkMagnifierImage);
        magnifierLandmarkOverlay = findViewById(R.id.landmarkMagnifierOverlay);
        magnifierLandmarkBadge = findViewById(R.id.landmarkMagnifierBadge);
        landmarkDisplayHint = findViewById(R.id.landmarkDisplayHint);
        trackingPreview = findViewById(R.id.detectionTrackingPreview);
        trackingGlOverlay = findViewById(R.id.detectionTrackingGlOverlay);
        trackingStatus = findViewById(R.id.detectionTrackingStatus);
        trackingSessionStatus = findViewById(R.id.trackingSessionStatus);
        trackingSettingsToggleText = findViewById(R.id.trackingSettingsToggleText);
        trackingSettingsContent = findViewById(R.id.trackingSettingsContent);
        trackingModeGroup = findViewById(R.id.trackingModeGroup);
        trackingInputPxGroup = findViewById(R.id.trackingInputPxGroup);
        trackingMaxFacesGroup = findViewById(R.id.trackingMaxFacesGroup);
        trackingMinFaceGroup = findViewById(R.id.trackingMinFaceGroup);
        resetTrackingButton = findViewById(R.id.btnResetTrackingSession);
        trackingLoadingIndicator = findViewById(R.id.trackingLoadingIndicator);
        flipTrackingButton = findViewById(R.id.btnFlipTrackingCamera);
    }

    private void configureTabs() {
        View imagePanel = findViewById(R.id.imageDetectionPanel);
        View trackingPanel = findViewById(R.id.videoTrackingPanel);
        ((TabLayout) findViewById(R.id.detectionTabs))
                .addOnTabSelectedListener(new TabLayout.OnTabSelectedListener() {
                    @Override
                    public void onTabSelected(TabLayout.Tab tab) {
                        boolean image = tab.getPosition() == 0;
                        videoSelected = !image;
                        imagePanel.setVisibility(image ? View.VISIBLE : View.GONE);
                        trackingPanel.setVisibility(image ? View.GONE : View.VISIBLE);
                        if (image) {
                            stopVideoTracking();
                        } else {
                            startVideoTracking();
                        }
                    }

                    @Override
                    public void onTabUnselected(TabLayout.Tab tab) {
                    }

                    @Override
                    public void onTabReselected(TabLayout.Tab tab) {
                    }
                });
    }

    private void initializeDetection() {
        StillImageSessionSettings.Values initial = savedSettings;
        boolean ready = FaceEngine.ensureLaunched(this);
        if (ready) {
            session = FaceEngine.createDetectionSession(
                    initial.inputPx, initial.maxFaces, initial.minFacePx);
            ready = session != null;
        }
        boolean finalReady = ready;
        postUi(() -> {
            sessionBusy = false;
            engineReady = finalReady;
            setControlsEnabled(finalReady);
            setTrackingControlsEnabled(finalReady);
            if (finalReady) {
                showSessionSummary(initial);
                if (videoSelected) {
                    startVideoTracking();
                }
            } else {
                setTrackingLoading(false);
                sessionStatus.setText(R.string.recognition_session_failed);
                sessionStatus.setTextColor(color(R.color.liveness_fail));
                setImageStatus(R.string.compare_engine_failed, R.color.liveness_fail);
            }
        });
    }

    private void loadPhoto(Uri uri) {
        int requestVersion = ++imageVersion;
        StillImageSessionSettings.Values settings = savedSettings;
        sessionBusy = true;
        candidates = null;
        selectedIndex = -1;
        faceOverlay.clearFace();
        clearLandmarkPresentation();
        setControlsEnabled(false);
        setImageStatus(R.string.image_analyzing, R.color.home_text_secondary);
        sdkExecutor.execute(() -> {
            Bitmap bitmap = null;
            FaceImageProcessor.Result result = null;
            try {
                bitmap = ImageBitmapLoader.decode(this, uri, MAX_IMAGE_DIMENSION);
                if (destroyed || imageVersion != requestVersion) {
                    recycle(bitmap);
                    return;
                }
                // The 1.2.0 landmark workaround uses LIGHT_TRACK. Start from a clean tracker
                // for every unrelated still image so cached boxes from the previous photo
                // cannot affect this detection.
                replaceDetectionSession(settings);
                if (session != null) {
                    result = FaceImageProcessor.detectWithLandmarks(session, bitmap);
                }
            } catch (Exception ignored) {
                // Null result is rendered as a load/detection failure.
            }
            Bitmap deliveredBitmap = bitmap;
            FaceImageProcessor.Result deliveredResult = result;
            runOnUiThread(() -> applyPhotoResult(
                    requestVersion, deliveredBitmap, deliveredResult));
        });
    }

    /** Called only from {@link #sdkExecutor}. */
    private void replaceDetectionSession(StillImageSessionSettings.Values settings) {
        if (session != null) {
            FaceEngine.releaseSession(session);
            session = null;
        }
        session = FaceEngine.createDetectionSession(
                settings.inputPx, settings.maxFaces, settings.minFacePx);
    }

    private void applyPhotoResult(int requestVersion, @Nullable Bitmap bitmap,
                                  @Nullable FaceImageProcessor.Result result) {
        if (destroyed || imageVersion != requestVersion) {
            recycle(bitmap);
            return;
        }
        Bitmap previous = imageBitmap;
        imageBitmap = bitmap;
        candidates = result == null ? null : result.candidates;
        selectedIndex = candidates != null && candidates.length > 0 ? 0 : -1;
        if (bitmap != null) {
            imageView.setImageBitmap(bitmap);
            imagePlaceholder.setVisibility(View.GONE);
            faceOverlay.showFaces(bitmap.getWidth(), bitmap.getHeight(),
                    result == null ? null : result.faceRects(), selectedIndex);
        } else {
            imageView.setImageDrawable(null);
            imagePlaceholder.setVisibility(View.VISIBLE);
            faceOverlay.clearFace();
        }
        if (previous != bitmap) {
            recycle(previous);
        }
        finishDetectionUi(result);
    }

    private void finishDetectionUi(@Nullable FaceImageProcessor.Result result) {
        sessionBusy = false;
        setControlsEnabled(session != null);
        if (result == null) {
            clearLandmarkPresentation();
            setImageStatus(R.string.image_load_failed, R.color.liveness_fail);
        } else if (result.status == FaceImageProcessor.Status.NO_FACE) {
            clearLandmarkPresentation();
            setImageStatus(R.string.image_no_face, R.color.liveness_fail);
        } else if (result.status != FaceImageProcessor.Status.READY
                || candidates == null || candidates.length == 0) {
            clearLandmarkPresentation();
            setImageStatus(R.string.face_detection_failed, R.color.liveness_fail);
        } else {
            updateSelectionStatus();
            renderSelectedLandmarks();
        }
    }

    private void selectFace(int index) {
        if (sessionBusy || candidates == null || index < 0 || index >= candidates.length) {
            return;
        }
        selectedIndex = index;
        faceOverlay.setSelectedIndex(index);
        updateSelectionStatus();
        renderSelectedLandmarks();
    }

    private void updateSelectionStatus() {
        if (candidates == null || candidates.length == 0) {
            return;
        }
        FaceImageProcessor.Candidate selected = selectedCandidate();
        int landmarkCount = selected == null || selected.denseLandmarks == null
                ? 0 : selected.denseLandmarks.length;
        imageStatus.setText(getString(R.string.detection_face_selected,
                selectedIndex + 1, candidates.length, landmarkCount));
        imageStatus.setTextColor(color(R.color.liveness_accent));
    }

    @Nullable
    private FaceImageProcessor.Candidate selectedCandidate() {
        return candidates != null && selectedIndex >= 0 && selectedIndex < candidates.length
                ? candidates[selectedIndex] : null;
    }

    private void renderSelectedLandmarks() {
        clearLandmarkPresentation();
        FaceImageProcessor.Candidate selected = selectedCandidate();
        Bitmap source = imageBitmap;
        if (selected == null || source == null || source.isRecycled()) {
            landmarkDisplayHint.setText(R.string.detection_landmarks_failed);
            return;
        }

        boolean showLandmarks = landmarkSwitch.isChecked();
        boolean landmarksAvailable = selected.denseLandmarks != null
                && selected.denseLandmarks.length > 0;
        float imageArea = (float) source.getWidth() * source.getHeight();
        float faceArea = Math.max(0f, selected.rect.width())
                * Math.max(0f, selected.rect.height());
        float ratio = imageArea <= 0f ? 0f : faceArea / imageArea;
        int percent = Math.round(ratio * 100f);
        boolean magnified = ratio < DIRECT_LANDMARK_FACE_AREA_RATIO
                && showLandmarkMagnifier(
                        selected, showLandmarks && landmarksAvailable);
        if (!showLandmarks) {
            landmarkDisplayHint.setText(R.string.detection_landmarks_hidden);
            return;
        }
        if (!landmarksAvailable) {
            landmarkDisplayHint.setText(R.string.detection_landmarks_failed);
            return;
        }
        if (magnified) {
            landmarkDisplayHint.setText(getString(
                    R.string.detection_landmarks_in_magnifier, percent));
        } else {
            sourceLandmarkOverlay.showPoints(source.getWidth(), source.getHeight(),
                    selected.denseLandmarks);
            landmarkDisplayHint.setText(getString(
                    R.string.detection_landmarks_on_source, percent));
        }
    }

    private boolean showLandmarkMagnifier(FaceImageProcessor.Candidate selected,
                                          boolean showLandmarks) {
        Bitmap source = imageBitmap;
        if (source == null) {
            return false;
        }
        FaceCropUtils.SquareCrop crop = FaceCropUtils.createSquare(
                source, selected.rect, LANDMARK_MAGNIFIER_CROP_SCALE);
        if (crop == null) {
            return false;
        }
        magnifierBitmap = crop.bitmap;
        magnifierImage.setImageBitmap(crop.bitmap);
        if (showLandmarks) {
            magnifierLandmarkOverlay.showPoints(crop.size, crop.size,
                    selected.denseLandmarks, crop.left, crop.top);
            magnifierLandmarkBadge.setVisibility(View.VISIBLE);
        }
        magnifierCard.setVisibility(View.VISIBLE);
        return true;
    }

    private void clearLandmarkPresentation() {
        sourceLandmarkOverlay.clearPoints();
        magnifierLandmarkOverlay.clearPoints();
        magnifierImage.setImageDrawable(null);
        magnifierLandmarkBadge.setVisibility(View.GONE);
        magnifierCard.setVisibility(View.GONE);
        recycle(magnifierBitmap);
        magnifierBitmap = null;
    }

    private void rebuildSession(StillImageSessionSettings.Values requested) {
        if (sessionBusy) {
            return;
        }
        int version = ++imageVersion;
        Bitmap currentImage = imageBitmap;
        sessionBusy = true;
        candidates = null;
        selectedIndex = -1;
        faceOverlay.clearFace();
        clearLandmarkPresentation();
        setControlsEnabled(false);
        sessionStatus.setText(R.string.recognition_session_rebuilding);
        sessionStatus.setTextColor(color(R.color.home_text_secondary));
        if (currentImage != null) {
            setImageStatus(R.string.image_analyzing, R.color.home_text_secondary);
        }
        sdkExecutor.execute(() -> {
            replaceDetectionSession(requested);
            FaceImageProcessor.Result result = session != null && currentImage != null
                    ? FaceImageProcessor.detectWithLandmarks(session, currentImage) : null;
            boolean ready = session != null;
            postUi(() -> {
                if (imageVersion != version) {
                    return;
                }
                sessionBusy = false;
                setControlsEnabled(ready);
                if (!ready) {
                    sessionStatus.setText(R.string.recognition_session_failed);
                    sessionStatus.setTextColor(color(R.color.liveness_fail));
                    if (currentImage != null) {
                        setImageStatus(R.string.compare_engine_failed, R.color.liveness_fail);
                    }
                    return;
                }
                savedSettings = requested;
                StillImageSessionSettings.save(this, requested);
                showSessionSummary(requested);
                if (currentImage != null) {
                    candidates = result == null ? null : result.candidates;
                    selectedIndex = candidates != null && candidates.length > 0 ? 0 : -1;
                    faceOverlay.showFaces(currentImage.getWidth(), currentImage.getHeight(),
                            result == null ? null : result.faceRects(), selectedIndex);
                    finishDetectionUi(result);
                }
            });
        });
    }

    private void showSessionSummary(StillImageSessionSettings.Values settings) {
        sessionStatus.setText(getString(R.string.recognition_session_summary,
                settings.inputPx, settings.maxFaces, settings.minFacePx));
        sessionStatus.setTextColor(color(R.color.liveness_accent));
    }

    private void selectParameters(StillImageSessionSettings.Values settings) {
        inputPxGroup.check(settings.inputPx == 320 ? R.id.inputPx320
                : settings.inputPx == 1280 ? R.id.inputPx1280 : R.id.inputPx640);
        maxFacesGroup.check(settings.maxFaces == 1 ? R.id.maxFaces1
                : settings.maxFaces == 3 ? R.id.maxFaces3
                : settings.maxFaces == 5 ? R.id.maxFaces5 : R.id.maxFaces10);
        minFaceGroup.check(settings.minFacePx == 48 ? R.id.minFace48
                : settings.minFacePx == 64 ? R.id.minFace64
                : settings.minFacePx == 128 ? R.id.minFace128 : R.id.minFace24);
    }

    private StillImageSessionSettings.Values selectedSettings() {
        int inputId = inputPxGroup.getCheckedChipId();
        int inputPx = inputId == R.id.inputPx320 ? 320
                : inputId == R.id.inputPx1280 ? 1280 : 640;
        int maxId = maxFacesGroup.getCheckedChipId();
        int maxFaces = maxId == R.id.maxFaces1 ? 1 : maxId == R.id.maxFaces3 ? 3
                : maxId == R.id.maxFaces5 ? 5 : 10;
        int minId = minFaceGroup.getCheckedChipId();
        int minFacePx = minId == R.id.minFace48 ? 48 : minId == R.id.minFace64 ? 64
                : minId == R.id.minFace128 ? 128 : 24;
        return new StillImageSessionSettings.Values(inputPx, maxFaces, minFacePx);
    }

    private void setSettingsExpanded(boolean expanded) {
        sessionSettingsContent.setVisibility(expanded ? View.VISIBLE : View.GONE);
        sessionSettingsToggleText.setText(expanded
                ? R.string.recognition_settings_collapse
                : R.string.recognition_settings_expand);
    }

    private void setTrackingSettingsExpanded(boolean expanded) {
        trackingSettingsContent.setVisibility(expanded ? View.VISIBLE : View.GONE);
        trackingSettingsToggleText.setText(expanded
                ? R.string.recognition_settings_collapse
                : R.string.recognition_settings_expand);
    }

    private void startVideoTracking() {
        if (!videoSelected || destroyed || trackingCameraController != null
                || trackingStarting) {
            return;
        }
        setTrackingLoading(true);
        if (!engineReady) {
            showTrackingMessage(
                    R.string.detection_tracking_initializing, R.color.white);
            return;
        }
        if (!cameraPermission.hasPermission()) {
            showTrackingMessage(
                    R.string.msg_permission_required, R.color.liveness_fail);
            cameraPermission.requestAccess();
            return;
        }

        trackingStarting = true;
        setTrackingControlsEnabled(false);
        showTrackingMessage(R.string.detection_tracking_initializing, R.color.white);
        int generation = ++trackingGeneration;
        FaceTrackingSessionSettings.Values settings = trackingSettings;
        FaceTrackingAnalyzer analyzer = new FaceTrackingAnalyzer(
                trackingGlOverlay, settings.mode, settings.inputPx,
                settings.maxFaces, settings.minFacePx,
                new FaceTrackingAnalyzer.Listener() {
                    @Override
                    public void onSessionReady() {
                        postTrackingUi(generation, () -> {
                            trackingStarting = false;
                            setTrackingLoading(false);
                            setTrackingControlsEnabled(engineReady);
                        });
                    }

                    @Override
                    public void onStats(double fps, long latencyMs) {
                        postTrackingUi(generation,
                                () -> renderTrackingStats(fps, latencyMs));
                    }

                    @Override
                    public void onSessionError() {
                        postTrackingUi(generation, () -> {
                            trackingStarting = false;
                            setTrackingLoading(false);
                            setTrackingControlsEnabled(engineReady);
                            showTrackingMessage(
                                    R.string.detection_tracking_session_failed,
                                    R.color.liveness_fail);
                        });
                    }
                });
        trackingAnalyzer = analyzer;
        trackingCameraController = new CameraPreviewController(
                this, this, trackingPreview, trackingExecutor, analyzer,
                new CameraPreviewController.Listener() {
                    @Override
                    public void onCameraReady(boolean frontCamera) {
                        if (!isCurrentTracking(generation)) {
                            return;
                        }
                        trackingFrontCamera = frontCamera;
                        analyzer.setMirrored(frontCamera);
                        showTrackingMessage(
                                R.string.detection_tracking_initializing, R.color.white);
                    }

                    @Override
                    public void onLensChanged(boolean frontCamera) {
                        if (!isCurrentTracking(generation)) {
                            return;
                        }
                        trackingFrontCamera = frontCamera;
                        analyzer.setMirrored(frontCamera);
                        showTrackingMessage(
                                R.string.detection_tracking_initializing, R.color.white);
                    }

                    @Override
                    public void onCameraError(int messageRes) {
                        if (!isCurrentTracking(generation)) {
                            return;
                        }
                        trackingStarting = false;
                        setTrackingLoading(false);
                        setTrackingControlsEnabled(engineReady);
                        showTrackingMessage(messageRes, R.color.liveness_fail);
                    }
                });
        trackingCameraController.start(trackingFrontCamera);
    }

    private void stopVideoTracking() {
        trackingGeneration++;
        trackingStarting = false;
        setTrackingLoading(false);
        if (trackingCameraController != null) {
            trackingCameraController.stop();
            trackingCameraController = null;
        }
        if (trackingAnalyzer != null) {
            FaceTrackingAnalyzer toRelease = trackingAnalyzer;
            trackingAnalyzer = null;
            trackingExecutor.execute(toRelease::release);
        }
        if (trackingGlOverlay != null) {
            trackingGlOverlay.clearTracking();
        }
        if (!destroyed) {
            setTrackingControlsEnabled(engineReady);
        }
    }

    private void flipTrackingCamera() {
        if (trackingCameraController == null || !trackingCameraController.flipCamera()) {
            Toast.makeText(this, R.string.msg_camera_unavailable, Toast.LENGTH_SHORT).show();
        }
    }

    private void applyTrackingSettings(FaceTrackingSessionSettings.Values requested) {
        if (!engineReady || trackingStarting) {
            return;
        }
        if (sameTrackingSettings(requested, trackingSettings)) {
            return;
        }
        setTrackingLoading(true);
        stopVideoTracking();
        trackingSettings = requested;
        FaceTrackingSessionSettings.save(this, requested);
        showTrackingSummary(requested);
        showTrackingMessage(
                R.string.detection_tracking_restarting, R.color.liveness_warn);
        if (videoSelected) {
            startVideoTracking();
        }
    }

    private boolean isCurrentTracking(int generation) {
        return !destroyed && videoSelected && generation == trackingGeneration
                && trackingAnalyzer != null;
    }

    private void postTrackingUi(int generation, Runnable action) {
        runOnUiThread(() -> {
            if (isCurrentTracking(generation)) {
                action.run();
            }
        });
    }

    private void renderTrackingStats(double fps, long latencyMs) {
        trackingStatus.setText(getString(
                R.string.detection_tracking_stats, fps, latencyMs));
        trackingStatus.setTextColor(color(R.color.liveness_accent));
    }

    private void showTrackingMessage(int statusRes, @ColorRes int colorRes) {
        trackingStatus.setText(statusRes);
        trackingStatus.setTextColor(color(colorRes));
    }

    private void showTrackingSummary(FaceTrackingSessionSettings.Values settings) {
        trackingSessionStatus.setText(getString(R.string.detection_tracking_summary,
                trackingModeLabel(settings.mode), settings.inputPx,
                settings.maxFaces, settings.minFacePx));
        trackingSessionStatus.setTextColor(color(R.color.liveness_accent));
    }

    private String trackingModeLabel(int mode) {
        return getString(mode == InspireFace.DETECT_MODE_TRACK_BY_DETECTION
                ? R.string.detection_tracking_mode_tbd
                : R.string.detection_tracking_mode_light);
    }

    private void selectTrackingParameters(FaceTrackingSessionSettings.Values settings) {
        suppressTrackingSettingChanges = true;
        trackingModeGroup.check(settings.mode == InspireFace.DETECT_MODE_TRACK_BY_DETECTION
                ? R.id.trackingModeTbd : R.id.trackingModeLight);
        trackingInputPxGroup.check(settings.inputPx == 320 ? R.id.trackingInputPx320
                : settings.inputPx == 1280
                ? R.id.trackingInputPx1280 : R.id.trackingInputPx640);
        trackingMaxFacesGroup.check(settings.maxFaces == 1 ? R.id.trackingMaxFaces1
                : settings.maxFaces == 3 ? R.id.trackingMaxFaces3
                : settings.maxFaces == 5
                ? R.id.trackingMaxFaces5 : R.id.trackingMaxFaces10);
        trackingMinFaceGroup.check(settings.minFacePx == 48 ? R.id.trackingMinFace48
                : settings.minFacePx == 64 ? R.id.trackingMinFace64
                : settings.minFacePx == 128
                ? R.id.trackingMinFace128 : R.id.trackingMinFace24);
        suppressTrackingSettingChanges = false;
    }

    private void configureTrackingAutoApply() {
        trackingModeGroup.setOnCheckedStateChangeListener(
                (group, checkedIds) -> onTrackingSettingChanged());
        trackingInputPxGroup.setOnCheckedStateChangeListener(
                (group, checkedIds) -> onTrackingSettingChanged());
        trackingMaxFacesGroup.setOnCheckedStateChangeListener(
                (group, checkedIds) -> onTrackingSettingChanged());
        trackingMinFaceGroup.setOnCheckedStateChangeListener(
                (group, checkedIds) -> onTrackingSettingChanged());
    }

    private void onTrackingSettingChanged() {
        if (!suppressTrackingSettingChanges) {
            applyTrackingSettings(selectedTrackingSettings());
        }
    }

    private static boolean sameTrackingSettings(
            FaceTrackingSessionSettings.Values left,
            FaceTrackingSessionSettings.Values right) {
        return left.mode == right.mode && left.inputPx == right.inputPx
                && left.maxFaces == right.maxFaces
                && left.minFacePx == right.minFacePx;
    }

    private FaceTrackingSessionSettings.Values selectedTrackingSettings() {
        int mode = trackingModeGroup.getCheckedChipId() == R.id.trackingModeTbd
                ? InspireFace.DETECT_MODE_TRACK_BY_DETECTION
                : InspireFace.DETECT_MODE_LIGHT_TRACK;
        int inputId = trackingInputPxGroup.getCheckedChipId();
        int inputPx = inputId == R.id.trackingInputPx320 ? 320
                : inputId == R.id.trackingInputPx1280 ? 1280 : 640;
        int maxId = trackingMaxFacesGroup.getCheckedChipId();
        int maxFaces = maxId == R.id.trackingMaxFaces1 ? 1
                : maxId == R.id.trackingMaxFaces3 ? 3
                : maxId == R.id.trackingMaxFaces5 ? 5 : 10;
        int minId = trackingMinFaceGroup.getCheckedChipId();
        int minFacePx = minId == R.id.trackingMinFace48 ? 48
                : minId == R.id.trackingMinFace64 ? 64
                : minId == R.id.trackingMinFace128 ? 128 : 24;
        return new FaceTrackingSessionSettings.Values(
                mode, inputPx, maxFaces, minFacePx);
    }

    private void setControlsEnabled(boolean enabled) {
        choosePhotoButton.setEnabled(enabled);
        applySessionButton.setEnabled(enabled);
        resetSessionButton.setEnabled(enabled);
        landmarkSwitch.setEnabled(enabled);
        setChipGroupEnabled(inputPxGroup, enabled);
        setChipGroupEnabled(maxFacesGroup, enabled);
        setChipGroupEnabled(minFaceGroup, enabled);
    }

    private void setTrackingControlsEnabled(boolean enabled) {
        resetTrackingButton.setEnabled(enabled);
        setChipGroupEnabled(trackingModeGroup, enabled);
        setChipGroupEnabled(trackingInputPxGroup, enabled);
        setChipGroupEnabled(trackingMaxFacesGroup, enabled);
        setChipGroupEnabled(trackingMinFaceGroup, enabled);
    }

    private void setTrackingLoading(boolean loading) {
        trackingLoadingIndicator.setVisibility(loading ? View.VISIBLE : View.GONE);
    }

    private static void setChipGroupEnabled(ChipGroup group, boolean enabled) {
        for (int i = 0; i < group.getChildCount(); i++) {
            group.getChildAt(i).setEnabled(enabled);
        }
    }

    private void setImageStatus(int textRes, @ColorRes int colorRes) {
        imageStatus.setText(textRes);
        imageStatus.setTextColor(color(colorRes));
    }

    private int color(@ColorRes int colorRes) {
        return ContextCompat.getColor(this, colorRes);
    }

    private void applyWindowInsets() {
        View root = findViewById(R.id.detectionRoot);
        int left = root.getPaddingLeft();
        int top = root.getPaddingTop();
        int right = root.getPaddingRight();
        int bottom = root.getPaddingBottom();
        ViewCompat.setOnApplyWindowInsetsListener(root, (v, insets) -> {
            Insets bars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(left + bars.left, top + bars.top,
                    right + bars.right, bottom + bars.bottom);
            return insets;
        });
    }

    private void postUi(Runnable action) {
        runOnUiThread(() -> {
            if (!destroyed && !isFinishing()) {
                action.run();
            }
        });
    }

    private static void recycle(@Nullable Bitmap bitmap) {
        if (bitmap != null && !bitmap.isRecycled()) {
            bitmap.recycle();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        cameraPermission.onResume();
    }

    @Override
    protected void onStart() {
        super.onStart();
        trackingGlOverlay.onResume();
    }

    @Override
    protected void onStop() {
        trackingGlOverlay.onPause();
        super.onStop();
    }

    @Override
    protected void onDestroy() {
        cameraPermission.close();
        destroyed = true;
        engineReady = false;
        imageVersion++;
        stopVideoTracking();
        Bitmap imageToRecycle = imageBitmap;
        Bitmap magnifierToRecycle = magnifierBitmap;
        imageBitmap = null;
        magnifierBitmap = null;
        sdkExecutor.execute(() -> {
            recycle(imageToRecycle);
            recycle(magnifierToRecycle);
            if (session != null) {
                FaceEngine.releaseSession(session);
                session = null;
            }
        });
        sdkExecutor.shutdown();
        trackingExecutor.shutdown();
        super.onDestroy();
    }
}
