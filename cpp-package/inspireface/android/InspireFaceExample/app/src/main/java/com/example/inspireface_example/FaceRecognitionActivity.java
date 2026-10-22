package com.example.inspireface_example;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
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
import com.example.inspireface_example.face.FaceRecord;
import com.example.inspireface_example.face.FaceRepository;
import com.example.inspireface_example.face.ImageBitmapLoader;
import com.example.inspireface_example.permission.CameraPermissionCoordinator;
import com.example.inspireface_example.view.CameraPreviewController;
import com.example.inspireface_example.view.FaceEngine;
import com.example.inspireface_example.view.FaceOverlayView;
import com.example.inspireface_example.view.RecognitionFaceAnalyzer;
import com.example.inspireface_example.widget.FaceImageOverlayView;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.chip.ChipGroup;
import com.google.android.material.tabs.TabLayout;
import com.insightface.sdk.inspireface.base.FaceFeature;
import com.insightface.sdk.inspireface.base.Session;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Model-scoped 1:N recognition from selectable still-image faces or camera face 0. */
public final class FaceRecognitionActivity extends AppCompatActivity {

    private static final int MAX_IMAGE_DIMENSION = 2048;
    private static final float SELECTED_CROP_HIDE_AREA_RATIO = 0.15f;
    private static final float SELECTED_CROP_SCALE = 1.5f;

    private final ExecutorService sdkExecutor = Executors.newSingleThreadExecutor();

    private FaceModelPrefs.Model model;
    private FaceRepository repository;
    private Session session;
    private volatile boolean destroyed;
    private volatile int imageVersion;
    private boolean libraryEmpty;
    private boolean sessionBusy = true;
    private boolean recognitionReady;
    private StillImageSessionSettings.Values savedSettings;

    private ChipGroup inputPxGroup;
    private ChipGroup maxFacesGroup;
    private ChipGroup minFaceGroup;
    private MaterialButton applySessionButton;
    private MaterialButton resetSessionButton;
    private MaterialButton choosePhotoButton;
    private TextView sessionStatus;
    private View emptyLibraryTip;
    private TextView emptyLibraryTipText;
    private ImageView imageView;
    private FaceImageOverlayView faceOverlay;
    private View selectedFaceCropCard;
    private ImageView selectedFaceCropView;
    private View imagePlaceholder;
    private TextView imageStatus;
    private View resultCard;
    private ImageView resultCropView;
    private TextView resultStatus;
    private TextView resultName;
    private TextView resultDetails;
    private View sessionSettingsContent;
    private TextView sessionSettingsToggleText;

    private PreviewView videoPreview;
    private FaceOverlayView videoOverlay;
    private TextView videoStatus;
    private TextView videoName;
    private TextView videoDetails;
    private View flipVideoButton;
    private CameraPreviewController videoCameraController;
    private RecognitionFaceAnalyzer videoAnalyzer;
    private boolean videoSelected;
    private boolean videoStarting;
    private CameraPermissionCoordinator cameraPermission;
    private int videoGeneration;

    private Bitmap imageBitmap;
    private Bitmap resultCropBitmap;
    private Bitmap selectedFaceCropBitmap;
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
                        startVideoRecognition();
                    }

                    @Override
                    public void onCameraPermissionBlocked(boolean requiresSettings) {
                        if (videoSelected) {
                            showVideoError(R.string.msg_permission_required,
                                    requiresSettings
                                            ? R.string.camera_permission_settings_hint
                                            : R.string.camera_permission_retry_hint);
                        }
                    }
                });
        WindowCompat.setDecorFitsSystemWindows(getWindow(), false);
        WindowCompat.getInsetsController(getWindow(), getWindow().getDecorView())
                .setAppearanceLightNavigationBars(false);
        setContentView(R.layout.activity_face_recognition);
        applyWindowInsets();

        model = FaceModelPrefs.get(this);
        repository = new FaceRepository(this, model);
        savedSettings = StillImageSessionSettings.load(this);
        bindViews();
        cameraPermission.bindRecoveryButton(
                findViewById(R.id.btnCameraPermissionAction));
        ((TextView) findViewById(R.id.currentModel)).setText(
                getString(R.string.current_model, model.sdkName()));
        findViewById(R.id.btnBack).setOnClickListener(v -> finish());
        choosePhotoButton.setOnClickListener(v -> photoPicker.launch("image/*"));
        findViewById(R.id.recognitionImageCard)
                .setOnClickListener(v -> {
                    if (!sessionBusy && session != null) {
                        photoPicker.launch("image/*");
                    }
                });
        faceOverlay.setOnFaceSelectedListener(this::selectFace);
        findViewById(R.id.sessionSettingsHeader).setOnClickListener(
                v -> setSettingsExpanded(
                        sessionSettingsContent.getVisibility() != View.VISIBLE));
        applySessionButton.setOnClickListener(v -> rebuildSession(
                selectedSettings(), true));
        resetSessionButton.setOnClickListener(v -> {
            StillImageSessionSettings.Values defaults = StillImageSessionSettings.defaults();
            selectParameters(defaults);
            rebuildSession(defaults, true);
        });
        flipVideoButton.setOnClickListener(v -> flipVideoCamera());
        configureTabs();
        selectParameters(savedSettings);
        setSettingsExpanded(false);
        setControlsEnabled(false);
        sessionStatus.setText(R.string.recognition_session_rebuilding);
        sdkExecutor.execute(this::initializeRecognition);
    }

    private void bindViews() {
        inputPxGroup = findViewById(R.id.inputPxGroup);
        maxFacesGroup = findViewById(R.id.maxFacesGroup);
        minFaceGroup = findViewById(R.id.minFaceGroup);
        applySessionButton = findViewById(R.id.btnApplySession);
        resetSessionButton = findViewById(R.id.btnResetSession);
        choosePhotoButton = findViewById(R.id.btnChooseRecognitionPhoto);
        sessionStatus = findViewById(R.id.sessionStatus);
        emptyLibraryTip = findViewById(R.id.emptyLibraryTip);
        emptyLibraryTipText = findViewById(R.id.emptyLibraryTipText);
        imageView = findViewById(R.id.recognitionImage);
        faceOverlay = findViewById(R.id.recognitionFaceOverlay);
        selectedFaceCropCard = findViewById(R.id.recognitionSelectedFaceCropCard);
        selectedFaceCropView = findViewById(R.id.recognitionSelectedFaceCrop);
        imagePlaceholder = findViewById(R.id.recognitionImagePlaceholder);
        imageStatus = findViewById(R.id.recognitionImageStatus);
        resultCard = findViewById(R.id.recognitionResultCard);
        resultCropView = findViewById(R.id.recognitionResultCrop);
        resultStatus = findViewById(R.id.recognitionResultStatus);
        resultName = findViewById(R.id.recognitionResultName);
        resultDetails = findViewById(R.id.recognitionResultDetails);
        sessionSettingsContent = findViewById(R.id.sessionSettingsContent);
        sessionSettingsToggleText = findViewById(R.id.sessionSettingsToggleText);
        videoPreview = findViewById(R.id.recognitionVideoPreview);
        videoOverlay = findViewById(R.id.recognitionVideoOverlay);
        videoStatus = findViewById(R.id.videoRecognitionStatus);
        videoName = findViewById(R.id.videoRecognitionName);
        videoDetails = findViewById(R.id.videoRecognitionDetails);
        flipVideoButton = findViewById(R.id.btnFlipRecognitionCamera);
    }

    private void configureTabs() {
        View photoPanel = findViewById(R.id.photoRecognitionPanel);
        View videoPanel = findViewById(R.id.videoRecognitionPanel);
        ((TabLayout) findViewById(R.id.recognitionTabs))
                .addOnTabSelectedListener(new TabLayout.OnTabSelectedListener() {
                    @Override
                    public void onTabSelected(TabLayout.Tab tab) {
                        boolean photo = tab.getPosition() == 0;
                        videoSelected = !photo;
                        photoPanel.setVisibility(photo ? View.VISIBLE : View.GONE);
                        videoPanel.setVisibility(photo ? View.GONE : View.VISIBLE);
                        if (photo) {
                            stopVideoRecognition();
                        } else {
                            startVideoRecognition();
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

    private void initializeRecognition() {
        StillImageSessionSettings.Values initialSettings = savedSettings;
        boolean ready = FaceEngine.ensureLaunched(this);
        if (ready) {
            session = FaceEngine.createRecognitionSession(
                    initialSettings.inputPx,
                    initialSettings.maxFaces,
                    initialSettings.minFacePx);
            ready = session != null;
        }
        if (ready) {
            ready = repository.open();
        }
        if (!ready) {
            repository.close();
            if (session != null) {
                FaceEngine.releaseSession(session);
                session = null;
            }
        }
        boolean finalReady = ready;
        boolean empty = ready && repository.query(null).isEmpty();
        postUi(() -> {
            sessionBusy = false;
            recognitionReady = finalReady;
            libraryEmpty = empty;
            updateEmptyLibraryTip();
            setControlsEnabled(finalReady);
            if (finalReady) {
                showSessionSummary(initialSettings);
                if (videoSelected) {
                    startVideoRecognition();
                }
            } else {
                sessionStatus.setText(R.string.recognition_session_failed);
                sessionStatus.setTextColor(color(R.color.liveness_fail));
                setImageStatus(R.string.compare_engine_failed, R.color.liveness_fail);
            }
        });
    }

    private void updateEmptyLibraryTip() {
        emptyLibraryTip.setVisibility(libraryEmpty ? View.VISIBLE : View.GONE);
        if (libraryEmpty) {
            emptyLibraryTipText.setText(
                    getString(R.string.recognition_empty_library, model.sdkName()));
        }
    }

    private void loadPhoto(Uri uri) {
        int requestVersion = ++imageVersion;
        sessionBusy = true;
        candidates = null;
        selectedIndex = -1;
        faceOverlay.clearFace();
        clearSelectedFaceCrop();
        hideResult();
        setControlsEnabled(false);
        setImageStatus(R.string.image_analyzing, R.color.home_text_secondary);
        sdkExecutor.execute(() -> {
            Bitmap bitmap = null;
            FaceImageProcessor.Result result = null;
            try {
                bitmap = ImageBitmapLoader.decode(this, uri, MAX_IMAGE_DIMENSION);
                if (destroyed || imageVersion != requestVersion) {
                    bitmap.recycle();
                    return;
                }
                result = FaceImageProcessor.detect(session, bitmap, false);
            } catch (Exception ignored) {
                // A null result is rendered as a load failure.
            }
            Bitmap deliveredBitmap = bitmap;
            FaceImageProcessor.Result deliveredResult = result;
            runOnUiThread(() -> applyPhotoResult(
                    requestVersion, deliveredBitmap, deliveredResult));
        });
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
            updateSelectedFaceCrop();
        } else {
            imageView.setImageDrawable(null);
            imagePlaceholder.setVisibility(View.VISIBLE);
            faceOverlay.clearFace();
            clearSelectedFaceCrop();
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
            setImageStatus(R.string.image_load_failed, R.color.liveness_fail);
        } else if (result.status == FaceImageProcessor.Status.NO_FACE) {
            setImageStatus(R.string.image_no_face, R.color.liveness_fail);
        } else if (result.status != FaceImageProcessor.Status.READY
                || candidates == null || candidates.length == 0) {
            setImageStatus(R.string.face_extract_failed, R.color.liveness_fail);
        } else {
            updateSelectedFaceStatus();
            recognizeSelectedFace();
        }
    }

    private void selectFace(int index) {
        if (sessionBusy || candidates == null || index < 0 || index >= candidates.length) {
            return;
        }
        selectedIndex = index;
        faceOverlay.setSelectedIndex(index);
        updateSelectedFaceCrop();
        updateSelectedFaceStatus();
        recognizeSelectedFace();
    }

    private void updateSelectedFaceStatus() {
        FaceImageProcessor.Candidate selected = selectedCandidate();
        if (selected == null || selected.feature == null) {
            setImageStatus(R.string.face_extract_failed, R.color.liveness_fail);
        } else if (candidates.length > 1) {
            imageStatus.setText(getString(R.string.image_face_selected,
                    selectedIndex + 1, candidates.length));
            imageStatus.setTextColor(color(R.color.liveness_accent));
        } else {
            setImageStatus(R.string.image_face_ready, R.color.liveness_accent);
        }
    }

    @Nullable
    private FaceImageProcessor.Candidate selectedCandidate() {
        return candidates != null && selectedIndex >= 0 && selectedIndex < candidates.length
                ? candidates[selectedIndex] : null;
    }

    private void updateSelectedFaceCrop() {
        clearSelectedFaceCrop();
        FaceImageProcessor.Candidate selected = selectedCandidate();
        Bitmap source = imageBitmap;
        if (selected == null || source == null || source.isRecycled()) {
            return;
        }
        RectF face = selected.rect;
        float imageArea = (float) source.getWidth() * source.getHeight();
        float faceArea = Math.max(0f, face.width()) * Math.max(0f, face.height());
        if (imageArea <= 0f || faceArea / imageArea >= SELECTED_CROP_HIDE_AREA_RATIO) {
            return;
        }

        int cropSize = Math.max(2, Math.round(
                Math.max(face.width(), face.height()) * SELECTED_CROP_SCALE));
        cropSize = Math.min(cropSize, Math.min(source.getWidth(), source.getHeight()));
        int left = Math.round(face.centerX() - cropSize / 2f);
        int top = Math.round(face.centerY() - cropSize / 2f);
        left = Math.max(0, Math.min(left, source.getWidth() - cropSize));
        top = Math.max(0, Math.min(top, source.getHeight() - cropSize));
        try {
            Bitmap crop = Bitmap.createBitmap(source, left, top, cropSize, cropSize);
            if (crop == source) {
                return;
            }
            selectedFaceCropBitmap = crop;
            selectedFaceCropView.setImageBitmap(crop);
            selectedFaceCropCard.setVisibility(View.VISIBLE);
        } catch (IllegalArgumentException ignored) {
            // A malformed SDK rectangle simply leaves the optional preview hidden.
        }
    }

    private void clearSelectedFaceCrop() {
        if (selectedFaceCropView != null) {
            selectedFaceCropView.setImageDrawable(null);
        }
        if (selectedFaceCropCard != null) {
            selectedFaceCropCard.setVisibility(View.GONE);
        }
        recycle(selectedFaceCropBitmap);
        selectedFaceCropBitmap = null;
    }

    private void recognizeSelectedFace() {
        FaceImageProcessor.Candidate selected = selectedCandidate();
        if (selected == null || selected.feature == null) {
            hideResult();
            return;
        }
        if (libraryEmpty) {
            showEmptyLibraryResult();
            return;
        }
        int version = imageVersion;
        int faceIndex = selectedIndex;
        FaceFeature feature = selected.feature;
        showSearching();
        sdkExecutor.execute(() -> {
            FaceRepository.SearchResult search = repository.search(feature);
            Bitmap crop = search.matched && search.record != null
                    ? BitmapFactory.decodeFile(search.record.cropPath) : null;
            runOnUiThread(() -> applySearchResult(
                    version, faceIndex, feature, search, crop));
        });
    }

    private void applySearchResult(int version, int faceIndex, FaceFeature feature,
                                   FaceRepository.SearchResult search,
                                   @Nullable Bitmap crop) {
        FaceImageProcessor.Candidate selected = selectedCandidate();
        if (destroyed || imageVersion != version || selectedIndex != faceIndex
                || selected == null || selected.feature != feature) {
            recycle(crop);
            return;
        }
        if (search.matched && search.record != null) {
            showMatch(search.record, search.confidence, search.threshold, crop);
        } else {
            recycle(crop);
            showNoMatch(search);
        }
    }

    private void showSearching() {
        clearResultCrop();
        resultCard.setVisibility(View.VISIBLE);
        resultStatus.setText(R.string.recognition_searching);
        resultStatus.setTextColor(color(R.color.home_text_secondary));
        resultName.setText(R.string.recognition_result_name_unknown);
        resultDetails.setText(null);
    }

    private void showMatch(FaceRecord record, float confidence,
                           float threshold, @Nullable Bitmap crop) {
        clearResultCrop();
        resultCropBitmap = crop;
        resultCropView.setImageBitmap(crop);
        resultCropView.setVisibility(crop == null ? View.GONE : View.VISIBLE);
        resultCard.setVisibility(View.VISIBLE);
        resultStatus.setText(R.string.recognition_match_found);
        resultStatus.setTextColor(color(R.color.liveness_accent));
        resultName.setText(record.name);
        resultDetails.setText(getString(R.string.recognition_result_details,
                record.id, confidence, threshold));
    }

    private void showNoMatch(FaceRepository.SearchResult search) {
        clearResultCrop();
        resultCard.setVisibility(View.VISIBLE);
        resultStatus.setText(R.string.recognition_no_match);
        resultStatus.setTextColor(color(R.color.liveness_fail));
        resultName.setText(R.string.recognition_result_name_unknown);
        if (!Float.isNaN(search.confidence)) {
            resultDetails.setText(getString(R.string.recognition_no_match_details,
                    search.confidence, search.threshold));
        } else {
            resultDetails.setText(R.string.recognition_no_confidence);
        }
    }

    private void showEmptyLibraryResult() {
        clearResultCrop();
        resultCard.setVisibility(View.VISIBLE);
        resultStatus.setText(R.string.recognition_library_empty_result);
        resultStatus.setTextColor(color(R.color.liveness_warn));
        resultName.setText(model.sdkName());
        resultDetails.setText(getString(
                R.string.recognition_empty_library, model.sdkName()));
    }

    private void hideResult() {
        clearResultCrop();
        resultCard.setVisibility(View.GONE);
    }

    private void clearResultCrop() {
        resultCropView.setImageDrawable(null);
        resultCropView.setVisibility(View.GONE);
        recycle(resultCropBitmap);
        resultCropBitmap = null;
    }

    private void rebuildSession(StillImageSessionSettings.Values requested,
                                boolean persistOnSuccess) {
        if (sessionBusy) {
            return;
        }
        int version = ++imageVersion;
        Bitmap currentImage = imageBitmap;
        sessionBusy = true;
        candidates = null;
        selectedIndex = -1;
        faceOverlay.clearFace();
        clearSelectedFaceCrop();
        hideResult();
        setControlsEnabled(false);
        sessionStatus.setText(R.string.recognition_session_rebuilding);
        sessionStatus.setTextColor(color(R.color.home_text_secondary));
        if (currentImage != null) {
            setImageStatus(R.string.image_analyzing, R.color.home_text_secondary);
        }
        sdkExecutor.execute(() -> {
            if (session != null) {
                FaceEngine.releaseSession(session);
                session = null;
            }
            session = FaceEngine.createRecognitionSession(
                    requested.inputPx, requested.maxFaces, requested.minFacePx);
            FaceImageProcessor.Result result = session != null && currentImage != null
                    ? FaceImageProcessor.detect(session, currentImage, false) : null;
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
                if (persistOnSuccess) {
                    StillImageSessionSettings.save(this, requested);
                }
                showSessionSummary(requested);
                if (currentImage != null) {
                    candidates = result == null ? null : result.candidates;
                    selectedIndex = candidates != null && candidates.length > 0 ? 0 : -1;
                    faceOverlay.showFaces(currentImage.getWidth(), currentImage.getHeight(),
                            result == null ? null : result.faceRects(), selectedIndex);
                    updateSelectedFaceCrop();
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
        if (settings.maxFaces == 1) {
            maxFacesGroup.check(R.id.maxFaces1);
        } else if (settings.maxFaces == 3) {
            maxFacesGroup.check(R.id.maxFaces3);
        } else if (settings.maxFaces == 5) {
            maxFacesGroup.check(R.id.maxFaces5);
        } else {
            maxFacesGroup.check(R.id.maxFaces10);
        }
        if (settings.minFacePx == 48) {
            minFaceGroup.check(R.id.minFace48);
        } else if (settings.minFacePx == 64) {
            minFaceGroup.check(R.id.minFace64);
        } else if (settings.minFacePx == 128) {
            minFaceGroup.check(R.id.minFace128);
        } else {
            minFaceGroup.check(R.id.minFace24);
        }
    }

    private StillImageSessionSettings.Values selectedSettings() {
        return new StillImageSessionSettings.Values(
                selectedInputPx(), selectedMaxFaces(), selectedMinFacePx());
    }

    private void setSettingsExpanded(boolean expanded) {
        sessionSettingsContent.setVisibility(expanded ? View.VISIBLE : View.GONE);
        sessionSettingsToggleText.setText(expanded
                ? R.string.recognition_settings_collapse
                : R.string.recognition_settings_expand);
    }

    private void startVideoRecognition() {
        if (!videoSelected || destroyed || videoCameraController != null || videoStarting) {
            return;
        }
        if (!recognitionReady) {
            showVideoError(R.string.recognition_video_initializing,
                    R.string.recognition_video_first_face_hint);
            return;
        }
        if (!cameraPermission.hasPermission()) {
            showVideoError(R.string.msg_permission_required,
                    R.string.capture_permission_hint);
            cameraPermission.requestAccess();
            return;
        }

        videoStarting = true;
        showVideoError(R.string.recognition_video_initializing,
                R.string.recognition_video_first_face_hint);
        int generation = ++videoGeneration;
        RecognitionFaceAnalyzer analyzer = new RecognitionFaceAnalyzer(
                this, videoOverlay, repository, libraryEmpty,
                new RecognitionFaceAnalyzer.Listener() {
                    @Override
                    public void onState(RecognitionFaceAnalyzer.State state,
                                        @Nullable FaceRecord record,
                                        float confidence, float threshold) {
                        postVideoUi(generation,
                                () -> renderVideoState(state, record, confidence, threshold));
                    }

                    @Override
                    public void onSessionError() {
                        postVideoUi(generation, () -> {
                            videoStarting = false;
                            showVideoError(R.string.compare_engine_failed,
                                    R.string.recognition_video_first_face_hint);
                        });
                    }
                });
        videoAnalyzer = analyzer;
        videoCameraController = new CameraPreviewController(
                this, this, videoPreview, sdkExecutor, analyzer,
                new CameraPreviewController.Listener() {
                    @Override
                    public void onCameraReady(boolean frontCamera) {
                        if (!isCurrentVideo(generation)) {
                            return;
                        }
                        videoStarting = false;
                        analyzer.setMirrored(frontCamera);
                        renderVideoState(RecognitionFaceAnalyzer.State.NO_FACE,
                                null, Float.NaN, Float.NaN);
                    }

                    @Override
                    public void onLensChanged(boolean frontCamera) {
                        if (!isCurrentVideo(generation)) {
                            return;
                        }
                        analyzer.setMirrored(frontCamera);
                        analyzer.resetTracking();
                        renderVideoState(RecognitionFaceAnalyzer.State.NO_FACE,
                                null, Float.NaN, Float.NaN);
                    }

                    @Override
                    public void onCameraError(int messageRes) {
                        if (!isCurrentVideo(generation)) {
                            return;
                        }
                        videoStarting = false;
                        showVideoError(messageRes,
                                R.string.recognition_video_first_face_hint);
                    }
                });
        videoCameraController.start();
    }

    private void stopVideoRecognition() {
        videoGeneration++;
        videoStarting = false;
        if (videoCameraController != null) {
            videoCameraController.stop();
            videoCameraController = null;
        }
        if (videoAnalyzer != null) {
            RecognitionFaceAnalyzer toRelease = videoAnalyzer;
            videoAnalyzer = null;
            sdkExecutor.execute(toRelease::release);
        }
        if (videoOverlay != null) {
            videoOverlay.submit(null);
        }
    }

    private void flipVideoCamera() {
        if (videoCameraController == null || !videoCameraController.flipCamera()) {
            Toast.makeText(this, R.string.msg_camera_unavailable, Toast.LENGTH_SHORT).show();
        }
    }

    private boolean isCurrentVideo(int generation) {
        return !destroyed && videoSelected && generation == videoGeneration
                && videoAnalyzer != null;
    }

    private void postVideoUi(int generation, Runnable action) {
        runOnUiThread(() -> {
            if (isCurrentVideo(generation)) {
                action.run();
            }
        });
    }

    private void renderVideoState(RecognitionFaceAnalyzer.State state,
                                  @Nullable FaceRecord record,
                                  float confidence, float threshold) {
        videoName.setVisibility(View.GONE);
        switch (state) {
            case MOVE_CLOSER:
                showVideoError(R.string.capture_move_closer,
                        R.string.recognition_video_first_face_hint);
                videoStatus.setTextColor(color(R.color.liveness_warn));
                break;
            case HOLD_STILL:
                showVideoError(R.string.recognition_video_hold_still,
                        R.string.recognition_video_first_face_hint);
                videoStatus.setTextColor(color(R.color.white));
                break;
            case SEARCHING:
                showVideoError(R.string.recognition_searching,
                        R.string.recognition_video_first_face_hint);
                videoStatus.setTextColor(color(R.color.liveness_warn));
                break;
            case MATCHED:
                videoStatus.setText(R.string.recognition_match_found);
                videoStatus.setTextColor(color(R.color.liveness_accent));
                videoName.setVisibility(View.VISIBLE);
                videoName.setText(record == null
                        ? getString(R.string.recognition_result_name_unknown) : record.name);
                if (record != null) {
                    videoDetails.setText(getString(R.string.recognition_result_details,
                            record.id, confidence, threshold));
                } else {
                    videoDetails.setText(R.string.recognition_no_confidence);
                }
                break;
            case NO_MATCH:
                videoStatus.setText(R.string.recognition_no_match);
                videoStatus.setTextColor(color(R.color.liveness_fail));
                videoName.setVisibility(View.VISIBLE);
                videoName.setText(R.string.recognition_result_name_unknown);
                videoDetails.setText(Float.isNaN(confidence)
                        ? getString(R.string.recognition_no_confidence)
                        : getString(R.string.recognition_no_match_details,
                        confidence, threshold));
                break;
            case EMPTY_LIBRARY:
                videoStatus.setText(R.string.recognition_library_empty_result);
                videoStatus.setTextColor(color(R.color.liveness_warn));
                videoName.setVisibility(View.VISIBLE);
                videoName.setText(model.sdkName());
                videoDetails.setText(getString(
                        R.string.recognition_empty_library, model.sdkName()));
                break;
            case NO_FACE:
            default:
                showVideoError(R.string.recognition_video_no_face,
                        R.string.recognition_video_first_face_hint);
                videoStatus.setTextColor(color(R.color.white));
                break;
        }
    }

    private void showVideoError(int titleRes, int detailsRes) {
        videoStatus.setText(titleRes);
        videoStatus.setTextColor(color(R.color.white));
        videoName.setVisibility(View.GONE);
        videoDetails.setText(detailsRes);
    }

    private int selectedInputPx() {
        int id = inputPxGroup.getCheckedChipId();
        return id == R.id.inputPx320 ? 320 : id == R.id.inputPx1280 ? 1280 : 640;
    }

    private int selectedMaxFaces() {
        int id = maxFacesGroup.getCheckedChipId();
        if (id == R.id.maxFaces1) return 1;
        if (id == R.id.maxFaces3) return 3;
        if (id == R.id.maxFaces5) return 5;
        return 10;
    }

    private int selectedMinFacePx() {
        int id = minFaceGroup.getCheckedChipId();
        if (id == R.id.minFace48) return 48;
        if (id == R.id.minFace64) return 64;
        if (id == R.id.minFace128) return 128;
        return 24;
    }

    private void setControlsEnabled(boolean enabled) {
        choosePhotoButton.setEnabled(enabled);
        applySessionButton.setEnabled(enabled);
        resetSessionButton.setEnabled(enabled);
        setChipGroupEnabled(inputPxGroup, enabled);
        setChipGroupEnabled(maxFacesGroup, enabled);
        setChipGroupEnabled(minFaceGroup, enabled);
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
        View root = findViewById(R.id.recognitionRoot);
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
    protected void onDestroy() {
        cameraPermission.close();
        destroyed = true;
        recognitionReady = false;
        imageVersion++;
        stopVideoRecognition();
        Bitmap imageToRecycle = imageBitmap;
        Bitmap cropToRecycle = resultCropBitmap;
        Bitmap selectedCropToRecycle = selectedFaceCropBitmap;
        imageBitmap = null;
        resultCropBitmap = null;
        selectedFaceCropBitmap = null;
        sdkExecutor.execute(() -> {
            recycle(imageToRecycle);
            recycle(cropToRecycle);
            recycle(selectedCropToRecycle);
            repository.close();
            if (session != null) {
                FaceEngine.releaseSession(session);
                session = null;
            }
        });
        sdkExecutor.shutdown();
        super.onDestroy();
    }
}
