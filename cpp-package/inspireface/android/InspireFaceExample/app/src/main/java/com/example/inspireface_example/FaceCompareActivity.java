package com.example.inspireface_example;

import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.ColorRes;
import androidx.annotation.Nullable;
import androidx.annotation.StringRes;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;

import com.example.inspireface_example.face.FaceImageProcessor;
import com.example.inspireface_example.face.ImageBitmapLoader;
import com.example.inspireface_example.view.FaceEngine;
import com.example.inspireface_example.widget.FaceImageOverlayView;
import com.example.inspireface_example.widget.SimilarityGaugeView;
import com.insightface.sdk.inspireface.InspireFace;
import com.insightface.sdk.inspireface.base.FaceFeature;
import com.insightface.sdk.inspireface.base.Session;
import com.insightface.sdk.inspireface.base.SimilarityConverterConfig;

import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Still-image 1:1 face comparison using the globally selected InspireFace model. */
public class FaceCompareActivity extends AppCompatActivity {

    private static final int MAX_IMAGE_DIMENSION = 2048;

    private final ExecutorService sdkExecutor = Executors.newSingleThreadExecutor();
    private final ImageSlot slotA = new ImageSlot();
    private final ImageSlot slotB = new ImageSlot();

    private SimilarityGaugeView similarityGauge;
    private TextView comparisonDetails;
    private Session session;
    private volatile boolean destroyed;

    private final ActivityResultLauncher<String> imageAPicker =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) {
                    selectImage(slotA, uri);
                }
            });

    private final ActivityResultLauncher<String> imageBPicker =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) {
                    selectImage(slotB, uri);
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        WindowCompat.setDecorFitsSystemWindows(getWindow(), false);
        WindowCompat.getInsetsController(getWindow(), getWindow().getDecorView())
                .setAppearanceLightNavigationBars(false);
        setContentView(R.layout.activity_face_compare);
        applyWindowInsets();

        bindSlot(slotA, R.id.imageA, R.id.faceOverlayA, R.id.placeholderA, R.id.statusA,
                R.id.compareCropCardA, R.id.compareCropA);
        bindSlot(slotB, R.id.imageB, R.id.faceOverlayB, R.id.placeholderB, R.id.statusB,
                R.id.compareCropCardB, R.id.compareCropB);
        similarityGauge = findViewById(R.id.similarityGauge);
        comparisonDetails = findViewById(R.id.comparisonDetails);

        ((TextView) findViewById(R.id.currentModel)).setText(
                getString(R.string.current_model, FaceModelPrefs.get(this).sdkName()));
        findViewById(R.id.btnBack).setOnClickListener(v -> finish());
        findViewById(R.id.cardImageA).setOnClickListener(v -> imageAPicker.launch("image/*"));
        findViewById(R.id.cardImageB).setOnClickListener(v -> imageBPicker.launch("image/*"));

        similarityGauge.showMessage(getString(R.string.compare_select_two));
        sdkExecutor.execute(this::initializeEngine);
    }

    private void bindSlot(ImageSlot slot, int imageId, int overlayId,
                          int placeholderId, int statusId, int cropCardId, int cropId) {
        slot.imageView = findViewById(imageId);
        slot.faceOverlay = findViewById(overlayId);
        slot.faceOverlay.setOnFaceSelectedListener(index -> selectFace(slot, index));
        slot.placeholder = findViewById(placeholderId);
        slot.statusView = findViewById(statusId);
        slot.cropCard = findViewById(cropCardId);
        slot.cropView = findViewById(cropId);
    }

    private void initializeEngine() {
        boolean launched = FaceEngine.ensureLaunched(this);
        if (launched) {
            session = FaceEngine.createRecognitionSession();
        }
        if (!launched || session == null) {
            postUi(() -> {
                similarityGauge.showMessage(getString(R.string.compare_engine_failed));
                setSlotStatus(slotA, R.string.compare_engine_failed, R.color.liveness_fail);
                setSlotStatus(slotB, R.string.compare_engine_failed, R.color.liveness_fail);
            });
        }
    }

    private void selectImage(ImageSlot slot, Uri uri) {
        int requestVersion = ++slot.version;
        FaceImageProcessor.Candidate[] previousCandidates = slot.candidates;
        clearSelectedCrop(slot);
        slot.feature = null;
        slot.candidates = null;
        slot.selectedIndex = -1;
        slot.pending = true;
        FaceImageProcessor.recycleCrops(previousCandidates, -1);
        slot.faceOverlay.clearFace();
        setSlotStatus(slot, R.string.image_analyzing, R.color.home_text_secondary);
        comparisonDetails.setText(null);
        updateGaugeForSlots();
        sdkExecutor.execute(() -> processImage(slot, uri, requestVersion));
    }

    private void processImage(ImageSlot slot, Uri uri, int requestVersion) {
        Bitmap bitmap = null;
        FaceImageProcessor.Candidate[] candidates = null;
        int statusRes;
        try {
            bitmap = ImageBitmapLoader.decode(this, uri, MAX_IMAGE_DIMENSION);
            if (slot.version != requestVersion || destroyed) {
                bitmap.recycle();
                return;
            }
            if (session == null) {
                statusRes = R.string.compare_engine_failed;
            } else {
                FaceImageProcessor.Result result =
                        FaceImageProcessor.detect(session, bitmap, true);
                candidates = result.candidates;
                if (result.status == FaceImageProcessor.Status.NO_FACE) {
                    statusRes = R.string.image_no_face;
                } else if (result.status != FaceImageProcessor.Status.READY) {
                    statusRes = R.string.face_extract_failed;
                } else if (candidates[0].feature == null) {
                    statusRes = R.string.face_extract_failed;
                } else {
                    statusRes = R.string.image_face_ready;
                }
            }
        } catch (Exception e) {
            statusRes = R.string.image_load_failed;
        }

        Bitmap deliveredBitmap = bitmap;
        FaceImageProcessor.Candidate[] deliveredCandidates = candidates;
        int deliveredStatus = statusRes;
        // applyImageResult owns cleanup for obsolete results, including Activity teardown.
        runOnUiThread(() -> applyImageResult(slot, requestVersion, deliveredBitmap,
                deliveredCandidates, deliveredStatus));
    }

    private void applyImageResult(ImageSlot slot, int requestVersion, Bitmap bitmap,
                                  FaceImageProcessor.Candidate[] candidates,
                                  @StringRes int statusRes) {
        if (destroyed || slot.version != requestVersion) {
            if (bitmap != null && !bitmap.isRecycled()) {
                bitmap.recycle();
            }
            FaceImageProcessor.recycleCrops(candidates, -1);
            return;
        }
        slot.pending = false;
        FaceImageProcessor.Candidate[] previousCandidates = slot.candidates;
        if (previousCandidates != candidates) {
            clearSelectedCrop(slot);
            FaceImageProcessor.recycleCrops(previousCandidates, -1);
        }
        slot.candidates = candidates;
        slot.selectedIndex = candidates != null && candidates.length > 0 ? 0 : -1;
        slot.feature = slot.selectedIndex >= 0 ? candidates[0].feature : null;
        if (bitmap != null) {
            Bitmap previous = slot.bitmap;
            slot.bitmap = bitmap;
            slot.imageView.setImageBitmap(bitmap);
            slot.faceOverlay.showFaces(bitmap.getWidth(), bitmap.getHeight(),
                    candidates == null ? null : faceRects(candidates), slot.selectedIndex);
            slot.placeholder.setVisibility(View.GONE);
            if (previous != null && previous != bitmap && !previous.isRecycled()) {
                previous.recycle();
            }
        }
        updateSelectedCrop(slot);
        if (slot.feature != null && candidates.length > 1) {
            setSlotStatus(slot, getString(R.string.image_face_selected,
                    slot.selectedIndex + 1, candidates.length), R.color.liveness_accent);
        } else {
            setSlotStatus(slot, getString(statusRes),
                    slot.feature != null ? R.color.liveness_accent : R.color.liveness_fail);
        }
        updateGaugeForSlots();
    }

    private void selectFace(ImageSlot slot, int index) {
        if (slot.pending || slot.candidates == null
                || index < 0 || index >= slot.candidates.length) {
            return;
        }
        slot.selectedIndex = index;
        slot.feature = slot.candidates[index].feature;
        slot.faceOverlay.setSelectedIndex(index);
        updateSelectedCrop(slot);
        comparisonDetails.setText(null);
        if (slot.feature == null) {
            setSlotStatus(slot, getString(R.string.face_extract_failed), R.color.liveness_fail);
        } else if (slot.candidates.length > 1) {
            setSlotStatus(slot, getString(R.string.image_face_selected,
                    index + 1, slot.candidates.length), R.color.liveness_accent);
        } else {
            setSlotStatus(slot, getString(R.string.image_face_ready), R.color.liveness_accent);
        }
        updateGaugeForSlots();
    }

    private void updateSelectedCrop(ImageSlot slot) {
        if (slot.candidates == null || slot.selectedIndex < 0
                || slot.selectedIndex >= slot.candidates.length) {
            clearSelectedCrop(slot);
            return;
        }
        Bitmap crop = slot.candidates[slot.selectedIndex].crop;
        if (crop == null || crop.isRecycled()) {
            clearSelectedCrop(slot);
            return;
        }
        slot.cropView.setImageBitmap(crop);
        slot.cropCard.setVisibility(View.VISIBLE);
    }

    private static void clearSelectedCrop(ImageSlot slot) {
        if (slot.cropView != null) {
            slot.cropView.setImageDrawable(null);
        }
        if (slot.cropCard != null) {
            slot.cropCard.setVisibility(View.INVISIBLE);
        }
    }

    private static android.graphics.RectF[] faceRects(
            FaceImageProcessor.Candidate[] candidates) {
        android.graphics.RectF[] rects = new android.graphics.RectF[candidates.length];
        for (int i = 0; i < candidates.length; i++) {
            rects[i] = candidates[i].rect;
        }
        return rects;
    }

    private void updateGaugeForSlots() {
        if (slotA.pending || slotB.pending) {
            similarityGauge.showMessage(getString(R.string.compare_analyzing));
            return;
        }
        if (slotA.feature == null || slotB.feature == null) {
            similarityGauge.showMessage(getString(
                    slotA.bitmap == null && slotB.bitmap == null
                            ? R.string.compare_select_two : R.string.compare_waiting_face));
            return;
        }

        int versionA = slotA.version;
        int versionB = slotB.version;
        FaceFeature featureA = slotA.feature;
        FaceFeature featureB = slotB.feature;
        similarityGauge.showMessage(getString(R.string.compare_comparing));
        sdkExecutor.execute(() -> compareFeatures(
                versionA, versionB, featureA, featureB));
    }

    private void compareFeatures(int versionA, int versionB,
                                 FaceFeature featureA, FaceFeature featureB) {
        float cosine = InspireFace.FaceComparison(featureA, featureB);
        float threshold = InspireFace.GetRecommendedCosineThreshold();
        SimilarityConverterConfig converter = InspireFace.GetCosineSimilarityConverter();
        float converted = InspireFace.CosineSimilarityConvertToPercentage(cosine);
        boolean converterUsesUnitRange = converter == null
                ? converted >= 0f && converted <= 1f
                : converter.outputMax <= 1.0001f;
        float percent = converterUsesUnitRange ? converted * 100f : converted;
        if (Float.isNaN(percent) || Float.isInfinite(percent)) {
            postUi(() -> similarityGauge.showMessage(getString(R.string.compare_failed)));
            return;
        }
        percent = Math.max(0f, Math.min(100f, percent));
        boolean matched = cosine >= threshold;
        float finalPercent = percent;
        postUi(() -> {
            if (slotA.version != versionA || slotB.version != versionB
                    || slotA.feature != featureA || slotB.feature != featureB) {
                return;
            }
            similarityGauge.showResult(finalPercent, matched,
                    getString(matched ? R.string.compare_same_person
                            : R.string.compare_different_person));
            comparisonDetails.setText(String.format(Locale.US,
                    getString(R.string.compare_details), cosine, threshold));
        });
    }

    private void setSlotStatus(ImageSlot slot, @StringRes int statusRes,
                               @ColorRes int colorRes) {
        setSlotStatus(slot, getString(statusRes), colorRes);
    }

    private void setSlotStatus(ImageSlot slot, CharSequence status,
                               @ColorRes int colorRes) {
        slot.statusView.setText(status);
        slot.statusView.setTextColor(ContextCompat.getColor(this, colorRes));
    }

    private void applyWindowInsets() {
        View root = findViewById(R.id.compareRoot);
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

    @Override
    protected void onDestroy() {
        destroyed = true;
        slotA.version++;
        slotB.version++;
        Bitmap bitmapA = slotA.bitmap;
        Bitmap bitmapB = slotB.bitmap;
        FaceImageProcessor.Candidate[] candidatesA = slotA.candidates;
        FaceImageProcessor.Candidate[] candidatesB = slotB.candidates;
        clearSelectedCrop(slotA);
        clearSelectedCrop(slotB);
        slotA.bitmap = null;
        slotB.bitmap = null;
        slotA.candidates = null;
        slotB.candidates = null;
        sdkExecutor.execute(() -> {
            if (session != null) {
                FaceEngine.releaseSession(session);
                session = null;
            }
            FaceImageProcessor.recycleCrops(candidatesA, -1);
            FaceImageProcessor.recycleCrops(candidatesB, -1);
            if (bitmapA != null && !bitmapA.isRecycled()) {
                bitmapA.recycle();
            }
            if (bitmapB != null && bitmapB != bitmapA && !bitmapB.isRecycled()) {
                bitmapB.recycle();
            }
        });
        sdkExecutor.shutdown();
        super.onDestroy();
    }

    private static final class ImageSlot {
        volatile int version;
        boolean pending;
        Bitmap bitmap;
        FaceFeature feature;
        FaceImageProcessor.Candidate[] candidates;
        int selectedIndex = -1;
        ImageView imageView;
        FaceImageOverlayView faceOverlay;
        View placeholder;
        TextView statusView;
        View cropCard;
        ImageView cropView;
    }

}
