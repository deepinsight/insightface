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

import com.example.inspireface_example.face.FaceAttributeProcessor;
import com.example.inspireface_example.face.FaceCropUtils;
import com.example.inspireface_example.face.FaceImageProcessor;
import com.example.inspireface_example.face.ImageBitmapLoader;
import com.example.inspireface_example.view.FaceEngine;
import com.example.inspireface_example.widget.FaceImageOverlayView;
import com.google.android.material.button.MaterialButton;
import com.insightface.sdk.inspireface.base.Session;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Selectable still-image face attribute analysis. */
public final class FaceAttributeActivity extends AppCompatActivity {

    private static final int MAX_IMAGE_DIMENSION = 2048;
    private static final float DIRECT_FACE_AREA_RATIO = 0.05f;
    private static final float MAGNIFIER_CROP_SCALE = 2.4f;
    private static final float BINARY_CONFIDENCE_THRESHOLD = 0.5f;

    private final ExecutorService sdkExecutor = Executors.newSingleThreadExecutor();

    private Session session;
    private volatile boolean destroyed;
    private volatile int imageVersion;
    private boolean busy = true;

    private MaterialButton choosePhotoButton;
    private ImageView imageView;
    private View imagePlaceholder;
    private TextView imageStatus;
    private FaceImageOverlayView faceOverlay;
    private View magnifierCard;
    private ImageView magnifierImage;
    private View resultValues;
    private TextView resultPlaceholder;
    private TextView maskValue;
    private TextView ageValue;
    private TextView qualityValue;
    private TextView expressionValue;
    private TextView raceValue;
    private TextView genderValue;
    private TextView leftEyeValue;
    private TextView rightEyeValue;

    private Bitmap imageBitmap;
    private Bitmap magnifierBitmap;
    private FaceAttributeProcessor.Result analysis;
    private int selectedIndex = -1;

    private final ActivityResultLauncher<String> photoPicker =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null && !busy && session != null) {
                    loadPhoto(uri);
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        WindowCompat.setDecorFitsSystemWindows(getWindow(), false);
        WindowCompat.getInsetsController(getWindow(), getWindow().getDecorView())
                .setAppearanceLightNavigationBars(false);
        setContentView(R.layout.activity_face_attribute);
        applyWindowInsets();
        bindViews();

        ((TextView) findViewById(R.id.currentModel)).setText(getString(
                R.string.current_model, FaceModelPrefs.get(this).sdkName()));
        findViewById(R.id.btnBack).setOnClickListener(v -> finish());
        choosePhotoButton.setOnClickListener(v -> photoPicker.launch("image/*"));
        findViewById(R.id.attributeImageCard).setOnClickListener(v -> {
            if (!busy && session != null) {
                photoPicker.launch("image/*");
            }
        });
        faceOverlay.setOnFaceSelectedListener(this::selectFace);
        setControlsEnabled(false);
        showResultPlaceholder(R.string.attribute_result_empty);
        sdkExecutor.execute(this::initializeEngine);
    }

    private void bindViews() {
        choosePhotoButton = findViewById(R.id.btnChooseAttributePhoto);
        imageView = findViewById(R.id.attributeImage);
        imagePlaceholder = findViewById(R.id.attributeImagePlaceholder);
        imageStatus = findViewById(R.id.attributeImageStatus);
        faceOverlay = findViewById(R.id.attributeFaceOverlay);
        magnifierCard = findViewById(R.id.attributeMagnifierCard);
        magnifierImage = findViewById(R.id.attributeMagnifierImage);
        resultValues = findViewById(R.id.attributeResultValues);
        resultPlaceholder = findViewById(R.id.attributeResultPlaceholder);
        maskValue = findViewById(R.id.attributeMaskValue);
        ageValue = findViewById(R.id.attributeAgeValue);
        qualityValue = findViewById(R.id.attributeQualityValue);
        expressionValue = findViewById(R.id.attributeExpressionValue);
        raceValue = findViewById(R.id.attributeRaceValue);
        genderValue = findViewById(R.id.attributeGenderValue);
        leftEyeValue = findViewById(R.id.attributeLeftEyeValue);
        rightEyeValue = findViewById(R.id.attributeRightEyeValue);
    }

    private void initializeEngine() {
        boolean ready = FaceEngine.ensureLaunched(this);
        if (ready) {
            session = FaceEngine.createAttributeSession();
            ready = session != null;
        }
        boolean finalReady = ready;
        runOnUiThread(() -> {
            if (destroyed) {
                return;
            }
            busy = false;
            setControlsEnabled(finalReady);
            if (finalReady) {
                setImageStatus(R.string.no_image_selected, R.color.home_text_secondary);
            } else {
                setImageStatus(R.string.compare_engine_failed, R.color.liveness_fail);
                showResultPlaceholder(R.string.attribute_analysis_failed);
            }
        });
    }

    private void loadPhoto(Uri uri) {
        int requestVersion = ++imageVersion;
        busy = true;
        analysis = null;
        selectedIndex = -1;
        faceOverlay.clearFace();
        clearMagnifier();
        showResultPlaceholder(R.string.attribute_analyzing);
        setImageStatus(R.string.attribute_analyzing, R.color.home_text_secondary);
        setControlsEnabled(false);

        sdkExecutor.execute(() -> {
            Bitmap bitmap = null;
            FaceAttributeProcessor.Result result = null;
            try {
                bitmap = ImageBitmapLoader.decode(this, uri, MAX_IMAGE_DIMENSION);
                if (destroyed || imageVersion != requestVersion) {
                    recycle(bitmap);
                    return;
                }
                if (session != null) {
                    result = FaceAttributeProcessor.analyze(session, bitmap);
                }
            } catch (Exception ignored) {
                // Null result is presented as a load or analysis failure.
            }
            Bitmap deliveredBitmap = bitmap;
            FaceAttributeProcessor.Result deliveredResult = result;
            runOnUiThread(() -> applyResult(
                    requestVersion, deliveredBitmap, deliveredResult));
        });
    }

    private void applyResult(int requestVersion, @Nullable Bitmap bitmap,
                             @Nullable FaceAttributeProcessor.Result result) {
        if (destroyed || imageVersion != requestVersion) {
            recycle(bitmap);
            return;
        }
        busy = false;
        setControlsEnabled(session != null);
        Bitmap previous = imageBitmap;
        imageBitmap = bitmap;
        analysis = result;
        selectedIndex = result != null && result.candidates.length > 0 ? 0 : -1;

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

        if (result == null) {
            setImageStatus(bitmap == null ? R.string.image_load_failed
                    : R.string.attribute_analysis_failed, R.color.liveness_fail);
            showResultPlaceholder(R.string.attribute_analysis_failed);
        } else if (result.status == FaceAttributeProcessor.Status.NO_FACE) {
            setImageStatus(R.string.image_no_face, R.color.liveness_fail);
            showResultPlaceholder(R.string.attribute_no_face_result);
        } else if (result.status != FaceAttributeProcessor.Status.READY
                || result.attributes.length != result.candidates.length) {
            setImageStatus(R.string.attribute_analysis_failed, R.color.liveness_fail);
            showResultPlaceholder(R.string.attribute_analysis_failed);
        } else {
            renderSelectedFace();
        }
    }

    private void selectFace(int index) {
        if (busy || analysis == null || index < 0
                || index >= analysis.candidates.length) {
            return;
        }
        selectedIndex = index;
        faceOverlay.setSelectedIndex(index);
        renderSelectedFace();
    }

    private void renderSelectedFace() {
        if (analysis == null || selectedIndex < 0
                || selectedIndex >= analysis.attributes.length) {
            showResultPlaceholder(R.string.attribute_analysis_failed);
            return;
        }
        setImageStatus(getString(R.string.attribute_face_selected,
                selectedIndex + 1, analysis.candidates.length), R.color.liveness_accent);
        updateMagnifier(analysis.candidates[selectedIndex]);
        FaceAttributeProcessor.Attribute attribute = analysis.attributes[selectedIndex];
        maskValue.setText(binaryValue(attribute.maskConfidence,
                R.string.attribute_mask_yes, R.string.attribute_mask_no));
        ageValue.setText(age(attribute.ageBracket));
        qualityValue.setText(validConfidence(attribute.qualityScore)
                ? getString(R.string.attribute_score_format, attribute.qualityScore)
                : getString(R.string.attribute_unknown));
        expressionValue.setText(attribute.jawOpen > 0
                ? R.string.attribute_expression_mouth_open
                : attribute.jawOpen == 0
                ? R.string.attribute_expression_neutral
                : R.string.attribute_unknown);
        raceValue.setText(race(attribute.race));
        genderValue.setText(gender(attribute.gender));
        leftEyeValue.setText(binaryValue(attribute.leftEyeConfidence,
                R.string.attribute_eye_open, R.string.attribute_eye_closed));
        rightEyeValue.setText(binaryValue(attribute.rightEyeConfidence,
                R.string.attribute_eye_open, R.string.attribute_eye_closed));
        resultPlaceholder.setVisibility(View.GONE);
        resultValues.setVisibility(View.VISIBLE);
    }

    private void updateMagnifier(FaceImageProcessor.Candidate selected) {
        clearMagnifier();
        Bitmap source = imageBitmap;
        if (source == null || source.isRecycled()) {
            return;
        }
        float imageArea = (float) source.getWidth() * source.getHeight();
        float faceArea = Math.max(0f, selected.rect.width())
                * Math.max(0f, selected.rect.height());
        if (imageArea > 0f && faceArea / imageArea >= DIRECT_FACE_AREA_RATIO) {
            return;
        }
        FaceCropUtils.SquareCrop crop = FaceCropUtils.createSquare(
                source, selected.rect, MAGNIFIER_CROP_SCALE);
        if (crop != null) {
            magnifierBitmap = crop.bitmap;
            magnifierImage.setImageBitmap(crop.bitmap);
            magnifierCard.setVisibility(View.VISIBLE);
        }
    }

    private CharSequence binaryValue(float confidence,
                                     @StringRes int high, @StringRes int low) {
        return validConfidence(confidence)
                ? getString(confidence >= BINARY_CONFIDENCE_THRESHOLD ? high : low)
                : getString(R.string.attribute_unknown);
    }

    private CharSequence age(int bracket) {
        switch (bracket) {
            case 0: return getString(R.string.attribute_age_0_2);
            case 1: return getString(R.string.attribute_age_3_9);
            case 2: return getString(R.string.attribute_age_10_19);
            case 3: return getString(R.string.attribute_age_20_29);
            case 4: return getString(R.string.attribute_age_30_39);
            case 5: return getString(R.string.attribute_age_40_49);
            case 6: return getString(R.string.attribute_age_50_59);
            case 7: return getString(R.string.attribute_age_60_69);
            case 8: return getString(R.string.attribute_age_70_plus);
            default: return getString(R.string.attribute_unknown);
        }
    }

    private CharSequence race(int value) {
        switch (value) {
            case 0: return getString(R.string.attribute_race_black);
            case 1: return getString(R.string.attribute_race_asian);
            case 2: return getString(R.string.attribute_race_latino);
            case 3: return getString(R.string.attribute_race_middle_eastern);
            case 4: return getString(R.string.attribute_race_white);
            default: return getString(R.string.attribute_unknown);
        }
    }

    private CharSequence gender(int value) {
        if (value == 0) {
            return getString(R.string.attribute_gender_female);
        }
        if (value == 1) {
            return getString(R.string.attribute_gender_male);
        }
        return getString(R.string.attribute_unknown);
    }

    private void showResultPlaceholder(@StringRes int message) {
        resultValues.setVisibility(View.GONE);
        resultPlaceholder.setText(message);
        resultPlaceholder.setVisibility(View.VISIBLE);
    }

    private void clearMagnifier() {
        magnifierImage.setImageDrawable(null);
        magnifierCard.setVisibility(View.GONE);
        recycle(magnifierBitmap);
        magnifierBitmap = null;
    }

    private void setControlsEnabled(boolean enabled) {
        choosePhotoButton.setEnabled(enabled && !busy);
    }

    private void setImageStatus(@StringRes int text, @ColorRes int colorRes) {
        setImageStatus(getString(text), colorRes);
    }

    private void setImageStatus(CharSequence text, @ColorRes int colorRes) {
        imageStatus.setText(text);
        imageStatus.setTextColor(ContextCompat.getColor(this, colorRes));
    }

    private static boolean validConfidence(float value) {
        return !Float.isNaN(value) && !Float.isInfinite(value)
                && value >= 0f && value <= 1f;
    }

    private static void recycle(@Nullable Bitmap bitmap) {
        if (bitmap != null && !bitmap.isRecycled()) {
            bitmap.recycle();
        }
    }

    private void applyWindowInsets() {
        View root = findViewById(R.id.attributeRoot);
        int horizontal = root.getPaddingLeft();
        int top = root.getPaddingTop();
        int bottom = root.getPaddingBottom();
        ViewCompat.setOnApplyWindowInsetsListener(root, (v, insets) -> {
            Insets bars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(horizontal, top + bars.top, horizontal, bottom + bars.bottom);
            return insets;
        });
    }

    @Override
    protected void onDestroy() {
        destroyed = true;
        imageVersion++;
        clearMagnifier();
        recycle(imageBitmap);
        imageBitmap = null;
        sdkExecutor.execute(() -> {
            if (session != null) {
                FaceEngine.releaseSession(session);
                session = null;
            }
        });
        sdkExecutor.shutdown();
        super.onDestroy();
    }
}
