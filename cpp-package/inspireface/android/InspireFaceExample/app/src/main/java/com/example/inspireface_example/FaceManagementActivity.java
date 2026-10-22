package com.example.inspireface_example;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.LruCache;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;

import com.example.inspireface_example.face.FaceImageProcessor;
import com.example.inspireface_example.face.FaceRecord;
import com.example.inspireface_example.face.FaceRepository;
import com.example.inspireface_example.face.ImageBitmapLoader;
import com.example.inspireface_example.view.FaceCaptureActivity;
import com.example.inspireface_example.view.FaceEngine;
import com.example.inspireface_example.widget.FaceImageOverlayView;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.dialog.MaterialAlertDialogBuilder;
import com.insightface.sdk.inspireface.base.FaceFeature;
import com.insightface.sdk.inspireface.base.Session;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** CRUD demo for a model-isolated persistent FeatureHub and its face crops. */
public class FaceManagementActivity extends AppCompatActivity {

    private static final int MAX_IMAGE_DIMENSION = 2048;

    private final ExecutorService sdkExecutor = Executors.newSingleThreadExecutor();
    private final FaceRecordAdapter adapter = new FaceRecordAdapter();

    private FaceModelPrefs.Model model;
    private FaceRepository repository;
    private Session session;
    private MaterialButton addButton;
    private EditText searchInput;
    private TextView faceCount;
    private View loadingIndicator;
    private View emptyState;
    private volatile boolean destroyed;

    // Editor dialog state
    private AlertDialog editorDialog;
    private FaceRecord editingRecord;
    private ImageView editorImage;
    private FaceImageOverlayView editorOverlay;
    private View editorPlaceholder;
    private TextView editorStatus;
    private EditText editorName;
    private MaterialButton captureFaceButton;
    private MaterialButton chooseImageButton;
    private Button saveButton;
    private Bitmap editorBitmap;
    private FaceImageProcessor.Candidate[] editorCandidates;
    private int editorSelectedIndex = -1;
    private volatile int editorVersion;
    private boolean editorSaving;

    private final ActivityResultLauncher<String> editorImagePicker =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null && editorDialog != null) {
                    loadEditorImage(uri);
                }
            });

    private final ActivityResultLauncher<Intent> editorCamera =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
                Intent data = result.getData();
                String path = data == null ? null
                        : data.getStringExtra(FaceCaptureActivity.EXTRA_CAPTURE_PATH);
                if (result.getResultCode() == RESULT_OK && path != null) {
                    if (editorDialog != null) {
                        loadEditorCapture(path);
                    } else {
                        new File(path).delete();
                    }
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        WindowCompat.setDecorFitsSystemWindows(getWindow(), false);
        WindowCompat.getInsetsController(getWindow(), getWindow().getDecorView())
                .setAppearanceLightNavigationBars(false);
        setContentView(R.layout.activity_face_management);
        applyWindowInsets();

        model = FaceModelPrefs.get(this);
        repository = new FaceRepository(this, model);
        addButton = findViewById(R.id.btnAddFace);
        searchInput = findViewById(R.id.searchInput);
        faceCount = findViewById(R.id.faceCount);
        loadingIndicator = findViewById(R.id.loadingIndicator);
        emptyState = findViewById(R.id.emptyState);
        ListView faceList = findViewById(R.id.faceList);
        faceList.setAdapter(adapter);
        faceList.setEmptyView(emptyState);
        emptyState.setVisibility(View.GONE);

        ((TextView) findViewById(R.id.currentModel)).setText(
                getString(R.string.current_model, model.sdkName()));
        ((TextView) findViewById(R.id.storageScope)).setText(
                getString(R.string.face_storage_scope, model.sdkName()));
        findViewById(R.id.btnBack).setOnClickListener(v -> finish());
        addButton.setOnClickListener(v -> openEditor(null));
        searchInput.addTextChangedListener(new SimpleTextWatcher() {
            @Override
            public void afterTextChanged(Editable editable) {
                refreshRecords();
            }
        });

        sdkExecutor.execute(this::initializeLibrary);
    }

    private void initializeLibrary() {
        boolean ready = FaceEngine.ensureLaunched(this);
        if (ready) {
            session = FaceEngine.createRecognitionSession();
            ready = session != null;
        }
        if (ready) {
            // Keep an active session for the whole lifetime of FeatureHub. FaceEngine then
            // cannot terminate or switch the loaded model while this repository is open.
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
        postUi(() -> {
            loadingIndicator.setVisibility(View.GONE);
            addButton.setEnabled(finalReady);
            if (finalReady) {
                refreshRecords();
            } else {
                emptyState.setVisibility(View.VISIBLE);
                Toast.makeText(this, R.string.face_library_failed, Toast.LENGTH_LONG).show();
            }
        });
    }

    private void refreshRecords() {
        if (repository == null) {
            return;
        }
        String keyword = searchInput == null ? null : searchInput.getText().toString();
        List<FaceRecord> records = repository.query(keyword);
        adapter.setRecords(records);
        faceCount.setText(getString(R.string.face_list_count,
                keyword == null || keyword.trim().isEmpty()
                        ? records.size() : repository.query(null).size()));
    }

    private void openEditor(@Nullable FaceRecord record) {
        if (session == null || editorDialog != null) {
            return;
        }
        editingRecord = record;
        editorCandidates = null;
        editorSelectedIndex = -1;
        editorSaving = false;
        editorVersion++;

        View content = getLayoutInflater().inflate(R.layout.dialog_face_editor, null, false);
        editorImage = content.findViewById(R.id.editorImage);
        editorOverlay = content.findViewById(R.id.editorFaceOverlay);
        editorPlaceholder = content.findViewById(R.id.editorPlaceholder);
        editorStatus = content.findViewById(R.id.editorImageStatus);
        editorName = content.findViewById(R.id.editorName);
        captureFaceButton = content.findViewById(R.id.btnCaptureEditorFace);
        chooseImageButton = content.findViewById(R.id.btnChooseEditorImage);
        editorOverlay.setOnFaceSelectedListener(this::selectEditorFace);

        if (record != null) {
            editorName.setText(record.name);
            editorBitmap = BitmapFactory.decodeFile(record.cropPath);
            if (editorBitmap != null) {
                editorImage.setImageBitmap(editorBitmap);
                editorPlaceholder.setVisibility(View.GONE);
                setEditorStatus(getString(R.string.stored_face_crop), true);
            }
        } else {
            editorBitmap = null;
        }

        editorDialog = new MaterialAlertDialogBuilder(this)
                .setTitle(record == null ? R.string.add_face_title : R.string.edit_face_title)
                .setView(content)
                .setNegativeButton(R.string.cancel, null)
                .setPositiveButton(R.string.save, null)
                .create();
        editorDialog.setOnShowListener(ignored -> {
            saveButton = editorDialog.getButton(AlertDialog.BUTTON_POSITIVE);
            saveButton.setEnabled(record != null);
            saveButton.setOnClickListener(v -> saveEditor());
            captureFaceButton.setOnClickListener(v -> editorCamera.launch(
                    new Intent(this, FaceCaptureActivity.class)));
            chooseImageButton.setOnClickListener(v -> editorImagePicker.launch("image/*"));
        });
        editorDialog.setOnDismissListener(ignored -> clearEditorState());
        editorDialog.show();
    }

    private void loadEditorImage(Uri uri) {
        loadEditorBitmap(() -> ImageBitmapLoader.decode(
                this, uri, MAX_IMAGE_DIMENSION), null);
    }

    private void loadEditorCapture(String path) {
        File temporaryCapture = new File(path);
        loadEditorBitmap(() -> {
            Bitmap bitmap = BitmapFactory.decodeFile(temporaryCapture.getAbsolutePath());
            if (bitmap == null) {
                throw new IOException("Unable to decode camera capture");
            }
            return bitmap;
        }, temporaryCapture);
    }

    private void loadEditorBitmap(EditorBitmapSource source,
                                  @Nullable File deleteAfterDecode) {
        int requestVersion = ++editorVersion;
        editorSelectedIndex = -1;
        editorOverlay.clearFace();
        saveButton.setEnabled(false);
        setEditorStatus(getString(R.string.image_analyzing), false);
        sdkExecutor.execute(() -> {
            Bitmap bitmap = null;
            FaceImageProcessor.Result result = null;
            try {
                bitmap = source.decode();
                if (editorVersion != requestVersion || destroyed) {
                    bitmap.recycle();
                    return;
                }
                result = FaceImageProcessor.detect(session, bitmap, true);
            } catch (Exception ignored) {
                // The UI maps a null result to a load failure below.
            } finally {
                if (deleteAfterDecode != null) {
                    deleteAfterDecode.delete();
                }
            }
            Bitmap deliveredBitmap = bitmap;
            FaceImageProcessor.Result deliveredResult = result;
            // Always deliver detector ownership to the UI callback. applyEditorImage also
            // handles a dismissed/destroyed editor and recycles an obsolete result.
            runOnUiThread(() -> applyEditorImage(
                    requestVersion, deliveredBitmap, deliveredResult));
        });
    }

    private void applyEditorImage(int requestVersion, @Nullable Bitmap bitmap,
                                  @Nullable FaceImageProcessor.Result result) {
        if (editorDialog == null || editorVersion != requestVersion) {
            recycleEditorResult(bitmap, result);
            return;
        }
        Bitmap previousBitmap = editorBitmap;
        FaceImageProcessor.Candidate[] previousCandidates = editorCandidates;
        editorBitmap = bitmap;
        editorCandidates = result == null ? null : result.candidates;
        editorSelectedIndex = editorCandidates != null && editorCandidates.length > 0 ? 0 : -1;
        if (bitmap != null) {
            editorImage.setImageBitmap(bitmap);
            editorPlaceholder.setVisibility(View.GONE);
            editorOverlay.showFaces(bitmap.getWidth(), bitmap.getHeight(),
                    result == null ? null : result.faceRects(), editorSelectedIndex);
        } else {
            editorImage.setImageDrawable(null);
            editorPlaceholder.setVisibility(View.VISIBLE);
            editorOverlay.clearFace();
        }
        recycleEditorAssets(previousBitmap, previousCandidates);
        if (result == null) {
            setEditorStatus(getString(R.string.image_load_failed), false);
        } else if (result.status == FaceImageProcessor.Status.NO_FACE) {
            setEditorStatus(getString(R.string.image_no_face), false);
        } else if (result.status != FaceImageProcessor.Status.READY) {
            setEditorStatus(getString(R.string.face_extract_failed), false);
        } else {
            updateEditorSelectionState();
        }
    }

    private void selectEditorFace(int index) {
        if (editorCandidates == null || index < 0 || index >= editorCandidates.length) {
            return;
        }
        editorSelectedIndex = index;
        editorOverlay.setSelectedIndex(index);
        updateEditorSelectionState();
    }

    private void updateEditorSelectionState() {
        FaceImageProcessor.Candidate candidate = selectedEditorCandidate();
        boolean valid = candidate != null && candidate.feature != null && candidate.crop != null;
        if (editorCandidates != null && editorCandidates.length > 1) {
            setEditorStatus(getString(R.string.image_face_selected,
                    editorSelectedIndex + 1, editorCandidates.length), valid);
        } else {
            setEditorStatus(getString(valid ? R.string.image_face_ready
                    : R.string.face_extract_failed), valid);
        }
        saveButton.setEnabled(valid);
    }

    @Nullable
    private FaceImageProcessor.Candidate selectedEditorCandidate() {
        return editorCandidates != null && editorSelectedIndex >= 0
                && editorSelectedIndex < editorCandidates.length
                ? editorCandidates[editorSelectedIndex] : null;
    }

    private void saveEditor() {
        FaceImageProcessor.Candidate selected = selectedEditorCandidate();
        if (editingRecord == null && selected == null) {
            return;
        }
        String name = editorName.getText() == null ? "" : editorName.getText().toString();
        FaceFeature feature = selected == null ? null : selected.feature;
        Bitmap crop = selected == null ? null : selected.crop;
        long editingId = editingRecord == null ? -1L : editingRecord.id;
        editorSaving = true;
        editorDialog.setCancelable(false);
        saveButton.setEnabled(false);
        editorDialog.getButton(AlertDialog.BUTTON_NEGATIVE).setEnabled(false);
        captureFaceButton.setEnabled(false);
        chooseImageButton.setEnabled(false);
        setEditorStatus(getString(R.string.saving_face), true);

        sdkExecutor.execute(() -> {
            boolean success;
            if (editingId < 0) {
                success = repository.insert(name, feature, crop).success;
            } else {
                success = repository.update(editingId, name, feature, crop);
            }
            postUi(() -> {
                editorSaving = false;
                if (success) {
                    Toast.makeText(this, R.string.face_saved, Toast.LENGTH_SHORT).show();
                    adapter.clearCropCache();
                    editorDialog.dismiss();
                    refreshRecords();
                } else {
                    editorDialog.setCancelable(true);
                    editorDialog.getButton(AlertDialog.BUTTON_NEGATIVE).setEnabled(true);
                    captureFaceButton.setEnabled(true);
                    chooseImageButton.setEnabled(true);
                    saveButton.setEnabled(editingRecord != null || selectedEditorCandidate() != null);
                    setEditorStatus(getString(R.string.face_save_failed), false);
                }
            });
        });
    }

    private void confirmDelete(FaceRecord record) {
        new MaterialAlertDialogBuilder(this)
                .setTitle(R.string.delete_face_title)
                .setMessage(getString(R.string.delete_face_message, record.name, model.sdkName()))
                .setNegativeButton(R.string.cancel, null)
                .setPositiveButton(R.string.delete, (dialog, which) -> deleteRecord(record))
                .show();
    }

    private void deleteRecord(FaceRecord record) {
        addButton.setEnabled(false);
        sdkExecutor.execute(() -> {
            boolean deleted = repository.delete(record.id);
            postUi(() -> {
                addButton.setEnabled(true);
                Toast.makeText(this, deleted ? R.string.face_deleted
                        : R.string.face_delete_failed, Toast.LENGTH_SHORT).show();
                if (deleted) {
                    adapter.removeCrop(record.cropPath);
                    refreshRecords();
                }
            });
        });
    }

    private void setEditorStatus(CharSequence message, boolean positive) {
        editorStatus.setText(message);
        editorStatus.setTextColor(ContextCompat.getColor(this,
                positive ? R.color.liveness_accent : R.color.home_text_secondary));
    }

    private void clearEditorState() {
        editorVersion++;
        if (!editorSaving) {
            recycleEditorAssets(editorBitmap, editorCandidates);
        }
        editorDialog = null;
        editingRecord = null;
        editorImage = null;
        editorOverlay = null;
        editorPlaceholder = null;
        editorStatus = null;
        editorName = null;
        captureFaceButton = null;
        chooseImageButton = null;
        saveButton = null;
        editorBitmap = null;
        editorCandidates = null;
        editorSelectedIndex = -1;
    }

    private static void recycleEditorResult(@Nullable Bitmap bitmap,
                                            @Nullable FaceImageProcessor.Result result) {
        recycleEditorAssets(bitmap, result == null ? null : result.candidates);
    }

    private static void recycleEditorAssets(@Nullable Bitmap bitmap,
                                            @Nullable FaceImageProcessor.Candidate[] candidates) {
        FaceImageProcessor.recycleCrops(candidates, -1);
        if (bitmap != null && !bitmap.isRecycled()) {
            bitmap.recycle();
        }
    }

    private void applyWindowInsets() {
        View root = findViewById(R.id.managementRoot);
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
        editorVersion++;
        Bitmap deferredEditorBitmap = editorBitmap;
        FaceImageProcessor.Candidate[] deferredEditorCandidates = editorCandidates;
        if (editorDialog != null && !editorSaving) {
            editorDialog.dismiss();
        }
        // The save task may still own the selected crop for JPEG compression. Queue a
        // second, idempotent cleanup behind all SDK tasks; this also covers dialog teardown.
        Bitmap bitmapToRecycle = deferredEditorBitmap;
        FaceImageProcessor.Candidate[] candidatesToRecycle = deferredEditorCandidates;
        sdkExecutor.execute(() -> {
            recycleEditorAssets(bitmapToRecycle, candidatesToRecycle);
            repository.close();
            if (session != null) {
                FaceEngine.releaseSession(session);
                session = null;
            }
        });
        sdkExecutor.shutdown();
        super.onDestroy();
    }

    private final class FaceRecordAdapter extends BaseAdapter {
        private final List<FaceRecord> records = new ArrayList<>();
        private final LruCache<String, Bitmap> cropCache = new LruCache<String, Bitmap>(8) {
            @Override
            protected int sizeOf(String key, Bitmap value) {
                return 1;
            }
        };

        void setRecords(List<FaceRecord> newRecords) {
            records.clear();
            records.addAll(newRecords);
            notifyDataSetChanged();
        }

        void removeCrop(String path) {
            cropCache.remove(path);
        }

        void clearCropCache() {
            cropCache.evictAll();
        }

        @Override
        public int getCount() {
            return records.size();
        }

        @Override
        public FaceRecord getItem(int position) {
            return records.get(position);
        }

        @Override
        public long getItemId(int position) {
            return getItem(position).id;
        }

        @Override
        public boolean hasStableIds() {
            return true;
        }

        @Override
        public View getView(int position, View convertView, ViewGroup parent) {
            ViewHolder holder;
            if (convertView == null) {
                convertView = LayoutInflater.from(parent.getContext())
                        .inflate(R.layout.item_face_record, parent, false);
                holder = new ViewHolder(convertView);
                convertView.setTag(holder);
            } else {
                holder = (ViewHolder) convertView.getTag();
            }
            FaceRecord record = getItem(position);
            holder.name.setText(record.name);
            holder.id.setText(getString(R.string.face_id_format, record.id));
            Bitmap crop = cropCache.get(record.cropPath);
            if (crop == null || crop.isRecycled()) {
                crop = BitmapFactory.decodeFile(record.cropPath);
                if (crop != null) {
                    cropCache.put(record.cropPath, crop);
                }
            }
            holder.crop.setImageBitmap(crop);
            holder.edit.setOnClickListener(v -> openEditor(record));
            holder.delete.setOnClickListener(v -> confirmDelete(record));
            return convertView;
        }
    }

    private static final class ViewHolder {
        final ImageView crop;
        final TextView name;
        final TextView id;
        final View edit;
        final View delete;

        ViewHolder(View root) {
            crop = root.findViewById(R.id.faceCrop);
            name = root.findViewById(R.id.faceName);
            id = root.findViewById(R.id.faceId);
            edit = root.findViewById(R.id.btnEditFace);
            delete = root.findViewById(R.id.btnDeleteFace);
        }
    }

    private abstract static class SimpleTextWatcher implements TextWatcher {
        @Override
        public void beforeTextChanged(CharSequence s, int start, int count, int after) {
        }

        @Override
        public void onTextChanged(CharSequence s, int start, int before, int count) {
        }
    }

    private interface EditorBitmapSource {
        Bitmap decode() throws Exception;
    }
}
