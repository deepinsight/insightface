package com.example.inspireface_example.face;

import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Bitmap;

import androidx.annotation.Nullable;

import com.example.inspireface_example.FaceModelPrefs;
import com.insightface.sdk.inspireface.InspireFace;
import com.insightface.sdk.inspireface.base.FaceFeature;
import com.insightface.sdk.inspireface.base.FaceFeatureIdentity;
import com.insightface.sdk.inspireface.base.FeatureHubConfiguration;
import com.insightface.sdk.inspireface.base.SearchTopKResults;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Model-scoped FeatureHub storage. Each model gets its own native DB, crop directory,
 * metadata preferences and ID sequence; no feature or image is shared across models.
 */
public final class FaceRepository {

    public static final class SearchResult {
        public final boolean matched;
        public final FaceRecord record;
        public final float confidence;
        public final float threshold;

        SearchResult(boolean matched, @Nullable FaceRecord record,
                     float confidence, float threshold) {
            this.matched = matched;
            this.record = record;
            this.confidence = confidence;
            this.threshold = threshold;
        }
    }

    public static final class InsertResult {
        public final boolean success;
        public final FaceRecord record;

        InsertResult(boolean success, @Nullable FaceRecord record) {
            this.success = success;
            this.record = record;
        }
    }

    private static final String KEY_PREFIX = "record.";
    private static final String KEY_NEXT_ID = "next_id";
    private static final Object HUB_LOCK = new Object();
    private static String activeDatabasePath;
    private static int hubReferences;

    private final SharedPreferences metadata;
    private final File modelDirectory;
    private final File cropDirectory;
    private final File databaseFile;
    private boolean hubAcquired;

    public FaceRepository(Context context, FaceModelPrefs.Model model) {
        Context app = context.getApplicationContext();
        modelDirectory = new File(new File(app.getFilesDir(), "face_hub"), model.sdkName());
        cropDirectory = new File(modelDirectory, "crops");
        databaseFile = new File(modelDirectory, "features.db");
        metadata = app.getSharedPreferences(
                "face_records_" + model.sdkName(), Context.MODE_PRIVATE);
    }

    /** Must be called after GlobalLaunch and from the repository's SDK executor. */
    public boolean open() {
        synchronized (HUB_LOCK) {
            if (hubAcquired) {
                return true;
            }
            String requestedPath = databaseFile.getAbsolutePath();
            if (hubReferences > 0) {
                if (!requestedPath.equals(activeDatabasePath)) {
                    return false;
                }
                hubReferences++;
                hubAcquired = true;
                return true;
            }
            if (!cropDirectory.exists() && !cropDirectory.mkdirs()) {
                return false;
            }
            FeatureHubConfiguration configuration = InspireFace.CreateFeatureHubConfiguration()
                    .setPrimaryKeyMode(InspireFace.PK_MANUAL_INPUT)
                    .setEnablePersistence(true)
                    .setPersistenceDbPath(requestedPath)
                    .setSearchThreshold(InspireFace.GetRecommendedCosineThreshold())
                    .setSearchMode(InspireFace.SEARCH_MODE_EXHAUSTIVE);
            if (!InspireFace.FeatureHubDataEnable(configuration)) {
                return false;
            }
            activeDatabasePath = requestedPath;
            hubReferences = 1;
            hubAcquired = true;
            return true;
        }
    }

    public void close() {
        synchronized (HUB_LOCK) {
            if (!hubAcquired) {
                return;
            }
            hubAcquired = false;
            hubReferences = Math.max(0, hubReferences - 1);
            if (hubReferences == 0) {
                InspireFace.FeatureHubDataDisable();
                activeDatabasePath = null;
            }
        }
    }

    public List<FaceRecord> query(@Nullable String keyword) {
        String normalized = keyword == null ? ""
                : keyword.trim().toLowerCase(Locale.ROOT);
        List<FaceRecord> records = new ArrayList<>();
        for (Map.Entry<String, ?> entry : metadata.getAll().entrySet()) {
            if (!entry.getKey().startsWith(KEY_PREFIX) || !(entry.getValue() instanceof String)) {
                continue;
            }
            FaceRecord record = decode((String) entry.getValue());
            if (record != null && (normalized.isEmpty()
                    || record.name.toLowerCase(Locale.ROOT).contains(normalized)
                    || String.valueOf(record.id).contains(normalized))) {
                records.add(record);
            }
        }
        Collections.sort(records,
                (left, right) -> Long.compare(right.updatedAt, left.updatedAt));
        return records;
    }

    /** Searches the active model's native FeatureHub and joins the best ID to metadata. */
    public SearchResult search(FaceFeature feature) {
        float threshold = InspireFace.GetRecommendedCosineThreshold();
        if (!hubAcquired || feature == null) {
            return new SearchResult(false, null, Float.NaN, threshold);
        }
        SearchTopKResults results = InspireFace.FeatureHubFaceSearchTopK(feature, 1);
        if (results == null || results.num <= 0 || results.ids == null
                || results.confidence == null || results.ids.length == 0
                || results.confidence.length == 0) {
            return new SearchResult(false, null, Float.NaN, threshold);
        }
        float confidence = results.confidence[0];
        FaceRecord record = get(results.ids[0]);
        return new SearchResult(record != null && confidence >= threshold,
                record, confidence, threshold);
    }

    @Nullable
    public FaceRecord get(long id) {
        return decode(metadata.getString(key(id), null));
    }

    public InsertResult insert(String name, FaceFeature feature, Bitmap crop) {
        if (!hubAcquired || feature == null || crop == null) {
            return new InsertResult(false, null);
        }
        long id = Math.max(1L, metadata.getLong(KEY_NEXT_ID, 1L));
        while (metadata.contains(key(id))) {
            id++;
        }
        File cropFile = cropFile(id);
        File stagedCrop = stageCrop(cropFile, crop);
        if (stagedCrop == null) {
            return new InsertResult(false, null);
        }
        File cropBackup = backupFile(cropFile);
        boolean hadPreviousCrop = cropFile.exists();
        FaceFeatureIdentity identity = FaceFeatureIdentity.create(id, feature);
        if (!InspireFace.FeatureHubInsertFeature(identity)) {
            stagedCrop.delete();
            return new InsertResult(false, null);
        }
        if (!commitStagedCrop(cropFile, stagedCrop, cropBackup)) {
            InspireFace.FeatureHubFaceRemove(id);
            return new InsertResult(false, null);
        }
        FaceRecord record = new FaceRecord(
                id, normalizedName(name, id), cropFile.getAbsolutePath(),
                System.currentTimeMillis());
        boolean saved = metadata.edit()
                .putString(key(id), encode(record))
                .putLong(KEY_NEXT_ID, id + 1)
                .commit();
        if (!saved) {
            InspireFace.FeatureHubFaceRemove(id);
            restoreCrop(cropFile, cropBackup, hadPreviousCrop);
            return new InsertResult(false, null);
        }
        cropBackup.delete();
        return new InsertResult(true, record);
    }

    /** Passing null feature/crop performs a metadata-only rename. */
    public boolean update(long id, String name,
                          @Nullable FaceFeature feature, @Nullable Bitmap crop) {
        FaceRecord old = get(id);
        if (!hubAcquired || old == null) {
            return false;
        }
        File cropFile = new File(old.cropPath);
        File stagedCrop = crop == null ? null : stageCrop(cropFile, crop);
        if (crop != null && stagedCrop == null) {
            return false;
        }
        File cropBackup = backupFile(cropFile);
        boolean hadPreviousCrop = cropFile.exists();

        FaceFeatureIdentity previousIdentity = null;
        if (feature != null) {
            previousIdentity = InspireFace.FeatureHubGetFaceIdentity(id);
            if (previousIdentity == null || !InspireFace.FeatureHubFaceUpdate(
                    FaceFeatureIdentity.create(id, feature))) {
                if (stagedCrop != null) {
                    stagedCrop.delete();
                }
                return false;
            }
        }
        if (stagedCrop != null
                && !commitStagedCrop(cropFile, stagedCrop, cropBackup)) {
            rollbackFeature(previousIdentity);
            return false;
        }
        FaceRecord updated = new FaceRecord(id, normalizedName(name, id),
                cropFile.getAbsolutePath(), System.currentTimeMillis());
        boolean saved = metadata.edit().putString(key(id), encode(updated)).commit();
        if (!saved) {
            if (stagedCrop != null) {
                restoreCrop(cropFile, cropBackup, hadPreviousCrop);
            }
            rollbackFeature(previousIdentity);
            return false;
        }
        cropBackup.delete();
        return true;
    }

    public boolean delete(long id) {
        FaceRecord record = get(id);
        if (!hubAcquired || record == null) {
            return false;
        }
        File crop = new File(record.cropPath);
        File pendingDelete = new File(crop.getParentFile(), crop.getName() + ".delete");
        if (!recoverPendingDelete(crop, pendingDelete)) {
            return false;
        }
        boolean hadCrop = crop.exists();
        if (hadCrop && !crop.renameTo(pendingDelete)) {
            return false;
        }

        FaceFeatureIdentity previousIdentity = InspireFace.FeatureHubGetFaceIdentity(id);
        if (previousIdentity != null && !InspireFace.FeatureHubFaceRemove(id)
                && InspireFace.FeatureHubGetFaceIdentity(id) != null) {
            restorePendingDelete(crop, pendingDelete, hadCrop);
            return false;
        }
        boolean metadataRemoved = metadata.edit().remove(key(id)).commit();
        if (!metadataRemoved) {
            if (previousIdentity != null
                    && InspireFace.FeatureHubGetFaceIdentity(id) == null) {
                InspireFace.FeatureHubInsertFeature(previousIdentity);
            }
            restorePendingDelete(crop, pendingDelete, hadCrop);
            return false;
        }
        // The identity is already logically deleted. A leftover tombstone is harmless and
        // will be cleaned the next time this path is touched.
        pendingDelete.delete();
        return true;
    }

    public File databaseFile() {
        return databaseFile;
    }

    public File cropDirectory() {
        return cropDirectory;
    }

    private File cropFile(long id) {
        return new File(cropDirectory, id + ".jpg");
    }

    @Nullable
    private static File stageCrop(File destination, Bitmap crop) {
        File parent = destination.getParentFile();
        if (parent == null || (!parent.exists() && !parent.mkdirs())) {
            return null;
        }
        File backup = backupFile(destination);
        if (!recoverCropReplacement(destination, backup)) {
            return null;
        }
        File temp = new File(parent, destination.getName() + ".tmp");
        if (temp.exists() && !temp.delete()) {
            return null;
        }
        try (FileOutputStream output = new FileOutputStream(temp)) {
            if (!crop.compress(Bitmap.CompressFormat.JPEG, 92, output)) {
                temp.delete();
                return null;
            }
            output.flush();
            output.getFD().sync();
        } catch (IOException e) {
            temp.delete();
            return null;
        }
        return temp;
    }

    private static boolean commitStagedCrop(File destination, File staged, File backup) {
        if (backup.exists() && !backup.delete()) {
            return false;
        }
        boolean hadDestination = destination.exists();
        if (hadDestination && !destination.renameTo(backup)) {
            staged.delete();
            return false;
        }
        if (!staged.renameTo(destination)) {
            if (hadDestination) {
                backup.renameTo(destination);
            }
            staged.delete();
            return false;
        }
        return true;
    }

    private static void restoreCrop(File destination, File backup, boolean hadPreviousCrop) {
        if (destination.exists()) {
            destination.delete();
        }
        if (hadPreviousCrop && backup.exists()) {
            backup.renameTo(destination);
        } else {
            backup.delete();
        }
    }

    private static boolean recoverCropReplacement(File destination, File backup) {
        if (!backup.exists()) {
            return true;
        }
        if (destination.exists()) {
            return backup.delete();
        }
        return backup.renameTo(destination);
    }

    private static boolean recoverPendingDelete(File crop, File pendingDelete) {
        if (!pendingDelete.exists()) {
            return true;
        }
        if (crop.exists()) {
            return pendingDelete.delete();
        }
        return pendingDelete.renameTo(crop);
    }

    private static void restorePendingDelete(File crop, File pendingDelete, boolean hadCrop) {
        if (hadCrop && pendingDelete.exists()) {
            pendingDelete.renameTo(crop);
        } else if (!hadCrop) {
            pendingDelete.delete();
        }
    }

    private static void rollbackFeature(@Nullable FaceFeatureIdentity previous) {
        if (previous != null) {
            InspireFace.FeatureHubFaceUpdate(previous);
        }
    }

    private static File backupFile(File destination) {
        return new File(destination.getParentFile(), destination.getName() + ".bak");
    }

    private static String normalizedName(String name, long id) {
        String trimmed = name == null ? "" : name.trim();
        return trimmed.isEmpty() ? "Face " + id : trimmed;
    }

    private static String key(long id) {
        return KEY_PREFIX + id;
    }

    private static String encode(FaceRecord record) {
        try {
            return new JSONObject()
                    .put("id", record.id)
                    .put("name", record.name)
                    .put("crop", record.cropPath)
                    .put("updated", record.updatedAt)
                    .toString();
        } catch (JSONException impossible) {
            throw new IllegalStateException(impossible);
        }
    }

    @Nullable
    private static FaceRecord decode(@Nullable String value) {
        if (value == null) {
            return null;
        }
        try {
            JSONObject json = new JSONObject(value);
            return new FaceRecord(json.getLong("id"), json.getString("name"),
                    json.getString("crop"), json.getLong("updated"));
        } catch (JSONException e) {
            return null;
        }
    }
}
