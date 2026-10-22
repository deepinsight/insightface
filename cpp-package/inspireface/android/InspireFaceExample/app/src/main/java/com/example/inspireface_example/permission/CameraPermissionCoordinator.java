package com.example.inspireface_example.permission;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.provider.Settings;
import android.view.View;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.inspireface_example.R;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.dialog.MaterialAlertDialogBuilder;

/**
 * Shared camera-permission UX for every live feature.
 *
 * <p>It owns the one-time on-device privacy notice, Android permission request,
 * rationale, permanent-denial settings recovery, and the return-from-settings check.</p>
 */
public final class CameraPermissionCoordinator {

    public interface Listener {
        void onCameraPermissionGranted();

        void onCameraPermissionBlocked(boolean requiresSettings);
    }

    public static final String PREFERENCES_NAME = "camera_permission";
    private static final String KEY_NOTICE_ACCEPTED = "notice_accepted";
    private static final String KEY_REQUESTED_BEFORE = "requested_before";

    private final AppCompatActivity activity;
    private final Listener listener;
    private final SharedPreferences preferences;
    private final ActivityResultLauncher<String> permissionLauncher;

    private MaterialButton recoveryButton;
    private AlertDialog activeDialog;
    private boolean permissionRequestInFlight;
    private boolean awaitingSettings;
    private boolean closed;

    public CameraPermissionCoordinator(AppCompatActivity activity, Listener listener) {
        this.activity = activity;
        this.listener = listener;
        preferences = activity.getSharedPreferences(
                PREFERENCES_NAME, Context.MODE_PRIVATE);
        permissionLauncher = activity.registerForActivityResult(
                new ActivityResultContracts.RequestPermission(), this::onPermissionResult);
    }

    /** Binds the persistent Try again / Open settings action shown by each camera page. */
    public void bindRecoveryButton(MaterialButton button) {
        recoveryButton = button;
        recoveryButton.setOnClickListener(v -> performRecoveryAction());
        recoveryButton.setVisibility(View.GONE);
    }

    /** Starts or resumes the permission flow without issuing duplicate requests/dialogs. */
    public void requestAccess() {
        if (closed || activity.isFinishing() || activity.isDestroyed()
                || permissionRequestInFlight || isDialogShowing()) {
            return;
        }
        CameraPermissionPolicy.Action action = CameraPermissionPolicy.nextAction(
                preferences.getBoolean(KEY_NOTICE_ACCEPTED, false),
                hasPermission(),
                preferences.getBoolean(KEY_REQUESTED_BEFORE, false),
                shouldShowRationale());
        switch (action) {
            case SHOW_NOTICE:
                showPrivacyNotice();
                break;
            case GRANTED:
                notifyGranted();
                break;
            case REQUEST_SYSTEM_PERMISSION:
                launchSystemPermission();
                break;
            case SHOW_RATIONALE:
                showRationale();
                break;
            case OPEN_SETTINGS:
                notifyBlocked(true);
                break;
        }
    }

    /** Call from Activity.onResume so a Settings grant takes effect without reopening the page. */
    public void onResume() {
        if (closed || !awaitingSettings) {
            return;
        }
        awaitingSettings = false;
        if (hasPermission()) {
            notifyGranted();
        } else {
            notifyBlocked(true);
        }
    }

    public boolean hasPermission() {
        return ContextCompat.checkSelfPermission(activity, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED;
    }

    public void close() {
        closed = true;
        if (activeDialog != null) {
            activeDialog.dismiss();
            activeDialog = null;
        }
        recoveryButton = null;
    }

    private void showPrivacyNotice() {
        activeDialog = new MaterialAlertDialogBuilder(activity)
                .setTitle(R.string.camera_privacy_notice_title)
                .setMessage(R.string.camera_privacy_notice_message)
                .setNegativeButton(R.string.camera_permission_not_now, (dialog, which) -> {
                    activeDialog = null;
                    notifyBlocked(false);
                })
                .setPositiveButton(R.string.camera_permission_continue, (dialog, which) -> {
                    activeDialog = null;
                    preferences.edit().putBoolean(KEY_NOTICE_ACCEPTED, true).apply();
                    requestAccess();
                })
                .setOnCancelListener(dialog -> {
                    activeDialog = null;
                    notifyBlocked(false);
                })
                .show();
    }

    private void showRationale() {
        activeDialog = new MaterialAlertDialogBuilder(activity)
                .setTitle(R.string.camera_permission_rationale_title)
                .setMessage(R.string.camera_permission_rationale_message)
                .setNegativeButton(android.R.string.cancel, (dialog, which) -> {
                    activeDialog = null;
                    notifyBlocked(false);
                })
                .setPositiveButton(R.string.camera_permission_try_again, (dialog, which) -> {
                    activeDialog = null;
                    launchSystemPermission();
                })
                .setOnCancelListener(dialog -> {
                    activeDialog = null;
                    notifyBlocked(false);
                })
                .show();
    }

    private void launchSystemPermission() {
        preferences.edit().putBoolean(KEY_REQUESTED_BEFORE, true).apply();
        permissionRequestInFlight = true;
        permissionLauncher.launch(Manifest.permission.CAMERA);
    }

    private void onPermissionResult(boolean granted) {
        permissionRequestInFlight = false;
        if (closed) {
            return;
        }
        if (granted) {
            notifyGranted();
        } else {
            notifyBlocked(!shouldShowRationale());
        }
    }

    private void performRecoveryAction() {
        if (closed) {
            return;
        }
        if (hasPermission()) {
            notifyGranted();
            return;
        }
        boolean requestedBefore = preferences.getBoolean(KEY_REQUESTED_BEFORE, false);
        if (requestedBefore && !shouldShowRationale()) {
            openApplicationSettings();
        } else {
            requestAccess();
        }
    }

    private void openApplicationSettings() {
        awaitingSettings = true;
        Intent intent = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                Uri.fromParts("package", activity.getPackageName(), null));
        activity.startActivity(intent);
    }

    private void notifyGranted() {
        if (closed) {
            return;
        }
        if (recoveryButton != null) {
            recoveryButton.setVisibility(View.GONE);
        }
        listener.onCameraPermissionGranted();
    }

    private void notifyBlocked(boolean requiresSettings) {
        if (closed) {
            return;
        }
        if (recoveryButton != null) {
            recoveryButton.setText(requiresSettings
                    ? R.string.camera_permission_open_settings
                    : R.string.camera_permission_try_again);
            recoveryButton.setVisibility(View.VISIBLE);
        }
        listener.onCameraPermissionBlocked(requiresSettings);
    }

    private boolean shouldShowRationale() {
        return ActivityCompat.shouldShowRequestPermissionRationale(
                activity, Manifest.permission.CAMERA);
    }

    private boolean isDialogShowing() {
        return activeDialog != null && activeDialog.isShowing();
    }
}
