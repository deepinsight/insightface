package com.example.inspireface_example.permission;

/** Pure decision table for the camera permission flow; kept Android-free for unit tests. */
public final class CameraPermissionPolicy {

    public enum Action {
        SHOW_NOTICE,
        GRANTED,
        REQUEST_SYSTEM_PERMISSION,
        SHOW_RATIONALE,
        OPEN_SETTINGS
    }

    private CameraPermissionPolicy() {
    }

    public static Action nextAction(boolean noticeAccepted,
                                    boolean permissionGranted,
                                    boolean requestedBefore,
                                    boolean shouldShowRationale) {
        if (!noticeAccepted) {
            return Action.SHOW_NOTICE;
        }
        if (permissionGranted) {
            return Action.GRANTED;
        }
        if (!requestedBefore) {
            return Action.REQUEST_SYSTEM_PERMISSION;
        }
        return shouldShowRationale ? Action.SHOW_RATIONALE : Action.OPEN_SETTINGS;
    }
}
