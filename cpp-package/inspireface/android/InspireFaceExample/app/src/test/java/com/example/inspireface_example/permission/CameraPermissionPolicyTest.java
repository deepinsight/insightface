package com.example.inspireface_example.permission;

import static com.example.inspireface_example.permission.CameraPermissionPolicy.Action.GRANTED;
import static com.example.inspireface_example.permission.CameraPermissionPolicy.Action.OPEN_SETTINGS;
import static com.example.inspireface_example.permission.CameraPermissionPolicy.Action.REQUEST_SYSTEM_PERMISSION;
import static com.example.inspireface_example.permission.CameraPermissionPolicy.Action.SHOW_NOTICE;
import static com.example.inspireface_example.permission.CameraPermissionPolicy.Action.SHOW_RATIONALE;
import static org.junit.Assert.assertEquals;

import org.junit.Test;

public final class CameraPermissionPolicyTest {

    @Test
    public void firstCameraUseAlwaysShowsPrivacyNotice() {
        assertEquals(SHOW_NOTICE,
                CameraPermissionPolicy.nextAction(false, false, false, false));
        assertEquals(SHOW_NOTICE,
                CameraPermissionPolicy.nextAction(false, true, true, false));
    }

    @Test
    public void acceptedNoticeAndGrantedPermissionStartsCamera() {
        assertEquals(GRANTED,
                CameraPermissionPolicy.nextAction(true, true, true, false));
    }

    @Test
    public void firstAndroidPermissionRequestUsesSystemDialog() {
        assertEquals(REQUEST_SYSTEM_PERMISSION,
                CameraPermissionPolicy.nextAction(true, false, false, false));
    }

    @Test
    public void retryableDenialShowsRationale() {
        assertEquals(SHOW_RATIONALE,
                CameraPermissionPolicy.nextAction(true, false, true, true));
    }

    @Test
    public void permanentDenialRequiresApplicationSettings() {
        assertEquals(OPEN_SETTINGS,
                CameraPermissionPolicy.nextAction(true, false, true, false));
    }
}
