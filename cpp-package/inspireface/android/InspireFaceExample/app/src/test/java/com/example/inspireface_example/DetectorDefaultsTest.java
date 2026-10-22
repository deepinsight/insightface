package com.example.inspireface_example;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public final class DetectorDefaultsTest {

    @Test
    public void allSessionDefaultsUse320InputPixels() {
        assertEquals(320, DetectorDefaults.INPUT_PX);
        assertEquals(DetectorDefaults.INPUT_PX,
                StillImageSessionSettings.defaults().inputPx);
        assertEquals(DetectorDefaults.INPUT_PX,
                FaceTrackingSessionSettings.defaults().inputPx);
    }
}
