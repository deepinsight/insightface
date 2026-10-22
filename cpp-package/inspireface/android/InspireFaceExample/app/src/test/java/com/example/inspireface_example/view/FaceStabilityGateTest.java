package com.example.inspireface_example.view;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class FaceStabilityGateTest {

    private static final float EPSILON = 0.0001f;

    @Test
    public void stableFaceWaitsOneSecondThenFillsForTwoSeconds() {
        FaceStabilityGate gate = new FaceStabilityGate();

        assertEquals(-1f, update(gate, 7, 0L), EPSILON);
        assertEquals(-1f, update(gate, 7, 999L), EPSILON);
        assertEquals(0f, update(gate, 7, 1_000L), EPSILON);
        assertEquals(0.5f, update(gate, 7, 2_000L), EPSILON);
        assertEquals(1f, update(gate, 7, 3_000L), EPSILON);
    }

    @Test
    public void motionHidesRingAndStartsANewWarmup() {
        FaceStabilityGate gate = new FaceStabilityGate();
        update(gate, 7, 0L);
        assertEquals(0.25f, update(gate, 7, 1_500L), EPSILON);

        assertEquals(-1f, gate.update(7, 40f, 0f, 140f, 100f, 1_600L), EPSILON);
        assertEquals(-1f, gate.update(7, 40f, 0f, 140f, 100f, 2_599L), EPSILON);
        assertEquals(0f, gate.update(7, 40f, 0f, 140f, 100f, 2_600L), EPSILON);
    }

    @Test
    public void switchingFirstFaceRestartsWarmup() {
        FaceStabilityGate gate = new FaceStabilityGate();
        update(gate, 7, 0L);
        assertEquals(0.25f, update(gate, 7, 1_500L), EPSILON);
        assertEquals(-1f, update(gate, 9, 1_600L), EPSILON);
    }

    @Test
    public void configurableGateCanTriggerRecognitionAfterOneSecond() {
        FaceStabilityGate gate = new FaceStabilityGate(1_000L, 1L);

        assertEquals(-1f, update(gate, 7, 0L), EPSILON);
        assertEquals(-1f, update(gate, 7, 999L), EPSILON);
        assertEquals(0f, update(gate, 7, 1_000L), EPSILON);
    }

    private static float update(FaceStabilityGate gate, int id, long timeMs) {
        return gate.update(id, 0f, 0f, 100f, 100f, timeMs);
    }
}
