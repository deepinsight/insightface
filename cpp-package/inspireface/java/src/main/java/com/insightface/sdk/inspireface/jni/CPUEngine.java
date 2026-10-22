package com.insightface.sdk.inspireface.jni;

/** Process-wide CPU inference configuration, shared by the JVM and Android adapters. */
public final class CPUEngine {
    public enum PowerMode {
        NORMAL(0), HIGH(1), LOW(2);

        private final int value;
        PowerMode(int value) { this.value = value; }
    }

    static { NativeLibrary.load(); }
    private CPUEngine() {}

    /**
     * Selects the CPU power/scheduling policy; the default is NORMAL.
     * Set this before creating sessions. Only subsequently initialized runtimes
     * read this value; existing runtimes, model thread counts and precision are unchanged.
     * Do not change it concurrently with session/model initialization.
     * Launch, reload and terminate preserve the setting.
     *
     * @param mode the desired policy (not null)
     * @throws IllegalArgumentException if mode is null
     * @throws InspireFaceException if the native configuration fails
     */
    public static void setGlobalPowerMode(PowerMode mode) {
        if (mode == null) throw new IllegalArgumentException("CPU power mode must not be null");
        InspireFaceException.check(nativeSetGlobalPowerMode(mode.value));
    }

    /** Returns the policy used when the next CPU runtime is initialized. */
    public static PowerMode getGlobalPowerMode() {
        int value = nativeGetGlobalPowerMode();
        switch (value) {
            case 0: return PowerMode.NORMAL;
            case 1: return PowerMode.HIGH;
            case 2: return PowerMode.LOW;
        }
        throw new IllegalStateException("Unknown native CPU power mode: " + value);
    }

    private static native int nativeSetGlobalPowerMode(int mode);
    private static native int nativeGetGlobalPowerMode();
}
