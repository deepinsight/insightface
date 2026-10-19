package com.insightface.sdk.inspireface.jni;

import java.io.File;

/** Android embeds JNI in the core; desktop JVMs use the separate JNI adapter. */
final class NativeLibrary {
    private NativeLibrary() {}
    static void load() {
        String path = System.getProperty("inspireface.native.path");
        if (path == null || path.isEmpty()) {
            boolean android = "Android Runtime".equals(System.getProperty("java.runtime.name"))
                    || "Dalvik".equals(System.getProperty("java.vm.name"));
            System.loadLibrary(android ? "InspireFace" : "InspireFaceJNI");
        } else {
            File library = new File(path);
            if (!library.isAbsolute()) {
                throw new IllegalArgumentException("inspireface.native.path must be an absolute library path");
            }
            System.load(library.getPath());
        }
    }
}
