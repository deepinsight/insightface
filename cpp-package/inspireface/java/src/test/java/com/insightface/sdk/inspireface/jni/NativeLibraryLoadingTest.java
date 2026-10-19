package com.insightface.sdk.inspireface.jni;

import java.io.File;
import java.net.URL;
import java.net.URLClassLoader;

/** Checks the Android library name on a host JVM, then exercises real JNI through that lookup. */
public final class NativeLibraryLoadingTest {
    private static final class Loader extends URLClassLoader {
        private final String library;
        private int lookups;
        Loader(File jar, File library) throws Exception {
            super(new URL[] {jar.toURI().toURL()}, null);
            this.library = library.getAbsolutePath();
        }
        @Override protected String findLibrary(String name) {
            if (!"InspireFace".equals(name)) throw new AssertionError("Android requested " + name);
            lookups++;
            // Resolve the host JNI binary after checking the Android-specific request.
            return library;
        }
    }

    public static void main(String[] args) throws Exception {
        // Each CTest invocation uses a separate JVM; these properties simulate the
        // Android loader branch only, not the Android runtime or instruction set.
        System.clearProperty("inspireface.native.path");
        System.setProperty("java.runtime.name", "runtime".equals(args[0]) ? "Android Runtime" : "Other Runtime");
        System.setProperty("java.vm.name", "vm".equals(args[0]) ? "Dalvik" : "Other VM");
        try (Loader loader = new Loader(new File(args[1]), new File(args[2]))) {
            String prefix = "com.insightface.sdk.inspireface.jni.";
            Class<?> cpu = Class.forName(prefix + "CPUEngine", true, loader);
            if (!"NORMAL".equals(cpu.getMethod("getGlobalPowerMode").invoke(null).toString())) {
                throw new AssertionError("Unexpected default CPU policy");
            }
            Class<?> nativeApi = Class.forName(prefix + "Native", true, loader);
            int[] level = new int[1];
            Object status = nativeApi.getMethod("HFQueryCAPILevel", int[].class).invoke(null, (Object) level);
            if (((Integer) status) != 0 || level[0] <= 0 || loader.lookups == 0) {
                throw new AssertionError("Android library lookup did not resolve CPU and C API JNI");
            }
            System.out.println("Android " + args[0] + " lookup resolved InspireFace and real JNI entry points");
        }
    }
}
