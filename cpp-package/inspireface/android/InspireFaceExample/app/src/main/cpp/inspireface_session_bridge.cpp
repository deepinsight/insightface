#include <android/log.h>
#include <dlfcn.h>
#include <jni.h>

namespace {

constexpr const char* kLogTag = "InspireFaceBridge";
using CreateSessionOptional = long (*)(int custom_option,
                                       int detect_mode,
                                       int max_detect_face_num,
                                       int detect_pixel_level,
                                       int track_by_detect_fps,
                                       void** session);

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_inspireface_1example_view_NativeSessionBridge_nativeCreateLandmarkSession(
        JNIEnv*, jclass, jint custom_options, jint detect_mode, jint max_faces,
        jint detect_pixel_level, jint track_by_detect_fps) {
    void* library = dlopen("libInspireFace.so", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, kLogTag,
                            "Could not load libInspireFace.so: %s", dlerror());
        return 0;
    }

    dlerror();
    auto create_session = reinterpret_cast<CreateSessionOptional>(
            dlsym(library, "HFCreateInspireFaceSessionOptional"));
    const char* symbol_error = dlerror();
    if (symbol_error != nullptr || create_session == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, kLogTag,
                            "Optional Session API unavailable: %s",
                            symbol_error == nullptr ? "unknown error" : symbol_error);
        dlclose(library);
        return 0;
    }

    void* session = nullptr;
    long result = create_session(custom_options, detect_mode, max_faces,
                                 detect_pixel_level, track_by_detect_fps, &session);
    dlclose(library);
    if (result != 0 || session == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, kLogTag,
                            "Could not create landmark Session, error=%ld", result);
        return 0;
    }
    return reinterpret_cast<jlong>(session);
}
