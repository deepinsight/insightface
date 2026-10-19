#include "support.h"
#include "launch.h"

using namespace isf_jni;
using inspire::Launch;

// Keep Java values independent of the inference backend's enum representation.
extern "C" JNIEXPORT jint JNICALL Java_com_insightface_sdk_inspireface_jni_CPUEngine_nativeSetGlobalPowerMode(
    JNIEnv* env, jclass, jint mode) {
    try {
        Launch::CPUEnginePowerMode policy;
        switch (mode) {
            case 0: policy = Launch::CPU_ENGINE_POWER_NORMAL; break;
            case 1: policy = Launch::CPU_ENGINE_POWER_HIGH; break;
            case 2: policy = Launch::CPU_ENGINE_POWER_LOW; break;
            default: return HERR_INVALID_PARAM;
        }
        return Launch::GetInstance()->SetGlobalCPUEnginePowerMode(policy);
    } catch (...) { TranslateException(env); }
    return HERR_INVALID_PARAM;
}

extern "C" JNIEXPORT jint JNICALL Java_com_insightface_sdk_inspireface_jni_CPUEngine_nativeGetGlobalPowerMode(
    JNIEnv* env, jclass) {
    try {
        switch (Launch::GetInstance()->GetGlobalCPUEnginePowerMode()) {
            case Launch::CPU_ENGINE_POWER_NORMAL: return 0;
            case Launch::CPU_ENGINE_POWER_HIGH: return 1;
            case Launch::CPU_ENGINE_POWER_LOW: return 2;
        }
        Fail(env, "java/lang/IllegalStateException", "Unknown native CPU power mode");
    } catch (...) { TranslateException(env); }
    return -1;
}
