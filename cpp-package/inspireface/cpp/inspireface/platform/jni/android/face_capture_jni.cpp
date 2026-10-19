#ifdef ANDROID

#include <jni.h>

#include <cstdint>
#include <vector>

#include "c_api/inspireface.h"

#define INSPIRE_FACE_JNI(sig) Java_com_insightface_sdk_inspireface_##sig

namespace {

jfieldID Field(JNIEnv* env, jobject object, const char* name, const char* signature) {
    if (env == nullptr || object == nullptr) return nullptr;
    jclass type = env->GetObjectClass(object);
    if (type == nullptr) return nullptr;
    jfieldID field = env->GetFieldID(type, name, signature);
    env->DeleteLocalRef(type);
    return field;
}

bool SetInt(JNIEnv* env, jobject object, const char* name, jint value) {
    jfieldID field = Field(env, object, name, "I");
    if (field == nullptr) return false;
    env->SetIntField(object, field, value);
    return !env->ExceptionCheck();
}

bool SetLong(JNIEnv* env, jobject object, const char* name, jlong value) {
    jfieldID field = Field(env, object, name, "J");
    if (field == nullptr) return false;
    env->SetLongField(object, field, value);
    return !env->ExceptionCheck();
}

bool SetFloat(JNIEnv* env, jobject object, const char* name, jfloat value) {
    jfieldID field = Field(env, object, name, "F");
    if (field == nullptr) return false;
    env->SetFloatField(object, field, value);
    return !env->ExceptionCheck();
}

bool SetObject(JNIEnv* env, jobject object, const char* name, const char* signature, jobject value) {
    jfieldID field = Field(env, object, name, signature);
    if (field == nullptr) return false;
    env->SetObjectField(object, field, value);
    return !env->ExceptionCheck();
}

jlong GetLong(JNIEnv* env, jobject object, const char* name, bool* ok) {
    jfieldID field = Field(env, object, name, "J");
    if (field == nullptr) {
        *ok = false;
        return 0;
    }
    return env->GetLongField(object, field);
}

jint GetInt(JNIEnv* env, jobject object, const char* name, bool* ok) {
    jfieldID field = Field(env, object, name, "I");
    if (field == nullptr) {
        *ok = false;
        return 0;
    }
    return env->GetIntField(object, field);
}

jfloat GetFloat(JNIEnv* env, jobject object, const char* name, bool* ok) {
    jfieldID field = Field(env, object, name, "F");
    if (field == nullptr) {
        *ok = false;
        return 0.0f;
    }
    return env->GetFloatField(object, field);
}

bool FillConfig(JNIEnv* env, jobject output, const HFFaceCaptureConfig& config) {
    return SetLong(env, output, "filterMask", static_cast<jlong>(config.filterMask)) &&
           SetInt(env, output, "outputCount", static_cast<jint>(config.outputCount)) &&
           SetInt(env, output, "minTrackCount", static_cast<jint>(config.minTrackCount)) &&
           SetLong(env, output, "stableDurationMs", static_cast<jlong>(config.stableDurationMs)) &&
           SetLong(env, output, "collectDurationMs", static_cast<jlong>(config.collectDurationMs)) &&
           SetLong(env, output, "maxCollectDurationMs", static_cast<jlong>(config.maxCollectDurationMs)) &&
           SetLong(env, output, "trackLostGraceMs", static_cast<jlong>(config.trackLostGraceMs)) &&
           SetLong(env, output, "minCandidateIntervalMs", static_cast<jlong>(config.minCandidateIntervalMs)) &&
           SetFloat(env, output, "minFaceWidthRatio", config.minFaceWidthRatio) &&
           SetFloat(env, output, "maxFaceWidthRatio", config.maxFaceWidthRatio) &&
           SetFloat(env, output, "maxCenterOffsetX", config.maxCenterOffsetX) &&
           SetFloat(env, output, "maxCenterOffsetY", config.maxCenterOffsetY) &&
           SetFloat(env, output, "boundaryMarginRatio", config.boundaryMarginRatio) &&
           SetFloat(env, output, "maxCenterMotionRatio", config.maxCenterMotionRatio) &&
           SetFloat(env, output, "maxSizeChangeRatio", config.maxSizeChangeRatio) &&
           SetFloat(env, output, "maxAbsYaw", config.maxAbsYaw) &&
           SetFloat(env, output, "maxAbsPitch", config.maxAbsPitch) &&
           SetFloat(env, output, "maxAbsRoll", config.maxAbsRoll) &&
           SetFloat(env, output, "minQualityScore", config.minQualityScore) &&
           SetFloat(env, output, "minSharpnessScore", config.minSharpnessScore) &&
           SetFloat(env, output, "minBrightnessScore", config.minBrightnessScore) &&
           SetFloat(env, output, "maxBrightnessScore", config.maxBrightnessScore);
}

bool ReadConfig(JNIEnv* env, jobject input, HFFaceCaptureConfig* config) {
    if (input == nullptr || config == nullptr || HFGetDefaultFaceCaptureConfig(config) != HSUCCEED) {
        return false;
    }
    bool ok = true;
    const jlong filterMask = GetLong(env, input, "filterMask", &ok);
    const jint outputCount = GetInt(env, input, "outputCount", &ok);
    const jint minTrackCount = GetInt(env, input, "minTrackCount", &ok);
    const jlong stableDurationMs = GetLong(env, input, "stableDurationMs", &ok);
    const jlong collectDurationMs = GetLong(env, input, "collectDurationMs", &ok);
    const jlong maxCollectDurationMs = GetLong(env, input, "maxCollectDurationMs", &ok);
    const jlong trackLostGraceMs = GetLong(env, input, "trackLostGraceMs", &ok);
    const jlong minCandidateIntervalMs = GetLong(env, input, "minCandidateIntervalMs", &ok);
    if (!ok || filterMask < 0 || outputCount <= 0 || minTrackCount < 0 ||
        (((static_cast<uint64_t>(filterMask) & HF_CAPTURE_FILTER_TRACK_COUNT) != 0) && minTrackCount == 0) ||
        stableDurationMs < 0 || collectDurationMs < 0 ||
        maxCollectDurationMs < 0 || trackLostGraceMs < 0 || minCandidateIntervalMs < 0) {
        return false;
    }
    config->filterMask = static_cast<HFUInt64>(filterMask);
    config->outputCount = static_cast<HFUInt32>(outputCount);
    config->minTrackCount = static_cast<HFUInt32>(minTrackCount);
    config->stableDurationMs = static_cast<HFUInt64>(stableDurationMs);
    config->collectDurationMs = static_cast<HFUInt64>(collectDurationMs);
    config->maxCollectDurationMs = static_cast<HFUInt64>(maxCollectDurationMs);
    config->trackLostGraceMs = static_cast<HFUInt64>(trackLostGraceMs);
    config->minCandidateIntervalMs = static_cast<HFUInt64>(minCandidateIntervalMs);
    config->minFaceWidthRatio = GetFloat(env, input, "minFaceWidthRatio", &ok);
    config->maxFaceWidthRatio = GetFloat(env, input, "maxFaceWidthRatio", &ok);
    config->maxCenterOffsetX = GetFloat(env, input, "maxCenterOffsetX", &ok);
    config->maxCenterOffsetY = GetFloat(env, input, "maxCenterOffsetY", &ok);
    config->boundaryMarginRatio = GetFloat(env, input, "boundaryMarginRatio", &ok);
    config->maxCenterMotionRatio = GetFloat(env, input, "maxCenterMotionRatio", &ok);
    config->maxSizeChangeRatio = GetFloat(env, input, "maxSizeChangeRatio", &ok);
    config->maxAbsYaw = GetFloat(env, input, "maxAbsYaw", &ok);
    config->maxAbsPitch = GetFloat(env, input, "maxAbsPitch", &ok);
    config->maxAbsRoll = GetFloat(env, input, "maxAbsRoll", &ok);
    config->minQualityScore = GetFloat(env, input, "minQualityScore", &ok);
    config->minSharpnessScore = GetFloat(env, input, "minSharpnessScore", &ok);
    config->minBrightnessScore = GetFloat(env, input, "minBrightnessScore", &ok);
    config->maxBrightnessScore = GetFloat(env, input, "maxBrightnessScore", &ok);
    return ok && !env->ExceptionCheck();
}

bool FillMetrics(JNIEnv* env, jobject output, const HFFaceCaptureMetrics& metrics) {
    return output != nullptr &&
           SetLong(env, output, "availableFilters", static_cast<jlong>(metrics.availableMetrics)) &&
           SetFloat(env, output, "faceWidthRatio", metrics.faceWidthRatio) &&
           SetFloat(env, output, "centerOffsetX", metrics.centerOffsetX) &&
           SetFloat(env, output, "centerOffsetY", metrics.centerOffsetY) &&
           SetFloat(env, output, "stabilityScore", metrics.stabilityScore) &&
           SetFloat(env, output, "poseScore", metrics.poseScore) &&
           SetFloat(env, output, "qualityScore", metrics.qualityScore) &&
           SetFloat(env, output, "sharpnessScore", metrics.sharpnessScore) &&
           SetFloat(env, output, "brightnessScore", metrics.brightnessScore);
}

bool FillProgress(JNIEnv* env, jobject output, const HFFaceCaptureProgress& progress) {
    if (output == nullptr || !SetInt(env, output, "state", progress.state) ||
        !SetInt(env, output, "candidateCount", static_cast<jint>(progress.candidateCount)) ||
        !SetLong(env, output, "frameId", static_cast<jlong>(progress.frameId)) ||
        !SetLong(env, output, "timestampMs", static_cast<jlong>(progress.timestampMs)) ||
        !SetInt(env, output, "trackId", progress.trackId) ||
        !SetInt(env, output, "trackCount", progress.trackCount) ||
        !SetLong(env, output, "evaluatedFilters", static_cast<jlong>(progress.evaluatedFilters)) ||
        !SetLong(env, output, "rejectReasons", static_cast<jlong>(progress.rejectReasons)) ||
        !SetFloat(env, output, "progress", progress.progress) ||
        !SetFloat(env, output, "currentScore", progress.currentScore)) {
        return false;
    }
    jfieldID metricsField = Field(env, output, "metrics", "Lcom/insightface/sdk/inspireface/base/FaceCaptureMetrics;");
    if (metricsField == nullptr) return false;
    jobject metrics = env->GetObjectField(output, metricsField);
    const bool success = FillMetrics(env, metrics, progress.metrics);
    if (metrics != nullptr) env->DeleteLocalRef(metrics);
    return success;
}

jobject NewObject(JNIEnv* env, const char* className) {
    jclass type = env->FindClass(className);
    if (type == nullptr) return nullptr;
    jmethodID constructor = env->GetMethodID(type, "<init>", "()V");
    jobject object = constructor == nullptr ? nullptr : env->NewObject(type, constructor);
    env->DeleteLocalRef(type);
    return object;
}

bool FillResult(JNIEnv* env, jobject output, const HFFaceCaptureResult& result) {
    if (output == nullptr || !SetLong(env, output, "frameId", static_cast<jlong>(result.frameId)) ||
        !SetLong(env, output, "timestampMs", static_cast<jlong>(result.timestampMs)) ||
        !SetInt(env, output, "trackId", result.trackId) || !SetInt(env, output, "trackCount", result.trackCount) ||
        !SetFloat(env, output, "score", result.score) || !SetFloat(env, output, "roll", result.roll) ||
        !SetFloat(env, output, "yaw", result.yaw) || !SetFloat(env, output, "pitch", result.pitch)) {
        return false;
    }

    jobject rect = NewObject(env, "com/insightface/sdk/inspireface/base/FaceRect");
    if (rect == nullptr || !SetInt(env, rect, "x", result.rect.x) || !SetInt(env, rect, "y", result.rect.y) ||
        !SetInt(env, rect, "width", result.rect.width) || !SetInt(env, rect, "height", result.rect.height) ||
        !SetObject(env, output, "rect", "Lcom/insightface/sdk/inspireface/base/FaceRect;", rect)) {
        if (rect != nullptr) env->DeleteLocalRef(rect);
        return false;
    }
    env->DeleteLocalRef(rect);

    if (result.token.size <= 0 || result.token.data == nullptr) return false;
    jbyteArray token = env->NewByteArray(result.token.size);
    if (token == nullptr) return false;
    env->SetByteArrayRegion(token, 0, result.token.size, static_cast<const jbyte*>(result.token.data));
    const bool tokenSet = !env->ExceptionCheck() && SetObject(env, output, "token", "[B", token);
    env->DeleteLocalRef(token);
    if (!tokenSet) return false;

    jfieldID metricsField = Field(env, output, "metrics", "Lcom/insightface/sdk/inspireface/base/FaceCaptureMetrics;");
    if (metricsField == nullptr) return false;
    jobject metrics = env->GetObjectField(output, metricsField);
    const bool success = FillMetrics(env, metrics, result.metrics);
    if (metrics != nullptr) env->DeleteLocalRef(metrics);
    return success;
}

jint Status(HResult result) {
    return static_cast<jint>(result);
}

}  // namespace

extern "C" {

JNIEXPORT jint INSPIRE_FACE_JNI(FaceCapture_nativeGetDefaultConfig)(JNIEnv* env, jclass, jobject output) {
    if (env == nullptr || output == nullptr) return HERR_INVALID_PARAM;
    HFFaceCaptureConfig config{};
    const HResult status = HFGetDefaultFaceCaptureConfig(&config);
    if (status != HSUCCEED) return Status(status);
    return FillConfig(env, output, config) ? HSUCCEED : HERR_UNKNOWN;
}

JNIEXPORT jint INSPIRE_FACE_JNI(FaceCapture_nativeCreate)(JNIEnv* env, jclass, jlong sessionHandle,
                                                          jobject configObject, jobject output) {
    if (env == nullptr || output == nullptr) return HERR_INVALID_PARAM;
    HFFaceCaptureConfig config{};
    if (!ReadConfig(env, configObject, &config)) return HERR_CAPTURE_INVALID_CONFIG;
    HFFaceCaptureSession capture = nullptr;
    const HResult status = HFCreateFaceCaptureSession(reinterpret_cast<HFSession>(sessionHandle), &config, &capture);
    if (status != HSUCCEED) return Status(status);
    if (!SetLong(env, output, "nativeHandle", reinterpret_cast<jlong>(capture))) {
        HFReleaseFaceCaptureSession(capture);
        return HERR_UNKNOWN;
    }
    return HSUCCEED;
}

JNIEXPORT jint INSPIRE_FACE_JNI(FaceCapture_nativeUpdate)(JNIEnv* env, jclass, jlong captureHandle,
                                                          jlong streamHandle, jlong frameId, jlong timestampMs,
                                                          jobject output) {
    if (env == nullptr || output == nullptr || frameId < 0 || timestampMs < 0) return HERR_INVALID_PARAM;
    HFFaceCaptureProgress progress{};
    const HResult status = HFUpdateFaceCaptureSession(
      reinterpret_cast<HFFaceCaptureSession>(captureHandle), reinterpret_cast<HFImageStream>(streamHandle),
      static_cast<HFUInt64>(frameId), static_cast<HFUInt64>(timestampMs), &progress);
    if (status != HSUCCEED) return Status(status);
    return FillProgress(env, output, progress) ? HSUCCEED : HERR_UNKNOWN;
}

JNIEXPORT jint INSPIRE_FACE_JNI(FaceCapture_nativeUpdateWithSnapshot)(
  JNIEnv* env, jclass, jlong captureHandle, jlong streamHandle, jlong snapshotHandle, jlong frameId,
  jlong timestampMs, jobject output) {
    if (env == nullptr || output == nullptr || frameId < 0 || timestampMs < 0) return HERR_INVALID_PARAM;
    HFFaceCaptureProgress progress{};
    const HResult status = HFUpdateFaceCaptureSessionWithSnapshot(
      reinterpret_cast<HFFaceCaptureSession>(captureHandle), reinterpret_cast<HFImageStream>(streamHandle),
      reinterpret_cast<HFFaceResultSnapshot>(snapshotHandle), static_cast<HFUInt64>(frameId),
      static_cast<HFUInt64>(timestampMs), &progress);
    if (status != HSUCCEED) return Status(status);
    return FillProgress(env, output, progress) ? HSUCCEED : HERR_UNKNOWN;
}

JNIEXPORT jint INSPIRE_FACE_JNI(FaceCapture_nativeGetResults)(JNIEnv* env, jclass, jlong captureHandle,
                                                              jobject output) {
    if (env == nullptr || output == nullptr) return HERR_INVALID_PARAM;
    HFUInt32 count = 0;
    HFFaceCaptureSession capture = reinterpret_cast<HFFaceCaptureSession>(captureHandle);
    HResult status = HFGetFaceCaptureResults(capture, nullptr, 0, &count);
    if (status != HSUCCEED) return Status(status);
    if (count > HF_FACE_CAPTURE_MAX_RESULTS) return HERR_INVALID_BUFFER_SIZE;
    std::vector<HFFaceCaptureResult> results(count);
    if (count > 0) {
        status = HFGetFaceCaptureResults(capture, results.data(), count, &count);
        if (status != HSUCCEED) return Status(status);
    }

    jclass resultClass = env->FindClass("com/insightface/sdk/inspireface/base/FaceCaptureResult");
    if (resultClass == nullptr) return HERR_UNKNOWN;
    jobjectArray array = env->NewObjectArray(static_cast<jsize>(count), resultClass, nullptr);
    jmethodID constructor = env->GetMethodID(resultClass, "<init>", "()V");
    if (array == nullptr || constructor == nullptr) {
        env->DeleteLocalRef(resultClass);
        return HERR_UNKNOWN;
    }
    for (HFUInt32 index = 0; index < count; ++index) {
        jobject item = env->NewObject(resultClass, constructor);
        if (item == nullptr || !FillResult(env, item, results[index])) {
            if (item != nullptr) env->DeleteLocalRef(item);
            env->DeleteLocalRef(array);
            env->DeleteLocalRef(resultClass);
            return HERR_UNKNOWN;
        }
        env->SetObjectArrayElement(array, static_cast<jsize>(index), item);
        env->DeleteLocalRef(item);
        if (env->ExceptionCheck()) {
            env->DeleteLocalRef(array);
            env->DeleteLocalRef(resultClass);
            return HERR_UNKNOWN;
        }
    }
    env->DeleteLocalRef(resultClass);
    const bool success = SetObject(
      env, output, "results", "[Lcom/insightface/sdk/inspireface/base/FaceCaptureResult;", array);
    env->DeleteLocalRef(array);
    return success ? HSUCCEED : HERR_UNKNOWN;
}

JNIEXPORT jint INSPIRE_FACE_JNI(FaceCapture_nativeFinish)(JNIEnv* env, jclass, jlong captureHandle,
                                                          jobject output) {
    if (env == nullptr || output == nullptr) return HERR_INVALID_PARAM;
    HFFaceCaptureProgress progress{};
    const HResult status = HFFinishFaceCaptureSession(
      reinterpret_cast<HFFaceCaptureSession>(captureHandle), &progress);
    if (status != HSUCCEED) return Status(status);
    return FillProgress(env, output, progress) ? HSUCCEED : HERR_UNKNOWN;
}

JNIEXPORT jint INSPIRE_FACE_JNI(FaceCapture_nativeReset)(JNIEnv*, jclass, jlong captureHandle) {
    return Status(HFResetFaceCaptureSession(reinterpret_cast<HFFaceCaptureSession>(captureHandle)));
}

JNIEXPORT jint INSPIRE_FACE_JNI(FaceCapture_nativeRelease)(JNIEnv*, jclass, jlong captureHandle) {
    return Status(HFReleaseFaceCaptureSession(reinterpret_cast<HFFaceCaptureSession>(captureHandle)));
}

JNIEXPORT jint INSPIRE_FACE_JNI(FaceDetectionSnapshot_nativeCreate)(JNIEnv* env, jclass, jlong sessionHandle,
                                                                    jlong streamHandle, jobject output) {
    if (env == nullptr || output == nullptr) return HERR_INVALID_PARAM;
    HFFaceResultSnapshot snapshot = nullptr;
    const HResult status = HFExecuteFaceTrackSnapshot(
      reinterpret_cast<HFSession>(sessionHandle), reinterpret_cast<HFImageStream>(streamHandle), &snapshot);
    if (status != HSUCCEED) return Status(status);
    if (!SetLong(env, output, "nativeHandle", reinterpret_cast<jlong>(snapshot))) {
        HFReleaseFaceResultSnapshot(snapshot);
        return HERR_UNKNOWN;
    }
    return HSUCCEED;
}

JNIEXPORT jint INSPIRE_FACE_JNI(FaceDetectionSnapshot_nativeRelease)(JNIEnv*, jclass, jlong snapshotHandle) {
    return Status(HFReleaseFaceResultSnapshot(reinterpret_cast<HFFaceResultSnapshot>(snapshotHandle)));
}

}  // extern "C"

#endif  // ANDROID
