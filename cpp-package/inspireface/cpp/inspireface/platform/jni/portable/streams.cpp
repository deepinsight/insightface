#include "support.h"
#include <map>
#include <mutex>

using namespace isf_jni;
#define JNI_METHOD(name) Java_com_insightface_sdk_inspireface_jni_Native_##name

namespace {
// A strong JNI reference pins Java-owned pixels until the stream is closed or
// its buffer is replaced. The recorded span is independent of later position changes.
struct StreamState {
    jobject pixels = nullptr;
    int width = 0, height = 0, format = HF_STREAM_YUV_NV21;
    int64_t bytes = 0;
};
std::mutex streamsMutex;
std::map<HFImageStream, StreamState> streams;

class Global {
    JNIEnv* env_;
    jobject reference_;
public:
    Global(JNIEnv* env, jobject object) : env_(env), reference_(object ? env->NewGlobalRef(object) : nullptr) { Check(env); }
    ~Global() { if (reference_) env_->DeleteGlobalRef(reference_); }
    jobject get() const { return reference_; }
    void release() { reference_ = nullptr; }
};
int64_t Remaining(JNIEnv* env, jobject buffer) {
    Local cls(env, env->GetObjectClass(buffer));
    auto method = env->GetMethodID(static_cast<jclass>(cls.get()), "remaining", "()I"); Check(env);
    auto remaining = env->CallIntMethod(buffer, method); Check(env);
    return remaining;
}
void Store(JNIEnv* env, HFImageStream stream, Global& pixels, int width, int height, int format, int64_t bytes) {
    try {
        std::lock_guard<std::mutex> lock(streamsMutex);
        StreamState state;
        state.pixels = pixels.get(); state.width = width; state.height = height;
        state.format = format; state.bytes = bytes;
        auto inserted = streams.emplace(stream, state);
        Require(env, inserted.second, "Stream handle already registered");
        pixels.release();
    } catch (...) {
        HFReleaseImageStream(stream);
        throw;
    }
}
void Output(JNIEnv* env, jlongArray output, HFImageStream handle) {
    jlong value = static_cast<jlong>(reinterpret_cast<uintptr_t>(handle));
    env->SetLongArrayRegion(output, 0, 1, &value); Check(env);
}
HFImageStream Handle(jlong value) { return reinterpret_cast<HFImageStream>(static_cast<uintptr_t>(value)); }
}  // namespace

extern "C" JNIEXPORT jlong JNICALL JNI_METHOD(HFCreateImageStream)(
    JNIEnv* env, jclass, jobject data, jlongArray handle) {
    try {
        RequireArray(env, handle, 1); Output(env, handle, nullptr);
        Require(env, data != nullptr, "Null image descriptor");
        HFImageData input{};
        input.width = env->GetIntField(data, Field(env, data, "width", "I")); Check(env);
        input.height = env->GetIntField(data, Field(env, data, "height", "I")); Check(env);
        input.format = static_cast<HFImageFormat>(env->GetIntField(data, Field(env, data, "format", "I"))); Check(env);
        input.rotation = static_cast<HFRotation>(env->GetIntField(data, Field(env, data, "rotation", "I"))); Check(env);
        Local buffer(env, env->GetObjectField(data, Field(env, data, "data", "Ljava/nio/ByteBuffer;")));
        input.data = Direct<uint8_t>(env, buffer.get(), ImageBytes(env, input.format, input.width, input.height));
        Global pixels(env, buffer.get());
        const int64_t bytes = Remaining(env, buffer.get());
        HFImageStream stream = nullptr;
        HResult status = HFCreateImageStream(&input, &stream);
        if (status == HSUCCEED) {
            Store(env, stream, pixels, input.width, input.height, input.format, bytes);
            Output(env, handle, stream);
        }
        return status;
    } catch (...) { TranslateException(env); }
    return HERR_INVALID_PARAM;
}

extern "C" JNIEXPORT jlong JNICALL JNI_METHOD(HFCreateImageStreamEmpty)(
    JNIEnv* env, jclass, jlongArray handle) {
    try {
        RequireArray(env, handle, 1); Output(env, handle, nullptr);
        HFImageStream stream = nullptr;
        HResult status = HFCreateImageStreamEmpty(&stream);
        if (status == HSUCCEED) {
            Global pixels(env, nullptr);
            Store(env, stream, pixels, 0, 0, HF_STREAM_YUV_NV21, 0);
            Output(env, handle, stream);
        }
        return status;
    } catch (...) { TranslateException(env); }
    return HERR_INVALID_PARAM;
}

extern "C" JNIEXPORT jlong JNICALL JNI_METHOD(HFImageStreamSetBuffer)(
    JNIEnv* env, jclass, jlong handle, jobject buffer, jint width, jint height) {
    try {
        std::lock_guard<std::mutex> lock(streamsMutex);
        auto entry = streams.find(Handle(handle));
        if (entry == streams.end()) return HERR_INVALID_IMAGE_STREAM_HANDLE;
        auto& state = entry->second;
        auto address = Direct<uint8_t>(env, buffer, ImageBytes(env, state.format, width, height));
        Global replacement(env, buffer);
        const int64_t bytes = Remaining(env, buffer);
        HResult status = HFImageStreamSetBuffer(Handle(handle), address, width, height);
        if (status == HSUCCEED) {
            if (state.pixels) env->DeleteGlobalRef(state.pixels);
            state.pixels = replacement.get(); replacement.release();
            state.width = width; state.height = height; state.bytes = bytes;
        }
        return status;
    } catch (...) { TranslateException(env); }
    return HERR_INVALID_PARAM;
}

extern "C" JNIEXPORT jlong JNICALL JNI_METHOD(HFImageStreamSetFormat)(
    JNIEnv* env, jclass, jlong handle, jint format) {
    try {
        std::lock_guard<std::mutex> lock(streamsMutex);
        auto entry = streams.find(Handle(handle));
        if (entry == streams.end()) return HERR_INVALID_IMAGE_STREAM_HANDLE;
        auto& state = entry->second;
        if (state.pixels) {
            Require(env, ImageBytes(env, format, state.width, state.height) <= state.bytes,
                    "New image format exceeds the borrowed buffer");
        }
        HResult status = HFImageStreamSetFormat(Handle(handle), static_cast<HFImageFormat>(format));
        if (status == HSUCCEED) state.format = format;
        return status;
    } catch (...) { TranslateException(env); }
    return HERR_INVALID_PARAM;
}

extern "C" JNIEXPORT jlong JNICALL JNI_METHOD(HFReleaseImageStream)(
    JNIEnv* env, jclass, jlong handle) {
    try {
        std::lock_guard<std::mutex> lock(streamsMutex);
        HResult status = HFReleaseImageStream(Handle(handle));
        if (status == HSUCCEED) {
            auto entry = streams.find(Handle(handle));
            if (entry != streams.end()) {
                if (entry->second.pixels) env->DeleteGlobalRef(entry->second.pixels);
                streams.erase(entry);
            }
        }
        return status;
    } catch (...) { TranslateException(env); }
    return HERR_INVALID_PARAM;
}

extern "C" JNIEXPORT jlong JNICALL JNI_METHOD(HFCreateImageStreamFromImageBitmap)(
    JNIEnv* env, jclass, jlong handle, jint rotation, jlongArray streamHandle) {
    try {
        RequireArray(env, streamHandle, 1); Output(env, streamHandle, nullptr);
        HFImageBitmap bitmap = reinterpret_cast<HFImageBitmap>(static_cast<uintptr_t>(handle));
        HFImageBitmapData data{};
        HResult status = HFImageBitmapGetData(bitmap, &data);
        if (status != HSUCCEED) return status;
        HFImageStream stream = nullptr;
        status = HFCreateImageStreamFromImageBitmap(bitmap, static_cast<HFRotation>(rotation), &stream);
        if (status == HSUCCEED) {
            Global pixels(env, nullptr);
            Store(env, stream, pixels, data.width, data.height,
                  data.channels == 1 ? HF_STREAM_GRAY : HF_STREAM_BGR, 0);
            Output(env, streamHandle, stream);
        }
        return status;
    } catch (...) { TranslateException(env); }
    return HERR_INVALID_PARAM;
}
