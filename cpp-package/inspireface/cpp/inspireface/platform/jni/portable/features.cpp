#include "support.h"
#include <map>
#include <mutex>

using namespace isf_jni;
namespace {
std::mutex featureMutex;
// The C resource manager keys allocated features by descriptor ADDRESS.
// Keep that address stable until HFReleaseFaceFeature, rather than marshalling
// the descriptor into a new stack temporary for each call.
std::map<jlong, std::unique_ptr<HFFaceFeature>> features;
}

extern "C" JNIEXPORT jlong JNICALL Java_com_insightface_sdk_inspireface_jni_Native_HFCreateFaceFeature(
    JNIEnv* env, jclass, jobject output) {
    try {
        Require(env, output != nullptr, "Null feature output");
        auto ownerField = Field(env, output, "nativeOwner", "J");
        auto owner = env->GetLongField(output, ownerField); Check(env);
        if (owner != 0) return HERR_INVALID_FACE_FEATURE;
        std::unique_ptr<HFFaceFeature> feature(new HFFaceFeature{});
        HResult status = HFCreateFaceFeature(feature.get());
        if (status != HSUCCEED) return status;
        try {
            Local buffer(env, Buffer(env, feature->data, Bytes(env, feature->size, sizeof(float))));
            auto sizeField = Field(env, output, "size", "I");
            auto dataField = Field(env, output, "data", "Ljava/nio/ByteBuffer;");
            std::lock_guard<std::mutex> lock(featureMutex);
            jlong key = static_cast<jlong>(reinterpret_cast<uintptr_t>(feature.get()));
            // Allocate the map node before transferring ownership.
            auto inserted = features.emplace(key, nullptr);
            Require(env, inserted.second, "Feature already registered");
            env->SetIntField(output, sizeField, feature->size);
            env->SetObjectField(output, dataField, buffer.get());
            env->SetLongField(output, ownerField, key);
            inserted.first->second = std::move(feature);
            Check(env);
        } catch (...) {
            if (feature) HFReleaseFaceFeature(feature.get());
            throw;
        }
        return status;
    } catch (...) { TranslateException(env); }
    return HERR_INVALID_PARAM;
}

extern "C" JNIEXPORT jlong JNICALL Java_com_insightface_sdk_inspireface_jni_Native_HFReleaseFaceFeature(
    JNIEnv* env, jclass, jobject output) {
    try {
        Require(env, output != nullptr, "Null feature output");
        auto ownerField = Field(env, output, "nativeOwner", "J");
        auto key = env->GetLongField(output, ownerField); Check(env);
        auto sizeField = Field(env, output, "size", "I");
        auto dataField = Field(env, output, "data", "Ljava/nio/ByteBuffer;");
        std::lock_guard<std::mutex> lock(featureMutex);
        auto entry = features.find(key);
        if (key == 0 || entry == features.end()) return HERR_INVALID_FACE_FEATURE;
        HResult status = HFReleaseFaceFeature(entry->second.get());
        if (status == HSUCCEED) {
            env->SetLongField(output, ownerField, 0);
            env->SetIntField(output, sizeField, 0);
            env->SetObjectField(output, dataField, nullptr);
            features.erase(entry);
            Check(env);
        }
        return status;
    } catch (...) { TranslateException(env); }
    return HERR_INVALID_PARAM;
}
