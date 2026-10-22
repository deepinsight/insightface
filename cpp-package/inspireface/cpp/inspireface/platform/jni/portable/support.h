#pragma once
#include <jni.h>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include "c_api/inspireface.h"
#include "herror.h"
#include "../common/jni_data_utils.h"

namespace isf_jni {
struct PendingException {};
inline void Check(JNIEnv* env) { if (env->ExceptionCheck()) throw PendingException{}; }
inline void Fail(JNIEnv* env, const char* type, const char* message) {
    if (!env->ExceptionCheck()) {
        jclass cls = env->FindClass(type);
        if (cls) { env->ThrowNew(cls, message); env->DeleteLocalRef(cls); }
    }
    throw PendingException{};
}
inline void Require(JNIEnv* env, bool condition, const char* message) {
    if (!condition) Fail(env, "java/lang/IllegalArgumentException", message);
}
inline void TranslateException(JNIEnv* env) noexcept {
    try { throw; }
    catch (const PendingException&) {}
    catch (const std::bad_alloc&) {
        try { Fail(env, "java/lang/OutOfMemoryError", "JNI allocation failed"); } catch (...) {}
    }
    catch (const std::exception& error) {
        try { Fail(env, "java/lang/IllegalStateException", error.what()); } catch (...) {}
    }
    catch (...) {
        try { Fail(env, "java/lang/IllegalStateException", "Unexpected native exception"); } catch (...) {}
    }
}
class Local {
    JNIEnv* env_;
    jobject value_;
public:
    Local(JNIEnv* env, jobject value) : env_(env), value_(value) { Check(env); }
    ~Local() { if (value_) env_->DeleteLocalRef(value_); }
    Local(const Local&) = delete;
    Local& operator=(const Local&) = delete;
    jobject get() const { return value_; }
    jobject release() { jobject value = value_; value_ = nullptr; return value; }
};
inline jfieldID Field(JNIEnv* env, jobject object, const char* name, const char* signature) {
    Require(env, object != nullptr, "Null descriptor");
    Local cls(env, env->GetObjectClass(object));
    jfieldID field = env->GetFieldID(static_cast<jclass>(cls.get()), name, signature);
    Check(env);
    return field;
}
inline int64_t Bytes(JNIEnv* env, int64_t count, size_t size) {
    Require(env, count >= 0 && static_cast<uint64_t>(count) <= INT32_MAX / size,
            "Negative or oversized native buffer");
    return count * size;
}
inline int64_t ImageBytes(JNIEnv* env, int format, int width, int height) {
    size_t size = 0;
    Require(env, inspire::jni::CheckedImageByteSize(format, width, height, &size) && size <= INT32_MAX,
            "Invalid image dimensions or format");
    return static_cast<int64_t>(size);
}
inline int64_t BitmapBytes(JNIEnv* env, int width, int height, int channels) {
    Require(env, channels == 1 || channels == 3 || channels == 4, "Invalid bitmap channel count");
    return ImageBytes(env, channels == 1 ? HF_STREAM_GRAY : channels == 3 ? HF_STREAM_BGR : HF_STREAM_BGRA, width, height);
}
inline void RequireArray(JNIEnv* env, jobject array, int64_t count) {
    Require(env, count >= 0 && count <= INT32_MAX, "Invalid array count");
    if (!array) { Require(env, count == 0, "Null or short array"); return; }
    jsize length = env->GetArrayLength(static_cast<jarray>(array)); Check(env);
    Require(env, length >= count, "Null or short array");
}
// ByteBuffer position/limit are honored; never cast a heap buffer or an unaligned slice.
template<class T> T* Direct(JNIEnv* env, jobject buffer, int64_t count) {
    const int64_t bytes = Bytes(env, count, sizeof(T));
    if (!buffer) { Require(env, bytes == 0, "Null direct buffer"); return nullptr; }
    Local cls(env, env->FindClass("java/nio/ByteBuffer"));
    Require(env, env->IsInstanceOf(buffer, static_cast<jclass>(cls.get())), "Expected ByteBuffer"); Check(env);
    auto positionMethod = env->GetMethodID(static_cast<jclass>(cls.get()), "position", "()I"); Check(env);
    auto remainingMethod = env->GetMethodID(static_cast<jclass>(cls.get()), "remaining", "()I"); Check(env);
    auto readOnlyMethod = env->GetMethodID(static_cast<jclass>(cls.get()), "isReadOnly", "()Z"); Check(env);
    const jint position = env->CallIntMethod(buffer, positionMethod); Check(env);
    const jint remaining = env->CallIntMethod(buffer, remainingMethod); Check(env);
    const bool readOnly = env->CallBooleanMethod(buffer, readOnlyMethod); Check(env);
    void* address = env->GetDirectBufferAddress(buffer); Check(env);
    const jlong capacity = env->GetDirectBufferCapacity(buffer); Check(env);
    Require(env, !readOnly && capacity >= 0 && remaining >= bytes &&
            static_cast<int64_t>(position) + remaining <= capacity && (address || bytes == 0),
            "Expected writable direct ByteBuffer with sufficient remaining bytes");
    auto value = reinterpret_cast<uintptr_t>(address) + position;
    Require(env, value % alignof(T) == 0, "Unaligned direct buffer");
    return reinterpret_cast<T*>(value);
}
inline jobject Buffer(JNIEnv* env, const void* data, int64_t size) {
    if (!data || size == 0) return nullptr;
    Bytes(env, size, 1);
    Local buffer(env, env->NewDirectByteBuffer(const_cast<void*>(data), size));
    Require(env, buffer.get() != nullptr, "VM does not support direct buffers");
    Local orderClass(env, env->FindClass("java/nio/ByteOrder"));
    auto nativeOrder = env->GetStaticMethodID(static_cast<jclass>(orderClass.get()), "nativeOrder", "()Ljava/nio/ByteOrder;"); Check(env);
    Local order(env, env->CallStaticObjectMethod(static_cast<jclass>(orderClass.get()), nativeOrder)); Check(env);
    Local cls(env, env->GetObjectClass(buffer.get()));
    auto method = env->GetMethodID(static_cast<jclass>(cls.get()), "order", "(Ljava/nio/ByteOrder;)Ljava/nio/ByteBuffer;"); Check(env);
    Local ordered(env, env->CallObjectMethod(buffer.get(), method, order.get())); Check(env);
    return buffer.release();
}

template<class J> struct ArrayOps;
#define ISF_ARRAY_OPS(J, NAME, ARRAY) \
template<> struct ArrayOps<J> { \
    using Array = ARRAY; \
    static ARRAY New(JNIEnv* e, jsize n) { auto a = e->New##NAME##Array(n); Check(e); return a; } \
    static void Get(JNIEnv* e, ARRAY a, jsize n, J* p) { if (n) e->Get##NAME##ArrayRegion(a, 0, n, p); Check(e); } \
    static void Set(JNIEnv* e, ARRAY a, jsize n, const J* p) { if (n) e->Set##NAME##ArrayRegion(a, 0, n, p); Check(e); } \
};
ISF_ARRAY_OPS(jint, Int, jintArray)
ISF_ARRAY_OPS(jlong, Long, jlongArray)
ISF_ARRAY_OPS(jfloat, Float, jfloatArray)
ISF_ARRAY_OPS(jbyte, Byte, jbyteArray)
#undef ISF_ARRAY_OPS
template<class T> using JavaScalar = typename std::conditional<std::is_pointer<T>::value || sizeof(T) == 8, jlong,
    typename std::conditional<std::is_floating_point<T>::value, jfloat,
    typename std::conditional<sizeof(T) == 1, jbyte, jint>::type>::type>::type;
template<class T> inline typename std::enable_if<!std::is_pointer<T>::value, JavaScalar<T>>::type ToJava(T value) {
    return static_cast<JavaScalar<T>>(value);
}
template<class T> inline typename std::enable_if<std::is_pointer<T>::value, jlong>::type ToJava(T value) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(value));
}
template<class T> void ReadArray(JNIEnv* env, jobject object, T* output, int64_t count) {
    Bytes(env, count, sizeof(T)); RequireArray(env, object, count);
    using J = JavaScalar<T>;
    std::vector<J> values(static_cast<size_t>(count));
    ArrayOps<J>::Get(env, static_cast<typename ArrayOps<J>::Array>(object), static_cast<jsize>(count), values.data());
    for (int64_t i = 0; i < count; ++i) output[i] = static_cast<T>(values[i]);
}
template<class T> void SetArray(JNIEnv* env, typename ArrayOps<JavaScalar<T>>::Array object, const T* input, int64_t count) {
    Bytes(env, count, sizeof(T)); RequireArray(env, object, count);
    using J = JavaScalar<T>;
    if (count == 1) {
        J value = ToJava(input[0]);
        ArrayOps<J>::Set(env, object, 1, &value);
        return;
    }
    std::vector<J> values(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) values[i] = ToJava(input[i]);
    ArrayOps<J>::Set(env, object, static_cast<jsize>(count), values.data());
}
template<class T> jobject WriteArray(JNIEnv* env, const T* input, int64_t count) {
    Bytes(env, count, sizeof(T));
    Local output(env, ArrayOps<JavaScalar<T>>::New(env, static_cast<jsize>(count)));
    SetArray(env, static_cast<typename ArrayOps<JavaScalar<T>>::Array>(output.get()), input, count);
    return output.release();
}
inline jobject Utf8String(JNIEnv* env, const char* value) {
    if (!value) return nullptr;
    const size_t size = std::strlen(value); Bytes(env, size, 1);
    Local bytes(env, env->NewByteArray(static_cast<jsize>(size)));
    env->SetByteArrayRegion(static_cast<jbyteArray>(bytes.get()), 0, static_cast<jsize>(size),
                           reinterpret_cast<const jbyte*>(value)); Check(env);
    Local cls(env, env->FindClass("java/lang/String"));
    auto constructor = env->GetMethodID(static_cast<jclass>(cls.get()), "<init>", "([BLjava/lang/String;)V"); Check(env);
    Local encoding(env, env->NewStringUTF("UTF-8"));
    auto result = env->NewObject(static_cast<jclass>(cls.get()), constructor, bytes.get(), encoding.get()); Check(env);
    return result;
}
class Context {
    std::vector<std::shared_ptr<void>> allocations_;
public:
    template<class T> T* array(JNIEnv* env, int64_t count) {
        Bytes(env, count, sizeof(T));
        if (!count) return nullptr;
        std::shared_ptr<T> value(new T[static_cast<size_t>(count)]{}, std::default_delete<T[]>());
        T* result = value.get(); allocations_.push_back(value); return result;
    }
    char* string(JNIEnv* env, jstring string) {
        if (!string) return nullptr;
        Local cls(env, env->FindClass("java/lang/String"));
        auto method = env->GetMethodID(static_cast<jclass>(cls.get()), "getBytes", "(Ljava/lang/String;)[B"); Check(env);
        Local encoding(env, env->NewStringUTF("UTF-8"));
        Local bytes(env, env->CallObjectMethod(string, method, encoding.get())); Check(env);
        jsize length = env->GetArrayLength(static_cast<jarray>(bytes.get())); Check(env);
        char* value = array<char>(env, static_cast<int64_t>(length) + 1);
        env->GetByteArrayRegion(static_cast<jbyteArray>(bytes.get()), 0, length, reinterpret_cast<jbyte*>(value)); Check(env);
        Require(env, std::memchr(value, 0, length) == nullptr, "Embedded NUL in native string");
        return value;
    }
};
inline jclass DescriptorClass(JNIEnv* env, const char* name) {
    std::string path = std::string("com/insightface/sdk/inspireface/jni/NativeTypes$") + name;
    jclass cls = env->FindClass(path.c_str()); Check(env); return cls;
}
inline jobject NewDescriptor(JNIEnv* env, const char* name) {
    Local cls(env, DescriptorClass(env, name));
    auto constructor = env->GetMethodID(static_cast<jclass>(cls.get()), "<init>", "()V"); Check(env);
    jobject result = env->NewObject(static_cast<jclass>(cls.get()), constructor); Check(env); return result;
}
template<class T, class Writer>
void FillStructArray(JNIEnv* env, jobjectArray array, const T* values, int64_t count, Writer writer) {
    RequireArray(env, array, count);
    for (jsize i = 0; i < count; ++i) {
        Local value(env, writer(env, values[i]));
        env->SetObjectArrayElement(array, i, value.get()); Check(env);
    }
}
template<class T, class Writer>
jobject WriteStructArray(JNIEnv* env, const T* values, int64_t count, const char* type, Writer writer) {
    Bytes(env, count, sizeof(T));
    Require(env, values || count == 0, "Native result array is null");
    Local cls(env, DescriptorClass(env, type));
    Local result(env, env->NewObjectArray(static_cast<jsize>(count), static_cast<jclass>(cls.get()), nullptr));
    FillStructArray(env, static_cast<jobjectArray>(result.get()), values, count, writer);
    return result.release();
}
}  // namespace isf_jni
