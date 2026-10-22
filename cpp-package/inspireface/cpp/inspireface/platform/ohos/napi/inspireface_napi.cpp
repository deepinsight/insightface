#include <napi/native_api.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "c_api/inspireface.h"
#include "ohos_napi_contract.h"

namespace {

constexpr napi_type_tag kSessionTypeTag = {0x8e68a46fd3b9460dULL, 0xa37adf709e45b06bULL};
constexpr napi_type_tag kImageStreamTypeTag = {0x20e8d7e8040f43b1ULL, 0xa2d11c912ea59bbfULL};
constexpr napi_type_tag kFaceResultTypeTag = {0xf8ea4a8ad49941ceULL, 0xbeb80ecf8b1752c8ULL};
constexpr napi_type_tag kImageBitmapTypeTag = {0x6621012739bd4ffbULL, 0x95589cfc1170c5a2ULL};
constexpr napi_type_tag kFaceCaptureTypeTag = {0x55e071e5df8d4fa8ULL, 0x871901c0e231550cULL};
std::atomic<uint64_t> g_next_state_identity{1};

struct SessionState {
    std::mutex mutex;
    HFSession handle = nullptr;
    const uint64_t identity = g_next_state_identity.fetch_add(1, std::memory_order_relaxed);

    ~SessionState() {
        std::lock_guard<std::mutex> lock(mutex);
        if (handle) {
            HFReleaseInspireFaceSession(handle);
            handle = nullptr;
        }
    }
};

struct ImageStreamState {
    std::mutex mutex;
    std::vector<uint8_t> bytes;
    HFImageStream handle = nullptr;
    const uint64_t identity = g_next_state_identity.fetch_add(1, std::memory_order_relaxed);

    ~ImageStreamState() {
        std::lock_guard<std::mutex> lock(mutex);
        if (handle) {
            HFReleaseImageStream(handle);
            handle = nullptr;
        }
    }
};

struct FaceResultState {
    std::mutex mutex;
    HFFaceResultSnapshot handle = nullptr;
    uint64_t owner_session_identity = 0;
    uint64_t owner_image_identity = 0;

    ~FaceResultState() {
        std::lock_guard<std::mutex> lock(mutex);
        if (handle) {
            HFReleaseFaceResultSnapshot(handle);
            handle = nullptr;
        }
    }
};

struct ImageBitmapState {
    std::mutex mutex;
    HFImageBitmap handle = nullptr;

    ~ImageBitmapState() {
        std::lock_guard<std::mutex> lock(mutex);
        if (handle) {
            HFReleaseImageBitmap(handle);
            handle = nullptr;
        }
    }
};

struct FaceCaptureState {
    std::mutex mutex;
    HFFaceCaptureSession handle = nullptr;

    ~FaceCaptureState() {
        std::lock_guard<std::mutex> lock(mutex);
        if (handle) {
            HFReleaseFaceCaptureSession(handle);
            handle = nullptr;
        }
    }
};

bool CheckNapi(napi_env env, napi_status status, const char* operation) {
    if (status == napi_ok) {
        return true;
    }
    const napi_extended_error_info* info = nullptr;
    napi_get_last_error_info(env, &info);
    std::string message(operation ? operation : "Node-API operation");
    message += " failed";
    if (info && info->error_message) {
        message += ": ";
        message += info->error_message;
    }
    napi_throw_error(env, "ERR_INSPIREFACE_NAPI", message.c_str());
    return false;
}

napi_value ThrowTypeError(napi_env env, const char* message) {
    napi_throw_type_error(env, "ERR_INSPIREFACE_ARGUMENT", message);
    return nullptr;
}

napi_value ThrowRangeError(napi_env env, const char* message) {
    napi_throw_range_error(env, "ERR_INSPIREFACE_RANGE", message);
    return nullptr;
}

std::string GetSdkErrorMessage(HResult result) {
    HInt32 required = 0;
    if (HFGetErrorMessage(result, nullptr, 0, &required) != HSUCCEED || required <= 0 || required > 4096) {
        return "InspireFace operation failed";
    }
    std::vector<char> buffer(static_cast<size_t>(required), '\0');
    if (HFGetErrorMessage(result, buffer.data(), required, &required) != HSUCCEED) {
        return "InspireFace operation failed";
    }
    return std::string(buffer.data());
}

napi_value ThrowSdkError(napi_env env, HResult result, const char* operation) {
    std::string message(operation ? operation : "InspireFace operation");
    message += " failed: ";
    message += GetSdkErrorMessage(result);
    const std::string code = std::to_string(static_cast<int64_t>(result));
    napi_throw_error(env, code.c_str(), message.c_str());
    return nullptr;
}

template <typename Callable>
napi_value GuardCallback(napi_env env, Callable&& callback) noexcept {
    try {
        return callback();
    } catch (const std::bad_alloc&) {
        napi_throw_error(env, "ERR_INSPIREFACE_NO_MEMORY", "Unable to allocate native bridge memory");
    } catch (const std::exception& error) {
        napi_throw_error(env, "ERR_INSPIREFACE_NATIVE", error.what());
    } catch (...) {
        napi_throw_error(env, "ERR_INSPIREFACE_NATIVE", "Unknown native bridge failure");
    }
    return nullptr;
}

napi_value Undefined(napi_env env) {
    napi_value value = nullptr;
    return CheckNapi(env, napi_get_undefined(env, &value), "get undefined") ? value : nullptr;
}

bool ReadArguments(napi_env env, napi_callback_info info, size_t required, napi_value* arguments) {
    size_t count = required;
    if (!CheckNapi(env, napi_get_cb_info(env, info, &count, arguments, nullptr, nullptr), "read arguments")) {
        return false;
    }
    if (count != required) {
        napi_throw_type_error(env, "ERR_INSPIREFACE_ARGUMENT", "Unexpected number of arguments");
        return false;
    }
    return true;
}

bool ReadString(napi_env env, napi_value value, std::string* result) {
    if (!result) {
        return false;
    }
    size_t length = 0;
    if (!CheckNapi(env, napi_get_value_string_utf8(env, value, nullptr, 0, &length), "read string length")) {
        return false;
    }
    std::vector<char> buffer(length + 1, '\0');
    size_t copied = 0;
    if (!CheckNapi(env, napi_get_value_string_utf8(env, value, buffer.data(), buffer.size(), &copied), "read string")) {
        return false;
    }
    result->assign(buffer.data(), copied);
    return true;
}

bool GetOptionalInt32(napi_env env, napi_value object, const char* name, int32_t* value, bool* present) {
    if (!value || !present) {
        return false;
    }
    *present = false;
    bool has_property = false;
    if (!CheckNapi(env, napi_has_named_property(env, object, name, &has_property), "check numeric option")) {
        return false;
    }
    if (!has_property) {
        return true;
    }
    napi_value property = nullptr;
    if (!CheckNapi(env, napi_get_named_property(env, object, name, &property), "read numeric option") ||
        !CheckNapi(env, napi_get_value_int32(env, property, value), "convert numeric option")) {
        return false;
    }
    *present = true;
    return true;
}

bool GetOptionalInt64(napi_env env, napi_value object, const char* name, int64_t* value, bool* present) {
    if (!value || !present) {
        return false;
    }
    *present = false;
    bool has_property = false;
    if (!CheckNapi(env, napi_has_named_property(env, object, name, &has_property), "check integer option")) {
        return false;
    }
    if (!has_property) {
        return true;
    }
    napi_value property = nullptr;
    if (!CheckNapi(env, napi_get_named_property(env, object, name, &property), "read integer option") ||
        !CheckNapi(env, napi_get_value_int64(env, property, value), "convert integer option")) {
        return false;
    }
    *present = true;
    return true;
}

bool GetOptionalDouble(napi_env env, napi_value object, const char* name, double* value, bool* present) {
    if (!value || !present) {
        return false;
    }
    *present = false;
    bool has_property = false;
    if (!CheckNapi(env, napi_has_named_property(env, object, name, &has_property), "check floating-point option")) {
        return false;
    }
    if (!has_property) {
        return true;
    }
    napi_value property = nullptr;
    if (!CheckNapi(env, napi_get_named_property(env, object, name, &property), "read floating-point option") ||
        !CheckNapi(env, napi_get_value_double(env, property, value), "convert floating-point option")) {
        return false;
    }
    *present = true;
    return true;
}

bool GetOptionalUInt64(napi_env env, napi_value object, const char* name, uint64_t* value, bool* present) {
    double numeric = 0.0;
    if (!value || !present || !GetOptionalDouble(env, object, name, &numeric, present)) return false;
    constexpr double kMaxSafeInteger = 9007199254740991.0;
    if (*present && (!std::isfinite(numeric) || numeric < 0.0 || numeric > kMaxSafeInteger ||
                     std::trunc(numeric) != numeric)) {
        napi_throw_range_error(env, "ERR_INSPIREFACE_RANGE", "Capture timing values must be non-negative safe integers");
        return false;
    }
    if (*present) *value = static_cast<uint64_t>(numeric);
    return true;
}

bool ReadUInt64(napi_env env, napi_value value, uint64_t* output) {
    double numeric = 0.0;
    constexpr double kMaxSafeInteger = 9007199254740991.0;
    if (!output || !CheckNapi(env, napi_get_value_double(env, value, &numeric), "read unsigned integer")) return false;
    if (!std::isfinite(numeric) || numeric < 0.0 || numeric > kMaxSafeInteger || std::trunc(numeric) != numeric) {
        napi_throw_range_error(env, "ERR_INSPIREFACE_RANGE", "Frame ID and timestamp must be non-negative safe integers");
        return false;
    }
    *output = static_cast<uint64_t>(numeric);
    return true;
}

bool GetOptionalBool(napi_env env, napi_value object, const char* name, bool* value, bool* present) {
    if (!value || !present) {
        return false;
    }
    *present = false;
    bool has_property = false;
    if (!CheckNapi(env, napi_has_named_property(env, object, name, &has_property), "check boolean option")) {
        return false;
    }
    if (!has_property) {
        return true;
    }
    napi_value property = nullptr;
    if (!CheckNapi(env, napi_get_named_property(env, object, name, &property), "read boolean option") ||
        !CheckNapi(env, napi_get_value_bool(env, property, value), "convert boolean option")) {
        return false;
    }
    *present = true;
    return true;
}

bool GetOptionalString(napi_env env, napi_value object, const char* name, std::string* value, bool* present) {
    if (!value || !present) {
        return false;
    }
    *present = false;
    bool has_property = false;
    if (!CheckNapi(env, napi_has_named_property(env, object, name, &has_property), "check string option")) {
        return false;
    }
    if (!has_property) {
        return true;
    }
    napi_value property = nullptr;
    if (!CheckNapi(env, napi_get_named_property(env, object, name, &property), "read string option") ||
        !ReadString(env, property, value)) {
        return false;
    }
    *present = true;
    return true;
}

bool GetRequiredInt32(napi_env env, napi_value object, const char* name, int32_t* value) {
    bool present = false;
    if (!GetOptionalInt32(env, object, name, value, &present)) {
        return false;
    }
    if (!present) {
        std::string message("Missing required integer property: ");
        message += name;
        napi_throw_type_error(env, "ERR_INSPIREFACE_ARGUMENT", message.c_str());
        return false;
    }
    return true;
}

bool GetRequiredDouble(napi_env env, napi_value object, const char* name, double* value) {
    bool present = false;
    if (!GetOptionalDouble(env, object, name, value, &present)) {
        return false;
    }
    if (!present) {
        std::string message("Missing required numeric property: ");
        message += name;
        napi_throw_type_error(env, "ERR_INSPIREFACE_ARGUMENT", message.c_str());
        return false;
    }
    return true;
}

template <typename State>
void FinalizeState(napi_env, void* data, void*) {
    delete static_cast<State*>(data);
}

template <typename State>
napi_value WrapState(napi_env env, State* state, const napi_type_tag& type_tag) {
    if (!state) {
        napi_throw_error(env, "ERR_INSPIREFACE_NO_MEMORY", "Unable to allocate native handle");
        return nullptr;
    }
    napi_value object = nullptr;
    if (!CheckNapi(env, napi_create_object(env, &object), "create native handle object") ||
        !CheckNapi(env, napi_type_tag_object(env, object, &type_tag), "tag native handle object") ||
        !CheckNapi(env, napi_wrap(env, object, state, FinalizeState<State>, nullptr, nullptr), "wrap native handle")) {
        delete state;
        return nullptr;
    }
    return object;
}

template <typename State>
State* UnwrapState(napi_env env, napi_value object, const napi_type_tag& type_tag, const char* expected_type) {
    bool tagged = false;
    if (!CheckNapi(env, napi_check_object_type_tag(env, object, &type_tag, &tagged), "check native handle type")) {
        return nullptr;
    }
    if (!tagged) {
        napi_throw_type_error(env, "ERR_INSPIREFACE_HANDLE", expected_type);
        return nullptr;
    }
    void* state = nullptr;
    if (!CheckNapi(env, napi_unwrap(env, object, &state), "unwrap native handle") || !state) {
        return nullptr;
    }
    return static_cast<State*>(state);
}

bool ReadUint8View(napi_env env, napi_value value, const uint8_t** data, size_t* length) {
    napi_typedarray_type type = napi_uint8_array;
    void* raw_data = nullptr;
    napi_value array_buffer = nullptr;
    size_t byte_offset = 0;
    if (!CheckNapi(env, napi_get_typedarray_info(env, value, &type, length, &raw_data, &array_buffer, &byte_offset),
                   "read Uint8Array")) {
        return false;
    }
    if (type != napi_uint8_array && type != napi_uint8_clamped_array) {
        napi_throw_type_error(env, "ERR_INSPIREFACE_ARGUMENT", "Expected Uint8Array or Uint8ClampedArray");
        return false;
    }
    if (*length > 0 && !raw_data) {
        napi_throw_type_error(env, "ERR_INSPIREFACE_ARGUMENT", "Typed array has no backing storage");
        return false;
    }
    *data = static_cast<const uint8_t*>(raw_data);
    return true;
}

bool ReadFloat32View(napi_env env, napi_value value, float** data, size_t* length) {
    napi_typedarray_type type = napi_float32_array;
    void* raw_data = nullptr;
    napi_value array_buffer = nullptr;
    size_t byte_offset = 0;
    if (!CheckNapi(env, napi_get_typedarray_info(env, value, &type, length, &raw_data, &array_buffer, &byte_offset),
                   "read Float32Array")) {
        return false;
    }
    if (type != napi_float32_array) {
        napi_throw_type_error(env, "ERR_INSPIREFACE_ARGUMENT", "Expected Float32Array");
        return false;
    }
    if (*length > 0 && !raw_data) {
        napi_throw_type_error(env, "ERR_INSPIREFACE_ARGUMENT", "Typed array has no backing storage");
        return false;
    }
    *data = static_cast<float*>(raw_data);
    return true;
}

bool ReadFaceId(napi_env env, napi_value value, HFaceId* result) {
    if (!result) {
        return false;
    }
    napi_valuetype type = napi_undefined;
    if (!CheckNapi(env, napi_typeof(env, value, &type), "read face ID type")) {
        return false;
    }
    if (type == napi_bigint) {
        bool lossless = false;
        int64_t id = 0;
        if (!CheckNapi(env, napi_get_value_bigint_int64(env, value, &id, &lossless), "read face ID") || !lossless) {
            napi_throw_range_error(env, "ERR_INSPIREFACE_RANGE", "Face ID is outside the signed 64-bit range");
            return false;
        }
        *result = static_cast<HFaceId>(id);
        return true;
    }
    if (type == napi_number) {
        double numeric_id = 0.0;
        if (!CheckNapi(env, napi_get_value_double(env, value, &numeric_id), "read numeric face ID")) {
            return false;
        }
        constexpr double kMaxSafeInteger = 9007199254740991.0;
        if (!std::isfinite(numeric_id) || std::trunc(numeric_id) != numeric_id ||
            std::abs(numeric_id) > kMaxSafeInteger) {
            napi_throw_range_error(env, "ERR_INSPIREFACE_RANGE", "Numeric face ID must be a safe integer; use bigint for 64-bit IDs");
            return false;
        }
        *result = static_cast<HFaceId>(numeric_id);
        return true;
    }
    napi_throw_type_error(env, "ERR_INSPIREFACE_ARGUMENT", "Face ID must be a bigint or integer number");
    return false;
}

napi_value CreateTypedArray(napi_env env, napi_typedarray_type type, const void* source, size_t element_count,
                            size_t element_size) {
    if (element_count > std::numeric_limits<size_t>::max() / element_size) {
        return ThrowRangeError(env, "Typed array size overflow");
    }
    const size_t byte_count = element_count * element_size;
    void* destination = nullptr;
    napi_value array_buffer = nullptr;
    napi_value typed_array = nullptr;
    if (!CheckNapi(env, napi_create_arraybuffer(env, byte_count, &destination, &array_buffer), "create ArrayBuffer")) {
        return nullptr;
    }
    if (byte_count > 0) {
        if (!source || !destination) {
            return ThrowTypeError(env, "Typed array data is unavailable");
        }
        std::memcpy(destination, source, byte_count);
    }
    if (!CheckNapi(env, napi_create_typedarray(env, type, element_count, array_buffer, 0, &typed_array),
                   "create typed array")) {
        return nullptr;
    }
    return typed_array;
}

bool SetInt32(napi_env env, napi_value object, const char* name, int32_t value) {
    napi_value number = nullptr;
    return CheckNapi(env, napi_create_int32(env, value, &number), "create integer") &&
           CheckNapi(env, napi_set_named_property(env, object, name, number), "set integer property");
}

bool SetDouble(napi_env env, napi_value object, const char* name, double value) {
    napi_value number = nullptr;
    return CheckNapi(env, napi_create_double(env, value, &number), "create number") &&
           CheckNapi(env, napi_set_named_property(env, object, name, number), "set number property");
}

bool SetBool(napi_env env, napi_value object, const char* name, bool value) {
    napi_value boolean = nullptr;
    return CheckNapi(env, napi_get_boolean(env, value, &boolean), "create boolean") &&
           CheckNapi(env, napi_set_named_property(env, object, name, boolean), "set boolean property");
}

bool SetString(napi_env env, napi_value object, const char* name, const char* value) {
    napi_value string = nullptr;
    return CheckNapi(env, napi_create_string_utf8(env, value ? value : "", NAPI_AUTO_LENGTH, &string), "create string") &&
           CheckNapi(env, napi_set_named_property(env, object, name, string), "set string property");
}

napi_value CreateCaptureMetrics(napi_env env, const HFFaceCaptureMetrics& metrics) {
    napi_value output = nullptr;
    if (!CheckNapi(env, napi_create_object(env, &output), "create capture metrics") ||
        !SetDouble(env, output, "availableMetrics", static_cast<double>(metrics.availableMetrics)) ||
        !SetDouble(env, output, "faceWidthRatio", metrics.faceWidthRatio) ||
        !SetDouble(env, output, "centerOffsetX", metrics.centerOffsetX) ||
        !SetDouble(env, output, "centerOffsetY", metrics.centerOffsetY) ||
        !SetDouble(env, output, "stabilityScore", metrics.stabilityScore) ||
        !SetDouble(env, output, "poseScore", metrics.poseScore) ||
        !SetDouble(env, output, "qualityScore", metrics.qualityScore) ||
        !SetDouble(env, output, "sharpnessScore", metrics.sharpnessScore) ||
        !SetDouble(env, output, "brightnessScore", metrics.brightnessScore)) {
        return nullptr;
    }
    return output;
}

napi_value CreateCaptureProgress(napi_env env, const HFFaceCaptureProgress& progress) {
    napi_value output = nullptr;
    napi_value metrics = CreateCaptureMetrics(env, progress.metrics);
    if (!metrics || !CheckNapi(env, napi_create_object(env, &output), "create capture progress") ||
        !SetInt32(env, output, "state", progress.state) ||
        !SetInt32(env, output, "candidateCount", static_cast<int32_t>(progress.candidateCount)) ||
        !SetDouble(env, output, "frameId", static_cast<double>(progress.frameId)) ||
        !SetDouble(env, output, "timestampMs", static_cast<double>(progress.timestampMs)) ||
        !SetInt32(env, output, "trackId", progress.trackId) ||
        !SetInt32(env, output, "trackCount", progress.trackCount) ||
        !SetDouble(env, output, "evaluatedFilters", static_cast<double>(progress.evaluatedFilters)) ||
        !SetDouble(env, output, "rejectReasons", static_cast<double>(progress.rejectReasons)) ||
        !SetDouble(env, output, "progress", progress.progress) ||
        !SetDouble(env, output, "currentScore", progress.currentScore) ||
        !CheckNapi(env, napi_set_named_property(env, output, "metrics", metrics), "set capture metrics")) {
        return nullptr;
    }
    return output;
}

napi_value CreateCaptureConfig(napi_env env, const HFFaceCaptureConfig& config) {
    napi_value output = nullptr;
    if (!CheckNapi(env, napi_create_object(env, &output), "create capture config") ||
        !SetDouble(env, output, "filterMask", static_cast<double>(config.filterMask)) ||
        !SetInt32(env, output, "outputCount", static_cast<int32_t>(config.outputCount)) ||
        !SetInt32(env, output, "minTrackCount", static_cast<int32_t>(config.minTrackCount)) ||
        !SetDouble(env, output, "stableDurationMs", static_cast<double>(config.stableDurationMs)) ||
        !SetDouble(env, output, "collectDurationMs", static_cast<double>(config.collectDurationMs)) ||
        !SetDouble(env, output, "maxCollectDurationMs", static_cast<double>(config.maxCollectDurationMs)) ||
        !SetDouble(env, output, "trackLostGraceMs", static_cast<double>(config.trackLostGraceMs)) ||
        !SetDouble(env, output, "minCandidateIntervalMs", static_cast<double>(config.minCandidateIntervalMs)) ||
        !SetDouble(env, output, "minFaceWidthRatio", config.minFaceWidthRatio) ||
        !SetDouble(env, output, "maxFaceWidthRatio", config.maxFaceWidthRatio) ||
        !SetDouble(env, output, "maxCenterOffsetX", config.maxCenterOffsetX) ||
        !SetDouble(env, output, "maxCenterOffsetY", config.maxCenterOffsetY) ||
        !SetDouble(env, output, "boundaryMarginRatio", config.boundaryMarginRatio) ||
        !SetDouble(env, output, "maxCenterMotionRatio", config.maxCenterMotionRatio) ||
        !SetDouble(env, output, "maxSizeChangeRatio", config.maxSizeChangeRatio) ||
        !SetDouble(env, output, "maxAbsYaw", config.maxAbsYaw) ||
        !SetDouble(env, output, "maxAbsPitch", config.maxAbsPitch) ||
        !SetDouble(env, output, "maxAbsRoll", config.maxAbsRoll) ||
        !SetDouble(env, output, "minQualityScore", config.minQualityScore) ||
        !SetDouble(env, output, "minSharpnessScore", config.minSharpnessScore) ||
        !SetDouble(env, output, "minBrightnessScore", config.minBrightnessScore) ||
        !SetDouble(env, output, "maxBrightnessScore", config.maxBrightnessScore)) {
        return nullptr;
    }
    return output;
}

bool SetFaceId(napi_env env, napi_value object, const char* name, HFaceId value) {
    napi_value bigint = nullptr;
    return CheckNapi(env, napi_create_bigint_int64(env, static_cast<int64_t>(value), &bigint), "create face ID") &&
           CheckNapi(env, napi_set_named_property(env, object, name, bigint), "set face ID property");
}

napi_value CreateFaceIdArray(napi_env env, const HFaceId* values, size_t count) {
    napi_value array = nullptr;
    if (!CheckNapi(env, napi_create_array_with_length(env, count, &array), "create face ID array")) {
        return nullptr;
    }
    for (size_t index = 0; index < count; ++index) {
        napi_value id = nullptr;
        if (!CheckNapi(env, napi_create_bigint_int64(env, static_cast<int64_t>(values[index]), &id), "create face ID") ||
            !CheckNapi(env, napi_set_element(env, array, static_cast<uint32_t>(index), id), "set face ID")) {
            return nullptr;
        }
    }
    return array;
}

bool SetFloatArray(napi_env env, napi_value object, const char* name, const float* values, int32_t count) {
    if (count < 0 || (count > 0 && !values)) {
        napi_throw_error(env, "ERR_INSPIREFACE_NATIVE", "SDK returned an invalid floating-point result array");
        return false;
    }
    napi_value array = CreateTypedArray(env, napi_float32_array, values, static_cast<size_t>(count), sizeof(float));
    return array && CheckNapi(env, napi_set_named_property(env, object, name, array), "set floating-point result array");
}

bool SetInt32Array(napi_env env, napi_value object, const char* name, const int32_t* values, int32_t count) {
    if (count < 0 || (count > 0 && !values)) {
        napi_throw_error(env, "ERR_INSPIREFACE_NATIVE", "SDK returned an invalid integer result array");
        return false;
    }
    napi_value array = CreateTypedArray(env, napi_int32_array, values, static_cast<size_t>(count), sizeof(int32_t));
    return array && CheckNapi(env, napi_set_named_property(env, object, name, array), "set integer result array");
}

template <typename Query>
napi_value QuerySdkString(napi_env env, const char* operation, Query&& query) {
    HInt32 required = 0;
    HResult result = query(nullptr, 0, &required);
    if (result != HSUCCEED) {
        return ThrowSdkError(env, result, operation);
    }
    if (required <= 0 || required > 1024 * 1024) {
        return ThrowRangeError(env, "SDK returned an invalid string size");
    }
    std::vector<char> buffer(static_cast<size_t>(required), '\0');
    result = query(buffer.data(), required, &required);
    if (result != HSUCCEED) {
        return ThrowSdkError(env, result, operation);
    }
    napi_value string = nullptr;
    return CheckNapi(env, napi_create_string_utf8(env, buffer.data(), NAPI_AUTO_LENGTH, &string), "create SDK string")
             ? string
             : nullptr;
}

napi_value Launch(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) {
            return nullptr;
        }
        std::string resource_path;
        if (!ReadString(env, arguments[0], &resource_path) || resource_path.empty()) {
            return resource_path.empty() ? ThrowTypeError(env, "resourcePath must not be empty") : nullptr;
        }
        const HResult result = HFLaunchInspireFace(resource_path.c_str());
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "launch");
    });
}

napi_value Reload(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) {
            return nullptr;
        }
        std::string resource_path;
        if (!ReadString(env, arguments[0], &resource_path) || resource_path.empty()) {
            return resource_path.empty() ? ThrowTypeError(env, "resourcePath must not be empty") : nullptr;
        }
        const HResult result = HFReloadInspireFace(resource_path.c_str());
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "reload");
    });
}

napi_value Terminate(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) {
            return nullptr;
        }
        const HResult result = HFTerminateInspireFace();
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "terminate");
    });
}

napi_value IsLaunched(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) {
            return nullptr;
        }
        HInt32 launched = 0;
        const HResult result = HFQueryInspireFaceLaunchStatus(&launched);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "query launch status");
        }
        napi_value value = nullptr;
        return CheckNapi(env, napi_get_boolean(env, launched != 0, &value), "create launch status") ? value : nullptr;
    });
}

napi_value GetVersion(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) {
            return nullptr;
        }
        HFInspireFaceVersion version{};
        const HResult result = HFQueryInspireFaceVersion(&version);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "query version");
        }
        napi_value object = nullptr;
        if (!CheckNapi(env, napi_create_object(env, &object), "create version object") ||
            !SetInt32(env, object, "major", version.major) || !SetInt32(env, object, "minor", version.minor) ||
            !SetInt32(env, object, "patch", version.patch)) {
            return nullptr;
        }
        return object;
    });
}

napi_value GetCapiLevel(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) {
            return nullptr;
        }
        HFUInt32 api_level = 0;
        const HFStatus result = HFQueryCAPILevel(&api_level);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "query C API level");
        }
        napi_value value = nullptr;
        return CheckNapi(env, napi_create_uint32(env, api_level, &value), "create C API level") ? value : nullptr;
    });
}

napi_value CreateSession(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) {
            return nullptr;
        }

        HFSessionConfigV2 config{};
        config.structSize = sizeof(config);
        config.structVersion = HF_SESSION_CONFIG_V2_VERSION;
        config.featureMask = HF_ENABLE_NONE;
        config.detectMode = HF_DETECT_MODE_ALWAYS_DETECT;
        config.maxDetectFaceNum = 1;
        config.detectPixelLevel = -1;
        config.trackByDetectModeFPS = -1;

        bool present = false;
        int64_t feature_mask = 0;
        if (!GetOptionalInt64(env, arguments[0], "featureMask", &feature_mask, &present)) {
            return nullptr;
        }
        if (present) {
            if (feature_mask < 0) {
                return ThrowRangeError(env, "featureMask must be non-negative");
            }
            config.featureMask = static_cast<HFUInt64>(feature_mask);
        }
        if (!GetOptionalInt32(env, arguments[0], "detectMode", &config.detectMode, &present) ||
            !GetOptionalInt32(env, arguments[0], "maxFaces", &config.maxDetectFaceNum, &present) ||
            !GetOptionalInt32(env, arguments[0], "detectPixelLevel", &config.detectPixelLevel, &present) ||
            !GetOptionalInt32(env, arguments[0], "trackFps", &config.trackByDetectModeFPS, &present)) {
            return nullptr;
        }

        auto* state = new (std::nothrow) SessionState();
        if (!state) {
            napi_throw_error(env, "ERR_INSPIREFACE_NO_MEMORY", "Unable to allocate session wrapper");
            return nullptr;
        }
        const HFStatus result = HFCreateInspireFaceSessionV2(&config, &state->handle);
        if (result != HSUCCEED) {
            delete state;
            return ThrowSdkError(env, result, "create session");
        }
        return WrapState(env, state, kSessionTypeTag);
    });
}

napi_value ReleaseSession(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) {
            return nullptr;
        }
        SessionState* state = UnwrapState<SessionState>(env, arguments[0], kSessionTypeTag, "Expected InspireFace session handle");
        if (!state) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!state->handle) {
            return Undefined(env);
        }
        const HResult result = HFReleaseInspireFaceSession(state->handle);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "release session");
        }
        state->handle = nullptr;
        return Undefined(env);
    });
}

napi_value ConfigureSession(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[2] = {nullptr, nullptr};
        if (!ReadArguments(env, info, 2, arguments)) {
            return nullptr;
        }
        SessionState* state = UnwrapState<SessionState>(env, arguments[0], kSessionTypeTag, "Expected InspireFace session handle");
        if (!state) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!state->handle) {
            return ThrowTypeError(env, "Session has already been released");
        }

        bool present = false;
        int32_t integer = 0;
        double decimal = 0.0;
        bool boolean = false;
        HResult result = HSUCCEED;

#define APPLY_INT_OPTION(property, function_name)                                                        \
    do {                                                                                                \
        if (!GetOptionalInt32(env, arguments[1], property, &integer, &present)) return nullptr;         \
        if (present && (result = function_name(state->handle, integer)) != HSUCCEED)                    \
            return ThrowSdkError(env, result, "configure session: " property);                         \
    } while (false)
#define APPLY_FLOAT_OPTION(property, function_name)                                                      \
    do {                                                                                                \
        if (!GetOptionalDouble(env, arguments[1], property, &decimal, &present)) return nullptr;        \
        if (present && (result = function_name(state->handle, static_cast<HFloat>(decimal))) != HSUCCEED) \
            return ThrowSdkError(env, result, "configure session: " property);                         \
    } while (false)

        APPLY_INT_OPTION("previewSize", HFSessionSetTrackPreviewSize);
        APPLY_INT_OPTION("minimumFaceSize", HFSessionSetFilterMinimumFacePixelSize);
        APPLY_INT_OPTION("smoothCacheFrames", HFSessionSetTrackModeNumSmoothCacheFrame);
        APPLY_INT_OPTION("detectInterval", HFSessionSetTrackModeDetectInterval);
        APPLY_INT_OPTION("landmarkAugmentation", HFSessionSetLandmarkAugmentationNum);
        APPLY_FLOAT_OPTION("detectThreshold", HFSessionSetFaceDetectThreshold);
        APPLY_FLOAT_OPTION("smoothRatio", HFSessionSetTrackModeSmoothRatio);
        APPLY_FLOAT_OPTION("lightTrackThreshold", HFSessionSetLightTrackConfidenceThreshold);
        if (!GetOptionalBool(env, arguments[1], "trackLostRecovery", &boolean, &present)) {
            return nullptr;
        }
        if (present && (result = HFSessionSetTrackLostRecoveryMode(state->handle, boolean ? 1 : 0)) != HSUCCEED) {
            return ThrowSdkError(env, result, "configure session: trackLostRecovery");
        }

#undef APPLY_FLOAT_OPTION
#undef APPLY_INT_OPTION
        return Undefined(env);
    });
}

napi_value ClearTracking(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) {
            return nullptr;
        }
        SessionState* state = UnwrapState<SessionState>(env, arguments[0], kSessionTypeTag, "Expected InspireFace session handle");
        if (!state) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!state->handle) {
            return ThrowTypeError(env, "Session has already been released");
        }
        const HResult result = HFSessionClearTrackingFace(state->handle);
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "clear tracking faces");
    });
}

napi_value CreateImageStream(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
        if (!ReadArguments(env, info, 5, arguments)) {
            return nullptr;
        }
        const uint8_t* source = nullptr;
        size_t source_size = 0;
        int32_t width = 0;
        int32_t height = 0;
        int32_t format_value = 0;
        int32_t rotation_value = 0;
        if (!ReadUint8View(env, arguments[0], &source, &source_size) ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[1], &width), "read image width") ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[2], &height), "read image height") ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[3], &format_value), "read image format") ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[4], &rotation_value), "read image rotation")) {
            return nullptr;
        }
        if (rotation_value < HF_CAMERA_ROTATION_0 || rotation_value > HF_CAMERA_ROTATION_270) {
            return ThrowRangeError(env, "Unsupported image rotation");
        }

        auto* state = new (std::nothrow) ImageStreamState();
        if (!state) {
            napi_throw_error(env, "ERR_INSPIREFACE_NO_MEMORY", "Unable to allocate image stream wrapper");
            return nullptr;
        }
        const auto format = static_cast<HFImageFormat>(format_value);
        if (!inspire::ohos::CopyExactImageBytes(source, source_size, format, width, height, &state->bytes)) {
            delete state;
            return ThrowRangeError(env, "Image byte length, dimensions, or format are invalid");
        }

        HFImageData image_data{};
        image_data.data = state->bytes.data();
        image_data.width = width;
        image_data.height = height;
        image_data.format = format;
        image_data.rotation = static_cast<HFRotation>(rotation_value);
        const HResult result = HFCreateImageStream(&image_data, &state->handle);
        if (result != HSUCCEED) {
            delete state;
            return ThrowSdkError(env, result, "create image stream");
        }
        return WrapState(env, state, kImageStreamTypeTag);
    });
}

napi_value ReleaseImageStream(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) {
            return nullptr;
        }
        ImageStreamState* state =
          UnwrapState<ImageStreamState>(env, arguments[0], kImageStreamTypeTag, "Expected InspireFace image stream handle");
        if (!state) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!state->handle) {
            return Undefined(env);
        }
        const HResult result = HFReleaseImageStream(state->handle);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "release image stream");
        }
        state->handle = nullptr;
        std::vector<uint8_t>().swap(state->bytes);
        return Undefined(env);
    });
}

napi_value UpdateImageStream(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        const uint8_t* source = nullptr;
        size_t source_size = 0;
        int32_t width = 0;
        int32_t height = 0;
        int32_t format_value = 0;
        int32_t rotation_value = 0;
        if (!ReadArguments(env, info, 6, arguments)) return nullptr;
        ImageStreamState* state =
          UnwrapState<ImageStreamState>(env, arguments[0], kImageStreamTypeTag, "Expected InspireFace image stream handle");
        if (!state || !ReadUint8View(env, arguments[1], &source, &source_size) ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[2], &width), "read image width") ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[3], &height), "read image height") ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[4], &format_value), "read image format") ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[5], &rotation_value), "read image rotation")) {
            return nullptr;
        }
        if (rotation_value < HF_CAMERA_ROTATION_0 || rotation_value > HF_CAMERA_ROTATION_270) {
            return ThrowRangeError(env, "Unsupported image rotation");
        }
        std::vector<uint8_t> replacement;
        const auto format = static_cast<HFImageFormat>(format_value);
        if (!inspire::ohos::CopyExactImageBytes(source, source_size, format, width, height, &replacement)) {
            return ThrowRangeError(env, "Image byte length, dimensions, or format are invalid");
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!state->handle) return ThrowTypeError(env, "Image stream has already been released");
        state->bytes.swap(replacement);
        HResult result = HFImageStreamSetBuffer(state->handle, state->bytes.data(), width, height);
        if (result != HSUCCEED) {
            state->bytes.swap(replacement);
            return ThrowSdkError(env, result, "update image stream buffer");
        }
        if (result == HSUCCEED) result = HFImageStreamSetFormat(state->handle, format);
        if (result == HSUCCEED) result = HFImageStreamSetRotation(state->handle, static_cast<HFRotation>(rotation_value));
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "update image stream");
    });
}

napi_value CreateImageBitmap(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[4] = {nullptr, nullptr, nullptr, nullptr};
        const uint8_t* source = nullptr;
        size_t source_size = 0;
        int32_t width = 0;
        int32_t height = 0;
        int32_t channels = 0;
        if (!ReadArguments(env, info, 4, arguments) || !ReadUint8View(env, arguments[0], &source, &source_size) ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[1], &width), "read bitmap width") ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[2], &height), "read bitmap height") ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[3], &channels), "read bitmap channels")) {
            return nullptr;
        }
        if (width <= 0 || height <= 0 || (channels != 1 && channels != 3)) {
            return ThrowRangeError(env, "Bitmap dimensions or channels are invalid");
        }
        const size_t safe_width = static_cast<size_t>(width);
        const size_t safe_height = static_cast<size_t>(height);
        if (safe_width > std::numeric_limits<size_t>::max() / safe_height ||
            safe_width * safe_height > std::numeric_limits<size_t>::max() / static_cast<size_t>(channels) ||
            safe_width * safe_height * static_cast<size_t>(channels) != source_size) {
            return ThrowRangeError(env, "Bitmap byte length does not match its dimensions");
        }
        HFImageBitmapData data{const_cast<uint8_t*>(source), width, height, channels};
        auto* state = new (std::nothrow) ImageBitmapState();
        if (!state) {
            napi_throw_error(env, "ERR_INSPIREFACE_NO_MEMORY", "Unable to allocate image bitmap wrapper");
            return nullptr;
        }
        const HResult result = HFCreateImageBitmap(&data, &state->handle);
        if (result != HSUCCEED) {
            delete state;
            return ThrowSdkError(env, result, "create image bitmap");
        }
        return WrapState(env, state, kImageBitmapTypeTag);
    });
}

napi_value CreateImageBitmapFromFile(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[2] = {nullptr, nullptr};
        std::string path;
        int32_t channels = 0;
        if (!ReadArguments(env, info, 2, arguments) || !ReadString(env, arguments[0], &path) ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[1], &channels), "read bitmap channels")) {
            return nullptr;
        }
        auto* state = new (std::nothrow) ImageBitmapState();
        if (!state) {
            napi_throw_error(env, "ERR_INSPIREFACE_NO_MEMORY", "Unable to allocate image bitmap wrapper");
            return nullptr;
        }
        const HResult result = HFCreateImageBitmapFromFilePath(path.c_str(), channels, &state->handle);
        if (result != HSUCCEED) {
            delete state;
            return ThrowSdkError(env, result, "create image bitmap from file");
        }
        return WrapState(env, state, kImageBitmapTypeTag);
    });
}

napi_value CopyImageBitmap(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) return nullptr;
        ImageBitmapState* source =
          UnwrapState<ImageBitmapState>(env, arguments[0], kImageBitmapTypeTag, "Expected InspireFace image bitmap handle");
        if (!source) return nullptr;
        std::lock_guard<std::mutex> lock(source->mutex);
        if (!source->handle) return ThrowTypeError(env, "Image bitmap has already been released");
        auto* copy = new (std::nothrow) ImageBitmapState();
        if (!copy) {
            napi_throw_error(env, "ERR_INSPIREFACE_NO_MEMORY", "Unable to allocate image bitmap wrapper");
            return nullptr;
        }
        const HResult result = HFImageBitmapCopy(source->handle, &copy->handle);
        if (result != HSUCCEED) {
            delete copy;
            return ThrowSdkError(env, result, "copy image bitmap");
        }
        return WrapState(env, copy, kImageBitmapTypeTag);
    });
}

napi_value ReleaseImageBitmap(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) return nullptr;
        ImageBitmapState* state =
          UnwrapState<ImageBitmapState>(env, arguments[0], kImageBitmapTypeTag, "Expected InspireFace image bitmap handle");
        if (!state) return nullptr;
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!state->handle) return Undefined(env);
        const HResult result = HFReleaseImageBitmap(state->handle);
        if (result != HSUCCEED) return ThrowSdkError(env, result, "release image bitmap");
        state->handle = nullptr;
        return Undefined(env);
    });
}

napi_value CreateImageStreamFromBitmap(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[2] = {nullptr, nullptr};
        int32_t rotation = 0;
        if (!ReadArguments(env, info, 2, arguments) ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[1], &rotation), "read image rotation")) {
            return nullptr;
        }
        ImageBitmapState* bitmap =
          UnwrapState<ImageBitmapState>(env, arguments[0], kImageBitmapTypeTag, "Expected InspireFace image bitmap handle");
        if (!bitmap) return nullptr;
        std::lock_guard<std::mutex> lock(bitmap->mutex);
        if (!bitmap->handle) return ThrowTypeError(env, "Image bitmap has already been released");
        auto* stream = new (std::nothrow) ImageStreamState();
        if (!stream) {
            napi_throw_error(env, "ERR_INSPIREFACE_NO_MEMORY", "Unable to allocate image stream wrapper");
            return nullptr;
        }
        const HResult result = HFCreateImageStreamFromImageBitmap(bitmap->handle, static_cast<HFRotation>(rotation),
                                                                  &stream->handle);
        if (result != HSUCCEED) {
            delete stream;
            return ThrowSdkError(env, result, "create image stream from bitmap");
        }
        return WrapState(env, stream, kImageStreamTypeTag);
    });
}

napi_value CreateImageBitmapFromStream(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[3] = {nullptr, nullptr, nullptr};
        int32_t rotate = 0;
        double scale = 1.0;
        if (!ReadArguments(env, info, 3, arguments) ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[1], &rotate), "read bitmap rotation option") ||
            !CheckNapi(env, napi_get_value_double(env, arguments[2], &scale), "read bitmap scale")) {
            return nullptr;
        }
        ImageStreamState* stream =
          UnwrapState<ImageStreamState>(env, arguments[0], kImageStreamTypeTag, "Expected InspireFace image stream handle");
        if (!stream) return nullptr;
        std::lock_guard<std::mutex> lock(stream->mutex);
        if (!stream->handle) return ThrowTypeError(env, "Image stream has already been released");
        auto* bitmap = new (std::nothrow) ImageBitmapState();
        if (!bitmap) {
            napi_throw_error(env, "ERR_INSPIREFACE_NO_MEMORY", "Unable to allocate image bitmap wrapper");
            return nullptr;
        }
        const HResult result = HFCreateImageBitmapFromImageStreamProcess(stream->handle, &bitmap->handle, rotate,
                                                                         static_cast<HFloat>(scale));
        if (result != HSUCCEED) {
            delete bitmap;
            return ThrowSdkError(env, result, "create image bitmap from stream");
        }
        return WrapState(env, bitmap, kImageBitmapTypeTag);
    });
}

napi_value GetImageBitmapData(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) return nullptr;
        ImageBitmapState* bitmap =
          UnwrapState<ImageBitmapState>(env, arguments[0], kImageBitmapTypeTag, "Expected InspireFace image bitmap handle");
        if (!bitmap) return nullptr;
        std::lock_guard<std::mutex> lock(bitmap->mutex);
        if (!bitmap->handle) return ThrowTypeError(env, "Image bitmap has already been released");
        HFImageBitmapData data{};
        const HResult result = HFImageBitmapGetData(bitmap->handle, &data);
        if (result != HSUCCEED) return ThrowSdkError(env, result, "get image bitmap data");
        if (data.width <= 0 || data.height <= 0 || data.channels <= 0) {
            return ThrowSdkError(env, HERR_INVALID_PARAM, "get image bitmap data");
        }
        const size_t width = static_cast<size_t>(data.width);
        const size_t height = static_cast<size_t>(data.height);
        const size_t channels = static_cast<size_t>(data.channels);
        if (width > std::numeric_limits<size_t>::max() / height || width * height > std::numeric_limits<size_t>::max() / channels) {
            return ThrowRangeError(env, "Bitmap size overflow");
        }
        napi_value output = nullptr;
        napi_value bytes = CreateTypedArray(env, napi_uint8_array, data.data, width * height * channels, sizeof(uint8_t));
        if (!bytes || !CheckNapi(env, napi_create_object(env, &output), "create bitmap data") ||
            !SetInt32(env, output, "width", data.width) || !SetInt32(env, output, "height", data.height) ||
            !SetInt32(env, output, "channels", data.channels) ||
            !CheckNapi(env, napi_set_named_property(env, output, "data", bytes), "set bitmap bytes")) {
            return nullptr;
        }
        return output;
    });
}

napi_value WriteImageBitmap(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[2] = {nullptr, nullptr};
        std::string path;
        if (!ReadArguments(env, info, 2, arguments) || !ReadString(env, arguments[1], &path)) return nullptr;
        ImageBitmapState* bitmap =
          UnwrapState<ImageBitmapState>(env, arguments[0], kImageBitmapTypeTag, "Expected InspireFace image bitmap handle");
        if (!bitmap) return nullptr;
        std::lock_guard<std::mutex> lock(bitmap->mutex);
        if (!bitmap->handle) return ThrowTypeError(env, "Image bitmap has already been released");
        const HResult result = HFImageBitmapWriteToFile(bitmap->handle, path.c_str());
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "write image bitmap");
    });
}

bool ReadColor(napi_env env, napi_value value, HColor* color) {
    double r = 0.0;
    double g = 0.0;
    double b = 0.0;
    if (!color || !GetRequiredDouble(env, value, "r", &r) || !GetRequiredDouble(env, value, "g", &g) ||
        !GetRequiredDouble(env, value, "b", &b)) {
        return false;
    }
    *color = {static_cast<float>(r), static_cast<float>(g), static_cast<float>(b)};
    return true;
}

napi_value DrawImageBitmapRect(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[4] = {nullptr, nullptr, nullptr, nullptr};
        HFaceRect rect{};
        HColor color{};
        int32_t thickness = 0;
        if (!ReadArguments(env, info, 4, arguments) || !GetRequiredInt32(env, arguments[1], "x", &rect.x) ||
            !GetRequiredInt32(env, arguments[1], "y", &rect.y) ||
            !GetRequiredInt32(env, arguments[1], "width", &rect.width) ||
            !GetRequiredInt32(env, arguments[1], "height", &rect.height) || !ReadColor(env, arguments[2], &color) ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[3], &thickness), "read rectangle thickness")) {
            return nullptr;
        }
        ImageBitmapState* bitmap =
          UnwrapState<ImageBitmapState>(env, arguments[0], kImageBitmapTypeTag, "Expected InspireFace image bitmap handle");
        if (!bitmap) return nullptr;
        std::lock_guard<std::mutex> lock(bitmap->mutex);
        if (!bitmap->handle) return ThrowTypeError(env, "Image bitmap has already been released");
        const HResult result = HFImageBitmapDrawRect(bitmap->handle, rect, color, thickness);
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "draw bitmap rectangle");
    });
}

napi_value DrawImageBitmapCircle(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
        HPoint2f point{};
        HColor color{};
        double x = 0.0;
        double y = 0.0;
        int32_t radius = 0;
        int32_t thickness = 0;
        if (!ReadArguments(env, info, 5, arguments) || !GetRequiredDouble(env, arguments[1], "x", &x) ||
            !GetRequiredDouble(env, arguments[1], "y", &y) ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[2], &radius), "read circle radius") ||
            !ReadColor(env, arguments[3], &color) ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[4], &thickness), "read circle thickness")) {
            return nullptr;
        }
        point = {static_cast<float>(x), static_cast<float>(y)};
        ImageBitmapState* bitmap =
          UnwrapState<ImageBitmapState>(env, arguments[0], kImageBitmapTypeTag, "Expected InspireFace image bitmap handle");
        if (!bitmap) return nullptr;
        std::lock_guard<std::mutex> lock(bitmap->mutex);
        if (!bitmap->handle) return ThrowTypeError(env, "Image bitmap has already been released");
        const HResult result = HFImageBitmapDrawCircleF(bitmap->handle, point, radius, color, thickness);
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "draw bitmap circle");
    });
}

napi_value ShowImageBitmap(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[3] = {nullptr, nullptr, nullptr};
        std::string title;
        int32_t delay = 0;
        if (!ReadArguments(env, info, 3, arguments) || !ReadString(env, arguments[1], &title) ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[2], &delay), "read bitmap display delay")) return nullptr;
        ImageBitmapState* bitmap =
          UnwrapState<ImageBitmapState>(env, arguments[0], kImageBitmapTypeTag, "Expected InspireFace image bitmap handle");
        if (!bitmap) return nullptr;
        std::lock_guard<std::mutex> lock(bitmap->mutex);
        if (!bitmap->handle) return ThrowTypeError(env, "Image bitmap has already been released");
        const HResult result = HFImageBitmapShow(bitmap->handle, const_cast<char*>(title.c_str()), delay);
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "show image bitmap");
    });
}

napi_value Track(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[2] = {nullptr, nullptr};
        if (!ReadArguments(env, info, 2, arguments)) {
            return nullptr;
        }
        SessionState* session = UnwrapState<SessionState>(env, arguments[0], kSessionTypeTag, "Expected InspireFace session handle");
        ImageStreamState* image =
          UnwrapState<ImageStreamState>(env, arguments[1], kImageStreamTypeTag, "Expected InspireFace image stream handle");
        if (!session || !image) {
            return nullptr;
        }

        auto* snapshot = new (std::nothrow) FaceResultState();
        if (!snapshot) {
            napi_throw_error(env, "ERR_INSPIREFACE_NO_MEMORY", "Unable to allocate face result wrapper");
            return nullptr;
        }
        {
            std::unique_lock<std::mutex> session_lock(session->mutex, std::defer_lock);
            std::unique_lock<std::mutex> image_lock(image->mutex, std::defer_lock);
            std::lock(session_lock, image_lock);
            if (!session->handle || !image->handle) {
                delete snapshot;
                return ThrowTypeError(env, "Session or image stream has already been released");
            }
            const HResult result = HFExecuteFaceTrackSnapshot(session->handle, image->handle, &snapshot->handle);
            if (result != HSUCCEED) {
                delete snapshot;
                return ThrowSdkError(env, result, "execute face track");
            }
            snapshot->owner_session_identity = session->identity;
            snapshot->owner_image_identity = image->identity;
        }

        HFMultipleFaceData faces{};
        HResult result = HFGetFaceResultSnapshotData(snapshot->handle, &faces);
        if (result != HSUCCEED) {
            delete snapshot;
            return ThrowSdkError(env, result, "read face track snapshot");
        }
        if (faces.detectedNum < 0) {
            delete snapshot;
            return ThrowSdkError(env, HERR_INVALID_FACE_LIST, "read face track snapshot");
        }
        if (faces.detectedNum > 0 &&
            (!faces.rects || !faces.trackIds || !faces.trackCounts || !faces.detConfidence || !faces.angles.roll ||
             !faces.angles.yaw || !faces.angles.pitch || !faces.tokens)) {
            delete snapshot;
            return ThrowSdkError(env, HERR_INVALID_FACE_LIST, "read face track snapshot");
        }

        napi_value output = nullptr;
        napi_value face_array = nullptr;
        napi_value native_handle = WrapState(env, snapshot, kFaceResultTypeTag);
        if (!native_handle) {
            return nullptr;
        }
        if (!CheckNapi(env, napi_create_object(env, &output), "create track result") ||
            !SetInt32(env, output, "detectedNum", faces.detectedNum) ||
            !CheckNapi(env, napi_set_named_property(env, output, "handle", native_handle), "set face result handle") ||
            !CheckNapi(env, napi_create_array_with_length(env, static_cast<size_t>(faces.detectedNum), &face_array),
                       "create face array")) {
            return nullptr;
        }

        for (int32_t index = 0; index < faces.detectedNum; ++index) {
            if (faces.tokens[index].size < 0 || (faces.tokens[index].size > 0 && !faces.tokens[index].data)) {
                return ThrowSdkError(env, HERR_INVALID_FACE_TOKEN, "read face token snapshot");
            }
            napi_value face = nullptr;
            napi_value rect = nullptr;
            napi_value token = CreateTypedArray(env, napi_uint8_array, faces.tokens[index].data,
                                                static_cast<size_t>(faces.tokens[index].size), sizeof(uint8_t));
            if (!token || !CheckNapi(env, napi_create_object(env, &face), "create face result") ||
                !CheckNapi(env, napi_create_object(env, &rect), "create face rectangle") ||
                !SetInt32(env, rect, "x", faces.rects[index].x) || !SetInt32(env, rect, "y", faces.rects[index].y) ||
                !SetInt32(env, rect, "width", faces.rects[index].width) ||
                !SetInt32(env, rect, "height", faces.rects[index].height) ||
                !CheckNapi(env, napi_set_named_property(env, face, "rect", rect), "set face rectangle") ||
                !SetInt32(env, face, "trackId", faces.trackIds[index]) ||
                !SetInt32(env, face, "trackCount", faces.trackCounts[index]) ||
                !SetDouble(env, face, "confidence", faces.detConfidence[index]) ||
                !SetDouble(env, face, "roll", faces.angles.roll[index]) ||
                !SetDouble(env, face, "yaw", faces.angles.yaw[index]) ||
                !SetDouble(env, face, "pitch", faces.angles.pitch[index]) ||
                !CheckNapi(env, napi_set_named_property(env, face, "token", token), "set face token") ||
                !CheckNapi(env, napi_set_element(env, face_array, static_cast<uint32_t>(index), face), "set face result")) {
                return nullptr;
            }
        }
        if (!CheckNapi(env, napi_set_named_property(env, output, "faces", face_array), "set face array")) {
            return nullptr;
        }
        return output;
    });
}

napi_value ReleaseFaceResult(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) {
            return nullptr;
        }
        FaceResultState* state =
          UnwrapState<FaceResultState>(env, arguments[0], kFaceResultTypeTag, "Expected InspireFace face result handle");
        if (!state) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!state->handle) {
            return Undefined(env);
        }
        const HResult result = HFReleaseFaceResultSnapshot(state->handle);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "release face result");
        }
        state->handle = nullptr;
        return Undefined(env);
    });
}

napi_value NapiGetDefaultFaceCaptureConfig(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) return nullptr;
        HFFaceCaptureConfig config{};
        const HResult result = HFGetDefaultFaceCaptureConfig(&config);
        return result == HSUCCEED ? CreateCaptureConfig(env, config)
                                  : ThrowSdkError(env, result, "get default face capture config");
    });
}

napi_value CreateFaceCaptureSession(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[2] = {nullptr, nullptr};
        if (!ReadArguments(env, info, 2, arguments)) return nullptr;
        SessionState* session =
          UnwrapState<SessionState>(env, arguments[0], kSessionTypeTag, "Expected InspireFace session handle");
        if (!session) return nullptr;

        HFFaceCaptureConfig config{};
        HResult result = HFGetDefaultFaceCaptureConfig(&config);
        if (result != HSUCCEED) return ThrowSdkError(env, result, "get default face capture config");
        bool present = false;
        int64_t filterMask = 0;
        int32_t outputCount = 0;
        int32_t minTrackCount = 0;
        if (!GetOptionalInt64(env, arguments[1], "filterMask", &filterMask, &present)) return nullptr;
        if (present) {
            if (filterMask < 0) return ThrowRangeError(env, "filterMask must be non-negative");
            config.filterMask = static_cast<HFUInt64>(filterMask);
        }
        if (!GetOptionalInt32(env, arguments[1], "outputCount", &outputCount, &present)) return nullptr;
        if (present) {
            if (outputCount < 0) return ThrowRangeError(env, "outputCount must be non-negative");
            config.outputCount = static_cast<HFUInt32>(outputCount);
        }
        if (!GetOptionalInt32(env, arguments[1], "minTrackCount", &minTrackCount, &present)) return nullptr;
        if (present) {
            if (minTrackCount < 0) return ThrowRangeError(env, "minTrackCount must be non-negative");
            config.minTrackCount = static_cast<HFUInt32>(minTrackCount);
        }
        if (!GetOptionalUInt64(env, arguments[1], "stableDurationMs", &config.stableDurationMs, &present) ||
            !GetOptionalUInt64(env, arguments[1], "collectDurationMs", &config.collectDurationMs, &present) ||
            !GetOptionalUInt64(env, arguments[1], "maxCollectDurationMs", &config.maxCollectDurationMs, &present) ||
            !GetOptionalUInt64(env, arguments[1], "trackLostGraceMs", &config.trackLostGraceMs, &present) ||
            !GetOptionalUInt64(env, arguments[1], "minCandidateIntervalMs", &config.minCandidateIntervalMs, &present)) {
            return nullptr;
        }
#define READ_CAPTURE_FLOAT(name, field)                                                        \
        do {                                                                                   \
            double numeric = 0.0;                                                              \
            if (!GetOptionalDouble(env, arguments[1], name, &numeric, &present)) return nullptr; \
            if (present) config.field = static_cast<HFloat>(numeric);                          \
        } while (false)
        READ_CAPTURE_FLOAT("minFaceWidthRatio", minFaceWidthRatio);
        READ_CAPTURE_FLOAT("maxFaceWidthRatio", maxFaceWidthRatio);
        READ_CAPTURE_FLOAT("maxCenterOffsetX", maxCenterOffsetX);
        READ_CAPTURE_FLOAT("maxCenterOffsetY", maxCenterOffsetY);
        READ_CAPTURE_FLOAT("boundaryMarginRatio", boundaryMarginRatio);
        READ_CAPTURE_FLOAT("maxCenterMotionRatio", maxCenterMotionRatio);
        READ_CAPTURE_FLOAT("maxSizeChangeRatio", maxSizeChangeRatio);
        READ_CAPTURE_FLOAT("maxAbsYaw", maxAbsYaw);
        READ_CAPTURE_FLOAT("maxAbsPitch", maxAbsPitch);
        READ_CAPTURE_FLOAT("maxAbsRoll", maxAbsRoll);
        READ_CAPTURE_FLOAT("minQualityScore", minQualityScore);
        READ_CAPTURE_FLOAT("minSharpnessScore", minSharpnessScore);
        READ_CAPTURE_FLOAT("minBrightnessScore", minBrightnessScore);
        READ_CAPTURE_FLOAT("maxBrightnessScore", maxBrightnessScore);
#undef READ_CAPTURE_FLOAT

        auto* capture = new (std::nothrow) FaceCaptureState();
        if (!capture) {
            napi_throw_error(env, "ERR_INSPIREFACE_NO_MEMORY", "Unable to allocate face capture wrapper");
            return nullptr;
        }
        {
            std::lock_guard<std::mutex> lock(session->mutex);
            if (!session->handle) {
                delete capture;
                return ThrowTypeError(env, "Session has already been released");
            }
            result = HFCreateFaceCaptureSession(session->handle, &config, &capture->handle);
        }
        if (result != HSUCCEED) {
            delete capture;
            return ThrowSdkError(env, result, "create face capture session");
        }
        return WrapState(env, capture, kFaceCaptureTypeTag);
    });
}

napi_value UpdateFaceCaptureSession(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
        if (!ReadArguments(env, info, 5, arguments)) return nullptr;
        FaceCaptureState* capture = UnwrapState<FaceCaptureState>(
          env, arguments[0], kFaceCaptureTypeTag, "Expected InspireFace face capture handle");
        ImageStreamState* image = UnwrapState<ImageStreamState>(
          env, arguments[1], kImageStreamTypeTag, "Expected InspireFace image stream handle");
        if (!capture || !image) return nullptr;
        uint64_t frameId = 0;
        uint64_t timestampMs = 0;
        if (!ReadUInt64(env, arguments[3], &frameId) || !ReadUInt64(env, arguments[4], &timestampMs)) return nullptr;
        napi_valuetype resultType = napi_undefined;
        if (!CheckNapi(env, napi_typeof(env, arguments[2], &resultType), "read optional face result type")) return nullptr;
        FaceResultState* faceResult = nullptr;
        if (resultType != napi_null && resultType != napi_undefined) {
            faceResult = UnwrapState<FaceResultState>(
              env, arguments[2], kFaceResultTypeTag, "Expected InspireFace face result handle or null");
            if (!faceResult) return nullptr;
        }

        HFFaceCaptureProgress progress{};
        HResult result = HERR_UNKNOWN;
        if (faceResult) {
            std::unique_lock<std::mutex> captureLock(capture->mutex, std::defer_lock);
            std::unique_lock<std::mutex> imageLock(image->mutex, std::defer_lock);
            std::unique_lock<std::mutex> resultLock(faceResult->mutex, std::defer_lock);
            std::lock(captureLock, imageLock, resultLock);
            if (!capture->handle || !image->handle || !faceResult->handle) {
                return ThrowTypeError(env, "Capture session, image stream, or face result has already been released");
            }
            result = HFUpdateFaceCaptureSessionWithSnapshot(capture->handle, image->handle, faceResult->handle,
                                                            frameId, timestampMs, &progress);
        } else {
            std::unique_lock<std::mutex> captureLock(capture->mutex, std::defer_lock);
            std::unique_lock<std::mutex> imageLock(image->mutex, std::defer_lock);
            std::lock(captureLock, imageLock);
            if (!capture->handle || !image->handle) {
                return ThrowTypeError(env, "Capture session or image stream has already been released");
            }
            result = HFUpdateFaceCaptureSession(capture->handle, image->handle, frameId, timestampMs, &progress);
        }
        return result == HSUCCEED ? CreateCaptureProgress(env, progress)
                                  : ThrowSdkError(env, result, "update face capture session");
    });
}

napi_value GetFaceCaptureResults(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) return nullptr;
        FaceCaptureState* capture = UnwrapState<FaceCaptureState>(
          env, arguments[0], kFaceCaptureTypeTag, "Expected InspireFace face capture handle");
        if (!capture) return nullptr;
        std::lock_guard<std::mutex> lock(capture->mutex);
        if (!capture->handle) return ThrowTypeError(env, "Face capture session has already been released");
        HFUInt32 count = 0;
        HResult result = HFGetFaceCaptureResults(capture->handle, nullptr, 0, &count);
        if (result != HSUCCEED) return ThrowSdkError(env, result, "query face capture results");
        std::vector<HFFaceCaptureResult> values(count);
        if (count > 0) {
            result = HFGetFaceCaptureResults(capture->handle, values.data(), count, &count);
            if (result != HSUCCEED) return ThrowSdkError(env, result, "get face capture results");
        }
        napi_value array = nullptr;
        if (!CheckNapi(env, napi_create_array_with_length(env, count, &array), "create face capture result array")) return nullptr;
        for (HFUInt32 index = 0; index < count; ++index) {
            const auto& value = values[index];
            napi_value item = nullptr;
            napi_value rect = nullptr;
            napi_value metrics = CreateCaptureMetrics(env, value.metrics);
            napi_value token = CreateTypedArray(env, napi_uint8_array, value.token.data,
                                                static_cast<size_t>(value.token.size), sizeof(uint8_t));
            if (!metrics || !token || !CheckNapi(env, napi_create_object(env, &item), "create face capture result") ||
                !CheckNapi(env, napi_create_object(env, &rect), "create face capture rectangle") ||
                !SetInt32(env, rect, "x", value.rect.x) || !SetInt32(env, rect, "y", value.rect.y) ||
                !SetInt32(env, rect, "width", value.rect.width) || !SetInt32(env, rect, "height", value.rect.height) ||
                !SetDouble(env, item, "frameId", static_cast<double>(value.frameId)) ||
                !SetDouble(env, item, "timestampMs", static_cast<double>(value.timestampMs)) ||
                !SetInt32(env, item, "trackId", value.trackId) || !SetInt32(env, item, "trackCount", value.trackCount) ||
                !SetDouble(env, item, "score", value.score) || !SetDouble(env, item, "roll", value.roll) ||
                !SetDouble(env, item, "yaw", value.yaw) || !SetDouble(env, item, "pitch", value.pitch) ||
                !CheckNapi(env, napi_set_named_property(env, item, "rect", rect), "set face capture rectangle") ||
                !CheckNapi(env, napi_set_named_property(env, item, "token", token), "set face capture token") ||
                !CheckNapi(env, napi_set_named_property(env, item, "metrics", metrics), "set face capture metrics") ||
                !CheckNapi(env, napi_set_element(env, array, index, item), "set face capture result")) {
                return nullptr;
            }
        }
        return array;
    });
}

napi_value FinishFaceCaptureSession(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) return nullptr;
        FaceCaptureState* capture = UnwrapState<FaceCaptureState>(
          env, arguments[0], kFaceCaptureTypeTag, "Expected InspireFace face capture handle");
        if (!capture) return nullptr;
        std::lock_guard<std::mutex> lock(capture->mutex);
        if (!capture->handle) return ThrowTypeError(env, "Face capture session has already been released");
        HFFaceCaptureProgress progress{};
        const HResult result = HFFinishFaceCaptureSession(capture->handle, &progress);
        return result == HSUCCEED ? CreateCaptureProgress(env, progress)
                                  : ThrowSdkError(env, result, "finish face capture session");
    });
}

napi_value ResetFaceCaptureSession(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) return nullptr;
        FaceCaptureState* capture = UnwrapState<FaceCaptureState>(
          env, arguments[0], kFaceCaptureTypeTag, "Expected InspireFace face capture handle");
        if (!capture) return nullptr;
        std::lock_guard<std::mutex> lock(capture->mutex);
        if (!capture->handle) return ThrowTypeError(env, "Face capture session has already been released");
        const HResult result = HFResetFaceCaptureSession(capture->handle);
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "reset face capture session");
    });
}

napi_value ReleaseFaceCaptureSession(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) return nullptr;
        FaceCaptureState* capture = UnwrapState<FaceCaptureState>(
          env, arguments[0], kFaceCaptureTypeTag, "Expected InspireFace face capture handle");
        if (!capture) return nullptr;
        std::lock_guard<std::mutex> lock(capture->mutex);
        if (!capture->handle) return Undefined(env);
        const HResult result = HFReleaseFaceCaptureSession(capture->handle);
        if (result != HSUCCEED) return ThrowSdkError(env, result, "release face capture session");
        capture->handle = nullptr;
        return Undefined(env);
    });
}

napi_value ProcessPipeline(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[4] = {nullptr, nullptr, nullptr, nullptr};
        if (!ReadArguments(env, info, 4, arguments)) {
            return nullptr;
        }
        SessionState* session =
          UnwrapState<SessionState>(env, arguments[0], kSessionTypeTag, "Expected InspireFace session handle");
        ImageStreamState* image =
          UnwrapState<ImageStreamState>(env, arguments[1], kImageStreamTypeTag, "Expected InspireFace image stream handle");
        FaceResultState* face_result =
          UnwrapState<FaceResultState>(env, arguments[2], kFaceResultTypeTag, "Expected InspireFace face result handle");
        int64_t option = 0;
        if (!session || !image || !face_result ||
            !CheckNapi(env, napi_get_value_int64(env, arguments[3], &option), "read pipeline feature mask")) {
            return nullptr;
        }
        if (option < 0) {
            return ThrowRangeError(env, "Pipeline feature mask must be non-negative");
        }
        if (face_result->owner_session_identity != session->identity ||
            face_result->owner_image_identity != image->identity) {
            return ThrowTypeError(env, "Face result must be processed with the session and image that created it");
        }

        std::unique_lock<std::mutex> session_lock(session->mutex, std::defer_lock);
        std::unique_lock<std::mutex> image_lock(image->mutex, std::defer_lock);
        std::unique_lock<std::mutex> face_lock(face_result->mutex, std::defer_lock);
        std::lock(session_lock, image_lock, face_lock);
        if (!session->handle || !image->handle || !face_result->handle) {
            return ThrowTypeError(env, "Session, image stream, or face result has already been released");
        }

        HFMultipleFaceData faces{};
        HResult result = HFGetFaceResultSnapshotData(face_result->handle, &faces);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "read pipeline face result");
        }
        result = HFMultipleFacePipelineProcessOptional(session->handle, image->handle, &faces,
                                                       static_cast<HOption>(option));
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "process face pipeline");
        }

        HFRGBLivenessConfidence liveness{};
        HFFaceMaskConfidence mask{};
        HFFaceQualityConfidence quality{};
        HFFaceInteractionState interaction{};
        HFFaceInteractionsActions actions{};
        HFFaceAttributeResult attributes{};
        HFFaceEmotionResult emotion{};
#define READ_PIPELINE_RESULT(call, output_name)                                      \
    do {                                                                             \
        result = (call);                                                              \
        if (result != HSUCCEED) return ThrowSdkError(env, result, (output_name));    \
    } while (false)
        READ_PIPELINE_RESULT(HFGetRGBLivenessConfidence(session->handle, &liveness), "read RGB liveness result");
        READ_PIPELINE_RESULT(HFGetFaceMaskConfidence(session->handle, &mask), "read mask result");
        READ_PIPELINE_RESULT(HFGetFaceQualityConfidence(session->handle, &quality), "read quality result");
        READ_PIPELINE_RESULT(HFGetFaceInteractionStateResult(session->handle, &interaction),
                             "read interaction state result");
        READ_PIPELINE_RESULT(HFGetFaceInteractionActionsResult(session->handle, &actions),
                             "read interaction action result");
        READ_PIPELINE_RESULT(HFGetFaceAttributeResult(session->handle, &attributes), "read face attribute result");
        READ_PIPELINE_RESULT(HFGetFaceEmotionResult(session->handle, &emotion), "read face emotion result");
#undef READ_PIPELINE_RESULT

        napi_value output = nullptr;
        if (!CheckNapi(env, napi_create_object(env, &output), "create pipeline result") ||
            !SetInt32(env, output, "detectedNum", faces.detectedNum) ||
            !SetFloatArray(env, output, "rgbLiveness", liveness.confidence, liveness.num) ||
            !SetFloatArray(env, output, "maskConfidence", mask.confidence, mask.num) ||
            !SetFloatArray(env, output, "qualityConfidence", quality.confidence, quality.num) ||
            !SetFloatArray(env, output, "leftEyeStatusConfidence", interaction.leftEyeStatusConfidence,
                           interaction.num) ||
            !SetFloatArray(env, output, "rightEyeStatusConfidence", interaction.rightEyeStatusConfidence,
                           interaction.num) ||
            !SetInt32Array(env, output, "normal", actions.normal, actions.num) ||
            !SetInt32Array(env, output, "shake", actions.shake, actions.num) ||
            !SetInt32Array(env, output, "jawOpen", actions.jawOpen, actions.num) ||
            !SetInt32Array(env, output, "headRaise", actions.headRaise, actions.num) ||
            !SetInt32Array(env, output, "blink", actions.blink, actions.num) ||
            !SetInt32Array(env, output, "race", attributes.race, attributes.num) ||
            !SetInt32Array(env, output, "gender", attributes.gender, attributes.num) ||
            !SetInt32Array(env, output, "ageBracket", attributes.ageBracket, attributes.num) ||
            !SetInt32Array(env, output, "emotion", emotion.emotion, emotion.num)) {
            return nullptr;
        }
        return output;
    });
}

napi_value DetectFaceQuality(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[2] = {nullptr, nullptr};
        if (!ReadArguments(env, info, 2, arguments)) {
            return nullptr;
        }
        SessionState* session =
          UnwrapState<SessionState>(env, arguments[0], kSessionTypeTag, "Expected InspireFace session handle");
        const uint8_t* token_data = nullptr;
        size_t token_size = 0;
        if (!session || !ReadUint8View(env, arguments[1], &token_data, &token_size)) {
            return nullptr;
        }
        if (token_size > static_cast<size_t>(std::numeric_limits<HInt32>::max())) {
            return ThrowRangeError(env, "Face token is too large");
        }
        std::lock_guard<std::mutex> lock(session->mutex);
        if (!session->handle) {
            return ThrowTypeError(env, "Session has already been released");
        }
        HFFaceBasicToken token{static_cast<HInt32>(token_size), const_cast<uint8_t*>(token_data)};
        HFloat confidence = 0.0f;
        const HResult result = HFFaceQualityDetect(session->handle, token, &confidence);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "detect face quality");
        }
        napi_value number = nullptr;
        return CheckNapi(env, napi_create_double(env, confidence, &number), "create quality confidence") ? number
                                                                                                         : nullptr;
    });
}

napi_value GetLandmarks(napi_env env, napi_callback_info info, bool dense) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) {
            return nullptr;
        }
        const uint8_t* token_data = nullptr;
        size_t token_size = 0;
        if (!ReadUint8View(env, arguments[0], &token_data, &token_size) ||
            token_size > static_cast<size_t>(std::numeric_limits<HInt32>::max())) {
            return token_size > static_cast<size_t>(std::numeric_limits<HInt32>::max())
                     ? ThrowRangeError(env, "Face token is too large")
                     : nullptr;
        }
        HFFaceBasicToken token{static_cast<HInt32>(token_size), const_cast<uint8_t*>(token_data)};
        HInt32 count = dense ? 0 : 5;
        HResult result = dense ? HFGetNumOfFaceDenseLandmark(&count) : HSUCCEED;
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "query landmark count");
        }
        std::vector<HPoint2f> points(static_cast<size_t>(count));
        result = dense ? HFGetFaceDenseLandmarkFromFaceToken(token, points.data(), count)
                       : HFGetFaceFiveKeyPointsFromFaceToken(token, points.data(), count);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, dense ? "get dense landmarks" : "get five key points");
        }
        std::vector<float> flattened(static_cast<size_t>(count) * 2);
        for (int32_t index = 0; index < count; ++index) {
            flattened[static_cast<size_t>(index) * 2] = points[static_cast<size_t>(index)].x;
            flattened[static_cast<size_t>(index) * 2 + 1] = points[static_cast<size_t>(index)].y;
        }
        return CreateTypedArray(env, napi_float32_array, flattened.data(), flattened.size(), sizeof(float));
    });
}

napi_value GetDenseLandmarks(napi_env env, napi_callback_info info) {
    return GetLandmarks(env, info, true);
}

napi_value GetFiveKeyPoints(napi_env env, napi_callback_info info) {
    return GetLandmarks(env, info, false);
}

napi_value ExtractFeature(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[3] = {nullptr, nullptr, nullptr};
        if (!ReadArguments(env, info, 3, arguments)) {
            return nullptr;
        }
        SessionState* session = UnwrapState<SessionState>(env, arguments[0], kSessionTypeTag, "Expected InspireFace session handle");
        ImageStreamState* image =
          UnwrapState<ImageStreamState>(env, arguments[1], kImageStreamTypeTag, "Expected InspireFace image stream handle");
        const uint8_t* token_data = nullptr;
        size_t token_size = 0;
        if (!session || !image || !ReadUint8View(env, arguments[2], &token_data, &token_size)) {
            return nullptr;
        }
        if (token_size > static_cast<size_t>(std::numeric_limits<HInt32>::max())) {
            return ThrowRangeError(env, "Face token is too large");
        }
        HFFaceBasicToken token{static_cast<HInt32>(token_size), const_cast<uint8_t*>(token_data)};
        std::vector<float> copied_feature;
        {
            std::unique_lock<std::mutex> session_lock(session->mutex, std::defer_lock);
            std::unique_lock<std::mutex> image_lock(image->mutex, std::defer_lock);
            std::lock(session_lock, image_lock);
            if (!session->handle || !image->handle) {
                return ThrowTypeError(env, "Session or image stream has already been released");
            }
            HFFaceFeature feature{};
            const HResult result = HFFaceFeatureExtract(session->handle, image->handle, token, &feature);
            if (result != HSUCCEED) {
                return ThrowSdkError(env, result, "extract face feature");
            }
            if (feature.size <= 0 || !feature.data) {
                return ThrowSdkError(env, HERR_INVALID_FACE_FEATURE, "extract face feature");
            }
            copied_feature.assign(feature.data, feature.data + feature.size);
        }
        return CreateTypedArray(env, napi_float32_array, copied_feature.data(), copied_feature.size(), sizeof(float));
    });
}

napi_value GetFaceAlignmentImage(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[3] = {nullptr, nullptr, nullptr};
        SessionState* session = nullptr;
        ImageStreamState* image = nullptr;
        const uint8_t* token_data = nullptr;
        size_t token_size = 0;
        if (!ReadArguments(env, info, 3, arguments) ||
            !(session = UnwrapState<SessionState>(env, arguments[0], kSessionTypeTag,
                                                  "Expected InspireFace session handle")) ||
            !(image = UnwrapState<ImageStreamState>(env, arguments[1], kImageStreamTypeTag,
                                                    "Expected InspireFace image stream handle")) ||
            !ReadUint8View(env, arguments[2], &token_data, &token_size)) {
            return nullptr;
        }
        if (token_size > static_cast<size_t>(std::numeric_limits<HInt32>::max())) {
            return ThrowRangeError(env, "Face token is too large");
        }
        HFFaceBasicToken token{static_cast<HInt32>(token_size), const_cast<uint8_t*>(token_data)};
        std::unique_lock<std::mutex> session_lock(session->mutex, std::defer_lock);
        std::unique_lock<std::mutex> image_lock(image->mutex, std::defer_lock);
        std::lock(session_lock, image_lock);
        if (!session->handle || !image->handle) {
            return ThrowTypeError(env, "Session or image stream has already been released");
        }
        auto* bitmap = new (std::nothrow) ImageBitmapState();
        if (!bitmap) {
            napi_throw_error(env, "ERR_INSPIREFACE_NO_MEMORY", "Unable to allocate image bitmap wrapper");
            return nullptr;
        }
        const HResult result = HFFaceGetFaceAlignmentImage(session->handle, image->handle, token, &bitmap->handle);
        if (result != HSUCCEED) {
            delete bitmap;
            return ThrowSdkError(env, result, "get face alignment image");
        }
        return WrapState(env, bitmap, kImageBitmapTypeTag);
    });
}

napi_value ExtractFeatureFromAlignmentImage(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[2] = {nullptr, nullptr};
        if (!ReadArguments(env, info, 2, arguments)) return nullptr;
        SessionState* session =
          UnwrapState<SessionState>(env, arguments[0], kSessionTypeTag, "Expected InspireFace session handle");
        ImageStreamState* image =
          UnwrapState<ImageStreamState>(env, arguments[1], kImageStreamTypeTag, "Expected InspireFace image stream handle");
        if (!session || !image) return nullptr;
        HInt32 feature_length = 0;
        HResult result = HFGetFeatureLength(&feature_length);
        if (result != HSUCCEED) return ThrowSdkError(env, result, "query feature length");
        if (feature_length <= 0) return ThrowRangeError(env, "SDK returned an invalid feature length");
        std::vector<float> feature_data(static_cast<size_t>(feature_length));
        HFFaceFeature feature{feature_length, feature_data.data()};
        std::unique_lock<std::mutex> session_lock(session->mutex, std::defer_lock);
        std::unique_lock<std::mutex> image_lock(image->mutex, std::defer_lock);
        std::lock(session_lock, image_lock);
        if (!session->handle || !image->handle) {
            return ThrowTypeError(env, "Session or image stream has already been released");
        }
        result = HFFaceFeatureExtractWithAlignmentImage(session->handle, image->handle, feature);
        if (result != HSUCCEED) return ThrowSdkError(env, result, "extract feature from alignment image");
        return CreateTypedArray(env, napi_float32_array, feature_data.data(), feature_data.size(), sizeof(float));
    });
}

napi_value GetFeatureLength(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) {
            return nullptr;
        }
        HInt32 length = 0;
        const HResult result = HFGetFeatureLength(&length);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "query feature length");
        }
        napi_value value = nullptr;
        return CheckNapi(env, napi_create_int32(env, length, &value), "create feature length") ? value : nullptr;
    });
}

napi_value CompareFeatures(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[2] = {nullptr, nullptr};
        if (!ReadArguments(env, info, 2, arguments)) {
            return nullptr;
        }
        float* first_data = nullptr;
        float* second_data = nullptr;
        size_t first_size = 0;
        size_t second_size = 0;
        if (!ReadFloat32View(env, arguments[0], &first_data, &first_size) ||
            !ReadFloat32View(env, arguments[1], &second_data, &second_size)) {
            return nullptr;
        }
        if (first_size != second_size || first_size == 0 ||
            first_size > static_cast<size_t>(std::numeric_limits<HInt32>::max())) {
            return ThrowRangeError(env, "Face features must have the same non-zero length");
        }
        HFFaceFeature first{static_cast<HInt32>(first_size), first_data};
        HFFaceFeature second{static_cast<HInt32>(second_size), second_data};
        HFloat similarity = 0.0f;
        const HResult result = HFFaceComparison(first, second, &similarity);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "compare face features");
        }
        napi_value value = nullptr;
        return CheckNapi(env, napi_create_double(env, similarity, &value), "create similarity") ? value : nullptr;
    });
}

napi_value GetRecommendedThreshold(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) {
            return nullptr;
        }
        HFloat threshold = 0.0f;
        const HResult result = HFGetRecommendedCosineThreshold(&threshold);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "query recommended threshold");
        }
        napi_value value = nullptr;
        return CheckNapi(env, napi_create_double(env, threshold, &value), "create threshold") ? value : nullptr;
    });
}

napi_value SimilarityToPercentage(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) {
            return nullptr;
        }
        double similarity = 0.0;
        if (!CheckNapi(env, napi_get_value_double(env, arguments[0], &similarity), "read similarity")) {
            return nullptr;
        }
        HFloat percentage = 0.0f;
        const HResult result = HFCosineSimilarityConvertToPercentage(static_cast<HFloat>(similarity), &percentage);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "convert similarity");
        }
        napi_value value = nullptr;
        return CheckNapi(env, napi_create_double(env, percentage, &value), "create percentage") ? value : nullptr;
    });
}

napi_value SetLogLevel(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        int32_t level = 0;
        if (!ReadArguments(env, info, 1, arguments) ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[0], &level), "read log level")) {
            return nullptr;
        }
        const HResult result = HFSetLogLevel(static_cast<HFLogLevel>(level));
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "set log level");
    });
}

napi_value DisableLog(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) {
            return nullptr;
        }
        const HResult result = HFLogDisable();
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "disable log");
    });
}

napi_value GetErrorMessage(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        int32_t code = 0;
        if (!ReadArguments(env, info, 1, arguments) ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[0], &code), "read error code")) {
            return nullptr;
        }
        const std::string message = GetSdkErrorMessage(code);
        napi_value value = nullptr;
        return CheckNapi(env, napi_create_string_utf8(env, message.c_str(), NAPI_AUTO_LENGTH, &value),
                         "create error message")
                 ? value
                 : nullptr;
    });
}

napi_value ValidateResourcePack(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        std::string path;
        if (!ReadArguments(env, info, 1, arguments) || !ReadString(env, arguments[0], &path)) {
            return nullptr;
        }
        HFResourcePackInfo resource{};
        resource.structSize = sizeof(resource);
        resource.structVersion = HF_RESOURCE_PACK_INFO_VERSION;
        const HResult result = HFValidateResourcePack(path.c_str(), &resource);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "validate resource pack");
        }
        napi_value output = nullptr;
        if (!CheckNapi(env, napi_create_object(env, &output), "create resource metadata") ||
            !SetInt32(env, output, "archiveFileCount", static_cast<int32_t>(resource.archiveFileCount)) ||
            !SetInt32(env, output, "modelCount", static_cast<int32_t>(resource.modelCount)) ||
            !SetString(env, output, "tag", resource.tag) || !SetString(env, output, "version", resource.version) ||
            !SetString(env, output, "major", resource.major) || !SetString(env, output, "releaseDate", resource.releaseDate)) {
            return nullptr;
        }
        return output;
    });
}

napi_value GetSupportedPixelLevels(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) {
            return nullptr;
        }
        HFFaceDetectPixelList levels{};
        const HResult result = HFQuerySupportedPixelLevelsForFaceDetection(&levels);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "query supported face detection pixel levels");
        }
        if (levels.size < 0 || levels.size > 20) {
            return ThrowSdkError(env, HERR_INVALID_PARAM, "query supported face detection pixel levels");
        }
        return CreateTypedArray(env, napi_int32_array, levels.pixel_level, static_cast<size_t>(levels.size),
                                sizeof(int32_t));
    });
}

napi_value SwitchLandmarkEngine(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        int32_t engine = 0;
        if (!ReadArguments(env, info, 1, arguments) ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[0], &engine), "read landmark engine")) {
            return nullptr;
        }
        const HResult result = HFSwitchLandmarkEngine(static_cast<HFSessionLandmarkEngine>(engine));
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "switch landmark engine");
    });
}

napi_value GetSessionPreviewSize(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) {
            return nullptr;
        }
        SessionState* session =
          UnwrapState<SessionState>(env, arguments[0], kSessionTypeTag, "Expected InspireFace session handle");
        if (!session) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(session->mutex);
        if (!session->handle) {
            return ThrowTypeError(env, "Session has already been released");
        }
        HInt32 preview_size = 0;
        const HResult result = HFSessionGetTrackPreviewSize(session->handle, &preview_size);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "get session preview size");
        }
        napi_value number = nullptr;
        return CheckNapi(env, napi_create_int32(env, preview_size, &number), "create preview size") ? number : nullptr;
    });
}

napi_value GetLastDetectionDebugPreviewSize(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) return nullptr;
        SessionState* session =
          UnwrapState<SessionState>(env, arguments[0], kSessionTypeTag, "Expected InspireFace session handle");
        if (!session) return nullptr;
        std::lock_guard<std::mutex> lock(session->mutex);
        if (!session->handle) return ThrowTypeError(env, "Session has already been released");
        HInt32 size = 0;
        const HResult result = HFSessionLastFaceDetectionGetDebugPreviewImageSize(session->handle, &size);
        if (result != HSUCCEED) return ThrowSdkError(env, result, "get last detection debug preview size");
        napi_value output = nullptr;
        return CheckNapi(env, napi_create_int32(env, size, &output), "create debug preview size") ? output : nullptr;
    });
}

napi_value GetSimilarityConverter(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) {
            return nullptr;
        }
        HFSimilarityConverterConfig config{};
        const HResult result = HFGetCosineSimilarityConverter(&config);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "get similarity converter");
        }
        napi_value output = nullptr;
        if (!CheckNapi(env, napi_create_object(env, &output), "create similarity converter") ||
            !SetDouble(env, output, "threshold", config.threshold) ||
            !SetDouble(env, output, "middleScore", config.middleScore) ||
            !SetDouble(env, output, "steepness", config.steepness) ||
            !SetDouble(env, output, "outputMin", config.outputMin) ||
            !SetDouble(env, output, "outputMax", config.outputMax)) {
            return nullptr;
        }
        return output;
    });
}

napi_value UpdateSimilarityConverter(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) {
            return nullptr;
        }
        HFSimilarityConverterConfig config{};
        double value = 0.0;
        bool present = false;
#define READ_REQUIRED_FLOAT(name, field)                                                       \
    do {                                                                                        \
        if (!GetOptionalDouble(env, arguments[0], (name), &value, &present)) return nullptr;   \
        if (!present) return ThrowTypeError(env, "Similarity converter requires all fields"); \
        config.field = static_cast<HFloat>(value);                                              \
    } while (false)
        READ_REQUIRED_FLOAT("threshold", threshold);
        READ_REQUIRED_FLOAT("middleScore", middleScore);
        READ_REQUIRED_FLOAT("steepness", steepness);
        READ_REQUIRED_FLOAT("outputMin", outputMin);
        READ_REQUIRED_FLOAT("outputMax", outputMax);
#undef READ_REQUIRED_FLOAT
        const HResult result = HFUpdateCosineSimilarityConverter(config);
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "update similarity converter");
    });
}

napi_value FeatureHubEnable(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) {
            return nullptr;
        }
        HFFeatureHubConfiguration config{};
        config.primaryKeyMode = HF_PK_AUTO_INCREMENT;
        config.enablePersistence = 0;
        config.persistenceDbPath = nullptr;
        config.searchThreshold = -1.0f;
        config.searchMode = HF_SEARCH_MODE_EXHAUSTIVE;
        bool present = false;
        bool enabled = false;
        int32_t integer = 0;
        double decimal = 0.0;
        std::string path;
        if (!GetOptionalInt32(env, arguments[0], "primaryKeyMode", &integer, &present)) return nullptr;
        if (present) config.primaryKeyMode = static_cast<HFPKMode>(integer);
        if (!GetOptionalBool(env, arguments[0], "enablePersistence", &enabled, &present)) return nullptr;
        if (present) config.enablePersistence = enabled ? 1 : 0;
        if (!GetOptionalString(env, arguments[0], "persistenceDbPath", &path, &present)) return nullptr;
        if (present) config.persistenceDbPath = const_cast<char*>(path.c_str());
        if (!GetOptionalDouble(env, arguments[0], "searchThreshold", &decimal, &present)) return nullptr;
        if (present) config.searchThreshold = static_cast<HFloat>(decimal);
        if (!GetOptionalInt32(env, arguments[0], "searchMode", &integer, &present)) return nullptr;
        if (present) config.searchMode = static_cast<HFSearchMode>(integer);
        const HResult result = HFFeatureHubDataEnable(config);
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "enable FeatureHub");
    });
}

napi_value FeatureHubDisable(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) return nullptr;
        const HResult result = HFFeatureHubDataDisable();
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "disable FeatureHub");
    });
}

napi_value FeatureHubViewTable(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) return nullptr;
        const HResult result = HFFeatureHubViewDBTable();
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "view FeatureHub table");
    });
}

napi_value FeatureHubSetThreshold(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        double threshold = 0.0;
        if (!ReadArguments(env, info, 1, arguments) ||
            !CheckNapi(env, napi_get_value_double(env, arguments[0], &threshold), "read FeatureHub threshold")) {
            return nullptr;
        }
        const HResult result = HFFeatureHubFaceSearchThresholdSetting(static_cast<HFloat>(threshold));
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "set FeatureHub threshold");
    });
}

bool ReadFeature(napi_env env, napi_value value, HFFaceFeature* feature) {
    float* data = nullptr;
    size_t size = 0;
    if (!feature || !ReadFloat32View(env, value, &data, &size)) {
        return false;
    }
    if (size > static_cast<size_t>(std::numeric_limits<HInt32>::max())) {
        napi_throw_range_error(env, "ERR_INSPIREFACE_RANGE", "Face feature is too large");
        return false;
    }
    feature->size = static_cast<HInt32>(size);
    feature->data = data;
    return true;
}

napi_value FeatureHubInsert(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[2] = {nullptr, nullptr};
        HFaceId id = HF_INVALID_FACE_ID;
        HFFaceFeature feature{};
        if (!ReadArguments(env, info, 2, arguments) || !ReadFaceId(env, arguments[0], &id) ||
            !ReadFeature(env, arguments[1], &feature)) {
            return nullptr;
        }
        HFFaceFeatureIdentity identity{id, &feature};
        HFaceId allocated = HF_INVALID_FACE_ID;
        const HResult result = HFFeatureHubInsertFeature(identity, &allocated);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "insert FeatureHub feature");
        }
        napi_value output = nullptr;
        return CheckNapi(env, napi_create_bigint_int64(env, static_cast<int64_t>(allocated), &output), "create allocated face ID")
                 ? output
                 : nullptr;
    });
}

napi_value FeatureHubUpdate(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[2] = {nullptr, nullptr};
        HFaceId id = HF_INVALID_FACE_ID;
        HFFaceFeature feature{};
        if (!ReadArguments(env, info, 2, arguments) || !ReadFaceId(env, arguments[0], &id) ||
            !ReadFeature(env, arguments[1], &feature)) {
            return nullptr;
        }
        HFFaceFeatureIdentity identity{id, &feature};
        const HResult result = HFFeatureHubFaceUpdate(identity);
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "update FeatureHub feature");
    });
}

napi_value FeatureHubRemove(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        HFaceId id = HF_INVALID_FACE_ID;
        if (!ReadArguments(env, info, 1, arguments) || !ReadFaceId(env, arguments[0], &id)) return nullptr;
        const HResult result = HFFeatureHubFaceRemove(id);
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "remove FeatureHub feature");
    });
}

napi_value FeatureHubGet(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        HFaceId id = HF_INVALID_FACE_ID;
        if (!ReadArguments(env, info, 1, arguments) || !ReadFaceId(env, arguments[0], &id)) return nullptr;
        HFFaceFeatureIdentity identity{};
        const HResult result = HFFeatureHubGetFaceIdentity(id, &identity);
        if (result != HSUCCEED) return ThrowSdkError(env, result, "get FeatureHub feature");
        if (!identity.feature || identity.feature->size < 0 ||
            (identity.feature->size > 0 && !identity.feature->data)) {
            return ThrowSdkError(env, HERR_FT_HUB_INVALID_FEATURE, "get FeatureHub feature");
        }
        return CreateTypedArray(env, napi_float32_array, identity.feature->data,
                                static_cast<size_t>(identity.feature->size), sizeof(float));
    });
}

napi_value FeatureHubGetCount(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) return nullptr;
        HInt32 count = 0;
        const HResult result = HFFeatureHubGetFaceCount(&count);
        if (result != HSUCCEED) return ThrowSdkError(env, result, "get FeatureHub count");
        napi_value output = nullptr;
        return CheckNapi(env, napi_create_int32(env, count, &output), "create FeatureHub count") ? output : nullptr;
    });
}

napi_value FeatureHubGetIds(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) return nullptr;
        HFFeatureHubExistingIds ids{};
        const HResult result = HFFeatureHubGetExistingIds(&ids);
        if (result != HSUCCEED) return ThrowSdkError(env, result, "get FeatureHub IDs");
        if (ids.size < 0 || (ids.size > 0 && !ids.ids)) {
            return ThrowSdkError(env, HERR_INVALID_PARAM, "get FeatureHub IDs");
        }
        return CreateFaceIdArray(env, ids.ids, static_cast<size_t>(ids.size));
    });
}

napi_value FeatureHubSearch(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        HFFaceFeature feature{};
        if (!ReadArguments(env, info, 1, arguments) || !ReadFeature(env, arguments[0], &feature)) return nullptr;
        HFFeatureHubSearchResultV2 result{};
        const HResult status = HFFeatureHubFaceSearchV2(feature, &result);
        if (status != HSUCCEED) return ThrowSdkError(env, status, "search FeatureHub");
        if (result.feature.size < 0 || (result.feature.size > 0 && !result.feature.data)) {
            return ThrowSdkError(env, HERR_FT_HUB_INVALID_FEATURE, "search FeatureHub");
        }
        napi_value output = nullptr;
        if (!CheckNapi(env, napi_create_object(env, &output), "create FeatureHub search result") ||
            !SetBool(env, output, "found", result.found != 0) || !SetFaceId(env, output, "id", result.id) ||
            !SetDouble(env, output, "confidence", result.confidence)) {
            return nullptr;
        }
        napi_value matched = CreateTypedArray(env, napi_float32_array, result.feature.data,
                                              static_cast<size_t>(result.feature.size), sizeof(float));
        if (!matched || !CheckNapi(env, napi_set_named_property(env, output, "feature", matched),
                                   "set FeatureHub matched feature")) {
            return nullptr;
        }
        return output;
    });
}

napi_value FeatureHubSearchTopK(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[2] = {nullptr, nullptr};
        HFFaceFeature feature{};
        int32_t top_k = 0;
        if (!ReadArguments(env, info, 2, arguments) || !ReadFeature(env, arguments[0], &feature) ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[1], &top_k), "read FeatureHub topK")) {
            return nullptr;
        }
        HFSearchTopKResults results{};
        const HResult status = HFFeatureHubFaceSearchTopK(feature, top_k, &results);
        if (status != HSUCCEED) return ThrowSdkError(env, status, "search FeatureHub topK");
        if (results.size < 0 || (results.size > 0 && (!results.ids || !results.confidence))) {
            return ThrowSdkError(env, HERR_INVALID_PARAM, "search FeatureHub topK");
        }
        napi_value output = nullptr;
        napi_value ids = CreateFaceIdArray(env, results.ids, static_cast<size_t>(results.size));
        napi_value confidence = CreateTypedArray(env, napi_float32_array, results.confidence,
                                                 static_cast<size_t>(results.size), sizeof(float));
        if (!ids || !confidence || !CheckNapi(env, napi_create_object(env, &output), "create topK result") ||
            !CheckNapi(env, napi_set_named_property(env, output, "ids", ids), "set topK IDs") ||
            !CheckNapi(env, napi_set_named_property(env, output, "confidence", confidence), "set topK confidence")) {
            return nullptr;
        }
        return output;
    });
}

napi_value GetComponentVersion(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        int32_t component = 0;
        if (!ReadArguments(env, info, 1, arguments) ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[0], &component), "read component type")) {
            return nullptr;
        }
        HFComponentVersion version{};
        const HResult result = HFQueryInspireFaceComponentVersion(static_cast<HFComponentType>(component), &version);
        if (result != HSUCCEED) return ThrowSdkError(env, result, "query component version");
        napi_value output = nullptr;
        if (!CheckNapi(env, napi_create_object(env, &output), "create component version") ||
            !SetInt32(env, output, "major", version.major) || !SetInt32(env, output, "minor", version.minor) ||
            !SetInt32(env, output, "patch", version.patch) || !SetInt32(env, output, "state", version.state)) {
            return nullptr;
        }
        return output;
    });
}

napi_value GetComponentVersions(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) return nullptr;
        return QuerySdkString(env, "query component versions", HFQueryInspireFaceComponentVersions);
    });
}

napi_value GetDiagnosticInformation(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) return nullptr;
        return QuerySdkString(env, "query diagnostic information", HFQueryInspireFaceDiagnosticInformation);
    });
}

napi_value GetExtendedInformation(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) return nullptr;
        HFInspireFaceExtendedInformation information{};
        const HResult result = HFQueryInspireFaceExtendedInformation(&information);
        if (result != HSUCCEED) return ThrowSdkError(env, result, "query extended information");
        napi_value output = nullptr;
        return CheckNapi(env, napi_create_string_utf8(env, information.information, NAPI_AUTO_LENGTH, &output),
                         "create extended information")
                 ? output
                 : nullptr;
    });
}

napi_value QueryRgaEnabled(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) return nullptr;
        HInt32 enabled = 0;
        const HResult result = HFQueryExpansiveHardwareRGACompileOption(&enabled);
        if (result != HSUCCEED) return ThrowSdkError(env, result, "query RGA support");
        napi_value output = nullptr;
        return CheckNapi(env, napi_get_boolean(env, enabled != 0, &output), "create RGA support status") ? output : nullptr;
    });
}

napi_value SetRgaDmaHeapPath(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        std::string path;
        if (!ReadArguments(env, info, 1, arguments) || !ReadString(env, arguments[0], &path)) return nullptr;
        const HResult result = HFSetExpansiveHardwareRockchipDmaHeapPath(path.c_str());
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "set RGA DMA heap path");
    });
}

napi_value GetRgaDmaHeapPath(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) return nullptr;
        std::vector<char> path(4096, '\0');
        const HResult result = HFQueryExpansiveHardwareRockchipDmaHeapPathWithSize(path.data(),
                                                                                  static_cast<HInt32>(path.size()));
        if (result != HSUCCEED) return ThrowSdkError(env, result, "get RGA DMA heap path");
        napi_value output = nullptr;
        return CheckNapi(env, napi_create_string_utf8(env, path.data(), NAPI_AUTO_LENGTH, &output),
                         "create RGA DMA heap path")
                 ? output
                 : nullptr;
    });
}

napi_value SwitchImageProcessingBackend(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        int32_t backend = 0;
        if (!ReadArguments(env, info, 1, arguments) ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[0], &backend), "read image processing backend")) {
            return nullptr;
        }
        const HResult result = HFSwitchImageProcessingBackend(static_cast<HFImageProcessingBackend>(backend));
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "switch image processing backend");
    });
}

napi_value SetImageProcessAlignedWidth(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        int32_t width = 0;
        if (!ReadArguments(env, info, 1, arguments) ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[0], &width), "read aligned width")) {
            return nullptr;
        }
        const HResult result = HFSetImageProcessAlignedWidth(width);
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "set image process aligned width");
    });
}

napi_value SetCoreMlInferenceMode(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        int32_t mode = 0;
        if (!ReadArguments(env, info, 1, arguments) ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[0], &mode), "read CoreML mode")) return nullptr;
        const HResult result = HFSetAppleCoreMLInferenceMode(static_cast<HFAppleCoreMLInferenceMode>(mode));
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "set CoreML inference mode");
    });
}

napi_value SetCudaDeviceId(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        int32_t id = 0;
        if (!ReadArguments(env, info, 1, arguments) ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[0], &id), "read CUDA device ID")) return nullptr;
        const HResult result = HFSetCudaDeviceId(id);
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "set CUDA device ID");
    });
}

napi_value GetCudaDeviceId(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) return nullptr;
        HInt32 id = 0;
        const HResult result = HFGetCudaDeviceId(&id);
        if (result != HSUCCEED) return ThrowSdkError(env, result, "get CUDA device ID");
        napi_value output = nullptr;
        return CheckNapi(env, napi_create_int32(env, id, &output), "create CUDA device ID") ? output : nullptr;
    });
}

napi_value PrintCudaDeviceInfo(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) return nullptr;
        const HResult result = HFPrintCudaDeviceInfo();
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "print CUDA device information");
    });
}

napi_value GetCudaDeviceCount(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) return nullptr;
        HInt32 count = 0;
        const HResult result = HFGetNumCudaDevices(&count);
        if (result != HSUCCEED) return ThrowSdkError(env, result, "get CUDA device count");
        napi_value output = nullptr;
        return CheckNapi(env, napi_create_int32(env, count, &output), "create CUDA device count") ? output : nullptr;
    });
}

napi_value IsCudaSupported(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) return nullptr;
        HInt32 supported = 0;
        const HResult result = HFCheckCudaDeviceSupport(&supported);
        if (result != HSUCCEED) return ThrowSdkError(env, result, "check CUDA support");
        napi_value output = nullptr;
        return CheckNapi(env, napi_get_boolean(env, supported != 0, &output), "create CUDA support status") ? output : nullptr;
    });
}

napi_value ConfigureTrackCost(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[2] = {nullptr, nullptr};
        bool enabled = false;
        if (!ReadArguments(env, info, 2, arguments)) return nullptr;
        SessionState* session =
          UnwrapState<SessionState>(env, arguments[0], kSessionTypeTag, "Expected InspireFace session handle");
        if (!session || !CheckNapi(env, napi_get_value_bool(env, arguments[1], &enabled), "read track cost option")) return nullptr;
        std::lock_guard<std::mutex> lock(session->mutex);
        if (!session->handle) return ThrowTypeError(env, "Session has already been released");
        const HResult result = HFSessionSetEnableTrackCostSpend(session->handle, enabled ? 1 : 0);
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "configure track cost measurement");
    });
}

napi_value PrintTrackCost(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) return nullptr;
        SessionState* session =
          UnwrapState<SessionState>(env, arguments[0], kSessionTypeTag, "Expected InspireFace session handle");
        if (!session) return nullptr;
        std::lock_guard<std::mutex> lock(session->mutex);
        if (!session->handle) return ThrowTypeError(env, "Session has already been released");
        const HResult result = HFSessionPrintTrackCostSpend(session->handle);
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "print track cost");
    });
}

napi_value GetDebugResourceCounts(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) return nullptr;
        HInt32 sessions = 0;
        HInt32 streams = 0;
        HResult result = HFDeBugGetUnreleasedSessionsCount(&sessions);
        if (result == HSUCCEED) result = HFDeBugGetUnreleasedStreamsCount(&streams);
        if (result != HSUCCEED) return ThrowSdkError(env, result, "get debug resource counts");
        napi_value output = nullptr;
        if (!CheckNapi(env, napi_create_object(env, &output), "create resource counts") ||
            !SetInt32(env, output, "sessions", sessions) || !SetInt32(env, output, "streams", streams)) return nullptr;
        return output;
    });
}

napi_value SaveDebugImageStream(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[2] = {nullptr, nullptr};
        std::string path;
        if (!ReadArguments(env, info, 2, arguments) || !ReadString(env, arguments[1], &path)) return nullptr;
        ImageStreamState* stream =
          UnwrapState<ImageStreamState>(env, arguments[0], kImageStreamTypeTag, "Expected InspireFace image stream handle");
        if (!stream) return nullptr;
        std::lock_guard<std::mutex> lock(stream->mutex);
        if (!stream->handle) return ThrowTypeError(env, "Image stream has already been released");
        const HResult result = HFDeBugImageStreamDecodeSave(stream->handle, path.c_str());
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "save debug image stream");
    });
}

napi_value ShowDebugImageStream(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) return nullptr;
        ImageStreamState* stream =
          UnwrapState<ImageStreamState>(env, arguments[0], kImageStreamTypeTag, "Expected InspireFace image stream handle");
        if (!stream) return nullptr;
        std::lock_guard<std::mutex> lock(stream->mutex);
        if (!stream->handle) return ThrowTypeError(env, "Image stream has already been released");
        HFDeBugImageStreamImShow(stream->handle);
        return Undefined(env);
    });
}

napi_value ShowDebugResourceStatistics(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) return nullptr;
        const HResult result = HFDeBugShowResourceStatistics();
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "show debug resource statistics");
    });
}

napi_value LogPrint(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[2] = {nullptr, nullptr};
        int32_t level = 0;
        std::string message;
        if (!ReadArguments(env, info, 2, arguments) ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[0], &level), "read log level") ||
            !ReadString(env, arguments[1], &message)) return nullptr;
        const HResult result = HFLogPrint(static_cast<HFLogLevel>(level), "%s", message.c_str());
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "print SDK log");
    });
}

napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor properties[] = {
      {"getErrorMessage", nullptr, GetErrorMessage, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"validateResourcePack", nullptr, ValidateResourcePack, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"launch", nullptr, Launch, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"reload", nullptr, Reload, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"terminate", nullptr, Terminate, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"isLaunched", nullptr, IsLaunched, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getVersion", nullptr, GetVersion, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getCapiLevel", nullptr, GetCapiLevel, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"createSession", nullptr, CreateSession, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"releaseSession", nullptr, ReleaseSession, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"configureSession", nullptr, ConfigureSession, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"clearTracking", nullptr, ClearTracking, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getSessionPreviewSize", nullptr, GetSessionPreviewSize, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getLastDetectionDebugPreviewSize", nullptr, GetLastDetectionDebugPreviewSize, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getSupportedPixelLevels", nullptr, GetSupportedPixelLevels, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"switchLandmarkEngine", nullptr, SwitchLandmarkEngine, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"createImageStream", nullptr, CreateImageStream, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"updateImageStream", nullptr, UpdateImageStream, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"releaseImageStream", nullptr, ReleaseImageStream, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"createImageBitmap", nullptr, CreateImageBitmap, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"createImageBitmapFromFile", nullptr, CreateImageBitmapFromFile, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"copyImageBitmap", nullptr, CopyImageBitmap, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"releaseImageBitmap", nullptr, ReleaseImageBitmap, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"createImageStreamFromBitmap", nullptr, CreateImageStreamFromBitmap, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"createImageBitmapFromStream", nullptr, CreateImageBitmapFromStream, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getImageBitmapData", nullptr, GetImageBitmapData, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"writeImageBitmap", nullptr, WriteImageBitmap, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"drawImageBitmapRect", nullptr, DrawImageBitmapRect, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"drawImageBitmapCircle", nullptr, DrawImageBitmapCircle, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"showImageBitmap", nullptr, ShowImageBitmap, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"track", nullptr, Track, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"releaseFaceResult", nullptr, ReleaseFaceResult, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getDefaultFaceCaptureConfig", nullptr, NapiGetDefaultFaceCaptureConfig, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"createFaceCaptureSession", nullptr, CreateFaceCaptureSession, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"updateFaceCaptureSession", nullptr, UpdateFaceCaptureSession, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getFaceCaptureResults", nullptr, GetFaceCaptureResults, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"finishFaceCaptureSession", nullptr, FinishFaceCaptureSession, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"resetFaceCaptureSession", nullptr, ResetFaceCaptureSession, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"releaseFaceCaptureSession", nullptr, ReleaseFaceCaptureSession, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"processPipeline", nullptr, ProcessPipeline, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"detectFaceQuality", nullptr, DetectFaceQuality, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getDenseLandmarks", nullptr, GetDenseLandmarks, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getFiveKeyPoints", nullptr, GetFiveKeyPoints, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"extractFeature", nullptr, ExtractFeature, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getFaceAlignmentImage", nullptr, GetFaceAlignmentImage, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"extractFeatureFromAlignmentImage", nullptr, ExtractFeatureFromAlignmentImage, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getFeatureLength", nullptr, GetFeatureLength, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"compareFeatures", nullptr, CompareFeatures, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getRecommendedThreshold", nullptr, GetRecommendedThreshold, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"similarityToPercentage", nullptr, SimilarityToPercentage, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getSimilarityConverter", nullptr, GetSimilarityConverter, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"updateSimilarityConverter", nullptr, UpdateSimilarityConverter, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"featureHubEnable", nullptr, FeatureHubEnable, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"featureHubDisable", nullptr, FeatureHubDisable, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"featureHubViewTable", nullptr, FeatureHubViewTable, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"featureHubSetThreshold", nullptr, FeatureHubSetThreshold, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"featureHubInsert", nullptr, FeatureHubInsert, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"featureHubUpdate", nullptr, FeatureHubUpdate, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"featureHubRemove", nullptr, FeatureHubRemove, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"featureHubGet", nullptr, FeatureHubGet, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"featureHubGetCount", nullptr, FeatureHubGetCount, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"featureHubGetIds", nullptr, FeatureHubGetIds, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"featureHubSearch", nullptr, FeatureHubSearch, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"featureHubSearchTopK", nullptr, FeatureHubSearchTopK, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getComponentVersion", nullptr, GetComponentVersion, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getComponentVersions", nullptr, GetComponentVersions, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getDiagnosticInformation", nullptr, GetDiagnosticInformation, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getExtendedInformation", nullptr, GetExtendedInformation, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"queryRgaEnabled", nullptr, QueryRgaEnabled, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"setRgaDmaHeapPath", nullptr, SetRgaDmaHeapPath, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getRgaDmaHeapPath", nullptr, GetRgaDmaHeapPath, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"switchImageProcessingBackend", nullptr, SwitchImageProcessingBackend, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"setImageProcessAlignedWidth", nullptr, SetImageProcessAlignedWidth, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"setCoreMlInferenceMode", nullptr, SetCoreMlInferenceMode, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"setCudaDeviceId", nullptr, SetCudaDeviceId, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getCudaDeviceId", nullptr, GetCudaDeviceId, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"printCudaDeviceInfo", nullptr, PrintCudaDeviceInfo, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getCudaDeviceCount", nullptr, GetCudaDeviceCount, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"isCudaSupported", nullptr, IsCudaSupported, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"configureTrackCost", nullptr, ConfigureTrackCost, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"printTrackCost", nullptr, PrintTrackCost, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getDebugResourceCounts", nullptr, GetDebugResourceCounts, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"saveDebugImageStream", nullptr, SaveDebugImageStream, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"showDebugImageStream", nullptr, ShowDebugImageStream, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"showDebugResourceStatistics", nullptr, ShowDebugResourceStatistics, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"setLogLevel", nullptr, SetLogLevel, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"disableLog", nullptr, DisableLog, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"logPrint", nullptr, LogPrint, nullptr, nullptr, nullptr, napi_default, nullptr},
    };
    if (!CheckNapi(env, napi_define_properties(env, exports, sizeof(properties) / sizeof(properties[0]), properties),
                   "define InspireFace exports")) {
        return nullptr;
    }
    return exports;
}

}  // namespace

NAPI_MODULE(inspireface_napi, Init)
