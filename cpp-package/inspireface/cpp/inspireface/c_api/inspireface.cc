/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "inspireface.h"
#include "intypedef.h"
#include "inspireface_internal.h"
#include "information.h"
#include "component_version.h"
#include "feature_hub_db.h"
#include <launch.h>
#include "runtime_module/resource_manage.h"
#include "similarity_converter.h"
#include "middleware/inference_wrapper/inference_wrapper.h"
#if defined(ISF_ENABLE_TENSORRT)
#include "cuda_toolkit.h"
#endif
#include <cmath>
#include <cstring>
#include <cstdarg>
#include <fstream>
#include <limits>
#include <memory>
#if defined(_WIN32)
#include <cstdio>
#include "middleware/system.h"
#endif

#define FACE_FEATURE_SIZE 512  ///< Temporary setup

using namespace inspire;

namespace {

constexpr size_t kRockchipDmaHeapPathCapacity = 256;

bool IsValidImageFormat(HFImageFormat format) {
    return format >= HF_STREAM_RGB && format <= HF_STREAM_GRAY;
}

bool RequiresEvenDimensions(HFImageFormat format) {
    return format == HF_STREAM_YUV_NV12 || format == HF_STREAM_YUV_NV21 || format == HF_STREAM_I420;
}

bool IsValidImageDimensionsForFormat(HFImageFormat format, HInt32 width, HInt32 height) {
    return width > 0 && height > 0 && (!RequiresEvenDimensions(format) || (width % 2 == 0 && height % 2 == 0));
}

bool CheckedImageByteSize(HFImageFormat format, HInt32 width, HInt32 height, size_t* byte_size) {
    if (byte_size == nullptr || !IsValidImageFormat(format) || !IsValidImageDimensionsForFormat(format, width, height)) {
        return false;
    }
    const uint64_t pixels = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    uint64_t bytes = 0;
    switch (format) {
        case HF_STREAM_RGB:
        case HF_STREAM_BGR: bytes = pixels * 3; break;
        case HF_STREAM_RGBA:
        case HF_STREAM_BGRA: bytes = pixels * 4; break;
        case HF_STREAM_GRAY: bytes = pixels; break;
        case HF_STREAM_YUV_NV12:
        case HF_STREAM_YUV_NV21:
        case HF_STREAM_I420: bytes = pixels + pixels / 2; break;
        default: return false;
    }
    if (bytes == 0 || bytes > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return false;
    }
    *byte_size = static_cast<size_t>(bytes);
    return true;
}

bool CheckedBitmapByteSize(HInt32 width, HInt32 height, HInt32 channels, size_t* byte_size) {
    if (byte_size == nullptr || width <= 0 || height <= 0 || (channels != 1 && channels != 3)) {
        return false;
    }
    const uint64_t bytes = static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * static_cast<uint64_t>(channels);
    if (bytes == 0 || bytes > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return false;
    }
    *byte_size = static_cast<size_t>(bytes);
    return true;
}

bool CanScaleImage(const inspirecv::FrameProcess& process, HInt32 is_rotate, HFloat scale) {
    if ((is_rotate != 0 && is_rotate != 1) || !std::isfinite(scale) || scale <= 0.0f ||
        process.GetWidth() <= 0 || process.GetHeight() <= 0) {
        return false;
    }
    const bool swaps_dimensions = is_rotate == 1 &&
                                  (process.getRotationMode() == inspirecv::ROTATION_90 ||
                                   process.getRotationMode() == inspirecv::ROTATION_270);
    const int source_width = swaps_dimensions ? process.GetHeight() : process.GetWidth();
    const int source_height = swaps_dimensions ? process.GetWidth() : process.GetHeight();
    const float output_width = static_cast<float>(source_width) * scale;
    const float output_height = static_cast<float>(source_height) * scale;
    if (!std::isfinite(output_width) || !std::isfinite(output_height) || output_width < 1.0f || output_height < 1.0f ||
        output_width >= static_cast<float>(std::numeric_limits<int>::max()) ||
        output_height >= static_cast<float>(std::numeric_limits<int>::max())) {
        return false;
    }
    const uint64_t elements = static_cast<uint64_t>(static_cast<int>(output_width)) *
                              static_cast<uint64_t>(static_cast<int>(output_height)) * 3;
    return elements <= static_cast<uint64_t>(std::numeric_limits<int>::max());
}

bool IsValidRotation(HFRotation rotation) {
    return rotation >= HF_CAMERA_ROTATION_0 && rotation <= HF_CAMERA_ROTATION_270;
}

HResult CopyQueryString(const std::string& value, HString buffer, HInt32 buffer_size, HPInt32 required_size) {
    if (required_size == nullptr || buffer_size < 0 || (buffer == nullptr && buffer_size != 0)) {
        return HERR_INVALID_PARAM;
    }
    if (value.size() >= static_cast<size_t>(std::numeric_limits<HInt32>::max())) {
        return HERR_INVALID_BUFFER_SIZE;
    }
    const HInt32 required = static_cast<HInt32>(value.size() + 1);
    *required_size = required;
    if (buffer == nullptr) {
        return HSUCCEED;
    }
    if (buffer_size < required) {
        if (buffer_size > 0) {
            buffer[0] = '\0';
        }
        return HERR_INVALID_BUFFER_SIZE;
    }
    std::memcpy(buffer, value.c_str(), static_cast<size_t>(required));
    return HSUCCEED;
}

inspire::ResourceManager::ResourceLease AcquireStream(HFImageStream handle) {
    return RESOURCE_MANAGE->acquireStream(reinterpret_cast<inspire::ResourceHandle>(handle));
}

inspire::ResourceManager::ResourceLease AcquireBitmap(HFImageBitmap handle) {
    return RESOURCE_MANAGE->acquireImageBitmap(reinterpret_cast<inspire::ResourceHandle>(handle));
}

inspire::ResourceManager::ResourceLease AcquireSession(HFSession handle) {
    return RESOURCE_MANAGE->acquireSession(reinterpret_cast<inspire::ResourceHandle>(handle));
}

inspire::ResourceManager::ResourceLease AcquireFaceResultSnapshot(HFFaceResultSnapshot handle) {
    return RESOURCE_MANAGE->acquireFaceResultSnapshot(reinterpret_cast<inspire::ResourceHandle>(handle));
}

inspire::ResourceManager::ResourceLease AcquireFaceCaptureSession(HFFaceCaptureSession handle) {
    return RESOURCE_MANAGE->acquireFaceCaptureSession(reinterpret_cast<inspire::ResourceHandle>(handle));
}

HFFaceCaptureMetrics ToCaptureMetrics(const inspire::FaceCaptureMetrics& source) {
    HFFaceCaptureMetrics target{};
    target.availableMetrics = source.availableMetrics;
    target.faceWidthRatio = source.faceWidthRatio;
    target.centerOffsetX = source.centerOffsetX;
    target.centerOffsetY = source.centerOffsetY;
    target.stabilityScore = source.stabilityScore;
    target.poseScore = source.poseScore;
    target.qualityScore = source.qualityScore;
    target.sharpnessScore = source.sharpnessScore;
    target.brightnessScore = source.brightnessScore;
    return target;
}

void ToCaptureProgress(const inspire::FaceCaptureUpdate& source, PHFFaceCaptureProgress target) {
    *target = HFFaceCaptureProgress{};
    target->state = static_cast<HInt32>(source.state);
    target->candidateCount = source.candidateCount;
    target->frameId = source.frameId;
    target->timestampMs = source.timestampMs;
    target->trackId = source.trackId;
    target->trackCount = source.trackCount;
    target->evaluatedFilters = source.evaluatedFilters;
    target->rejectReasons = source.rejectReasons;
    target->progress = source.progress;
    target->currentScore = source.currentScore;
    target->metrics = ToCaptureMetrics(source.metrics);
}

inspire::FaceCaptureConfig DefaultCaptureConfig() {
    return inspire::FaceCaptureConfig{};
}

void ToCCaptureConfig(const inspire::FaceCaptureConfig& source, PHFFaceCaptureConfig target) {
    *target = HFFaceCaptureConfig{};
    target->structSize = sizeof(HFFaceCaptureConfig);
    target->structVersion = HF_FACE_CAPTURE_CONFIG_VERSION;
    target->filterMask = source.filterMask;
    target->outputCount = source.outputCount;
    target->minTrackCount = source.minTrackCount;
    target->stableDurationMs = source.stableDurationMs;
    target->collectDurationMs = source.collectDurationMs;
    target->maxCollectDurationMs = source.maxCollectDurationMs;
    target->trackLostGraceMs = source.trackLostGraceMs;
    target->minCandidateIntervalMs = source.minCandidateIntervalMs;
    target->minFaceWidthRatio = source.minFaceWidthRatio;
    target->maxFaceWidthRatio = source.maxFaceWidthRatio;
    target->maxCenterOffsetX = source.maxCenterOffsetX;
    target->maxCenterOffsetY = source.maxCenterOffsetY;
    target->boundaryMarginRatio = source.boundaryMarginRatio;
    target->maxCenterMotionRatio = source.maxCenterMotionRatio;
    target->maxSizeChangeRatio = source.maxSizeChangeRatio;
    target->maxAbsYaw = source.maxAbsYaw;
    target->maxAbsPitch = source.maxAbsPitch;
    target->maxAbsRoll = source.maxAbsRoll;
    target->minQualityScore = source.minQualityScore;
    target->minSharpnessScore = source.minSharpnessScore;
    target->minBrightnessScore = source.minBrightnessScore;
    target->maxBrightnessScore = source.maxBrightnessScore;
}

inspire::FaceCaptureConfig FromCCaptureConfig(const HFFaceCaptureConfig& source) {
    inspire::FaceCaptureConfig target;
    target.filterMask = source.filterMask;
    target.outputCount = source.outputCount;
    target.minTrackCount = source.minTrackCount;
    target.stableDurationMs = source.stableDurationMs;
    target.collectDurationMs = source.collectDurationMs;
    target.maxCollectDurationMs = source.maxCollectDurationMs;
    target.trackLostGraceMs = source.trackLostGraceMs;
    target.minCandidateIntervalMs = source.minCandidateIntervalMs;
    target.minFaceWidthRatio = source.minFaceWidthRatio;
    target.maxFaceWidthRatio = source.maxFaceWidthRatio;
    target.maxCenterOffsetX = source.maxCenterOffsetX;
    target.maxCenterOffsetY = source.maxCenterOffsetY;
    target.boundaryMarginRatio = source.boundaryMarginRatio;
    target.maxCenterMotionRatio = source.maxCenterMotionRatio;
    target.maxSizeChangeRatio = source.maxSizeChangeRatio;
    target.maxAbsYaw = source.maxAbsYaw;
    target.maxAbsPitch = source.maxAbsPitch;
    target.maxAbsRoll = source.maxAbsRoll;
    target.minQualityScore = source.minQualityScore;
    target.minSharpnessScore = source.minSharpnessScore;
    target.minBrightnessScore = source.minBrightnessScore;
    target.maxBrightnessScore = source.maxBrightnessScore;
    return target;
}

bool IsCaptureConfigHeaderValid(const HFFaceCaptureConfig* config) {
    if (config == nullptr || config->structSize < sizeof(HFFaceCaptureConfig) ||
        config->structVersion != HF_FACE_CAPTURE_CONFIG_VERSION) {
        return false;
    }
    for (HFUInt32 value : config->reserved) {
        if (value != 0) return false;
    }
    return true;
}

HResult DeserializeFaces(const HFMultipleFaceData& source, std::vector<inspire::FaceTrackWrap>* faces) {
    if (faces == nullptr || source.detectedNum < 0 ||
        (source.detectedNum > 0 && source.tokens == nullptr)) {
        return HERR_INVALID_PARAM;
    }
    faces->clear();
    faces->reserve(static_cast<size_t>(source.detectedNum));
    for (HInt32 index = 0; index < source.detectedNum; ++index) {
        const HFFaceBasicToken& token = source.tokens[index];
        if (token.data == nullptr || token.size != static_cast<HInt32>(sizeof(inspire::FaceTrackWrap))) {
            return HERR_SESS_FACE_DATA_ERROR;
        }
        inspire::FaceTrackWrap face{};
        std::memcpy(&face, token.data, sizeof(face));
        faces->push_back(face);
    }
    return HSUCCEED;
}

bool IsValidDetectMode(HFDetectMode mode) {
    return mode >= HF_DETECT_MODE_ALWAYS_DETECT && mode <= HF_DETECT_MODE_TRACK_BY_DETECTION;
}

bool IsValidFaceToken(HFFaceBasicToken token) {
    return token.data != nullptr && token.size == static_cast<HInt32>(sizeof(inspire::FaceTrackWrap));
}

}  // namespace

HYPER_CAPI_EXPORT extern HResult HFCreateImageStream(PHFImageData data, PHFImageStream handle) {
    if (data == nullptr || handle == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    *handle = nullptr;
    size_t byte_size = 0;
    if (data->data == nullptr || !CheckedImageByteSize(data->format, data->width, data->height, &byte_size) ||
        !IsValidRotation(data->rotation)) {
        return HERR_INVALID_IMAGE_STREAM_PARAM;
    }

    try {
        auto stream = std::make_shared<HF_CameraStream>();
    switch (data->rotation) {
        case HF_CAMERA_ROTATION_90:
            stream->impl.SetRotationMode(inspirecv::ROTATION_90);
            break;
        case HF_CAMERA_ROTATION_180:
            stream->impl.SetRotationMode(inspirecv::ROTATION_180);
            break;
        case HF_CAMERA_ROTATION_270:
            stream->impl.SetRotationMode(inspirecv::ROTATION_270);
            break;
        default:
            stream->impl.SetRotationMode(inspirecv::ROTATION_0);
            break;
    }
    switch (data->format) {
        case HF_STREAM_RGB:
            stream->impl.SetDataFormat(inspirecv::RGB);
            break;
        case HF_STREAM_BGR:
            stream->impl.SetDataFormat(inspirecv::BGR);
            break;
        case HF_STREAM_RGBA:
            stream->impl.SetDataFormat(inspirecv::RGBA);
            break;
        case HF_STREAM_BGRA:
            stream->impl.SetDataFormat(inspirecv::BGRA);
            break;
        case HF_STREAM_YUV_NV12:
            stream->impl.SetDataFormat(inspirecv::NV12);
            break;
        case HF_STREAM_YUV_NV21:
            stream->impl.SetDataFormat(inspirecv::NV21);
            break;
        case HF_STREAM_I420:
            stream->impl.SetDataFormat(inspirecv::I420);
            break;
        case HF_STREAM_GRAY:
            stream->impl.SetDataFormat(inspirecv::GRAY);
            break;
        default:
            return HERR_INVALID_IMAGE_STREAM_PARAM;  // Assume there's a return code for unsupported
                                                     // formats
    }
    stream->format = data->format;
    stream->impl.SetDataBuffer(data->data, data->height, data->width);

    const auto resource_handle = reinterpret_cast<inspire::ResourceHandle>(stream.get());
    if (!RESOURCE_MANAGE->createStream(resource_handle, stream)) {
        return HERR_UNKNOWN;
    }
    *handle = static_cast<HFImageStream>(stream.get());

    return HSUCCEED;
    } catch (const std::exception& error) {
        INSPIRE_LOGE("Failed to create image stream: %s", error.what());
        return HERR_UNKNOWN;
    } catch (...) {
        return HERR_UNKNOWN;
    }
}

HYPER_CAPI_EXPORT extern HResult HFCreateImageStreamEmpty(PHFImageStream handle) {
    if (handle == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    *handle = nullptr;
    try {
        auto stream = std::make_shared<HF_CameraStream>();
        const auto resource_handle = reinterpret_cast<inspire::ResourceHandle>(stream.get());
        if (!RESOURCE_MANAGE->createStream(resource_handle, stream)) {
            return HERR_UNKNOWN;
        }
        *handle = static_cast<HFImageStream>(stream.get());
        return HSUCCEED;
    } catch (...) {
        return HERR_UNKNOWN;
    }
}

HYPER_CAPI_EXPORT extern HResult HFImageStreamSetBuffer(HFImageStream handle, HPUInt8 buffer, HInt32 width, HInt32 height) {
    auto stream_lease = AcquireStream(handle);
    if (!stream_lease) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    auto *stream = (HF_CameraStream *)handle;
    size_t byte_size = 0;
    if (buffer == nullptr || !CheckedImageByteSize(stream->format, width, height, &byte_size)) {
        return HERR_INVALID_IMAGE_STREAM_PARAM;
    }
    stream->owned_buffer.clear();
    stream->impl.SetDataBuffer(buffer, height, width);
    return HSUCCEED;
}

HYPER_CAPI_EXPORT extern HResult HFImageStreamSetRotation(HFImageStream handle, HFRotation rotation) {
    auto stream_lease = AcquireStream(handle);
    if (!stream_lease) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    if (!IsValidRotation(rotation)) {
        return HERR_INVALID_IMAGE_STREAM_PARAM;
    }
    switch (rotation) {
        case HF_CAMERA_ROTATION_90:
            ((HF_CameraStream *)handle)->impl.SetRotationMode(inspirecv::ROTATION_90);
            break;
        case HF_CAMERA_ROTATION_180:
            ((HF_CameraStream *)handle)->impl.SetRotationMode(inspirecv::ROTATION_180);
            break;
        case HF_CAMERA_ROTATION_270:
            ((HF_CameraStream *)handle)->impl.SetRotationMode(inspirecv::ROTATION_270);
            break;
        default:
            ((HF_CameraStream *)handle)->impl.SetRotationMode(inspirecv::ROTATION_0);
            break;
    }
    return HSUCCEED;
}

HYPER_CAPI_EXPORT extern HResult HFImageStreamSetFormat(HFImageStream handle, HFImageFormat format) {
    auto stream_lease = AcquireStream(handle);
    if (!stream_lease) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    if (!IsValidImageFormat(format)) {
        return HERR_INVALID_IMAGE_STREAM_PARAM;
    }
    auto *stream = (HF_CameraStream *)handle;
    if ((stream->impl.GetWidth() > 0 || stream->impl.GetHeight() > 0) &&
        !IsValidImageDimensionsForFormat(format, stream->impl.GetWidth(), stream->impl.GetHeight())) {
        return HERR_INVALID_IMAGE_STREAM_PARAM;
    }
    if (!stream->owned_buffer.empty()) {
        size_t required_size = 0;
        if (!CheckedImageByteSize(format, stream->impl.GetWidth(), stream->impl.GetHeight(), &required_size) ||
            required_size > stream->owned_buffer.size()) {
            return HERR_INVALID_IMAGE_STREAM_PARAM;
        }
    }
    switch (format) {
        case HF_STREAM_RGB:
            ((HF_CameraStream *)handle)->impl.SetDataFormat(inspirecv::RGB);
            break;
        case HF_STREAM_BGR:
            ((HF_CameraStream *)handle)->impl.SetDataFormat(inspirecv::BGR);
            break;
        case HF_STREAM_RGBA:
            ((HF_CameraStream *)handle)->impl.SetDataFormat(inspirecv::RGBA);
            break;
        case HF_STREAM_BGRA:
            ((HF_CameraStream *)handle)->impl.SetDataFormat(inspirecv::BGRA);
            break;
        case HF_STREAM_YUV_NV12:
            ((HF_CameraStream *)handle)->impl.SetDataFormat(inspirecv::NV12);
            break;
        case HF_STREAM_YUV_NV21:
            ((HF_CameraStream *)handle)->impl.SetDataFormat(inspirecv::NV21);
            break;
        case HF_STREAM_I420:
            ((HF_CameraStream *)handle)->impl.SetDataFormat(inspirecv::I420);
            break;
        case HF_STREAM_GRAY:
            ((HF_CameraStream *)handle)->impl.SetDataFormat(inspirecv::GRAY);
            break;
        default:
            return HERR_INVALID_IMAGE_STREAM_PARAM;  // Assume there's a return code for unsupported
                                                     // formats
    }
    stream->format = format;
    return HSUCCEED;
}

HYPER_CAPI_EXPORT extern HResult HFReleaseImageStream(HFImageStream streamHandle) {
    if (streamHandle == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    // Check and mark this stream as released in the ResourceManager
    if (!RESOURCE_MANAGE->releaseStream(reinterpret_cast<inspire::ResourceHandle>(streamHandle))) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;  // or other appropriate error code
    }
    return HSUCCEED;
}

HYPER_CAPI_EXPORT extern HResult HFCreateImageBitmap(PHFImageBitmapData data, PHFImageBitmap handle) {
    if (data == nullptr || handle == nullptr) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
    *handle = nullptr;
    size_t byte_size = 0;
    if (data->data == nullptr || !CheckedBitmapByteSize(data->width, data->height, data->channels, &byte_size)) {
        return HERR_INVALID_PARAM;
    }
    try {
        auto bitmap = std::make_shared<HF_ImageBitmap>();
        bitmap->impl.Reset(data->width, data->height, data->channels, data->data);
        if (bitmap->impl.Empty()) {
            return HERR_INVALID_PARAM;
        }
        const auto resource_handle = reinterpret_cast<inspire::ResourceHandle>(bitmap.get());
        if (!RESOURCE_MANAGE->createImageBitmap(resource_handle, bitmap)) {
            return HERR_UNKNOWN;
        }
        *handle = static_cast<HFImageBitmap>(bitmap.get());
        return HSUCCEED;
    } catch (const std::exception& error) {
        INSPIRE_LOGE("Failed to create image bitmap: %s", error.what());
        return HERR_UNKNOWN;
    } catch (...) {
        return HERR_UNKNOWN;
    }
}

HYPER_CAPI_EXPORT extern HResult HFCreateImageBitmapFromFilePath(HPath filePath, HInt32 channels, PHFImageBitmap handle) {
    if (handle == nullptr) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
    *handle = nullptr;
    if (filePath == nullptr || filePath[0] == '\0' || (channels != 1 && channels != 3)) {
        return HERR_INVALID_PARAM;
    }
#if defined(_WIN32)
    const auto wide_path = inspire::os::Utf8ToWideChar(filePath);
    FILE* image_file = wide_path.empty() ? nullptr : _wfopen(wide_path.c_str(), L"rb");
    if (image_file == nullptr) {
        return HERR_IMAGE_STREAM_DECODE_FAILED;
    }
    std::fclose(image_file);
#else
    std::ifstream image_file(filePath, std::ios::binary);
    if (!image_file.good()) {
        return HERR_IMAGE_STREAM_DECODE_FAILED;
    }
#endif
    try {
        auto image = inspirecv::Image::Create(filePath, channels);
        size_t byte_size = 0;
        if (image.Empty() || !CheckedBitmapByteSize(image.Width(), image.Height(), image.Channels(), &byte_size)) {
            return HERR_IMAGE_STREAM_DECODE_FAILED;
        }
        auto bitmap = std::make_shared<HF_ImageBitmap>();
        bitmap->impl.Reset(image.Width(), image.Height(), image.Channels(), image.Data());
        const auto resource_handle = reinterpret_cast<inspire::ResourceHandle>(bitmap.get());
        if (!RESOURCE_MANAGE->createImageBitmap(resource_handle, bitmap)) {
            return HERR_UNKNOWN;
        }
        *handle = static_cast<HFImageBitmap>(bitmap.get());
        return HSUCCEED;
    } catch (const std::exception& error) {
        INSPIRE_LOGE("Failed to decode image bitmap: %s", error.what());
        return HERR_IMAGE_STREAM_DECODE_FAILED;
    } catch (...) {
        return HERR_IMAGE_STREAM_DECODE_FAILED;
    }
}

HYPER_CAPI_EXPORT extern HResult HFImageBitmapCopy(HFImageBitmap handle, PHFImageBitmap copyHandle) {
    if (copyHandle == nullptr) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
    *copyHandle = nullptr;
    auto bitmap_lease = AcquireBitmap(handle);
    if (!bitmap_lease) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
    try {
        auto bitmap = std::make_shared<HF_ImageBitmap>();
        bitmap->impl.Reset(((HF_ImageBitmap *)handle)->impl.Width(), ((HF_ImageBitmap *)handle)->impl.Height(),
                           ((HF_ImageBitmap *)handle)->impl.Channels(), ((HF_ImageBitmap *)handle)->impl.Data());
        const auto resource_handle = reinterpret_cast<inspire::ResourceHandle>(bitmap.get());
        if (!RESOURCE_MANAGE->createImageBitmap(resource_handle, bitmap)) {
            return HERR_UNKNOWN;
        }
        *copyHandle = static_cast<HFImageBitmap>(bitmap.get());
        return HSUCCEED;
    } catch (...) {
        return HERR_UNKNOWN;
    }
}

HYPER_CAPI_EXPORT extern HResult HFReleaseImageBitmap(HFImageBitmap handle) {
    if (handle == nullptr) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
    // Check and mark this image bitmap as released in the ResourceManager
    if (!RESOURCE_MANAGE->releaseImageBitmap(reinterpret_cast<inspire::ResourceHandle>(handle))) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;  // or other appropriate error code
    }
    return HSUCCEED;
}

HYPER_CAPI_EXPORT extern HResult HFCreateImageStreamFromImageBitmap(HFImageBitmap handle, HFRotation rotation, PHFImageStream streamHandle) {
    if (streamHandle == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    *streamHandle = nullptr;
    auto bitmap_lease = AcquireBitmap(handle);
    if (!bitmap_lease) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    if (!IsValidRotation(rotation)) {
        return HERR_INVALID_IMAGE_STREAM_PARAM;
    }
    try {
        auto stream = std::make_shared<HF_CameraStream>();
    switch (rotation) {
        case HF_CAMERA_ROTATION_90:
            stream->impl.SetRotationMode(inspirecv::ROTATION_90);
            break;
        case HF_CAMERA_ROTATION_180:
            stream->impl.SetRotationMode(inspirecv::ROTATION_180);
            break;
        case HF_CAMERA_ROTATION_270:
            stream->impl.SetRotationMode(inspirecv::ROTATION_270);
            break;
        default:
            stream->impl.SetRotationMode(inspirecv::ROTATION_0);
            break;
    }
    if (((HF_ImageBitmap *)handle)->impl.Channels() == 1) {
        stream->impl.SetDataFormat(inspirecv::GRAY);
        stream->format = HF_STREAM_GRAY;
    } else {
        stream->impl.SetDataFormat(inspirecv::BGR);
        stream->format = HF_STREAM_BGR;
    }
    const auto& image = ((HF_ImageBitmap *)handle)->impl;
    size_t byte_size = 0;
    if (!CheckedBitmapByteSize(image.Width(), image.Height(), image.Channels(), &byte_size)) {
        return HERR_INVALID_IMAGE_STREAM_PARAM;
    }
    stream->owned_buffer.assign(image.Data(), image.Data() + byte_size);
    stream->impl.SetDataBuffer(stream->owned_buffer.data(), image.Height(), image.Width());
    const auto resource_handle = reinterpret_cast<inspire::ResourceHandle>(stream.get());
    if (!RESOURCE_MANAGE->createStream(resource_handle, stream)) {
        return HERR_UNKNOWN;
    }
    *streamHandle = static_cast<HFImageStream>(stream.get());
    return HSUCCEED;
    } catch (const std::exception& error) {
        INSPIRE_LOGE("Failed to create image stream from bitmap: %s", error.what());
        return HERR_UNKNOWN;
    } catch (...) {
        return HERR_UNKNOWN;
    }
}

HYPER_CAPI_EXPORT extern HResult HFCreateImageBitmapFromImageStreamProcess(HFImageStream streamHandle, PHFImageBitmap handle, HInt32 is_rotate,
                                                                           HFloat scale) {
    if (handle == nullptr) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
    *handle = nullptr;
    auto stream_lease = AcquireStream(streamHandle);
    if (!stream_lease) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
    auto* stream = static_cast<HF_CameraStream*>(streamHandle);
    if (!CanScaleImage(stream->impl, is_rotate, scale)) {
        return HERR_INVALID_PARAM;
    }
    try {
        auto bitmap = std::make_shared<HF_ImageBitmap>();
        auto img = stream->impl.ExecuteImageScaleProcessing(scale, is_rotate == 1);
        if (img.Empty()) {
            return HERR_DEVICE_IMAGE_PROCESS_FAILURE;
        }
        bitmap->impl.Reset(img.Width(), img.Height(), img.Channels(), img.Data());
        if (bitmap->impl.Empty()) {
            return HERR_DEVICE_IMAGE_PROCESS_FAILURE;
        }
        const auto resource_handle = reinterpret_cast<inspire::ResourceHandle>(bitmap.get());
        if (!RESOURCE_MANAGE->createImageBitmap(resource_handle, bitmap)) {
            return HERR_UNKNOWN;
        }
        *handle = static_cast<HFImageBitmap>(bitmap.get());
        return HSUCCEED;
    } catch (const std::exception& error) {
        INSPIRE_LOGE("Failed to process image stream: %s", error.what());
        return HERR_DEVICE_IMAGE_PROCESS_FAILURE;
    } catch (...) {
        return HERR_DEVICE_IMAGE_PROCESS_FAILURE;
    }
}

HYPER_CAPI_EXPORT extern HResult HFImageBitmapWriteToFile(HFImageBitmap handle, HPath filePath) {
    auto bitmap_lease = AcquireBitmap(handle);
    if (!bitmap_lease) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
    if (filePath == nullptr || filePath[0] == '\0') {
        return HERR_INVALID_PARAM;
    }
    auto success = ((HF_ImageBitmap *)handle)->impl.Write(filePath);
    if (success) {
        return HSUCCEED;
    } else {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
}

HYPER_CAPI_EXPORT extern HResult HFImageBitmapDrawRect(HFImageBitmap handle, HFaceRect rect, HColor color, HInt32 thickness) {
    auto bitmap_lease = AcquireBitmap(handle);
    if (!bitmap_lease) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
    if (rect.width <= 0 || rect.height <= 0 || thickness == 0) {
        return HERR_INVALID_PARAM;
    }
    inspirecv::Rect<int> rect_inner(rect.x, rect.y, rect.width, rect.height);
    ((HF_ImageBitmap *)handle)->impl.DrawRect(rect_inner, {color.r, color.g, color.b}, thickness);
    return HSUCCEED;
}

HYPER_CAPI_EXPORT extern HResult HFImageBitmapDrawCircle(HFImageBitmap handle, HPoint2i point, HInt32 radius, HColor color, HInt32 thickness) {
    auto bitmap_lease = AcquireBitmap(handle);
    if (!bitmap_lease) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
    if (radius < 0 || thickness == 0) {
        return HERR_INVALID_PARAM;
    }
    ((HF_ImageBitmap *)handle)->impl.DrawCircle({point.x, point.y}, radius, {color.r, color.g, color.b}, thickness);
    return HSUCCEED;
}

HYPER_CAPI_EXPORT extern HResult HFImageBitmapDrawCircleF(HFImageBitmap handle, HPoint2f point, HInt32 radius, HColor color, HInt32 thickness) {
    auto bitmap_lease = AcquireBitmap(handle);
    if (!bitmap_lease) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || radius < 0 || thickness == 0) {
        return HERR_INVALID_PARAM;
    }
    ((HF_ImageBitmap *)handle)->impl.DrawCircle({(int)point.x, (int)point.y}, radius, {color.r, color.g, color.b}, thickness);
    return HSUCCEED;
}

HYPER_CAPI_EXPORT extern HResult HFImageBitmapGetData(HFImageBitmap handle, PHFImageBitmapData data) {
    if (data == nullptr) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
    *data = {};
    auto bitmap_lease = AcquireBitmap(handle);
    if (!bitmap_lease) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
    data->width = ((HF_ImageBitmap *)handle)->impl.Width();
    data->height = ((HF_ImageBitmap *)handle)->impl.Height();
    data->channels = ((HF_ImageBitmap *)handle)->impl.Channels();
    data->data = (uint8_t *)((HF_ImageBitmap *)handle)->impl.Data();
    return HSUCCEED;
}

HYPER_CAPI_EXPORT extern HResult HFImageBitmapShow(HFImageBitmap handle, HString title, HInt32 delay) {
    auto bitmap_lease = AcquireBitmap(handle);
    if (!bitmap_lease) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
    if (title == nullptr) {
        return HERR_INVALID_PARAM;
    }
    ((HF_ImageBitmap *)handle)->impl.Show(title, delay);
    return HSUCCEED;
}

void HFDeBugImageStreamImShow(HFImageStream streamHandle) {
    auto stream_lease = AcquireStream(streamHandle);
    if (!stream_lease) {
        INSPIRE_LOGE("Handle error");
        return;
    }
    HF_CameraStream *stream = (HF_CameraStream *)streamHandle;
    if (stream == nullptr) {
        INSPIRE_LOGE("Image error");
        return;
    }
    auto image = stream->impl.ExecuteImageScaleProcessing(1.0f, true);
#ifdef DISABLE_GUI
    image.Write("tmp.jpg");
#else
    image.Show();
#endif
}

HResult HFDeBugImageStreamDecodeSave(HFImageStream streamHandle, HPath savePath) {
    auto stream_lease = AcquireStream(streamHandle);
    if (!stream_lease) {
        INSPIRE_LOGE("Handle error");
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    if (savePath == nullptr || savePath[0] == '\0') {
        return HERR_INVALID_PARAM;
    }
    HF_CameraStream *stream = (HF_CameraStream *)streamHandle;
    if (stream == nullptr) {
        INSPIRE_LOGE("Image error");
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    auto image = stream->impl.ExecuteImageScaleProcessing(1.0f, true);
    auto ret = image.Write(savePath);
    if (ret) {
        INSPIRE_LOGI("Image saved successfully to %s", savePath);
        return HSUCCEED;
    } else {
        INSPIRE_LOGE("Failed to save image to %s", savePath);
        return HERR_IMAGE_STREAM_DECODE_FAILED;
    }
}

HResult HFReleaseInspireFaceSession(HFSession handle) {
    if (handle == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    // Check and mark this session as released in the ResourceManager
    if (!RESOURCE_MANAGE->releaseSession(reinterpret_cast<inspire::ResourceHandle>(handle))) {
        return HERR_INVALID_CONTEXT_HANDLE;  // or other appropriate error code
    }
    return HSUCCEED;
}

HResult HFSessionClearTrackingFace(HFSession session) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    ctx->impl.ClearTrackingFace();
    return HSUCCEED;
}

HResult HFSessionSetTrackLostRecoveryMode(HFSession session, HInt32 enable) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (enable != 0 && enable != 1) {
        return HERR_INVALID_PARAM;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    ctx->impl.SetTrackLostRecoveryMode(enable);
    return HSUCCEED;
}

HResult HFSessionSetLightTrackConfidenceThreshold(HFSession session, HFloat value) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
        return HERR_INVALID_PARAM;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    ctx->impl.SetLightTrackConfidenceThreshold(value);
    return HSUCCEED;
}

HResult HFSwitchLandmarkEngine(HFSessionLandmarkEngine engine) {
    inspire::Launch::LandmarkEngine type;
    if (engine == HF_LANDMARK_HYPLMV2_0_25) {
        type = inspire::Launch::LANDMARK_HYPLMV2_0_25;
    } else if (engine == HF_LANDMARK_HYPLMV2_0_50) {
        type = inspire::Launch::LANDMARK_HYPLMV2_0_50;
    } else if (engine == HF_LANDMARK_INSIGHTFACE_2D106_TRACK) {
        type = inspire::Launch::LANDMARK_INSIGHTFACE_2D106_TRACK;
    } else {
        INSPIRE_LOGE("Unsupported Landmark engine.");
        return HERR_INVALID_PARAM;
    }
    return INSPIREFACE_CONTEXT->SwitchLandmarkEngine(type);
}

HResult HFQuerySupportedPixelLevelsForFaceDetection(PHFFaceDetectPixelList pixel_levels) {
    if (pixel_levels == nullptr) {
        return HERR_INVALID_PARAM;
    }
    auto ret = INSPIREFACE_CONTEXT->GetFaceDetectPixelList();
    if (ret.size() > sizeof(pixel_levels->pixel_level) / sizeof(pixel_levels->pixel_level[0])) {
        return HERR_INVALID_BUFFER_SIZE;
    }
    pixel_levels->size = ret.size();
    for (int i = 0; i < ret.size(); i++) {
        pixel_levels->pixel_level[i] = ret[i];
    }
    return HSUCCEED;
}

HResult HFCreateInspireFaceSession(HFSessionCustomParameter parameter, HFDetectMode detectMode, HInt32 maxDetectFaceNum, HInt32 detectPixelLevel,
                                   HInt32 trackByDetectModeFPS, PHFSession handle) {
    if (handle == nullptr) {
        return HERR_INVALID_PARAM;
    }
    *handle = nullptr;
    if (!IsValidDetectMode(detectMode) || maxDetectFaceNum <= 0) {
        return HERR_INVALID_PARAM;
    }
    inspire::ContextCustomParameter param;
    param.enable_mask_detect = parameter.enable_mask_detect;
    param.enable_liveness = parameter.enable_liveness;
    param.enable_face_quality = parameter.enable_face_quality;
    param.enable_interaction_liveness = parameter.enable_interaction_liveness;
    param.enable_ir_liveness = parameter.enable_ir_liveness;
    param.enable_recognition = parameter.enable_recognition;
    param.enable_face_attribute = parameter.enable_face_attribute;
    param.enable_face_pose = parameter.enable_face_pose;
    param.enable_face_emotion = parameter.enable_face_emotion;
    inspire::DetectModuleMode detMode = inspire::DETECT_MODE_ALWAYS_DETECT;
    if (detectMode == HF_DETECT_MODE_LIGHT_TRACK) {
        detMode = inspire::DETECT_MODE_LIGHT_TRACK;
    } else if (detectMode == HF_DETECT_MODE_TRACK_BY_DETECTION) {
        detMode = inspire::DETECT_MODE_TRACK_BY_DETECT;
    }

    try {
        auto ctx = std::make_shared<HF_FaceAlgorithmSession>();
        const HResult ret = ctx->impl.Configuration(detMode, maxDetectFaceNum, param, detectPixelLevel, trackByDetectModeFPS);
        if (ret != HSUCCEED) {
            return ret;
        }
        const auto resource_handle = reinterpret_cast<inspire::ResourceHandle>(ctx.get());
        if (!RESOURCE_MANAGE->createSession(resource_handle, ctx)) {
            return HERR_UNKNOWN;
        }
        *handle = static_cast<HFSession>(ctx.get());
        return HSUCCEED;
    } catch (const std::exception& error) {
        INSPIRE_LOGE("Failed to create session: %s", error.what());
        return HERR_UNKNOWN;
    } catch (...) {
        return HERR_UNKNOWN;
    }
}

HResult HFCreateInspireFaceSessionOptional(HOption customOption, HFDetectMode detectMode, HInt32 maxDetectFaceNum, HInt32 detectPixelLevel,
                                           HInt32 trackByDetectModeFPS, PHFSession handle) {
    if (handle == nullptr) {
        return HERR_INVALID_PARAM;
    }
    *handle = nullptr;
    constexpr HOption kSupportedOptions = HF_ENABLE_FACE_RECOGNITION | HF_ENABLE_LIVENESS | HF_ENABLE_IR_LIVENESS |
                                           HF_ENABLE_MASK_DETECT | HF_ENABLE_FACE_ATTRIBUTE | HF_ENABLE_PLACEHOLDER_ |
                                           HF_ENABLE_QUALITY | HF_ENABLE_INTERACTION | HF_ENABLE_FACE_POSE |
                                           HF_ENABLE_FACE_EMOTION;
    if (!IsValidDetectMode(detectMode) || maxDetectFaceNum <= 0 || (customOption & ~kSupportedOptions) != 0) {
        return HERR_INVALID_PARAM;
    }
    inspire::ContextCustomParameter param;
    if (customOption & HF_ENABLE_FACE_RECOGNITION) {
        param.enable_recognition = true;
    }
    if (customOption & HF_ENABLE_LIVENESS) {
        param.enable_liveness = true;
    }
    if (customOption & HF_ENABLE_IR_LIVENESS) {
        param.enable_ir_liveness = true;
    }
    if (customOption & HF_ENABLE_FACE_ATTRIBUTE) {
        param.enable_face_attribute = true;
    }
    if (customOption & HF_ENABLE_MASK_DETECT) {
        param.enable_mask_detect = true;
    }
    if (customOption & HF_ENABLE_QUALITY) {
        param.enable_face_quality = true;
    }
    if (customOption & HF_ENABLE_INTERACTION) {
        param.enable_interaction_liveness = true;
    }
    if (customOption & HF_ENABLE_FACE_POSE) {
        param.enable_face_pose = true;
    }
    if (customOption & HF_ENABLE_FACE_EMOTION) {
        param.enable_face_emotion = true;
    }
    inspire::DetectModuleMode detMode = inspire::DETECT_MODE_ALWAYS_DETECT;
    if (detectMode == HF_DETECT_MODE_LIGHT_TRACK) {
        detMode = inspire::DETECT_MODE_LIGHT_TRACK;
    } else if (detectMode == HF_DETECT_MODE_TRACK_BY_DETECTION) {
        detMode = inspire::DETECT_MODE_TRACK_BY_DETECT;
    }

    try {
        auto ctx = std::make_shared<HF_FaceAlgorithmSession>();
        const HResult ret = ctx->impl.Configuration(detMode, maxDetectFaceNum, param, detectPixelLevel, trackByDetectModeFPS);
        if (ret != HSUCCEED) {
            return ret;
        }
        const auto resource_handle = reinterpret_cast<inspire::ResourceHandle>(ctx.get());
        if (!RESOURCE_MANAGE->createSession(resource_handle, ctx)) {
            return HERR_UNKNOWN;
        }
        *handle = static_cast<HFSession>(ctx.get());
        return HSUCCEED;
    } catch (const std::exception& error) {
        INSPIRE_LOGE("Failed to create optional session: %s", error.what());
        return HERR_UNKNOWN;
    } catch (...) {
        return HERR_UNKNOWN;
    }
}

HFStatus HFCreateInspireFaceSessionV2(const HFSessionConfigV2* config, PHFSession handle) {
    if (handle == nullptr) {
        return HERR_INVALID_PARAM;
    }
    *handle = nullptr;
    if (config == nullptr || config->structSize < sizeof(HFSessionConfigV2) ||
        config->structVersion != HF_SESSION_CONFIG_V2_VERSION) {
        return HERR_INVALID_PARAM;
    }
    for (const HFUInt32 value : config->reserved) {
        if (value != 0) {
            return HERR_INVALID_PARAM;
        }
    }

    constexpr HFUInt64 kSupportedFeatureMask =
      static_cast<HFUInt64>(HF_ENABLE_FACE_RECOGNITION | HF_ENABLE_LIVENESS | HF_ENABLE_MASK_DETECT |
                            HF_ENABLE_FACE_ATTRIBUTE | HF_ENABLE_QUALITY | HF_ENABLE_INTERACTION |
                            HF_ENABLE_FACE_POSE | HF_ENABLE_FACE_EMOTION);
    if ((config->featureMask & ~kSupportedFeatureMask) != 0) {
        return HERR_UNSUPPORTED;
    }
    if (config->detectMode < HF_DETECT_MODE_ALWAYS_DETECT || config->detectMode > HF_DETECT_MODE_TRACK_BY_DETECTION ||
        config->maxDetectFaceNum <= 0) {
        return HERR_INVALID_PARAM;
    }

    return static_cast<HFStatus>(HFCreateInspireFaceSessionOptional(
      static_cast<HOption>(config->featureMask), static_cast<HFDetectMode>(config->detectMode), config->maxDetectFaceNum,
      config->detectPixelLevel, config->trackByDetectModeFPS, handle));
}

HResult HFLaunchInspireFace(HPath resourcePath) {
    if (resourcePath == nullptr) {
        return HERR_INVALID_PARAM;
    }
    return INSPIREFACE_CONTEXT->Load(resourcePath);
}

HResult HFReloadInspireFace(HPath resourcePath) {
    if (resourcePath == nullptr) {
        return HERR_INVALID_PARAM;
    }
    return INSPIREFACE_CONTEXT->Reload(resourcePath);
}

HResult HFTerminateInspireFace() {
    INSPIREFACE_CONTEXT->Unload();
    return HSUCCEED;
}

HResult HFQueryInspireFaceLaunchStatus(HPInt32 status) {
    if (status == nullptr) {
        return HERR_INVALID_PARAM;
    }
    *status = INSPIREFACE_CONTEXT->isMLoad();
    return HSUCCEED;
}

HResult HFFeatureHubDataDisable() {
    return INSPIREFACE_FEATURE_HUB->DisableHub();
}

HResult HFQueryExpansiveHardwareRGACompileOption(HPInt32 enable) {
    if (enable == nullptr) {
        return HERR_INVALID_PARAM;
    }
#if defined(ISF_ENABLE_RGA)
    INSPIRE_LOGI("RGA is enabled during compilation");
    *enable = 1;
#else
    INSPIRE_LOGW("RGA is not enabled during compilation");
    *enable = 0;
#endif
    return HSUCCEED;
}

HResult HFSetExpansiveHardwareRockchipDmaHeapPath(HPath path) {
    if (path == nullptr || std::strlen(path) >= kRockchipDmaHeapPathCapacity) {
        return HERR_INVALID_PARAM;
    }
    INSPIREFACE_CONTEXT->SetRockchipDmaHeapPath(path);
    return HSUCCEED;
}

HResult HFQueryExpansiveHardwareRockchipDmaHeapPath(HString path) {
    return HFQueryExpansiveHardwareRockchipDmaHeapPathWithSize(path, static_cast<HInt32>(kRockchipDmaHeapPathCapacity));
}

HResult HFQueryExpansiveHardwareRockchipDmaHeapPathWithSize(HString path, HInt32 bufferSize) {
    if (path == nullptr || bufferSize <= 0) {
        return HERR_INVALID_PARAM;
    }
    path[0] = '\0';
    const std::string value = INSPIREFACE_CONTEXT->GetRockchipDmaHeapPath();
    if (value.size() + 1 > static_cast<size_t>(bufferSize)) {
        return HERR_INVALID_BUFFER_SIZE;
    }
    std::memcpy(path, value.c_str(), value.size() + 1);
    return HSUCCEED;
}

HResult HFSetAppleCoreMLInferenceMode(HFAppleCoreMLInferenceMode mode) {
    if (mode == HF_APPLE_COREML_INFERENCE_MODE_CPU) {
        INSPIREFACE_CONTEXT->SetGlobalCoreMLInferenceMode(inspire::Launch::NN_INFERENCE_CPU);
    } else if (mode == HF_APPLE_COREML_INFERENCE_MODE_GPU) {
        INSPIREFACE_CONTEXT->SetGlobalCoreMLInferenceMode(inspire::Launch::NN_INFERENCE_COREML_GPU);
    } else if (mode == HF_APPLE_COREML_INFERENCE_MODE_ANE) {
        INSPIREFACE_CONTEXT->SetGlobalCoreMLInferenceMode(inspire::Launch::NN_INFERENCE_COREML_ANE);
    } else {
        INSPIRE_LOGE("Unsupported Apple CoreML inference mode.");
        return HERR_INVALID_PARAM;
    }
    return HSUCCEED;
}

HResult HFSwitchImageProcessingBackend(HFImageProcessingBackend backend) {
    if (backend == HF_IMAGE_PROCESSING_CPU) {
        INSPIREFACE_CONTEXT->SwitchImageProcessingBackend(inspire::Launch::IMAGE_PROCESSING_CPU);
    } else if (backend == HF_IMAGE_PROCESSING_RGA) {
        INSPIREFACE_CONTEXT->SwitchImageProcessingBackend(inspire::Launch::IMAGE_PROCESSING_RGA);
    } else {
        INSPIRE_LOGE("Unsupported image processing backend.");
        return HERR_INVALID_PARAM;
    }
    return HSUCCEED;
}

HResult HFSetImageProcessAlignedWidth(HInt32 width) {
    if (width <= 0) {
        return HERR_INVALID_PARAM;
    }
    INSPIREFACE_CONTEXT->SetImageProcessAlignedWidth(width);
    return HSUCCEED;
}

HResult HFSetCudaDeviceId(HInt32 device_id) {
    if (device_id < 0) {
        return HERR_INVALID_PARAM;
    }
    INSPIREFACE_CONTEXT->SetCudaDeviceId(device_id);
    return HSUCCEED;
}

HResult HFGetCudaDeviceId(HPInt32 device_id) {
    if (device_id == nullptr) {
        return HERR_INVALID_PARAM;
    }
    *device_id = INSPIREFACE_CONTEXT->GetCudaDeviceId();
    return HSUCCEED;
}

HResult HFPrintCudaDeviceInfo() {
#if defined(ISF_ENABLE_TENSORRT)
    return inspire::PrintCudaDeviceInfo();
#else
    INSPIRE_LOGW("CUDA is not supported, you need to enable the compile option that supports TensorRT");
    return HERR_DEVICE_CUDA_DISABLE;
#endif
}

HResult HFGetNumCudaDevices(HPInt32 num_devices) {
    if (num_devices == nullptr) {
        return HERR_INVALID_PARAM;
    }
#if defined(ISF_ENABLE_TENSORRT)
    return inspire::GetCudaDeviceCount(num_devices);
#else
    INSPIRE_LOGW("CUDA is not supported, you need to enable the compile option that supports TensorRT");
    return HERR_DEVICE_CUDA_DISABLE;
#endif
}

HResult HFCheckCudaDeviceSupport(HPInt32 is_support) {
    if (is_support == nullptr) {
        return HERR_INVALID_PARAM;
    }
#if defined(ISF_ENABLE_TENSORRT)
    return inspire::CheckCudaUsability(is_support);
#else
    INSPIRE_LOGW("CUDA is not supported, you need to enable the compile option that supports TensorRT");
    return HERR_DEVICE_CUDA_DISABLE;
#endif
}

HResult HFFeatureHubDataEnable(HFFeatureHubConfiguration configuration) {
    if (configuration.primaryKeyMode != HF_PK_AUTO_INCREMENT && configuration.primaryKeyMode != HF_PK_MANUAL_INPUT) {
        return HERR_INVALID_PARAM;
    }
    if (configuration.searchMode != HF_SEARCH_MODE_EAGER && configuration.searchMode != HF_SEARCH_MODE_EXHAUSTIVE) {
        return HERR_INVALID_PARAM;
    }
    if (configuration.enablePersistence != 0 && configuration.enablePersistence != 1) {
        return HERR_INVALID_PARAM;
    }
    if (!std::isfinite(configuration.searchThreshold) || configuration.searchThreshold < -1.0f || configuration.searchThreshold > 1.0f) {
        return HERR_INVALID_PARAM;
    }
    if (configuration.enablePersistence && (configuration.persistenceDbPath == nullptr || configuration.persistenceDbPath[0] == '\0')) {
        return HERR_INVALID_PARAM;
    }
    inspire::DatabaseConfiguration param;
    param.primary_key_mode = inspire::PrimaryKeyMode(configuration.primaryKeyMode);
    // Add validation for persistenceDbPath
    if (configuration.enablePersistence) {
        if (configuration.persistenceDbPath == nullptr) {
            param.persistence_db_path = std::string("");
        } else {
            param.persistence_db_path = std::string(configuration.persistenceDbPath);
        }
    } else {
        param.persistence_db_path = std::string("");  // Empty string for in-memory mode
    }
    param.enable_persistence = configuration.enablePersistence;
    param.recognition_threshold = configuration.searchThreshold;
    param.search_mode = static_cast<inspire::SearchMode>(configuration.searchMode);
    auto ret = INSPIREFACE_FEATURE_HUB->EnableHub(param);
    return ret;
}

HResult HFSessionSetTrackPreviewSize(HFSession session, HInt32 previewSize) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (previewSize == 0 || previewSize < -1) {
        return HERR_INVALID_PARAM;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    return ctx->impl.SetTrackPreviewSize(previewSize);
}

HResult HFSessionGetTrackPreviewSize(HFSession session, HPInt32 previewSize) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (previewSize == nullptr) {
        return HERR_INVALID_PARAM;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    *previewSize = ctx->impl.GetTrackPreviewSize();
    return HSUCCEED;
}

HResult HFSessionSetFilterMinimumFacePixelSize(HFSession session, HInt32 minSize) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (minSize < 0) {
        return HERR_INVALID_PARAM;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    return ctx->impl.SetTrackFaceMinimumSize(minSize);
}

HResult HFSessionSetFaceTrackMode(HFSession session, HFDetectMode detectMode) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (!IsValidDetectMode(detectMode)) {
        return HERR_INVALID_PARAM;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    inspire::DetectModuleMode detMode = inspire::DETECT_MODE_ALWAYS_DETECT;
    if (detectMode == HF_DETECT_MODE_LIGHT_TRACK) {
        detMode = inspire::DETECT_MODE_LIGHT_TRACK;
    } else if (detectMode == HF_DETECT_MODE_TRACK_BY_DETECTION) {
        detMode = inspire::DETECT_MODE_TRACK_BY_DETECT;
    }
    return ctx->impl.SetDetectMode(detMode);
}

HResult HFSessionSetFaceDetectThreshold(HFSession session, HFloat threshold) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (!std::isfinite(threshold) || threshold < 0.0f || threshold > 1.0f) {
        return HERR_INVALID_PARAM;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    return ctx->impl.SetFaceDetectThreshold(threshold);
}

HResult HFSessionSetTrackModeSmoothRatio(HFSession session, HFloat ratio) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (!std::isfinite(ratio) || ratio < 0.0f || ratio > 1.0f) {
        return HERR_INVALID_PARAM;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    return ctx->impl.SetTrackModeSmoothRatio(ratio);
}

HResult HFSessionSetTrackModeNumSmoothCacheFrame(HFSession session, HInt32 num) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (num <= 0) {
        return HERR_INVALID_PARAM;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    return ctx->impl.SetTrackModeNumSmoothCacheFrame(num);
}

HResult HFSessionSetTrackModeDetectInterval(HFSession session, HInt32 num) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (num <= 0) {
        return HERR_INVALID_PARAM;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    return ctx->impl.SetTrackModeDetectInterval(num);
}

HResult HFSessionSetLandmarkAugmentationNum(HFSession session, HInt32 num) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (num <= 0) {
        return HERR_INVALID_PARAM;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    return ctx->impl.SetLandmarkLoop(num);
}

HResult HFExecuteFaceTrack(HFSession session, HFImageStream streamHandle, PHFMultipleFaceData results) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    auto stream_lease = AcquireStream(streamHandle);
    if (!stream_lease) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    if (results == nullptr) {
        return HERR_INVALID_PARAM;
    }
    *results = HFMultipleFaceData{};
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_CameraStream *stream = (HF_CameraStream *)streamHandle;
    if (stream == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    auto ret = ctx->impl.FaceDetectAndTrack(stream->impl);
    if (ret != HSUCCEED) {
        return ret;
    }
    results->detectedNum = ctx->impl.GetNumberOfFacesCurrentlyDetected();
    if (results->detectedNum > 0) {
        results->rects = (HFaceRect *)ctx->impl.GetFaceRectsCache().data();
        results->trackIds = (HInt32 *)ctx->impl.GetTrackIDCache().data();
        results->detConfidence = (HFloat *)ctx->impl.GetDetConfidenceCache().data();
        results->trackCounts = (HInt32 *)ctx->impl.GetTrackCountCache().data();
        results->angles.pitch = (HFloat *)ctx->impl.GetPitchResultsCache().data();
        results->angles.roll = (HFloat *)ctx->impl.GetRollResultsCache().data();
        results->angles.yaw = (HFloat *)ctx->impl.GetYawResultsCache().data();
        results->tokens = (HFFaceBasicToken *)ctx->impl.GetFaceBasicDataCache().data();
    }

    return HSUCCEED;
}

HResult HFExecuteFaceTrackSnapshot(HFSession session, HFImageStream streamHandle, PHFFaceResultSnapshot snapshot) {
    if (snapshot == nullptr) {
        return HERR_INVALID_PARAM;
    }
    *snapshot = nullptr;
    // Keep the borrowed result cache alive until it has been copied into the
    // owned snapshot, even if another thread releases either public handle.
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    auto stream_lease = AcquireStream(streamHandle);
    if (!stream_lease) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    HFMultipleFaceData borrowed = {};
    const HResult detection_status = HFExecuteFaceTrack(session, streamHandle, &borrowed);
    if (detection_status != HSUCCEED) {
        return detection_status;
    }

    try {
        auto owned = std::make_shared<HF_FaceResultSnapshot>();
        const HResult copy_status = owned->CopyFrom(borrowed);
        if (copy_status != HSUCCEED) {
            return copy_status;
        }
        const auto resource_handle = reinterpret_cast<inspire::ResourceHandle>(owned.get());
        if (!RESOURCE_MANAGE->createFaceResultSnapshot(resource_handle, owned)) {
            return HERR_UNKNOWN;
        }
        *snapshot = static_cast<HFFaceResultSnapshot>(owned.get());
        return HSUCCEED;
    } catch (const std::exception& error) {
        INSPIRE_LOGE("Failed to create face result snapshot: %s", error.what());
        return HERR_UNKNOWN;
    } catch (...) {
        return HERR_UNKNOWN;
    }
}

HResult HFGetFaceResultSnapshotData(HFFaceResultSnapshot snapshot, PHFMultipleFaceData results) {
    if (results == nullptr) {
        return HERR_INVALID_PARAM;
    }
    *results = HFMultipleFaceData{};
    auto snapshot_lease = AcquireFaceResultSnapshot(snapshot);
    if (!snapshot_lease) {
        return HERR_INVALID_PARAM;
    }
    auto* owned = static_cast<HF_FaceResultSnapshot*>(snapshot);
    owned->GetView(results);
    return HSUCCEED;
}

HResult HFReleaseFaceResultSnapshot(HFFaceResultSnapshot snapshot) {
    if (!RESOURCE_MANAGE->releaseFaceResultSnapshot(reinterpret_cast<inspire::ResourceHandle>(snapshot))) {
        return HERR_INVALID_PARAM;
    }
    return HSUCCEED;
}

HResult HFGetDefaultFaceCaptureConfig(PHFFaceCaptureConfig config) {
    if (config == nullptr) {
        return HERR_INVALID_PARAM;
    }
    ToCCaptureConfig(DefaultCaptureConfig(), config);
    return HSUCCEED;
}

HResult HFCreateFaceCaptureSession(HFSession session, const HFFaceCaptureConfig* config,
                                   PHFFaceCaptureSession captureSession) {
    if (captureSession == nullptr) {
        return HERR_INVALID_PARAM;
    }
    *captureSession = nullptr;
    if (!IsCaptureConfigHeaderValid(config)) {
        return HERR_CAPTURE_INVALID_CONFIG;
    }
    auto sessionLease = AcquireSession(session);
    if (!sessionLease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    auto* algorithmSession = static_cast<HF_FaceAlgorithmSession*>(session);
    try {
        auto capture = std::make_shared<HF_FaceCaptureSession>();
        capture->session = algorithmSession;
        capture->session_lease = sessionLease;
        const HResult status = capture->selector.Configure(FromCCaptureConfig(*config),
                                                           algorithmSession->impl.getMParameter());
        if (status != HSUCCEED) {
            return status;
        }
        const auto resourceHandle = reinterpret_cast<inspire::ResourceHandle>(capture.get());
        if (!RESOURCE_MANAGE->createFaceCaptureSession(resourceHandle, capture)) {
            return HERR_UNKNOWN;
        }
        *captureSession = static_cast<HFFaceCaptureSession>(capture.get());
        return HSUCCEED;
    } catch (const std::exception& error) {
        INSPIRE_LOGE("Failed to create face capture session: %s", error.what());
        return HERR_UNKNOWN;
    } catch (...) {
        return HERR_UNKNOWN;
    }
}

HResult HFUpdateFaceCaptureSession(HFFaceCaptureSession captureSession, HFImageStream stream,
                                   HFUInt64 frameId, HFUInt64 timestampMs,
                                   PHFFaceCaptureProgress progress) {
    if (progress == nullptr) {
        return HERR_INVALID_PARAM;
    }
    *progress = HFFaceCaptureProgress{};
    auto captureLease = AcquireFaceCaptureSession(captureSession);
    if (!captureLease) {
        return HERR_CAPTURE_INVALID_HANDLE;
    }
    auto streamLease = AcquireStream(stream);
    if (!streamLease) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    auto* capture = static_cast<HF_FaceCaptureSession*>(captureSession);
    auto* cameraStream = static_cast<HF_CameraStream*>(stream);
    std::lock_guard<std::mutex> lock(capture->mutex);
    const HResult trackStatus = capture->session->impl.FaceDetectAndTrack(cameraStream->impl);
    if (trackStatus != HSUCCEED) {
        return trackStatus;
    }
    std::vector<inspire::FaceTrackWrap> faces;
    const auto& cache = capture->session->impl.GetDetectCache();
    faces.reserve(cache.size());
    for (const auto& serialized : cache) {
        inspire::FaceTrackWrap face{};
        const HResult decodeStatus = inspire::RunDeserializeHyperFaceData(serialized, face);
        if (decodeStatus != HSUCCEED) return decodeStatus;
        faces.push_back(face);
    }
    inspire::FaceCaptureUpdate update;
    const HResult status = capture->selector.Update(cameraStream->impl, faces, frameId, timestampMs, update);
    if (status == HSUCCEED) {
        capture->result_cache_dirty = true;
        ToCaptureProgress(update, progress);
    }
    return status;
}

HResult HFUpdateFaceCaptureSessionWithSnapshot(HFFaceCaptureSession captureSession,
                                               HFImageStream stream, HFFaceResultSnapshot snapshot,
                                               HFUInt64 frameId, HFUInt64 timestampMs,
                                               PHFFaceCaptureProgress progress) {
    if (progress == nullptr) {
        return HERR_INVALID_PARAM;
    }
    *progress = HFFaceCaptureProgress{};
    auto captureLease = AcquireFaceCaptureSession(captureSession);
    if (!captureLease) return HERR_CAPTURE_INVALID_HANDLE;
    auto streamLease = AcquireStream(stream);
    if (!streamLease) return HERR_INVALID_IMAGE_STREAM_HANDLE;
    auto snapshotLease = AcquireFaceResultSnapshot(snapshot);
    if (!snapshotLease) return HERR_INVALID_PARAM;

    HFMultipleFaceData view{};
    static_cast<HF_FaceResultSnapshot*>(snapshot)->GetView(&view);
    std::vector<inspire::FaceTrackWrap> faces;
    const HResult decodeStatus = DeserializeFaces(view, &faces);
    if (decodeStatus != HSUCCEED) return decodeStatus;

    auto* capture = static_cast<HF_FaceCaptureSession*>(captureSession);
    auto* cameraStream = static_cast<HF_CameraStream*>(stream);
    std::lock_guard<std::mutex> lock(capture->mutex);
    inspire::FaceCaptureUpdate update;
    const HResult status = capture->selector.Update(cameraStream->impl, faces, frameId, timestampMs, update);
    if (status == HSUCCEED) {
        capture->result_cache_dirty = true;
        ToCaptureProgress(update, progress);
    }
    return status;
}

HResult HFGetFaceCaptureResults(HFFaceCaptureSession captureSession,
                                PHFFaceCaptureResult results, HFUInt32 capacity,
                                HFUInt32* resultCount) {
    if (resultCount == nullptr || (results == nullptr && capacity != 0)) {
        return HERR_INVALID_PARAM;
    }
    *resultCount = 0;
    auto captureLease = AcquireFaceCaptureSession(captureSession);
    if (!captureLease) return HERR_CAPTURE_INVALID_HANDLE;
    auto* capture = static_cast<HF_FaceCaptureSession*>(captureSession);
    std::lock_guard<std::mutex> lock(capture->mutex);
    if (capture->result_cache_dirty) {
        capture->result_cache = capture->selector.GetResults();
        capture->result_cache_dirty = false;
    }
    const HFUInt32 required = static_cast<HFUInt32>(capture->result_cache.size());
    *resultCount = required;
    if (results == nullptr) return HSUCCEED;
    if (capacity < required) return HERR_INVALID_BUFFER_SIZE;
    for (HFUInt32 index = 0; index < required; ++index) {
        const inspire::FaceCaptureCandidate& source = capture->result_cache[index];
        HFFaceCaptureResult& target = results[index];
        target = HFFaceCaptureResult{};
        target.frameId = source.frameId;
        target.timestampMs = source.timestampMs;
        target.trackId = source.trackId;
        target.trackCount = source.face.trackCount;
        target.score = source.score;
        target.rect.x = source.face.rect.x;
        target.rect.y = source.face.rect.y;
        target.rect.width = source.face.rect.width;
        target.rect.height = source.face.rect.height;
        target.roll = source.face.face3DAngle.roll;
        target.yaw = source.face.face3DAngle.yaw;
        target.pitch = source.face.face3DAngle.pitch;
        target.token.size = static_cast<HInt32>(sizeof(source.face));
        target.token.data = static_cast<HPVoid>(&capture->result_cache[index].face);
        target.metrics = ToCaptureMetrics(source.metrics);
    }
    return HSUCCEED;
}

HResult HFFinishFaceCaptureSession(HFFaceCaptureSession captureSession,
                                   PHFFaceCaptureProgress progress) {
    if (progress == nullptr) return HERR_INVALID_PARAM;
    *progress = HFFaceCaptureProgress{};
    auto captureLease = AcquireFaceCaptureSession(captureSession);
    if (!captureLease) return HERR_CAPTURE_INVALID_HANDLE;
    auto* capture = static_cast<HF_FaceCaptureSession*>(captureSession);
    std::lock_guard<std::mutex> lock(capture->mutex);
    inspire::FaceCaptureUpdate update;
    const HResult status = capture->selector.Finish(update);
    if (status == HSUCCEED) {
        capture->result_cache_dirty = true;
        ToCaptureProgress(update, progress);
    }
    return status;
}

HResult HFResetFaceCaptureSession(HFFaceCaptureSession captureSession) {
    auto captureLease = AcquireFaceCaptureSession(captureSession);
    if (!captureLease) return HERR_CAPTURE_INVALID_HANDLE;
    auto* capture = static_cast<HF_FaceCaptureSession*>(captureSession);
    std::lock_guard<std::mutex> lock(capture->mutex);
    capture->selector.Reset();
    capture->result_cache.clear();
    capture->result_cache_dirty = true;
    return HSUCCEED;
}

HResult HFReleaseFaceCaptureSession(HFFaceCaptureSession captureSession) {
    if (!RESOURCE_MANAGE->releaseFaceCaptureSession(
          reinterpret_cast<inspire::ResourceHandle>(captureSession))) {
        return HERR_CAPTURE_INVALID_HANDLE;
    }
    return HSUCCEED;
}

HResult HFSessionLastFaceDetectionGetDebugPreviewImageSize(HFSession session, HPInt32 size) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (size == nullptr) {
        return HERR_INVALID_PARAM;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    *size = ctx->impl.GetDebugPreviewImageSize();
    return HSUCCEED;
}

HResult HFCopyFaceBasicToken(HFFaceBasicToken token, HPBuffer buffer, HInt32 bufferSize) {
    if (!IsValidFaceToken(token)) {
        return HERR_INVALID_FACE_TOKEN;
    }
    if (bufferSize < sizeof(inspire::FaceTrackWrap)) {
        return HERR_INVALID_BUFFER_SIZE;
    }
    if (buffer == nullptr) {
        return HERR_INVALID_PARAM;
    }
    std::memcpy(buffer, token.data, sizeof(inspire::FaceTrackWrap));
    return HSUCCEED;
}

HResult HFGetFaceBasicTokenSize(HPInt32 bufferSize) {
    if (bufferSize == nullptr) {
        return HERR_INVALID_PARAM;
    }
    *bufferSize = sizeof(inspire::FaceTrackWrap);
    return HSUCCEED;
}

HResult HFGetNumOfFaceDenseLandmark(HPInt32 num) {
    if (num == nullptr) {
        return HERR_INVALID_PARAM;
    }
    *num = 106;
    return HSUCCEED;
}

HResult HFGetFaceDenseLandmarkFromFaceToken(HFFaceBasicToken singleFace, PHPoint2f landmarks, HInt32 num) {
    if (num != 106) {
        return HERR_SESS_LANDMARK_NUM_NOT_MATCH;
    }
    if (!IsValidFaceToken(singleFace)) {
        return HERR_INVALID_FACE_TOKEN;
    }
    if (landmarks == nullptr) {
        return HERR_INVALID_PARAM;
    }
    inspire::FaceBasicData data;
    data.dataSize = singleFace.size;
    data.data = singleFace.data;
    FaceTrackWrap face = {0};
    HInt32 ret;
    ret = RunDeserializeHyperFaceData((char *)data.data, data.dataSize, face);
    if (ret != HSUCCEED) {
        return ret;
    }
    if (face.densityLandmarkEnable == HF_STATUS_DISABLE) {
        INSPIRE_LOGW("To get dense landmarks in always-detect mode, you need to enable HF_ENABLE_DETECT_MODE_LANDMARK");
        return HERR_SESS_LANDMARK_NOT_ENABLE;
    }
    for (size_t i = 0; i < num; i++) {
        landmarks[i].x = face.densityLandmark[i].x;
        landmarks[i].y = face.densityLandmark[i].y;
    }
    return HSUCCEED;
}

HResult HFGetFaceFiveKeyPointsFromFaceToken(HFFaceBasicToken singleFace, PHPoint2f landmarks, HInt32 num) {
    if (num != 5) {
        return HERR_SESS_KEY_POINT_NUM_NOT_MATCH;
    }
    if (!IsValidFaceToken(singleFace)) {
        return HERR_INVALID_FACE_TOKEN;
    }
    if (landmarks == nullptr) {
        return HERR_INVALID_PARAM;
    }
    inspire::FaceBasicData data;
    data.dataSize = singleFace.size;
    data.data = singleFace.data;
    FaceTrackWrap face = {0};
    HInt32 ret;
    ret = RunDeserializeHyperFaceData((char *)data.data, data.dataSize, face);
    if (ret != HSUCCEED) {
        return ret;
    }
    for (size_t i = 0; i < num; i++) {
        landmarks[i].x = face.keyPoints[i].x;
        landmarks[i].y = face.keyPoints[i].y;
    }
    return HSUCCEED;
}

HResult HFSessionSetEnableTrackCostSpend(HFSession session, HInt32 value) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (value != 0 && value != 1) {
        return HERR_INVALID_PARAM;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    ctx->impl.SetEnableTrackCostSpend(value);
    return HSUCCEED;
}

HResult HFSessionPrintTrackCostSpend(HFSession session) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    ctx->impl.PrintTrackCostSpend();
    return HSUCCEED;
}

HResult HFFeatureHubFaceSearchThresholdSetting(HFloat threshold) {
    if (!std::isfinite(threshold) || threshold < -1.0f || threshold > 1.0f) {
        return HERR_INVALID_PARAM;
    }
    INSPIREFACE_FEATURE_HUB->SetRecognitionThreshold(threshold);
    return HSUCCEED;
}

HResult HFFaceFeatureExtract(HFSession session, HFImageStream streamHandle, HFFaceBasicToken singleFace, PHFFaceFeature feature) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    auto stream_lease = AcquireStream(streamHandle);
    if (!stream_lease) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    if (feature == nullptr) {
        return HERR_INVALID_FACE_FEATURE;
    }
    feature->size = 0;
    feature->data = nullptr;
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_CameraStream *stream = (HF_CameraStream *)streamHandle;
    if (stream == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    if (!IsValidFaceToken(singleFace)) {
        return HERR_INVALID_FACE_TOKEN;
    }
    inspire::FaceBasicData data;
    data.dataSize = singleFace.size;
    data.data = singleFace.data;
    auto ret = ctx->impl.FaceFeatureExtract(stream->impl, data);
    if (ret != HSUCCEED) {
        return ret;
    }
    feature->size = ctx->impl.GetFaceFeatureCache().size();
    feature->data = (HFloat *)ctx->impl.GetFaceFeatureCache().data();

    return HSUCCEED;
}

HResult HFFaceFeatureExtractTo(HFSession session, HFImageStream streamHandle, HFFaceBasicToken singleFace, HFFaceFeature feature) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    auto stream_lease = AcquireStream(streamHandle);
    if (!stream_lease) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    if (feature.data == nullptr || feature.size < FACE_FEATURE_SIZE) {
        return HERR_INVALID_FACE_FEATURE;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_CameraStream *stream = (HF_CameraStream *)streamHandle;
    if (stream == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    if (!IsValidFaceToken(singleFace)) {
        return HERR_INVALID_FACE_TOKEN;
    }
    inspire::FaceBasicData data;
    data.dataSize = singleFace.size;
    data.data = singleFace.data;
    auto ret = ctx->impl.FaceFeatureExtract(stream->impl, data);
    if (ret != HSUCCEED) {
        return ret;
    }
    if (ctx->impl.GetFaceFeatureCache().size() > static_cast<size_t>(feature.size)) {
        return HERR_INVALID_BUFFER_SIZE;
    }
    for (int i = 0; i < ctx->impl.GetFaceFeatureCache().size(); ++i) {
        feature.data[i] = ctx->impl.GetFaceFeatureCache()[i];
    }

    return ret;
}

HResult HFFaceFeatureExtractCpy(HFSession session, HFImageStream streamHandle, HFFaceBasicToken singleFace, HPFloat feature) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    auto stream_lease = AcquireStream(streamHandle);
    if (!stream_lease) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    if (feature == nullptr) {
        return HERR_INVALID_FACE_FEATURE;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_CameraStream *stream = (HF_CameraStream *)streamHandle;
    if (stream == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    if (!IsValidFaceToken(singleFace)) {
        return HERR_INVALID_FACE_TOKEN;
    }
    inspire::FaceBasicData data;
    data.dataSize = singleFace.size;
    data.data = singleFace.data;
    auto ret = ctx->impl.FaceFeatureExtract(stream->impl, data);
    if (ret != HSUCCEED) {
        return ret;
    }
    for (int i = 0; i < ctx->impl.GetFaceFeatureCache().size(); ++i) {
        feature[i] = ctx->impl.GetFaceFeatureCache()[i];
    }

    return ret;
}

HResult HFCreateFaceFeature(PHFFaceFeature feature) {
    if (feature == nullptr) {
        return HERR_INVALID_FACE_FEATURE;
    }
    std::unique_ptr<HFloat[]> data(new HFloat[FACE_FEATURE_SIZE]);
    if (!RESOURCE_MANAGE->createFaceFeature(reinterpret_cast<inspire::ResourceHandle>(feature))) {
        return HERR_INVALID_FACE_FEATURE;
    }
    feature->size = FACE_FEATURE_SIZE;
    feature->data = data.release();
    return HSUCCEED;
}

HResult HFReleaseFaceFeature(PHFFaceFeature feature) {
    if (feature == nullptr) {
        return HERR_INVALID_FACE_FEATURE;
    }
    if (!RESOURCE_MANAGE->releaseFaceFeature(reinterpret_cast<inspire::ResourceHandle>(feature))) {
        return HERR_INVALID_FACE_FEATURE;
    }
    delete[] feature->data;
    feature->data = nullptr;
    feature->size = 0;
    return HSUCCEED;
}

HResult HFFaceGetFaceAlignmentImage(HFSession session, HFImageStream streamHandle, HFFaceBasicToken singleFace, PHFImageBitmap handle) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    auto stream_lease = AcquireStream(streamHandle);
    if (!stream_lease) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    if (handle == nullptr) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
    *handle = nullptr;
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_CameraStream *stream = (HF_CameraStream *)streamHandle;
    if (stream == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    if (!IsValidFaceToken(singleFace)) {
        return HERR_INVALID_FACE_TOKEN;
    }
    inspire::FaceBasicData data;
    data.dataSize = singleFace.size;
    data.data = singleFace.data;
    auto bitmap = std::make_shared<HF_ImageBitmap>();
    auto ret = ctx->impl.FaceGetFaceAlignmentImage(stream->impl, data, bitmap->impl);
    if (ret != HSUCCEED) {
        return ret;
    }
    const auto resource_handle = reinterpret_cast<inspire::ResourceHandle>(bitmap.get());
    if (!RESOURCE_MANAGE->createImageBitmap(resource_handle, bitmap)) {
        return HERR_UNKNOWN;
    }
    *handle = bitmap.get();
    return HSUCCEED;
}

HResult HFFaceFeatureExtractWithAlignmentImage(HFSession session, HFImageStream streamHandle, HFFaceFeature feature) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    auto stream_lease = AcquireStream(streamHandle);
    if (!stream_lease) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    if (feature.data == nullptr || feature.size < FACE_FEATURE_SIZE) {
        return HERR_INVALID_FACE_FEATURE;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_CameraStream *stream = (HF_CameraStream *)streamHandle;
    if (stream == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    Embedded embedded;
    float norm;
    auto ret = ctx->impl.FaceRecognitionModule()->FaceExtractWithAlignmentImage(stream->impl, embedded, norm);
    if (ret != HSUCCEED) {
        return ret;
    }
    if (embedded.size() > static_cast<size_t>(feature.size)) {
        return HERR_INVALID_BUFFER_SIZE;
    }
    for (int i = 0; i < embedded.size(); ++i) {
        feature.data[i] = embedded[i];
    }
    return ret;
}

HResult HFFaceComparison(HFFaceFeature feature1, HFFaceFeature feature2, HPFloat result) {
    if (result == nullptr) {
        return HERR_INVALID_PARAM;
    }
    if (feature1.data == nullptr || feature2.data == nullptr || feature1.size != FACE_FEATURE_SIZE || feature2.size != FACE_FEATURE_SIZE) {
        return HERR_INVALID_FACE_FEATURE;
    }
    if (feature1.size != feature2.size) {
        INSPIRE_LOGE("feature1.size: %d, feature2.size: %d", feature1.size, feature2.size);
        return HERR_INVALID_FACE_FEATURE;
    }
    *result = 0.0f;
    float res = -1.0f;
    auto ret = INSPIREFACE_FEATURE_HUB->CosineSimilarity(feature1.data, feature2.data, feature1.size, res);
    *result = res;

    return ret;
}

HResult HFGetRecommendedCosineThreshold(HPFloat threshold) {
    if (threshold == nullptr) {
        return HERR_INVALID_PARAM;
    }
    if (!INSPIREFACE_CONTEXT->isMLoad()) {
        INSPIRE_LOGW("Inspireface is not launched, using default threshold 0.48");
    }
    *threshold = SIMILARITY_CONVERTER_GET_RECOMMENDED_COSINE_THRESHOLD();
    return HSUCCEED;
}

HResult HFCosineSimilarityConvertToPercentage(HFloat similarity, HPFloat result) {
    if (result == nullptr) {
        return HERR_INVALID_PARAM;
    }
    *result = 0.0f;
    if (!std::isfinite(similarity)) {
        return HERR_INVALID_PARAM;
    }
    if (!INSPIREFACE_CONTEXT->isMLoad()) {
        INSPIRE_LOGW("Inspireface is not launched.");
    }
    *result = static_cast<HFloat>(SIMILARITY_CONVERTER_RUN(similarity));
    return HSUCCEED;
}

HResult HFUpdateCosineSimilarityConverter(HFSimilarityConverterConfig config) {
    if (!INSPIREFACE_CONTEXT->isMLoad()) {
        INSPIRE_LOGW("Inspireface is not launched.");
    }
    inspire::SimilarityConverterConfig cfg;
    cfg.threshold = config.threshold;
    cfg.middleScore = config.middleScore;
    cfg.steepness = config.steepness;
    cfg.outputMin = config.outputMin;
    cfg.outputMax = config.outputMax;
    if (!SIMILARITY_CONVERTER_UPDATE_CONFIG(cfg)) {
        return HERR_INVALID_PARAM;
    }
    return HSUCCEED;
}

HResult HFGetCosineSimilarityConverter(PHFSimilarityConverterConfig config) {
    if (config == nullptr) {
        return HERR_INVALID_PARAM;
    }
    if (!INSPIREFACE_CONTEXT->isMLoad()) {
        INSPIRE_LOGW("Inspireface is not launched.");
    }
    inspire::SimilarityConverterConfig cfg = SIMILARITY_CONVERTER_GET_CONFIG();
    config->threshold = cfg.threshold;
    config->middleScore = cfg.middleScore;
    config->steepness = cfg.steepness;
    config->outputMin = cfg.outputMin;
    config->outputMax = cfg.outputMax;
    return HSUCCEED;
}

HResult HFGetFeatureLength(HPInt32 num) {
    if (num == nullptr) {
        return HERR_INVALID_PARAM;
    }
    *num = FACE_FEATURE_SIZE;

    return HSUCCEED;
}

HResult HFFeatureHubInsertFeature(HFFaceFeatureIdentity featureIdentity, HPFaceId allocId) {
    if (allocId == nullptr) {
        return HERR_INVALID_PARAM;
    }
    *allocId = -1;
    if (featureIdentity.feature == nullptr || featureIdentity.feature->data == nullptr) {
        return HERR_INVALID_FACE_FEATURE;
    }
    if (featureIdentity.feature->size != FACE_FEATURE_SIZE) {
        return HERR_FT_HUB_INVALID_FEATURE;
    }
    std::vector<float> feat;
    feat.reserve(featureIdentity.feature->size);
    for (int i = 0; i < featureIdentity.feature->size; ++i) {
        feat.push_back(featureIdentity.feature->data[i]);
    }
    HInt32 ret = INSPIREFACE_FEATURE_HUB->FaceFeatureInsert(feat, featureIdentity.id, *allocId);

    return ret;
}

HResult HFFeatureHubFaceSearch(HFFaceFeature searchFeature, HPFloat confidence, PHFFaceFeatureIdentity mostSimilar) {
    if (confidence == nullptr || mostSimilar == nullptr) {
        return HERR_INVALID_PARAM;
    }
    *confidence = -1.0f;
    mostSimilar->id = HF_INVALID_FACE_ID;
    mostSimilar->feature = nullptr;

    HFFeatureHubSearchResultV2 result = {};
    const HResult ret = HFFeatureHubFaceSearchV2(searchFeature, &result);
    if (ret != HSUCCEED) {
        return ret;
    }

    // Preserve the legacy sentinel contract even for a persisted record whose
    // row ID is -1. V2 callers can distinguish that record using `found`.
    if (result.found && result.id != HF_INVALID_FACE_ID) {
        static thread_local HFFaceFeature legacy_feature_view = {0, nullptr};
        legacy_feature_view = result.feature;
        mostSimilar->id = result.id;
        mostSimilar->feature = &legacy_feature_view;
        *confidence = result.confidence;
    }
    return HSUCCEED;
}

HResult HFFeatureHubFaceSearchV2(HFFaceFeature searchFeature, PHFFeatureHubSearchResultV2 result) {
    if (result == nullptr) {
        return HERR_INVALID_PARAM;
    }
    result->found = 0;
    result->id = HF_INVALID_FACE_ID;
    result->confidence = -1.0f;
    result->feature = {0, nullptr};

    static thread_local std::vector<float> result_feature;
    result_feature.clear();
    if (searchFeature.data == nullptr) {
        return HERR_INVALID_FACE_FEATURE;
    }
    if (searchFeature.size != FACE_FEATURE_SIZE) {
        return HERR_FT_HUB_INVALID_FEATURE;
    }
    std::vector<float> feat(searchFeature.data, searchFeature.data + searchFeature.size);
    inspire::FaceSearchResult search_result{HF_INVALID_FACE_ID, -1.0, {}};
    bool found = false;
    const HResult ret = INSPIREFACE_FEATURE_HUB->SearchFaceFeatureV2(feat, search_result, found, true);
    if (ret != HSUCCEED) {
        return ret;
    }
    if (!found) {
        return HSUCCEED;
    }

    result_feature = std::move(search_result.feature);
    result->found = 1;
    result->id = search_result.id;
    result->confidence = static_cast<HFloat>(search_result.similarity);
    result->feature.size = static_cast<HInt32>(result_feature.size());
    result->feature.data = result_feature.empty() ? nullptr : result_feature.data();
    return HSUCCEED;
}

HResult HFFeatureHubFaceSearchTopK(HFFaceFeature searchFeature, HInt32 topK, PHFSearchTopKResults results) {
    if (topK <= 0 || results == nullptr) {
        return HERR_INVALID_PARAM;
    }
    results->size = 0;
    results->confidence = nullptr;
    results->ids = nullptr;
    if (searchFeature.data == nullptr) {
        return HERR_INVALID_FACE_FEATURE;
    }
    if (searchFeature.size != FACE_FEATURE_SIZE) {
        return HERR_FT_HUB_INVALID_FEATURE;
    }
    std::vector<float> feat;
    feat.reserve(searchFeature.size);
    for (int i = 0; i < searchFeature.size; ++i) {
        feat.push_back(searchFeature.data[i]);
    }
    std::vector<inspire::FaceSearchResult> search_results;
    HInt32 ret = INSPIREFACE_FEATURE_HUB->SearchFaceFeatureTopK(feat, search_results, static_cast<size_t>(topK), false);
    if (ret != HSUCCEED) {
        return ret;
    }

    static thread_local std::vector<HFloat> confidence_cache;
    static thread_local std::vector<HFaceId> id_cache;
    confidence_cache.clear();
    id_cache.clear();
    confidence_cache.reserve(search_results.size());
    id_cache.reserve(search_results.size());
    for (const inspire::FaceSearchResult &result : search_results) {
        confidence_cache.push_back(static_cast<HFloat>(result.similarity));
        id_cache.push_back(static_cast<HFaceId>(result.id));
    }
    results->size = static_cast<HInt32>(search_results.size());
    results->confidence = confidence_cache.empty() ? nullptr : confidence_cache.data();
    results->ids = id_cache.empty() ? nullptr : id_cache.data();
    return HSUCCEED;
}

HResult HFFeatureHubFaceRemove(HFaceId id) {
    auto ret = INSPIREFACE_FEATURE_HUB->FaceFeatureRemove(id);
    return ret;
}

HResult HFFeatureHubFaceUpdate(HFFaceFeatureIdentity featureIdentity) {
    if (featureIdentity.feature == nullptr || featureIdentity.feature->data == nullptr) {
        return HERR_INVALID_FACE_FEATURE;
    }
    if (featureIdentity.feature->size != FACE_FEATURE_SIZE) {
        return HERR_FT_HUB_INVALID_FEATURE;
    }
    std::vector<float> feat;
    feat.reserve(featureIdentity.feature->size);
    for (int i = 0; i < featureIdentity.feature->size; ++i) {
        feat.push_back(featureIdentity.feature->data[i]);
    }

    auto ret = INSPIREFACE_FEATURE_HUB->FaceFeatureUpdate(feat, featureIdentity.id);

    return ret;
}

HResult HFFeatureHubGetFaceIdentity(HFaceId id, PHFFaceFeatureIdentity identity) {
    if (identity == nullptr) {
        return HERR_INVALID_PARAM;
    }
    identity->id = -1;
    identity->feature = nullptr;
    static thread_local std::vector<float> feature_cache;
    static thread_local HFFaceFeature feature_view = {0, nullptr};
    auto ret = INSPIREFACE_FEATURE_HUB->GetFaceFeature(id, feature_cache);
    if (ret == HSUCCEED) {
        identity->id = id;
        feature_view.data = feature_cache.data();
        feature_view.size = static_cast<HInt32>(feature_cache.size());
        identity->feature = &feature_view;
    }

    return ret;
}

HResult HFMultipleFacePipelineProcess(HFSession session, HFImageStream streamHandle, PHFMultipleFaceData faces, HFSessionCustomParameter parameter) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    auto stream_lease = AcquireStream(streamHandle);
    if (!stream_lease) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    if (faces == nullptr || faces->detectedNum < 0 || (faces->detectedNum > 0 && faces->tokens == nullptr)) {
        return HERR_INVALID_FACE_LIST;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_CameraStream *stream = (HF_CameraStream *)streamHandle;
    if (stream == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }

    inspire::ContextCustomParameter param;
    param.enable_mask_detect = parameter.enable_mask_detect;
    param.enable_face_attribute = parameter.enable_face_quality;
    param.enable_liveness = parameter.enable_liveness;
    param.enable_face_quality = parameter.enable_face_quality;
    param.enable_interaction_liveness = parameter.enable_interaction_liveness;
    param.enable_ir_liveness = parameter.enable_ir_liveness;
    param.enable_recognition = parameter.enable_recognition;
    param.enable_face_attribute = parameter.enable_face_attribute;
    param.enable_face_emotion = parameter.enable_face_emotion;

    HResult ret;
    std::vector<inspire::FaceTrackWrap> data;
    data.resize(faces->detectedNum);
    for (int i = 0; i < faces->detectedNum; ++i) {
        if (!IsValidFaceToken(faces->tokens[i])) {
            return HERR_INVALID_FACE_TOKEN;
        }
        auto &face = data[i];
        ret = RunDeserializeHyperFaceData((char *)faces->tokens[i].data, faces->tokens[i].size, face);
        if (ret != HSUCCEED) {
            return HERR_INVALID_FACE_TOKEN;
        }
    }

    ret = ctx->impl.FacesProcess(stream->impl, data, param);

    return ret;
}

HResult HFMultipleFacePipelineProcessOptional(HFSession session, HFImageStream streamHandle, PHFMultipleFaceData faces, HInt32 customOption) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    auto stream_lease = AcquireStream(streamHandle);
    if (!stream_lease) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    if (faces == nullptr || faces->detectedNum < 0 || (faces->detectedNum > 0 && faces->tokens == nullptr)) {
        return HERR_INVALID_FACE_LIST;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_CameraStream *stream = (HF_CameraStream *)streamHandle;
    if (stream == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }

    inspire::ContextCustomParameter param;
    if (customOption & HF_ENABLE_FACE_RECOGNITION) {
        param.enable_recognition = true;
    }
    if (customOption & HF_ENABLE_LIVENESS) {
        param.enable_liveness = true;
    }
    if (customOption & HF_ENABLE_IR_LIVENESS) {
        param.enable_ir_liveness = true;
    }
    if (customOption & HF_ENABLE_FACE_ATTRIBUTE) {
        param.enable_face_attribute = true;
    }
    if (customOption & HF_ENABLE_MASK_DETECT) {
        param.enable_mask_detect = true;
    }
    if (customOption & HF_ENABLE_QUALITY) {
        param.enable_face_quality = true;
    }
    if (customOption & HF_ENABLE_INTERACTION) {
        param.enable_interaction_liveness = true;
    }
    if (customOption & HF_ENABLE_FACE_POSE) {
        param.enable_face_pose = true;
    }
    if (customOption & HF_ENABLE_FACE_EMOTION) {
        param.enable_face_emotion = true;
    }

    HResult ret;
    std::vector<inspire::FaceTrackWrap> data;
    data.resize(faces->detectedNum);
    for (int i = 0; i < faces->detectedNum; ++i) {
        if (!IsValidFaceToken(faces->tokens[i])) {
            return HERR_INVALID_FACE_TOKEN;
        }
        auto &face = data[i];
        ret = RunDeserializeHyperFaceData((char *)faces->tokens[i].data, faces->tokens[i].size, face);
        if (ret != HSUCCEED) {
            return HERR_INVALID_FACE_TOKEN;
        }
    }

    ret = ctx->impl.FacesProcess(stream->impl, data, param);

    return ret;
}

HResult HFGetRGBLivenessConfidence(HFSession session, PHFRGBLivenessConfidence confidence) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (confidence == nullptr) {
        return HERR_INVALID_PARAM;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }

    confidence->num = ctx->impl.GetRgbLivenessResultsCache().size();
    confidence->confidence = (HFloat *)ctx->impl.GetRgbLivenessResultsCache().data();

    return HSUCCEED;
}

HResult HFGetFaceMaskConfidence(HFSession session, PHFFaceMaskConfidence confidence) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (confidence == nullptr) {
        return HERR_INVALID_PARAM;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }

    confidence->num = ctx->impl.GetMaskResultsCache().size();
    confidence->confidence = (HFloat *)ctx->impl.GetMaskResultsCache().data();

    return HSUCCEED;
}

HResult HFGetFaceQualityConfidence(HFSession session, PHFFaceQualityConfidence confidence) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (confidence == nullptr) {
        return HERR_INVALID_PARAM;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }

    confidence->num = ctx->impl.GetFaceQualityScoresResultsCache().size();
    confidence->confidence = (HFloat *)ctx->impl.GetFaceQualityScoresResultsCache().data();

    return HSUCCEED;
}

HResult HFFaceQualityDetect(HFSession session, HFFaceBasicToken singleFace, HPFloat confidence) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (!IsValidFaceToken(singleFace)) {
        return HERR_INVALID_FACE_TOKEN;
    }
    if (confidence == nullptr) {
        return HERR_INVALID_PARAM;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }

    inspire::FaceBasicData data;
    data.dataSize = singleFace.size;
    data.data = singleFace.data;

    auto ret = inspire::FaceSession::FaceQualityDetect(data, *confidence);

    return ret;
}

HResult HFGetFaceInteractionStateResult(HFSession session, PHFFaceInteractionState result) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (result == nullptr) {
        return HERR_INVALID_PARAM;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    result->num = ctx->impl.GetFaceInteractionLeftEyeStatusCache().size();
    result->leftEyeStatusConfidence = (HFloat *)ctx->impl.GetFaceInteractionLeftEyeStatusCache().data();
    result->rightEyeStatusConfidence = (HFloat *)ctx->impl.GetFaceInteractionRightEyeStatusCache().data();

    return HSUCCEED;
}

HResult HFGetFaceInteractionActionsResult(HFSession session, PHFFaceInteractionsActions actions) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (actions == nullptr) {
        return HERR_INVALID_PARAM;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    actions->num = ctx->impl.GetFaceNormalAactionsResultCache().size();
    actions->normal = (HInt32 *)ctx->impl.GetFaceNormalAactionsResultCache().data();
    actions->blink = (HInt32 *)ctx->impl.GetFaceBlinkAactionsResultCache().data();
    actions->shake = (HInt32 *)ctx->impl.GetFaceShakeAactionsResultCache().data();
    actions->headRaise = (HInt32 *)ctx->impl.GetFaceRaiseHeadAactionsResultCache().data();
    actions->jawOpen = (HInt32 *)ctx->impl.GetFaceJawOpenAactionsResultCache().data();

    return HSUCCEED;
}

HResult HFGetFaceAttributeResult(HFSession session, PHFFaceAttributeResult results) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (results == nullptr) {
        return HERR_INVALID_PARAM;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }

    results->num = ctx->impl.GetFaceAgeBracketResultsCache().size();
    results->race = (HPInt32)ctx->impl.GetFaceRaceResultsCache().data();
    results->gender = (HPInt32)ctx->impl.GetFaceGenderResultsCache().data();
    results->ageBracket = (HPInt32)ctx->impl.GetFaceAgeBracketResultsCache().data();

    return HSUCCEED;
}

HResult HFGetFaceEmotionResult(HFSession session, PHFFaceEmotionResult result) {
    auto session_lease = AcquireSession(session);
    if (!session_lease) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (result == nullptr) {
        return HERR_INVALID_PARAM;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }

    result->num = ctx->impl.GetFaceEmotionResultsCache().size();
    result->emotion = (HPInt32)ctx->impl.GetFaceEmotionResultsCache().data();

    return HSUCCEED;
}

HResult HFFeatureHubGetFaceCount(HPInt32 count) {
    if (count == nullptr) {
        return HERR_INVALID_PARAM;
    }
    return INSPIREFACE_FEATURE_HUB->GetFaceFeatureCount(*count);
}

HResult HFFeatureHubViewDBTable() {
    return INSPIREFACE_FEATURE_HUB->ViewDBTable();
}

HResult HFFeatureHubGetExistingIds(PHFFeatureHubExistingIds ids) {
    if (ids == nullptr) {
        return HERR_INVALID_PARAM;
    }
    ids->size = 0;
    ids->ids = nullptr;
    static thread_local std::vector<int64_t> id_cache;
    auto ret = INSPIREFACE_FEATURE_HUB->GetAllIds(id_cache);
    if (ret == HSUCCEED) {
        if (id_cache.size() > static_cast<size_t>(std::numeric_limits<HInt32>::max())) {
            id_cache.clear();
            return HERR_FT_HUB_DATABASE_FAILURE;
        }
        ids->size = static_cast<HInt32>(id_cache.size());
        ids->ids = id_cache.empty() ? nullptr : id_cache.data();
    }
    return ret;
}

HResult HFQueryInspireFaceVersion(PHFInspireFaceVersion version) {
    if (version == nullptr) {
        return HERR_INVALID_PARAM;
    }
    version->major = atoi(INSPIRE_FACE_VERSION_MAJOR_STR);
    version->minor = atoi(INSPIRE_FACE_VERSION_MINOR_STR);
    version->patch = atoi(INSPIRE_FACE_VERSION_PATCH_STR);

    return HSUCCEED;
}

HFStatus HFQueryCAPILevel(HFUInt32* apiLevel) {
    if (apiLevel == nullptr) {
        return HERR_INVALID_PARAM;
    }
    *apiLevel = HF_C_API_LEVEL;
    return HSUCCEED;
}

HResult HFQueryInspireFaceComponentVersion(HFComponentType component, PHFComponentVersion version) {
    static_assert(HF_COMPONENT_COUNT == static_cast<int>(inspire::ComponentType::COUNT), "Component enums must remain aligned");
    if (version == nullptr || component < HF_COMPONENT_MNN || component >= HF_COMPONENT_COUNT) {
        return HERR_INVALID_PARAM;
    }
    const auto component_version = inspire::GetComponentVersion(static_cast<inspire::ComponentType>(component));
    version->major = component_version.major;
    version->minor = component_version.minor;
    version->patch = component_version.patch;
    version->state = static_cast<HFComponentVersionState>(component_version.state);
    return HSUCCEED;
}

HResult HFQueryInspireFaceComponentVersions(HString buffer, HInt32 bufferSize, HPInt32 requiredSize) {
    return CopyQueryString(inspire::GetComponentVersionsString(), buffer, bufferSize, requiredSize);
}

HResult HFQueryInspireFaceDiagnosticInformation(HString buffer, HInt32 bufferSize, HPInt32 requiredSize) {
    return CopyQueryString(inspire::GetDiagnosticInfo(), buffer, bufferSize, requiredSize);
}

HResult HFQueryInspireFaceExtendedInformation(PHFInspireFaceExtendedInformation information) {
    if (information == nullptr) {
        return HERR_INVALID_PARAM;
    }
    std::memset(information->information, 0, sizeof(information->information));
    std::strncpy(information->information, INSPIRE_FACE_EXTENDED_INFORMATION, sizeof(information->information) - 1);
    return HSUCCEED;
}

HResult HFSetLogLevel(HFLogLevel level) {
    if (level < HF_LOG_NONE || level > HF_LOG_FATAL) {
        return HERR_INVALID_PARAM;
    }
    INSPIRE_SET_LOG_LEVEL(LogLevel(level));
    return HSUCCEED;
}

HResult HFLogDisable() {
    INSPIRE_SET_LOG_LEVEL(inspire::ISF_LOG_NONE);

    return HSUCCEED;
}

HResult HFLogPrint(HFLogLevel level, HFormat format, ...) {
    if (level < HF_LOG_NONE || level > HF_LOG_FATAL || format == nullptr) {
        return HERR_INVALID_PARAM;
    }
    inspire::LogLevel logLevel = static_cast<inspire::LogLevel>(level);
    if (inspire::LogManager::getInstance()->getLogLevel() == inspire::ISF_LOG_NONE || logLevel < inspire::LogManager::getInstance()->getLogLevel()) {
        return HSUCCEED;
    }
    char buffer[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    switch (logLevel) {
        case inspire::ISF_LOG_DEBUG:
            INSPIRE_LOGD("%s", buffer);
            break;
        case inspire::ISF_LOG_INFO:
            INSPIRE_LOGI("%s", buffer);
            break;
        case inspire::ISF_LOG_WARN:
            INSPIRE_LOGW("%s", buffer);
            break;
        case inspire::ISF_LOG_ERROR:
            INSPIRE_LOGE("%s", buffer);
            break;
        case inspire::ISF_LOG_FATAL:
            INSPIRE_LOGF("%s", buffer);
            break;
        default:
            break;
    }

    return HSUCCEED;
}

HResult HFDeBugShowResourceStatistics() {
    RESOURCE_MANAGE->printResourceStatistics();
    return HSUCCEED;
}

HResult HFDeBugGetUnreleasedSessionsCount(HPInt32 count) {
    if (count == nullptr) {
        return HERR_INVALID_PARAM;
    }
    *count = static_cast<HInt32>(RESOURCE_MANAGE->getUnreleasedSessions().size());
    return HSUCCEED;
}

HResult HFDeBugGetUnreleasedSessions(PHFSession sessions, HInt32 count) {
    if (count < 0 || (count > 0 && sessions == nullptr)) {
        return HERR_INVALID_PARAM;
    }
    const std::vector<inspire::ResourceHandle> unreleasedSessions = RESOURCE_MANAGE->getUnreleasedSessions();
    const size_t copy_count = std::min(static_cast<size_t>(count), unreleasedSessions.size());
    for (size_t i = 0; i < copy_count; ++i) {
        sessions[i] = reinterpret_cast<HFSession>(unreleasedSessions[i]);
    }
    return HSUCCEED;
}

HResult HFDeBugGetUnreleasedStreamsCount(HPInt32 count) {
    if (count == nullptr) {
        return HERR_INVALID_PARAM;
    }
    *count = static_cast<HInt32>(RESOURCE_MANAGE->getUnreleasedStreams().size());
    return HSUCCEED;
}

HResult HFDeBugGetUnreleasedStreams(PHFImageStream streams, HInt32 count) {
    if (count < 0 || (count > 0 && streams == nullptr)) {
        return HERR_INVALID_PARAM;
    }
    const std::vector<inspire::ResourceHandle> unreleasedStreams = RESOURCE_MANAGE->getUnreleasedStreams();
    const size_t copy_count = std::min(static_cast<size_t>(count), unreleasedStreams.size());
    for (size_t i = 0; i < copy_count; ++i) {
        streams[i] = reinterpret_cast<HFImageStream>(unreleasedStreams[i]);
    }
    return HSUCCEED;
}
