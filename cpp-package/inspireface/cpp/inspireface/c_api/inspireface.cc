/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "inspireface.h"
#include "intypedef.h"
#include "inspireface_internal.h"
#include "information.h"
#include "feature_hub_db.h"
#include <launch.h>
#include "runtime_module/resource_manage.h"
#include "similarity_converter.h"
#include "middleware/inference_wrapper/inference_wrapper.h"
#if defined(ISF_ENABLE_TENSORRT)
#include "cuda_toolkit.h"
#endif
#include <cstdarg>

#define FACE_FEATURE_SIZE 512  ///< Temporary setup

using namespace inspire;

HYPER_CAPI_EXPORT extern HResult HFCreateImageStream(PHFImageData data, PHFImageStream handle) {
    if (data == nullptr || handle == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }

    auto stream = new HF_CameraStream();
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
    stream->impl.SetDataBuffer(data->data, data->height, data->width);

    *handle = (HFImageStream)stream;

    // Record the creation of this stream in the ResourceManager
    RESOURCE_MANAGE->createStream((long)*handle);

    return HSUCCEED;
}

HYPER_CAPI_EXPORT extern HResult HFCreateImageStreamEmpty(PHFImageStream handle) {
    if (handle == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    auto stream = new HF_CameraStream();
    *handle = (HFImageStream)stream;
    return HSUCCEED;
}

HYPER_CAPI_EXPORT extern HResult HFImageStreamSetBuffer(HFImageStream handle, HPUInt8 buffer, HInt32 width, HInt32 height) {
    if (handle == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    ((HF_CameraStream *)handle)->impl.SetDataBuffer(buffer, width, height);
    return HSUCCEED;
}

HYPER_CAPI_EXPORT extern HResult HFImageStreamSetRotation(HFImageStream handle, HFRotation rotation) {
    if (handle == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
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
    if (handle == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
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
    return HSUCCEED;
}

HYPER_CAPI_EXPORT extern HResult HFReleaseImageStream(HFImageStream streamHandle) {
    if (streamHandle == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    // Check and mark this stream as released in the ResourceManager
    if (!RESOURCE_MANAGE->releaseStream((long)streamHandle)) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;  // or other appropriate error code
    }
    delete (HF_CameraStream *)streamHandle;
    return HSUCCEED;
}

HYPER_CAPI_EXPORT extern HResult HFCreateImageBitmap(PHFImageBitmapData data, PHFImageBitmap handle) {
    if (data == nullptr || handle == nullptr) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
    auto bitmap = new HF_ImageBitmap();
    bitmap->impl.Reset(data->width, data->height, data->channels, data->data);
    *handle = (HFImageBitmap)bitmap;
    // Record the creation of this image bitmap in the ResourceManager
    RESOURCE_MANAGE->createImageBitmap((long)*handle);
    return HSUCCEED;
}

HYPER_CAPI_EXPORT extern HResult HFCreateImageBitmapFromFilePath(HPath filePath, HInt32 channels, PHFImageBitmap handle) {
    if (handle == nullptr) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
    auto image = inspirecv::Image::Create(filePath, channels);
    auto bitmap = new HF_ImageBitmap();
    bitmap->impl.Reset(image.Width(), image.Height(), image.Channels(), image.Data());
    *handle = (HFImageBitmap)bitmap;
    // Record the creation of this image bitmap in the ResourceManager
    RESOURCE_MANAGE->createImageBitmap((long)*handle);
    return HSUCCEED;
}

HYPER_CAPI_EXPORT extern HResult HFImageBitmapCopy(HFImageBitmap handle, PHFImageBitmap copyHandle) {
    if (handle == nullptr || copyHandle == nullptr) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
    auto bitmap = new HF_ImageBitmap();
    bitmap->impl.Reset(((HF_ImageBitmap *)handle)->impl.Width(), ((HF_ImageBitmap *)handle)->impl.Height(),
                       ((HF_ImageBitmap *)handle)->impl.Channels(), ((HF_ImageBitmap *)handle)->impl.Data());
    *copyHandle = (HFImageBitmap)bitmap;
    // Record the creation of this image bitmap in the ResourceManager
    RESOURCE_MANAGE->createImageBitmap((long)*copyHandle);
    return HSUCCEED;
}

HYPER_CAPI_EXPORT extern HResult HFReleaseImageBitmap(HFImageBitmap handle) {
    if (handle == nullptr) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
    // Check and mark this image bitmap as released in the ResourceManager
    if (!RESOURCE_MANAGE->releaseImageBitmap((long)handle)) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;  // or other appropriate error code
    }
    delete (HF_ImageBitmap *)handle;
    return HSUCCEED;
}

HYPER_CAPI_EXPORT extern HResult HFCreateImageStreamFromImageBitmap(HFImageBitmap handle, HFRotation rotation, PHFImageStream streamHandle) {
    if (handle == nullptr || streamHandle == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    auto stream = new HF_CameraStream();
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
    } else {
        stream->impl.SetDataFormat(inspirecv::BGR);
    }
    stream->impl.SetDataBuffer(((HF_ImageBitmap *)handle)->impl.Data(), ((HF_ImageBitmap *)handle)->impl.Height(),
                               ((HF_ImageBitmap *)handle)->impl.Width());
    *streamHandle = (HFImageStream)stream;

    // Record the creation of this stream in the ResourceManager
    RESOURCE_MANAGE->createStream((long)*streamHandle);
    return HSUCCEED;
}

HYPER_CAPI_EXPORT extern HResult HFCreateImageBitmapFromImageStreamProcess(HFImageStream streamHandle, PHFImageBitmap handle, HInt32 is_rotate,
                                                                           HFloat scale) {
    if (streamHandle == nullptr || handle == nullptr) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
    auto bitmap = new HF_ImageBitmap();
    auto img = ((HF_CameraStream *)streamHandle)->impl.ExecuteImageScaleProcessing(scale, is_rotate);
    bitmap->impl.Reset(img.Width(), img.Height(), img.Channels(), img.Data());
    *handle = (HFImageBitmap)bitmap;
    // Record the creation of this image bitmap in the ResourceManager
    RESOURCE_MANAGE->createImageBitmap((long)*handle);
    return HSUCCEED;
}

HYPER_CAPI_EXPORT extern HResult HFImageBitmapWriteToFile(HFImageBitmap handle, HPath filePath) {
    if (handle == nullptr) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
    auto success = ((HF_ImageBitmap *)handle)->impl.Write(filePath);
    if (success) {
        return HSUCCEED;
    } else {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
}

HYPER_CAPI_EXPORT extern HResult HFImageBitmapDrawRect(HFImageBitmap handle, HFaceRect rect, HColor color, HInt32 thickness) {
    if (handle == nullptr) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
    inspirecv::Rect<int> rect_inner(rect.x, rect.y, rect.width, rect.height);
    ((HF_ImageBitmap *)handle)->impl.DrawRect(rect_inner, {color.r, color.g, color.b}, thickness);
    return HSUCCEED;
}

HYPER_CAPI_EXPORT extern HResult HFImageBitmapDrawCircle(HFImageBitmap handle, HPoint2i point, HInt32 radius, HColor color, HInt32 thickness) {
    if (handle == nullptr) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
    ((HF_ImageBitmap *)handle)->impl.DrawCircle({point.x, point.y}, radius, {color.r, color.g, color.b}, thickness);
    return HSUCCEED;
}

HYPER_CAPI_EXPORT extern HResult HFImageBitmapDrawCircleF(HFImageBitmap handle, HPoint2f point, HInt32 radius, HColor color, HInt32 thickness) {
    if (handle == nullptr) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
    ((HF_ImageBitmap *)handle)->impl.DrawCircle({(int)point.x, (int)point.y}, radius, {color.r, color.g, color.b}, thickness);
    return HSUCCEED;
}

HYPER_CAPI_EXPORT extern HResult HFImageBitmapGetData(HFImageBitmap handle, PHFImageBitmapData data) {
    if (handle == nullptr || data == nullptr) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
    data->width = ((HF_ImageBitmap *)handle)->impl.Width();
    data->height = ((HF_ImageBitmap *)handle)->impl.Height();
    data->channels = ((HF_ImageBitmap *)handle)->impl.Channels();
    data->data = (uint8_t *)((HF_ImageBitmap *)handle)->impl.Data();
    return HSUCCEED;
}

HYPER_CAPI_EXPORT extern HResult HFImageBitmapShow(HFImageBitmap handle, HString title, HInt32 delay) {
    if (handle == nullptr) {
        return HERR_INVALID_IMAGE_BITMAP_HANDLE;
    }
    ((HF_ImageBitmap *)handle)->impl.Show(title, delay);
    return HSUCCEED;
}

void HFDeBugImageStreamImShow(HFImageStream streamHandle) {
    if (streamHandle == nullptr) {
        INSPIRE_LOGE("Handle error");
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
    if (streamHandle == nullptr) {
        INSPIRE_LOGE("Handle error");
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
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
    if (!RESOURCE_MANAGE->releaseSession((long)handle)) {
        return HERR_INVALID_CONTEXT_HANDLE;  // or other appropriate error code
    }
    delete (HF_FaceAlgorithmSession *)handle;
    return HSUCCEED;
}

HResult HFSessionClearTrackingFace(HFSession session) {
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    ctx->impl.ClearTrackingFace();
    return HSUCCEED;
}

HResult HFSessionSetTrackLostRecoveryMode(HFSession session, HInt32 enable) {
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    ctx->impl.SetTrackLostRecoveryMode(enable);
    return HSUCCEED;
}

HResult HFSessionSetLightTrackConfidenceThreshold(HFSession session, HFloat value) {
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
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
    INSPIREFACE_CONTEXT->SwitchLandmarkEngine(type);
    return HSUCCEED;
}

HResult HFQuerySupportedPixelLevelsForFaceDetection(PHFFaceDetectPixelList pixel_levels) {
    auto ret = INSPIREFACE_CONTEXT->GetFaceDetectPixelList();
    pixel_levels->size = ret.size();
    for (int i = 0; i < ret.size(); i++) {
        pixel_levels->pixel_level[i] = ret[i];
    }
    return HSUCCEED;
}

HResult HFCreateInspireFaceSession(HFSessionCustomParameter parameter, HFDetectMode detectMode, HInt32 maxDetectFaceNum, HInt32 detectPixelLevel,
                                   HInt32 trackByDetectModeFPS, PHFSession handle) {
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

    HF_FaceAlgorithmSession *ctx = new HF_FaceAlgorithmSession();
    auto ret = ctx->impl.Configuration(detMode, maxDetectFaceNum, param, detectPixelLevel, trackByDetectModeFPS);
    if (ret != HSUCCEED) {
        delete ctx;
        *handle = nullptr;
    } else {
        *handle = ctx;
        // Record the creation of this session in the ResourceManager
        RESOURCE_MANAGE->createSession((long)*handle);
    }

    return ret;
}

HResult HFCreateInspireFaceSessionOptional(HOption customOption, HFDetectMode detectMode, HInt32 maxDetectFaceNum, HInt32 detectPixelLevel,
                                           HInt32 trackByDetectModeFPS, PHFSession handle) {
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

    HF_FaceAlgorithmSession *ctx = new HF_FaceAlgorithmSession();
    auto ret = ctx->impl.Configuration(detMode, maxDetectFaceNum, param, detectPixelLevel, trackByDetectModeFPS);
    if (ret != HSUCCEED) {
        delete ctx;
        *handle = nullptr;
    } else {
        *handle = ctx;
        // Record the creation of this session in the ResourceManager
        RESOURCE_MANAGE->createSession((long)*handle);
    }

    return ret;
}

HResult HFLaunchInspireFace(HPath resourcePath) {
    std::string path(resourcePath);
    return INSPIREFACE_CONTEXT->Load(resourcePath);
}

HResult HFReloadInspireFace(HPath resourcePath) {
    std::string path(resourcePath);
    return INSPIREFACE_CONTEXT->Reload(resourcePath);
}

HResult HFTerminateInspireFace() {
    INSPIREFACE_CONTEXT->Unload();
    return HSUCCEED;
}

HResult HFQueryInspireFaceLaunchStatus(HPInt32 status) {
    *status = INSPIREFACE_CONTEXT->isMLoad();
    return HSUCCEED;
}

HResult HFFeatureHubDataDisable() {
    return INSPIREFACE_FEATURE_HUB->DisableHub();
}

HResult HFQueryExpansiveHardwareRGACompileOption(HPInt32 enable) {
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
    INSPIREFACE_CONTEXT->SetRockchipDmaHeapPath(path);
    return HSUCCEED;
}

HResult HFQueryExpansiveHardwareRockchipDmaHeapPath(HString path) {
    strcpy(path, INSPIREFACE_CONTEXT->GetRockchipDmaHeapPath().c_str());
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
    INSPIREFACE_CONTEXT->SetImageProcessAlignedWidth(width);
    return HSUCCEED;
}

HResult HFSetCudaDeviceId(HInt32 device_id) {
    INSPIREFACE_CONTEXT->SetCudaDeviceId(device_id);
    return HSUCCEED;
}

HResult HFGetCudaDeviceId(HPInt32 device_id) {
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
#if defined(ISF_ENABLE_TENSORRT)
    return inspire::GetCudaDeviceCount(num_devices);
#else
    INSPIRE_LOGW("CUDA is not supported, you need to enable the compile option that supports TensorRT");
    return HERR_DEVICE_CUDA_DISABLE;
#endif
}

HResult HFCheckCudaDeviceSupport(HPInt32 is_support) {
#if defined(ISF_ENABLE_TENSORRT)
    return inspire::CheckCudaUsability(is_support);
#else
    INSPIRE_LOGW("CUDA is not supported, you need to enable the compile option that supports TensorRT");
    return HERR_DEVICE_CUDA_DISABLE;
#endif
}

HResult HFFeatureHubDataEnable(HFFeatureHubConfiguration configuration) {
    inspire::DatabaseConfiguration param;
    if (configuration.primaryKeyMode != HF_PK_AUTO_INCREMENT && configuration.primaryKeyMode != HF_PK_MANUAL_INPUT) {
        param.primary_key_mode = inspire::PrimaryKeyMode::AUTO_INCREMENT;
    } else {
        param.primary_key_mode = inspire::PrimaryKeyMode(configuration.primaryKeyMode);
    }
    if (configuration.persistenceDbPath == nullptr) {
        INSPIRE_LOGE("persistenceDbPath is null, use default path");
    }
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
    param.search_mode = (inspire::SearchMode)configuration.searchMode;
    auto ret = INSPIREFACE_FEATURE_HUB->EnableHub(param);
    return ret;
}

HResult HFSessionSetTrackPreviewSize(HFSession session, HInt32 previewSize) {
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    return ctx->impl.SetTrackPreviewSize(previewSize);
}

HResult HFSessionGetTrackPreviewSize(HFSession session, HPInt32 previewSize) {
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    *previewSize = ctx->impl.GetTrackPreviewSize();
    return HSUCCEED;
}

HResult HFSessionSetFilterMinimumFacePixelSize(HFSession session, HInt32 minSize) {
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    return ctx->impl.SetTrackFaceMinimumSize(minSize);
}

HResult HFSessionSetFaceTrackMode(HFSession session, HFDetectMode detectMode) {
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    inspire::DetectModuleMode detMode = inspire::DETECT_MODE_ALWAYS_DETECT;
    if (detectMode == HF_DETECT_MODE_LIGHT_TRACK) {
        detMode = inspire::DETECT_MODE_LIGHT_TRACK;
    }
    return ctx->impl.SetDetectMode(detMode);
}

HResult HFSessionSetFaceDetectThreshold(HFSession session, HFloat threshold) {
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    return ctx->impl.SetFaceDetectThreshold(threshold);
}

HResult HFSessionSetTrackModeSmoothRatio(HFSession session, HFloat ratio) {
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    return ctx->impl.SetTrackModeSmoothRatio(ratio);
}

HResult HFSessionSetTrackModeNumSmoothCacheFrame(HFSession session, HInt32 num) {
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    return ctx->impl.SetTrackModeNumSmoothCacheFrame(num);
}

HResult HFSessionSetTrackModeDetectInterval(HFSession session, HInt32 num) {
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    return ctx->impl.SetTrackModeDetectInterval(num);
}

HResult HFExecuteFaceTrack(HFSession session, HFImageStream streamHandle, PHFMultipleFaceData results) {
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (streamHandle == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_CameraStream *stream = (HF_CameraStream *)streamHandle;
    if (stream == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    auto ret = ctx->impl.FaceDetectAndTrack(stream->impl);
    results->detectedNum = ctx->impl.GetNumberOfFacesCurrentlyDetected();
    results->rects = (HFaceRect *)ctx->impl.GetFaceRectsCache().data();
    results->trackIds = (HInt32 *)ctx->impl.GetTrackIDCache().data();
    results->detConfidence = (HFloat *)ctx->impl.GetDetConfidenceCache().data();
    results->trackCounts = (HInt32 *)ctx->impl.GetTrackCountCache().data();
    results->angles.pitch = (HFloat *)ctx->impl.GetPitchResultsCache().data();
    results->angles.roll = (HFloat *)ctx->impl.GetRollResultsCache().data();
    results->angles.yaw = (HFloat *)ctx->impl.GetYawResultsCache().data();
    results->tokens = (HFFaceBasicToken *)ctx->impl.GetFaceBasicDataCache().data();

    return ret;
}

HResult HFSessionLastFaceDetectionGetDebugPreviewImageSize(HFSession session, HPInt32 size) {
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    *size = ctx->impl.GetDebugPreviewImageSize();
    return HSUCCEED;
}

HResult HFCopyFaceBasicToken(HFFaceBasicToken token, HPBuffer buffer, HInt32 bufferSize) {
    if (bufferSize < sizeof(inspire::FaceTrackWrap)) {
        return HERR_INVALID_BUFFER_SIZE;
    }
    std::memcpy(buffer, token.data, sizeof(inspire::FaceTrackWrap));
    return HSUCCEED;
}

HResult HFGetFaceBasicTokenSize(HPInt32 bufferSize) {
    *bufferSize = sizeof(inspire::FaceTrackWrap);
    return HSUCCEED;
}

HResult HFGetNumOfFaceDenseLandmark(HPInt32 num) {
    *num = 106;
    return HSUCCEED;
}

HResult HFGetFaceDenseLandmarkFromFaceToken(HFFaceBasicToken singleFace, PHPoint2f landmarks, HInt32 num) {
    if (num != 106) {
        return HERR_SESS_LANDMARK_NUM_NOT_MATCH;
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
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    ctx->impl.SetEnableTrackCostSpend(value);
    return HSUCCEED;
}

HResult HFSessionPrintTrackCostSpend(HFSession session) {
    if (session == nullptr) {
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
    INSPIREFACE_FEATURE_HUB->SetRecognitionThreshold(threshold);
    return HSUCCEED;
}

HResult HFFaceFeatureExtract(HFSession session, HFImageStream streamHandle, HFFaceBasicToken singleFace, PHFFaceFeature feature) {
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (streamHandle == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_CameraStream *stream = (HF_CameraStream *)streamHandle;
    if (stream == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    if (singleFace.data == nullptr || singleFace.size <= 0) {
        return HERR_INVALID_FACE_TOKEN;
    }
    inspire::FaceBasicData data;
    data.dataSize = singleFace.size;
    data.data = singleFace.data;
    auto ret = ctx->impl.FaceFeatureExtract(stream->impl, data);
    feature->size = ctx->impl.GetFaceFeatureCache().size();
    feature->data = (HFloat *)ctx->impl.GetFaceFeatureCache().data();

    return ret;
}

HResult HFFaceFeatureExtractTo(HFSession session, HFImageStream streamHandle, HFFaceBasicToken singleFace, HFFaceFeature feature) {
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (streamHandle == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_CameraStream *stream = (HF_CameraStream *)streamHandle;
    if (stream == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    if (singleFace.data == nullptr || singleFace.size <= 0) {
        return HERR_INVALID_FACE_TOKEN;
    }
    inspire::FaceBasicData data;
    data.dataSize = singleFace.size;
    data.data = singleFace.data;
    auto ret = ctx->impl.FaceFeatureExtract(stream->impl, data);
    for (int i = 0; i < ctx->impl.GetFaceFeatureCache().size(); ++i) {
        feature.data[i] = ctx->impl.GetFaceFeatureCache()[i];
    }

    return HSUCCEED;
}

HResult HFFaceFeatureExtractCpy(HFSession session, HFImageStream streamHandle, HFFaceBasicToken singleFace, HPFloat feature) {
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (streamHandle == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_CameraStream *stream = (HF_CameraStream *)streamHandle;
    if (stream == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    if (singleFace.data == nullptr || singleFace.size <= 0) {
        return HERR_INVALID_FACE_TOKEN;
    }
    inspire::FaceBasicData data;
    data.dataSize = singleFace.size;
    data.data = singleFace.data;
    auto ret = ctx->impl.FaceFeatureExtract(stream->impl, data);
    for (int i = 0; i < ctx->impl.GetFaceFeatureCache().size(); ++i) {
        feature[i] = ctx->impl.GetFaceFeatureCache()[i];
    }

    return ret;
}

HResult HFCreateFaceFeature(PHFFaceFeature feature) {
    if (feature == nullptr) {
        return HERR_INVALID_FACE_FEATURE;
    }
    feature->size = FACE_FEATURE_SIZE;
    feature->data = new HFloat[FACE_FEATURE_SIZE];
    RESOURCE_MANAGE->createFaceFeature((long)feature);
    return HSUCCEED;
}

HResult HFReleaseFaceFeature(PHFFaceFeature feature) {
    if (feature == nullptr) {
        return HERR_INVALID_FACE_FEATURE;
    }
    delete[] feature->data;
    RESOURCE_MANAGE->releaseFaceFeature((long)feature);
    return HSUCCEED;
}

HResult HFFaceGetFaceAlignmentImage(HFSession session, HFImageStream streamHandle, HFFaceBasicToken singleFace, PHFImageBitmap handle) {
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (streamHandle == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_CameraStream *stream = (HF_CameraStream *)streamHandle;
    if (stream == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    if (singleFace.data == nullptr || singleFace.size <= 0) {
        return HERR_INVALID_FACE_TOKEN;
    }
    inspire::FaceBasicData data;
    data.dataSize = singleFace.size;
    data.data = singleFace.data;
    auto bitmap = new HF_ImageBitmap();
    auto ret = ctx->impl.FaceGetFaceAlignmentImage(stream->impl, data, bitmap->impl);
    if (ret != HSUCCEED) {
        delete bitmap;
        return ret;
    }
    *handle = bitmap;
    // Record the creation of this image bitmap in the ResourceManager
    RESOURCE_MANAGE->createImageBitmap((long)*handle);
    return HSUCCEED;
}

HResult HFFaceFeatureExtractWithAlignmentImage(HFSession session, HFImageStream streamHandle, HFFaceFeature feature) {
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (streamHandle == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
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
    for (int i = 0; i < embedded.size(); ++i) {
        feature.data[i] = embedded[i];
    }
    return ret;
}

HResult HFFaceComparison(HFFaceFeature feature1, HFFaceFeature feature2, HPFloat result) {
    if (feature1.data == nullptr || feature2.data == nullptr) {
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
    if (!INSPIREFACE_CONTEXT->isMLoad()) {
        INSPIRE_LOGW("Inspireface is not launched, using default threshold 0.48");
    }
    *threshold = SIMILARITY_CONVERTER_GET_RECOMMENDED_COSINE_THRESHOLD();
    return HSUCCEED;
}

HResult HFCosineSimilarityConvertToPercentage(HFloat similarity, HPFloat result) {
    if (!INSPIREFACE_CONTEXT->isMLoad()) {
        INSPIRE_LOGW("Inspireface is not launched.");
    }
    *result = SIMILARITY_CONVERTER_RUN(similarity);
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
    SIMILARITY_CONVERTER_UPDATE_CONFIG(cfg);
    return HSUCCEED;
}

HResult HFGetCosineSimilarityConverter(PHFSimilarityConverterConfig config) {
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
    *num = FACE_FEATURE_SIZE;

    return HSUCCEED;
}

HResult HFFeatureHubInsertFeature(HFFaceFeatureIdentity featureIdentity, HPFaceId allocId) {
    if (featureIdentity.feature->data == nullptr) {
        return HERR_INVALID_FACE_FEATURE;
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
    if (searchFeature.data == nullptr) {
        return HERR_INVALID_FACE_FEATURE;
    }
    std::vector<float> feat;
    feat.reserve(searchFeature.size);
    for (int i = 0; i < searchFeature.size; ++i) {
        feat.push_back(searchFeature.data[i]);
    }
    *confidence = -1.0f;
    inspire::FaceSearchResult result;
    HInt32 ret = INSPIREFACE_FEATURE_HUB->SearchFaceFeature(feat, result);
    mostSimilar->feature = (HFFaceFeature *)INSPIREFACE_FEATURE_HUB->GetFaceFeaturePtrCache().get();
    mostSimilar->feature->data = (HFloat *)INSPIREFACE_FEATURE_HUB->GetSearchFaceFeatureCache().data();
    mostSimilar->feature->size = INSPIREFACE_FEATURE_HUB->GetSearchFaceFeatureCache().size();
    mostSimilar->id = result.id;
    if (mostSimilar->id != -1) {
        *confidence = result.similarity;
    }

    return ret;
}

HResult HFFeatureHubFaceSearchTopK(HFFaceFeature searchFeature, HInt32 topK, PHFSearchTopKResults results) {
    if (searchFeature.data == nullptr) {
        return HERR_INVALID_FACE_FEATURE;
    }
    std::vector<float> feat;
    feat.reserve(searchFeature.size);
    for (int i = 0; i < searchFeature.size; ++i) {
        feat.push_back(searchFeature.data[i]);
    }
    HInt32 ret = INSPIREFACE_FEATURE_HUB->SearchFaceFeatureTopKCache(feat, topK);
    if (ret == HSUCCEED) {
        results->size = INSPIREFACE_FEATURE_HUB->GetTopKConfidence().size();
        results->confidence = INSPIREFACE_FEATURE_HUB->GetTopKConfidence().data();
        results->ids = INSPIREFACE_FEATURE_HUB->GetTopKCustomIdsCache().data();
    }

    return ret;
}

HResult HFFeatureHubFaceRemove(HFaceId id) {
    auto ret = INSPIREFACE_FEATURE_HUB->FaceFeatureRemove(id);
    return ret;
}

HResult HFFeatureHubFaceUpdate(HFFaceFeatureIdentity featureIdentity) {
    if (featureIdentity.feature->data == nullptr) {
        return HERR_INVALID_FACE_FEATURE;
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
    auto ret = INSPIREFACE_FEATURE_HUB->GetFaceFeature(id);
    if (ret == HSUCCEED) {
        identity->id = id;
        identity->feature = (HFFaceFeature *)INSPIREFACE_FEATURE_HUB->GetFaceFeaturePtrCache().get();
        identity->feature->data = (HFloat *)INSPIREFACE_FEATURE_HUB->GetFaceFeaturePtrCache()->data;
        identity->feature->size = INSPIREFACE_FEATURE_HUB->GetFaceFeaturePtrCache()->dataSize;
    } else {
        identity->id = -1;
    }

    return ret;
}

HResult HFMultipleFacePipelineProcess(HFSession session, HFImageStream streamHandle, PHFMultipleFaceData faces, HFSessionCustomParameter parameter) {
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (streamHandle == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (faces->detectedNum == 0) {
        return HSUCCEED;
    }
    HF_CameraStream *stream = (HF_CameraStream *)streamHandle;
    if (stream == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    if (faces == nullptr || faces->tokens == nullptr || faces->tokens->data == nullptr) {
        return HERR_INVALID_FACE_LIST;
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
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    if (streamHandle == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (faces->detectedNum == 0) {
        return HSUCCEED;
    }
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_CameraStream *stream = (HF_CameraStream *)streamHandle;
    if (stream == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }
    if (faces == nullptr || faces->tokens == nullptr || faces->tokens->data == nullptr) {
        return HERR_INVALID_FACE_LIST;
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
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
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
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
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
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
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
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
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
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
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
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
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
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
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
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
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
    *count = INSPIREFACE_FEATURE_HUB->GetFaceFeatureCount();
    return HSUCCEED;
}

HResult HFFeatureHubViewDBTable() {
    INSPIREFACE_FEATURE_HUB->ViewDBTable();
    return HSUCCEED;
}

HResult HFFeatureHubGetExistingIds(PHFFeatureHubExistingIds ids) {
    auto ret = INSPIREFACE_FEATURE_HUB->GetAllIds();
    if (ret == HSUCCEED) {
        ids->size = INSPIREFACE_FEATURE_HUB->GetExistingIds().size();
        ids->ids = INSPIREFACE_FEATURE_HUB->GetExistingIds().data();
    }
    return ret;
}

HResult HFQueryInspireFaceVersion(PHFInspireFaceVersion version) {
    version->major = atoi(INSPIRE_FACE_VERSION_MAJOR_STR);
    version->minor = atoi(INSPIRE_FACE_VERSION_MINOR_STR);
    version->patch = atoi(INSPIRE_FACE_VERSION_PATCH_STR);

    return HSUCCEED;
}

HResult HFQueryInspireFaceExtendedInformation(PHFInspireFaceExtendedInformation information) {
    strncpy(information->information, INSPIRE_FACE_EXTENDED_INFORMATION, strlen(INSPIRE_FACE_EXTENDED_INFORMATION));
    return HSUCCEED;
}

HResult HFSetLogLevel(HFLogLevel level) {
    INSPIRE_SET_LOG_LEVEL(LogLevel(level));
    return HSUCCEED;
}

HResult HFLogDisable() {
    INSPIRE_SET_LOG_LEVEL(inspire::ISF_LOG_NONE);

    return HSUCCEED;
}

HResult HFLogPrint(HFLogLevel level, HFormat format, ...) {
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
    *count = RESOURCE_MANAGE->getUnreleasedSessions().size();
    return HSUCCEED;
}

HResult HFDeBugGetUnreleasedSessions(PHFSession sessions, HInt32 count) {
    std::vector<long> unreleasedSessions = RESOURCE_MANAGE->getUnreleasedSessions();
    for (int i = 0; i < count; ++i) {
        sessions[i] = (HFSession)unreleasedSessions[i];
    }
    return HSUCCEED;
}

HResult HFDeBugGetUnreleasedStreamsCount(HPInt32 count) {
    *count = RESOURCE_MANAGE->getUnreleasedStreams().size();
    return HSUCCEED;
}

HResult HFDeBugGetUnreleasedStreams(PHFImageStream streams, HInt32 count) {
    std::vector<long> unreleasedStreams = RESOURCE_MANAGE->getUnreleasedStreams();
    for (int i = 0; i < count; ++i) {
        streams[i] = (HFImageStream)unreleasedStreams[i];
    }
    return HSUCCEED;
}
