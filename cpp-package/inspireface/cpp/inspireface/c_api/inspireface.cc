//
// Created by tunm on 2023/10/3.
//

#include "inspireface.h"
#include "intypedef.h"
#include "inspireface_internal.h"
#include "information.h"
#include "feature_hub/feature_hub.h"
#include "Initialization_module/launch.h"
#include "Initialization_module/resource_manage.h"

using namespace inspire;

HYPER_CAPI_EXPORT extern HResult HFCreateImageStream(PHFImageData data, HFImageStream *handle) {
    if (data == nullptr || handle == nullptr) {
        return HERR_INVALID_IMAGE_STREAM_HANDLE;
    }

    auto stream = new HF_CameraStream();
    switch (data->rotation) {
        case HF_CAMERA_ROTATION_90:
            stream->impl.SetRotationMode(ROTATION_90);
            break;
        case HF_CAMERA_ROTATION_180:
            stream->impl.SetRotationMode(ROTATION_180);
            break;
        case HF_CAMERA_ROTATION_270:
            stream->impl.SetRotationMode(ROTATION_270);
            break;
        default:
            stream->impl.SetRotationMode(ROTATION_0);
            break;
    }
    switch (data->format) {
        case HF_STREAM_RGB:
            stream->impl.SetDataFormat(RGB);
            break;
        case HF_STREAM_BGR:
            stream->impl.SetDataFormat(BGR);
            break;
        case HF_STREAM_RGBA:
            stream->impl.SetDataFormat(RGBA);
            break;
        case HF_STREAM_BGRA:
            stream->impl.SetDataFormat(BGRA);
            break;
        case HF_STREAM_YUV_NV12:
            stream->impl.SetDataFormat(NV12);
            break;
        case HF_STREAM_YUV_NV21:
            stream->impl.SetDataFormat(NV21);
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

void HFDeBugImageStreamImShow(HFImageStream streamHandle) {
    if (streamHandle == nullptr) {
        INSPIRE_LOGE("Handle error");
    }
    HF_CameraStream *stream = (HF_CameraStream *)streamHandle;
    if (stream == nullptr) {
        INSPIRE_LOGE("Image error");
        return;
    }
    auto image = stream->impl.GetScaledImage(1.0f, true);
#ifdef DISABLE_GUI
    cv::imwrite("tmp.jpg", image);
#else
    cv::imshow("Debug", image);
    cv::waitKey(0);
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
    auto image = stream->impl.GetScaledImage(1.0f, true);
    auto ret = cv::imwrite(savePath, image);
    if (ret) {
        INSPIRE_LOGE("Image saved successfully to %s", savePath);
        return HSUCCEED;
    } else {
        INSPIRE_LOGE("Failed to save image to %s", savePath);
        return -1;
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

HResult HFCreateInspireFaceSession(HFSessionCustomParameter parameter, HFDetectMode detectMode,
                                   HInt32 maxDetectFaceNum, HInt32 detectPixelLevel,
                                   HInt32 trackByDetectModeFPS, HFSession *handle) {
    inspire::ContextCustomParameter param;
    param.enable_mask_detect = parameter.enable_mask_detect;
    param.enable_face_attribute = parameter.enable_face_quality;
    param.enable_liveness = parameter.enable_liveness;
    param.enable_face_quality = parameter.enable_face_quality;
    param.enable_interaction_liveness = parameter.enable_interaction_liveness;
    param.enable_ir_liveness = parameter.enable_ir_liveness;
    param.enable_recognition = parameter.enable_recognition;
    param.enable_face_attribute = parameter.enable_face_attribute;
    inspire::DetectMode detMode = inspire::DETECT_MODE_ALWAYS_DETECT;
    if (detectMode == HF_DETECT_MODE_LIGHT_TRACK) {
        detMode = inspire::DETECT_MODE_LIGHT_TRACK;
    } else if (detectMode == HF_DETECT_MODE_TRACK_BY_DETECTION) {
        detMode = inspire::DETECT_MODE_TRACK_BY_DETECT;
    }

    HF_FaceAlgorithmSession *ctx = new HF_FaceAlgorithmSession();
    auto ret = ctx->impl.Configuration(detMode, maxDetectFaceNum, param, detectPixelLevel,
                                       trackByDetectModeFPS);
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

HResult HFCreateInspireFaceSessionOptional(HOption customOption, HFDetectMode detectMode,
                                           HInt32 maxDetectFaceNum, HInt32 detectPixelLevel,
                                           HInt32 trackByDetectModeFPS, HFSession *handle) {
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
    inspire::DetectMode detMode = inspire::DETECT_MODE_ALWAYS_DETECT;
    if (detectMode == HF_DETECT_MODE_LIGHT_TRACK) {
        detMode = inspire::DETECT_MODE_LIGHT_TRACK;
    } else if (detectMode == HF_DETECT_MODE_TRACK_BY_DETECTION) {
        detMode = inspire::DETECT_MODE_TRACK_BY_DETECT;
    }

    HF_FaceAlgorithmSession *ctx = new HF_FaceAlgorithmSession();
    auto ret = ctx->impl.Configuration(detMode, maxDetectFaceNum, param, detectPixelLevel,
                                       trackByDetectModeFPS);
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
    return INSPIRE_LAUNCH->Load(resourcePath);
}

HResult HFTerminateInspireFace() {
    INSPIRE_LAUNCH->Unload();
    return HSUCCEED;
}

HResult HFFeatureHubDataDisable() {
    return FEATURE_HUB->DisableHub();
}

HResult HFFeatureHubDataEnable(HFFeatureHubConfiguration configuration) {
    inspire::DatabaseConfiguration param;
    param.db_path =
      (configuration.dbPath != nullptr) ? std::string(configuration.dbPath) : std::string();
    param.enable_use_db = configuration.enablePersistence;
    param.feature_block_num = configuration.featureBlockNum;
    param.recognition_threshold = configuration.searchThreshold;
    param.search_mode = (SearchMode)configuration.searchMode;
    auto ret = FEATURE_HUB->EnableHub(param);

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
    inspire::DetectMode detMode = inspire::DETECT_MODE_ALWAYS_DETECT;
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

HResult HFExecuteFaceTrack(HFSession session, HFImageStream streamHandle,
                           PHFMultipleFaceData results) {
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
    results->angles.pitch = (HFloat *)ctx->impl.GetPitchResultsCache().data();
    results->angles.roll = (HFloat *)ctx->impl.GetRollResultsCache().data();
    results->angles.yaw = (HFloat *)ctx->impl.GetYawResultsCache().data();
    results->tokens = (HFFaceBasicToken *)ctx->impl.GetFaceBasicDataCache().data();

    return ret;
}

HResult HFCopyFaceBasicToken(HFFaceBasicToken token, HPBuffer buffer, HInt32 bufferSize) {
    if (bufferSize < sizeof(inspire::HyperFaceData)) {
        return HERR_INVALID_BUFFER_SIZE;
    }
    std::memcpy(buffer, token.data, sizeof(inspire::HyperFaceData));
    return HSUCCEED;
}

HResult HFGetFaceBasicTokenSize(HPInt32 bufferSize) {
    *bufferSize = sizeof(inspire::HyperFaceData);
    return HSUCCEED;
}

HResult HFGetNumOfFaceDenseLandmark(HPInt32 num) {
    *num = 106;
    return HSUCCEED;
}

HResult HFGetFaceDenseLandmarkFromFaceToken(HFFaceBasicToken singleFace, HPoint2f *landmarks,
                                            HInt32 num) {
    if (num != 106) {
        return HERR_SESS_LANDMARK_NUM_NOT_MATCH;
    }
    inspire::FaceBasicData data;
    data.dataSize = singleFace.size;
    data.data = singleFace.data;
    HyperFaceData face = {0};
    HInt32 ret;
    ret = DeserializeHyperFaceData((char *)data.data, data.dataSize, face);
    if (ret != HSUCCEED) {
        return ret;
    }
    for (size_t i = 0; i < num; i++) {
        landmarks[i].x = face.densityLandmark[i].x;
        landmarks[i].y = face.densityLandmark[i].y;
    }

    return HSUCCEED;
}

HResult HFFeatureHubFaceSearchThresholdSetting(float threshold) {
    FEATURE_HUB->SetRecognitionThreshold(threshold);
    return HSUCCEED;
}

HResult HFFaceFeatureExtract(HFSession session, HFImageStream streamHandle,
                             HFFaceBasicToken singleFace, PHFFaceFeature feature) {
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

HResult HFFaceFeatureExtractCpy(HFSession session, HFImageStream streamHandle,
                                HFFaceBasicToken singleFace, HPFloat feature) {
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
    auto ret = FEATURE_HUB->CosineSimilarity(feature1.data, feature2.data, feature1.size, res);
    *result = res;

    return ret;
}

HResult HFGetFeatureLength(HPInt32 num) {
    *num = FEATURE_HUB->GetFeatureNum();

    return HSUCCEED;
}

HResult HFFeatureHubInsertFeature(HFFaceFeatureIdentity featureIdentity) {
    if (featureIdentity.feature->data == nullptr) {
        return HERR_INVALID_FACE_FEATURE;
    }
    std::vector<float> feat;
    feat.reserve(featureIdentity.feature->size);
    for (int i = 0; i < featureIdentity.feature->size; ++i) {
        feat.push_back(featureIdentity.feature->data[i]);
    }
    std::string tag(featureIdentity.tag);
    HInt32 ret = FEATURE_HUB->FaceFeatureInsertFromCustomId(feat, tag, featureIdentity.customId);

    return ret;
}

HResult HFFeatureHubFaceSearch(HFFaceFeature searchFeature, HPFloat confidence,
                               PHFFaceFeatureIdentity mostSimilar) {
    if (searchFeature.data == nullptr) {
        return HERR_INVALID_FACE_FEATURE;
    }
    std::vector<float> feat;
    feat.reserve(searchFeature.size);
    for (int i = 0; i < searchFeature.size; ++i) {
        feat.push_back(searchFeature.data[i]);
    }
    inspire::SearchResult result;
    HInt32 ret = FEATURE_HUB->SearchFaceFeature(feat, result);
    mostSimilar->feature = (HFFaceFeature *)FEATURE_HUB->GetFaceFeaturePtrCache().get();
    mostSimilar->feature->data = (HFloat *)FEATURE_HUB->GetSearchFaceFeatureCache().data();
    mostSimilar->feature->size = FEATURE_HUB->GetSearchFaceFeatureCache().size();
    mostSimilar->tag = FEATURE_HUB->GetStringCache();
    mostSimilar->customId = result.customId;
    *confidence = result.score;

    return ret;
}

HResult HFFeatureHubFaceSearchTopK(HFFaceFeature searchFeature, HInt32 topK,
                                   PHFSearchTopKResults results) {
    if (searchFeature.data == nullptr) {
        return HERR_INVALID_FACE_FEATURE;
    }
    std::vector<float> feat;
    feat.reserve(searchFeature.size);
    for (int i = 0; i < searchFeature.size; ++i) {
        feat.push_back(searchFeature.data[i]);
    }
    HInt32 ret = FEATURE_HUB->SearchFaceFeatureTopK(feat, topK);
    if (ret == HSUCCEED) {
        results->size = FEATURE_HUB->GetTopKConfidence().size();
        results->confidence = FEATURE_HUB->GetTopKConfidence().data();
        results->customIds = FEATURE_HUB->GetTopKCustomIdsCache().data();
    }

    return ret;
}

HResult HFFeatureHubFaceRemove(HInt32 customId) {
    auto ret = FEATURE_HUB->FaceFeatureRemoveFromCustomId(customId);
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
    std::string tag(featureIdentity.tag);

    auto ret = FEATURE_HUB->FaceFeatureUpdateFromCustomId(feat, tag, featureIdentity.customId);

    return ret;
}

HResult HFFeatureHubGetFaceIdentity(HInt32 customId, PHFFaceFeatureIdentity identity) {
    auto ret = FEATURE_HUB->GetFaceFeatureFromCustomId(customId);
    if (ret == HSUCCEED) {
        identity->tag = FEATURE_HUB->GetStringCache();
        identity->customId = customId;
        identity->feature = (HFFaceFeature *)FEATURE_HUB->GetFaceFeaturePtrCache().get();
        identity->feature->data = (HFloat *)FEATURE_HUB->GetFaceFeaturePtrCache()->data;
        identity->feature->size = FEATURE_HUB->GetFaceFeaturePtrCache()->dataSize;
    } else {
        identity->customId = -1;
    }

    return ret;
}

HResult HFMultipleFacePipelineProcess(HFSession session, HFImageStream streamHandle,
                                      PHFMultipleFaceData faces,
                                      HFSessionCustomParameter parameter) {
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

    HResult ret;
    std::vector<inspire::HyperFaceData> data;
    data.resize(faces->detectedNum);
    for (int i = 0; i < faces->detectedNum; ++i) {
        auto &face = data[i];
        ret = DeserializeHyperFaceData((char *)faces->tokens[i].data, faces->tokens[i].size, face);
        if (ret != HSUCCEED) {
            return HERR_INVALID_FACE_TOKEN;
        }
    }

    ret = ctx->impl.FacesProcess(stream->impl, data, param);

    return ret;
}

HResult HFMultipleFacePipelineProcessOptional(HFSession session, HFImageStream streamHandle,
                                              PHFMultipleFaceData faces, HInt32 customOption) {
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

    HResult ret;
    std::vector<inspire::HyperFaceData> data;
    data.resize(faces->detectedNum);
    for (int i = 0; i < faces->detectedNum; ++i) {
        auto &face = data[i];
        ret = DeserializeHyperFaceData((char *)faces->tokens[i].data, faces->tokens[i].size, face);
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

HResult HFFaceQualityDetect(HFSession session, HFFaceBasicToken singleFace, HFloat *confidence) {
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

    auto ret = inspire::FaceContext::FaceQualityDetect(data, *confidence);

    return ret;
}

HResult HFGetFaceIntereactionStateResult(HFSession session, PHFFaceIntereactionState result) {
    if (session == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    HF_FaceAlgorithmSession *ctx = (HF_FaceAlgorithmSession *)session;
    if (ctx == nullptr) {
        return HERR_INVALID_CONTEXT_HANDLE;
    }
    result->num = ctx->impl.GetFaceInteractionLeftEyeStatusCache().size();
    result->leftEyeStatusConfidence =
      (HFloat *)ctx->impl.GetFaceInteractionLeftEyeStatusCache().data();
    result->rightEyeStatusConfidence =
      (HFloat *)ctx->impl.GetFaceInteractionRightEyeStatusCache().data();

    return HSUCCEED;
}

HResult HFGetFaceIntereactionActionsResult(HFSession session, PHFFaceIntereactionsActions actions) {
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
    actions->headRiase = (HInt32 *)ctx->impl.GetFaceRaiseHeadAactionsResultCache().data();
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

HResult HFFeatureHubGetFaceCount(HInt32 *count) {
    *count = FEATURE_HUB->GetFaceFeatureCount();
    return HSUCCEED;
}

HResult HFFeatureHubViewDBTable() {
    return FEATURE_HUB->ViewDBTable();
}

HResult HFQueryInspireFaceVersion(PHFInspireFaceVersion version) {
    version->major = std::stoi(INSPIRE_FACE_VERSION_MAJOR_STR);
    version->minor = std::stoi(INSPIRE_FACE_VERSION_MINOR_STR);
    version->patch = std::stoi(INSPIRE_FACE_VERSION_PATCH_STR);

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

HResult HFDeBugShowResourceStatistics() {
    RESOURCE_MANAGE->printResourceStatistics();
    return HSUCCEED;
}