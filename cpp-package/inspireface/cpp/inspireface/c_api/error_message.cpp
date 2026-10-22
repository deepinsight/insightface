#include "inspireface.h"

#include <cstring>

namespace {

const char* ResolveErrorMessage(HResult error_code) {
    switch (error_code) {
        case HSUCCEED: return "Success";
        case HERR_UNKNOWN: return "Unspecified SDK error";
        case HERR_INVALID_PARAM: return "Invalid parameter";
        case HERR_INVALID_IMAGE_STREAM_HANDLE: return "Invalid image stream handle";
        case HERR_INVALID_CONTEXT_HANDLE: return "Invalid session handle";
        case HERR_INVALID_FACE_TOKEN: return "Invalid face token";
        case HERR_INVALID_FACE_FEATURE: return "Invalid face feature";
        case HERR_INVALID_FACE_LIST: return "Invalid face list";
        case HERR_INVALID_BUFFER_SIZE: return "Invalid buffer size";
        case HERR_INVALID_IMAGE_STREAM_PARAM: return "Invalid image stream parameter";
        case HERR_INVALID_SERIALIZATION_FAILED: return "Face data serialization failed";
        case HERR_INVALID_DETECTION_INPUT: return "Invalid face detection input";
        case HERR_INVALID_IMAGE_BITMAP_HANDLE: return "Invalid image bitmap handle";
        case HERR_IMAGE_STREAM_DECODE_FAILED: return "Image stream decoding failed";
        case HERR_UNSUPPORTED: return "Requested operation is unsupported";

        case HERR_SESS_FUNCTION_UNUSABLE: return "Session function is unavailable";
        case HERR_SESS_TRACKER_FAILURE: return "Session tracker is not initialized";
        case HERR_SESS_PIPELINE_FAILURE: return "Session pipeline is not initialized";
        case HERR_SESS_INVALID_RESOURCE: return "Session resource is invalid";
        case HERR_SESS_LANDMARK_NUM_NOT_MATCH: return "Landmark count does not match";
        case HERR_SESS_LANDMARK_NOT_ENABLE: return "Landmark processing is not enabled";
        case HERR_SESS_KEY_POINT_NUM_NOT_MATCH: return "Key point count does not match";
        case HERR_SESS_REC_EXTRACT_FAILURE: return "Face feature extraction is unavailable";
        case HERR_SESS_REC_CONTRAST_FEAT_ERR: return "Face feature comparison input is invalid";
        case HERR_SESS_FACE_DATA_ERROR: return "Face data could not be parsed";
        case HERR_SESS_FACE_REC_OPTION_ERROR: return "Face recognition option is invalid";

        case HERR_FT_HUB_DISABLE: return "FeatureHub is disabled";
        case HERR_FT_HUB_INSERT_FAILURE: return "FeatureHub insertion failed";
        case HERR_FT_HUB_NOT_FOUND_FEATURE: return "FeatureHub feature was not found";
        case HERR_FT_HUB_INVALID_FEATURE: return "FeatureHub feature is invalid";
        case HERR_FT_HUB_DATABASE_FAILURE: return "FeatureHub database operation failed";

        case HERR_ARCHIVE_LOAD_FAILURE: return "Resource archive loading failed";
        case HERR_ARCHIVE_LOAD_MODEL_FAILURE: return "Model loading from the resource archive failed";
        case HERR_ARCHIVE_FILE_FORMAT_ERROR: return "Resource archive format is invalid";
        case HERR_ARCHIVE_REPETITION_LOAD: return "Resource archive is already loaded";
        case HERR_ARCHIVE_NOT_LOAD: return "Resource archive is not loaded";

        case HERR_DEVICE_CUDA_NOT_SUPPORT: return "CUDA is unsupported on this device";
        case HERR_DEVICE_CUDA_TENSORRT_NOT_SUPPORT: return "TensorRT is unsupported on this CUDA device";
        case HERR_DEVICE_CUDA_UNKNOWN_ERROR: return "CUDA reported an unknown error";
        case HERR_DEVICE_CUDA_DISABLE: return "CUDA support is disabled";
        case HERR_DEVICE_IMAGE_PROCESS_FAILURE: return "Device image processing failed";

        case HERR_EXTENSION_ERROR: return "Extension module error";
        case HERR_EXTENSION_MLMODEL_LOAD_FAILED: return "Extension ML model loading failed";
        case HERR_EXTENSION_HETERO_MODEL_TAG_ERROR: return "Extension heterogeneous model tag is invalid";
        case HERR_EXTENSION_HETERO_REC_HEAD_CONFIG_ERROR: return "Extension recognition head configuration is invalid";
        case HERR_EXTENSION_HETERO_MODEL_NOT_MATCH: return "Extension heterogeneous model does not match";
        case HERR_EXTENSION_HETERO_MODEL_NOT_LOADED: return "Extension heterogeneous model is not loaded";

        case HERR_CAPTURE_INVALID_CONFIG: return "Face capture configuration is invalid";
        case HERR_CAPTURE_REQUIRED_FEATURE_OFF: return "A face capture filter requires a disabled session feature";
        case HERR_CAPTURE_FRAME_OUT_OF_ORDER: return "Face capture frame ID or timestamp is out of order";
        case HERR_CAPTURE_INVALID_HANDLE: return "Invalid face capture session handle";

        default: return "Unknown error code";
    }
}

}  // namespace

HResult HFGetErrorMessage(HResult errorCode, HString buffer, HInt32 bufferSize, HPInt32 requiredSize) {
    if (requiredSize == nullptr || bufferSize < 0 || (buffer == nullptr && bufferSize != 0)) {
        return HERR_INVALID_PARAM;
    }

    const char* message = ResolveErrorMessage(errorCode);
    const size_t required = std::strlen(message) + 1;
    *requiredSize = static_cast<HInt32>(required);
    if (buffer == nullptr) {
        return HSUCCEED;
    }
    if (static_cast<size_t>(bufferSize) < required) {
        if (bufferSize > 0) {
            buffer[0] = '\0';
        }
        return HERR_INVALID_BUFFER_SIZE;
    }

    std::memcpy(buffer, message, required);
    return HSUCCEED;
}
