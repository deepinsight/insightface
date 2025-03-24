#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include "inference_wrapper_log.h"
#include "inference_wrapper.h"

#ifdef INFERENCE_WRAPPER_ENABLE_MNN
#include "inference_wrapper_mnn.h"
#endif
#ifdef INFERENCE_WRAPPER_ENABLE_RKNN
// #include "inference_wrapper_rknn.h"
#include "inference_wrapper_rknn_adapter.h"
#endif
#ifdef INFERENCE_WRAPPER_ENABLE_RKNN2
#include "inference_wrapper_rknn_adapter_nano.h"
#endif
#ifdef INFERENCE_WRAPPER_ENABLE_COREML
#include "inference_wrapper_coreml.h"
#endif
#ifdef INFERENCE_WRAPPER_ENABLE_TENSORRT
#include "inference_wrapper_tensorrt.h"
#endif

#define TAG "InferenceWrapper"
#define PRINT(...) INFERENCE_WRAPPER_LOG_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) INFERENCE_WRAPPER_LOG_PRINT_E(TAG, __VA_ARGS__)

InferenceWrapper* InferenceWrapper::Create(const InferenceWrapper::EngineType helper_type) {
    InferenceWrapper* p = nullptr;
    switch (helper_type) {
#ifdef INFERENCE_WRAPPER_ENABLE_MNN
        case INFER_MNN:
            //        PRINT("Use General Inference\n");
            p = new InferenceWrapperMNN();
            break;
#endif
#ifdef INFERENCE_WRAPPER_ENABLE_RKNN
        case INFER_RKNN:
            //        PRINT("Use Rknn\n")
            //        p = new InferenceWrapperRKNN();
            p = new InferenceWrapperRKNNAdapter();
            break;
#endif
#ifdef INFERENCE_WRAPPER_ENABLE_RKNN2
        case INFER_RKNN:
            // PRINT("Use Rknn2\n");
            p = new InferenceWrapperRKNNAdapter();
            break;
#endif
#ifdef INFERENCE_WRAPPER_ENABLE_COREML
        case INFER_COREML:
            // PRINT("Use CoreML\n");
            p = new InferenceWrapperCoreML();
            break;
#endif
#ifdef INFERENCE_WRAPPER_ENABLE_TENSORRT
        case INFER_TENSORRT:
            // PRINT("Use TensorRT\n");
            p = new InferenceWrapperTensorRT();
            break;
#endif
        default:
            PRINT_E("Unsupported inference helper type (%d)\n", helper_type)
            break;
    }
    if (p == nullptr) {
        PRINT_E("Failed to create inference helper\n")
    } else {
        p->helper_type_ = helper_type;
    }
    return p;
}

void InferenceWrapper::ConvertNormalizeParameters(InputTensorInfo& tensor_info) {
    if (tensor_info.data_type != InputTensorInfo::DataTypeImage)
        return;

    /* Convert to speeden up normalization:  ((src / 255) - mean) / norm = (src  - (mean * 255))  * (1 / (255 * norm)) */
    for (int32_t i = 0; i < 3; i++) {
        tensor_info.normalize.mean[i] *= 255.0f;
        tensor_info.normalize.norm[i] *= 255.0f;
        tensor_info.normalize.norm[i] = 1.0f / tensor_info.normalize.norm[i];
    }
}

void InferenceWrapper::PreProcessImage(int32_t num_thread, const InputTensorInfo& input_tensor_info, float* dst) {
    const int32_t img_width = input_tensor_info.GetWidth();
    const int32_t img_height = input_tensor_info.GetHeight();
    const int32_t img_channel = input_tensor_info.GetChannel();
    uint8_t* src = (uint8_t*)(input_tensor_info.data);
    if (input_tensor_info.is_nchw == true) {
        /* convert NHWC to NCHW */
#pragma omp parallel for num_threads(num_thread)
        for (int32_t c = 0; c < img_channel; c++) {
            for (int32_t i = 0; i < img_width * img_height; i++) {
                dst[c * img_width * img_height + i] =
                  (src[i * img_channel + c] - input_tensor_info.normalize.mean[c]) * input_tensor_info.normalize.norm[c];
            }
        }
    } else {
        /* convert NHWC to NHWC */
#pragma omp parallel for num_threads(num_thread)
        for (int32_t i = 0; i < img_width * img_height; i++) {
            for (int32_t c = 0; c < img_channel; c++) {
                dst[i * img_channel + c] = (src[i * img_channel + c] - input_tensor_info.normalize.mean[c]) * input_tensor_info.normalize.norm[c];
            }
        }
    }
}

void InferenceWrapper::PreProcessImage(int32_t num_thread, const InputTensorInfo& input_tensor_info, uint8_t* dst) {
    const int32_t img_width = input_tensor_info.GetWidth();
    const int32_t img_height = input_tensor_info.GetHeight();
    const int32_t img_channel = input_tensor_info.GetChannel();
    uint8_t* src = (uint8_t*)(input_tensor_info.data);
    if (input_tensor_info.is_nchw == true) {
        /* convert NHWC to NCHW */
#pragma omp parallel for num_threads(num_thread)
        for (int32_t c = 0; c < img_channel; c++) {
            for (int32_t i = 0; i < img_width * img_height; i++) {
                dst[c * img_width * img_height + i] = src[i * img_channel + c];
            }
        }
    } else {
        /* convert NHWC to NHWC */
        std::copy(src, src + input_tensor_info.GetElementNum(), dst);
    }
}

void InferenceWrapper::PreProcessImage(int32_t num_thread, const InputTensorInfo& input_tensor_info, int8_t* dst) {
    const int32_t img_width = input_tensor_info.GetWidth();
    const int32_t img_height = input_tensor_info.GetHeight();
    const int32_t img_channel = input_tensor_info.GetChannel();
    uint8_t* src = (uint8_t*)(input_tensor_info.data);
    if (input_tensor_info.is_nchw == true) {
        /* convert NHWC to NCHW */
#pragma omp parallel for num_threads(num_thread)
        for (int32_t c = 0; c < img_channel; c++) {
            for (int32_t i = 0; i < img_width * img_height; i++) {
                dst[c * img_width * img_height + i] = src[i * img_channel + c] - 128;
            }
        }
    } else {
#pragma omp parallel for num_threads(num_thread)
        for (int32_t i = 0; i < img_width * img_height; i++) {
            for (int32_t c = 0; c < img_channel; c++) {
                dst[i * img_channel + c] = src[i * img_channel + c] - 128;
            }
        }
    }
}

template <typename T>
void InferenceWrapper::PreProcessBlob(int32_t num_thread, const InputTensorInfo& input_tensor_info, T* dst) {
    const int32_t img_width = input_tensor_info.GetWidth();
    const int32_t img_height = input_tensor_info.GetHeight();
    const int32_t img_channel = input_tensor_info.GetChannel();
    T* src = static_cast<T*>(input_tensor_info.data);
    if ((input_tensor_info.data_type == InputTensorInfo::DataTypeBlobNchw && input_tensor_info.is_nchw) ||
        (input_tensor_info.data_type == InputTensorInfo::DataTypeBlobNhwc && !input_tensor_info.is_nchw)) {
        std::copy(src, src + input_tensor_info.GetElementNum(), dst);
    } else if (input_tensor_info.data_type == InputTensorInfo::DataTypeBlobNchw) {
        /* NCHW -> NHWC */
#pragma omp parallel for num_threads(num_thread)
        for (int32_t i = 0; i < img_width * img_height; i++) {
            for (int32_t c = 0; c < img_channel; c++) {
                dst[i * img_channel + c] = src[c * (img_width * img_height) + i];
            }
        }
    } else if (input_tensor_info.data_type == InputTensorInfo::DataTypeBlobNhwc) {
        /* NHWC -> NCHW */
#pragma omp parallel for num_threads(num_thread)
        for (int32_t i = 0; i < img_width * img_height; i++) {
            for (int32_t c = 0; c < img_channel; c++) {
                dst[c * (img_width * img_height) + i] = src[i * img_channel + c];
            }
        }
    }
}

template void InferenceWrapper::PreProcessBlob<float>(int32_t num_thread, const InputTensorInfo& input_tensor_info, float* dst);
template void InferenceWrapper::PreProcessBlob<int32_t>(int32_t num_thread, const InputTensorInfo& input_tensor_info, int32_t* dst);
template void InferenceWrapper::PreProcessBlob<int64_t>(int32_t num_thread, const InputTensorInfo& input_tensor_info, int64_t* dst);
template void InferenceWrapper::PreProcessBlob<uint8_t>(int32_t num_thread, const InputTensorInfo& input_tensor_info, uint8_t* dst);
template void InferenceWrapper::PreProcessBlob<int8_t>(int32_t num_thread, const InputTensorInfo& input_tensor_info, int8_t* dst);
