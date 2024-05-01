/* Copyright 2021 iwatake2222

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/*** Include ***/
/* for general */
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>

/* for My modules */
#include "inference_helper_log.h"
#include "inference_helper.h"

#ifdef INFERENCE_HELPER_ENABLE_OPENCV
#include "inference_helper_opencv.h"
#endif
#if defined(INFERENCE_HELPER_ENABLE_TFLITE) || defined(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK) || defined(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU) || defined(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU)
#include "inference_helper_tensorflow_lite.h"
#endif
#ifdef INFERENCE_HELPER_ENABLE_TENSORRT
#include "inference_helper_tensorrt.h"
#endif
#ifdef INFERENCE_HELPER_ENABLE_NCNN
#include "inference_helper_ncnn.h"
#endif
#ifdef INFERENCE_HELPER_ENABLE_MNN
#include "inference_helper_mnn.h"
#endif
#ifdef INFERENCE_HELPER_ENABLE_SNPE
#include "inference_helper_snpe.h"
#endif
#ifdef INFERENCE_HELPER_ENABLE_ARMNN
#include "inference_helper_armnn.h"
#endif
#if defined(INFERENCE_HELPER_ENABLE_NNABLA) || defined(INFERENCE_HELPER_ENABLE_NNABLA_CUDA)
#include "inference_helper_nnabla.h"
#endif
#if defined(INFERENCE_HELPER_ENABLE_ONNX_RUNTIME) || defined(INFERENCE_HELPER_ENABLE_ONNX_RUNTIME_CUDA)
#include "inference_helper_onnx_runtime.h"
#endif
#if defined(INFERENCE_HELPER_ENABLE_LIBTORCH) || defined(INFERENCE_HELPER_ENABLE_LIBTORCH_CUDA)
#include "inference_helper_libtorch.h"
#endif
#if defined(INFERENCE_HELPER_ENABLE_TENSORFLOW) || defined(INFERENCE_HELPER_ENABLE_TENSORFLOW_GPU)
#include "inference_helper_tensorflow.h"
#endif
#ifdef INFERENCE_HELPER_ENABLE_SAMPLE
#include "inference_helper_sample.h"
#endif
#ifdef INFERENCE_HELPER_ENABLE_RKNN
//#include "inference_helper_rknn.h"
#include "inference_helper_rknn_adapter.h"
#endif

/*** Macro ***/
#define TAG "InferenceHelper"
#define PRINT(...)   INFERENCE_HELPER_LOG_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) INFERENCE_HELPER_LOG_PRINT_E(TAG, __VA_ARGS__)


InferenceHelper* InferenceHelper::Create(const InferenceHelper::HelperType helper_type)
{
    InferenceHelper* p = nullptr;
    switch (helper_type) {
#ifdef INFERENCE_HELPER_ENABLE_OPENCV
    case kOpencv:
    case kOpencvGpu:
        PRINT("Use OpenCV \n");
        p = new InferenceHelperOpenCV();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE
    case kTensorflowLite:
        PRINT("Use TensorflowLite\n");
        p = new InferenceHelperTensorflowLite();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK
    case kTensorflowLiteXnnpack:
        PRINT("Use TensorflowLite XNNPACK Delegate\n");
        p = new InferenceHelperTensorflowLite();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU
    case kTensorflowLiteGpu:
        PRINT("Use TensorflowLite GPU Delegate\n");
        p = new InferenceHelperTensorflowLite();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU
    case kTensorflowLiteEdgetpu:
        PRINT("Use TensorflowLite EdgeTPU Delegate\n");
        p = new InferenceHelperTensorflowLite();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_NNAPI
    case kTensorflowLiteNnapi:
        PRINT("Use TensorflowLite NNAPI Delegate\n");
        p = new InferenceHelperTensorflowLite();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_TENSORRT
    case kTensorrt:
        PRINT("Use TensorRT \n");
        p = new InferenceHelperTensorRt();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_NCNN
    case kNcnn:
    case kNcnnVulkan:
        PRINT("Use NCNN\n");
        p = new InferenceHelperNcnn();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_MNN
    case kMnn:
//        PRINT("Use General Inference\n");
        p = new InferenceHelperMnn();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_SNPE
    case kSnpe:
        PRINT("Use SNPE\n");
        p = new InferenceHelperSnpe();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_ARMNN
    case kArmnn:
        PRINT("Use ARMNN\n");
        p = new InferenceHelperArmnn();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_NNABLA
    case kNnabla:
        PRINT("Use NNabla\n");
        p = new InferenceHelperNnabla();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_NNABLA_CUDA
    case kNnablaCuda:
        PRINT("Use NNabla_CUDA\n");
        p = new InferenceHelperNnabla();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_ONNX_RUNTIME
    case kOnnxRuntime:
        PRINT("Use ONNX Runtime\n");
        p = new InferenceHelperOnnxRuntime();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_ONNX_RUNTIME_CUDA
    case kOnnxRuntimeCuda:
        PRINT("Use ONNX Runtime_CUDA\n");
        p = new InferenceHelperOnnxRuntime();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_LIBTORCH
    case kLibtorch:
        PRINT("Use LibTorch\n");
        p = new InferenceHelperLibtorch();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_LIBTORCH_CUDA
    case kLibtorchCuda:
        PRINT("Use LibTorch CUDA\n");
        p = new InferenceHelperLibtorch();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_TENSORFLOW
    case kTensorflow:
        PRINT("Use TensorFlow\n");
        p = new InferenceHelperTensorflow();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_TENSORFLOW_GPU
    case kTensorflowGpu:
        PRINT("Use TensorFlow GPU\n");
        p = new InferenceHelperTensorflow();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_SAMPLE
    case kSample:
        PRINT("Do not use this. this is just a reference code\n");
        p = new InferenceHelperSample();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_RKNN
    case kRknn:
//        PRINT("Use Rknn\n")
//        p = new InferenceHelperRKNN();
        p = new InferenceHelperRknnAdapter();
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

#ifdef INFERENCE_HELPER_ENABLE_PRE_PROCESS_BY_OPENCV
#include <opencv2/opencv.hpp>
void InferenceHelper::PreProcessByOpenCV(const InputTensorInfo& input_tensor_info, bool is_nchw, cv::Mat& img_blob)
{
    /* Generate mat from original data */
    cv::Mat img_src = cv::Mat(cv::Size(input_tensor_info.image_info.width, input_tensor_info.image_info.height), (input_tensor_info.image_info.channel == 3) ? CV_8UC3 : CV_8UC1, input_tensor_info.data);

    /* Crop image */
    if (input_tensor_info.image_info.width == input_tensor_info.image_info.crop_width && input_tensor_info.image_info.height == input_tensor_info.image_info.crop_height) {
        /* do nothing */
    } else {
        img_src = img_src(cv::Rect(input_tensor_info.image_info.crop_x, input_tensor_info.image_info.crop_y, input_tensor_info.image_info.crop_width, input_tensor_info.image_info.crop_height));
    }

    /* Resize image */
    if (input_tensor_info.image_info.crop_width == input_tensor_info.GetWidth() && input_tensor_info.image_info.crop_height == input_tensor_info.GetHeight()) {
        /* do nothing */
    } else {
        cv::resize(img_src, img_src, cv::Size(input_tensor_info.GetWidth(), input_tensor_info.GetHeight()));
    }

    /* Convert color type */
    if (input_tensor_info.image_info.channel == input_tensor_info.GetChannel()) {
        if (input_tensor_info.image_info.channel == 3 && input_tensor_info.image_info.swap_color) {
            cv::cvtColor(img_src, img_src, cv::COLOR_BGR2RGB);
        }
    } else if (input_tensor_info.image_info.channel == 3 && input_tensor_info.GetChannel() == 1) {
        cv::cvtColor(img_src, img_src, (input_tensor_info.image_info.is_bgr) ? cv::COLOR_BGR2GRAY : cv::COLOR_RGB2GRAY);
    } else if (input_tensor_info.image_info.channel == 1 && input_tensor_info.GetChannel() == 3) {
        cv::cvtColor(img_src, img_src, cv::COLOR_GRAY2BGR);
    }

    if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeFp32) {
        /* Normalize image */
        if (input_tensor_info.GetChannel() == 3) {
#if 1
            img_src.convertTo(img_src, CV_32FC3);
            cv::subtract(img_src, cv::Scalar(cv::Vec<float, 3>(input_tensor_info.normalize.mean)), img_src);
            cv::multiply(img_src, cv::Scalar(cv::Vec<float, 3>(input_tensor_info.normalize.norm)), img_src);
#else
            img_src.convertTo(img_src, CV_32FC3, 1.0 / 255);
            cv::subtract(img_src, cv::Scalar(cv::Vec<float, 3>(input_tensor_info.normalize.mean)), img_src);
            cv::divide(img_src, cv::Scalar(cv::Vec<float, 3>(input_tensor_info.normalize.norm)), img_src);
#endif
        } else {
#if 1
            img_src.convertTo(img_src, CV_32FC1);
            cv::subtract(img_src, cv::Scalar(cv::Vec<float, 1>(input_tensor_info.normalize.mean)), img_src);
            cv::multiply(img_src, cv::Scalar(cv::Vec<float, 1>(input_tensor_info.normalize.norm)), img_src);
#else
            img_src.convertTo(img_src, CV_32FC1, 1.0 / 255);
            cv::subtract(img_src, cv::Scalar(cv::Vec<float, 1>(input_tensor_info.normalize.mean)), img_src);
            cv::divide(img_src, cv::Scalar(cv::Vec<float, 1>(input_tensor_info.normalize.norm)), img_src);
#endif
        }
    } else {
        /* do nothing */
    }

    if (is_nchw) {
        /* Convert to 4-dimensional Mat in NCHW */
        img_src = cv::dnn::blobFromImage(img_src);
    }

    img_blob = img_src;
    //memcpy(blobData, img_src.data, img_src.cols * img_src.rows * img_src.channels());

}

#else 
/* For the environment where OpenCV is not supported */
void InferenceHelper::PreProcessByOpenCV(const InputTensorInfo& input_tensor_info, bool is_nchw, cv::Mat& img_blob)
{
    PRINT_E("[PreProcessByOpenCV] Unsupported function called\n");
    exit(-1);
}
#endif



void InferenceHelper::ConvertNormalizeParameters(InputTensorInfo& tensor_info)
{
    if (tensor_info.data_type != InputTensorInfo::kDataTypeImage) return;

#if 0
    /* Convert to speeden up normalization:  ((src / 255) - mean) / norm  = src * 1 / (255 * norm) - (mean / norm) */
    for (int32_t i = 0; i < 3; i++) {
        tensor_info.normalize.mean[i] /= tensor_info.normalize.norm[i];
        tensor_info.normalize.norm[i] *= 255.0f;
        tensor_info.normalize.norm[i] = 1.0f / tensor_info.normalize.norm[i];
    }
#endif
#if 1
    /* Convert to speeden up normalization:  ((src / 255) - mean) / norm = (src  - (mean * 255))  * (1 / (255 * norm)) */
    for (int32_t i = 0; i < 3; i++) {
        tensor_info.normalize.mean[i] *= 255.0f;
        tensor_info.normalize.norm[i] *= 255.0f;
        tensor_info.normalize.norm[i] = 1.0f / tensor_info.normalize.norm[i];
    }
#endif
}


void InferenceHelper::PreProcessImage(int32_t num_thread, const InputTensorInfo& input_tensor_info, float* dst)
{
    const int32_t img_width = input_tensor_info.GetWidth();
    const int32_t img_height = input_tensor_info.GetHeight();
    const int32_t img_channel = input_tensor_info.GetChannel();
    uint8_t* src = (uint8_t*)(input_tensor_info.data);
    if (input_tensor_info.is_nchw == true) {
        /* convert NHWC to NCHW */
#pragma omp parallel for num_threads(num_thread)
        for (int32_t c = 0; c < img_channel; c++) {
            for (int32_t i = 0; i < img_width * img_height; i++) {
                dst[c * img_width * img_height + i] = (src[i * img_channel + c] - input_tensor_info.normalize.mean[c]) * input_tensor_info.normalize.norm[c];
            }
        }
    } else {
        /* convert NHWC to NHWC */
#pragma omp parallel for num_threads(num_thread)
        for (int32_t i = 0; i < img_width * img_height; i++) {
            for (int32_t c = 0; c < img_channel; c++) {
#if 1
                dst[i * img_channel + c] = (src[i * img_channel + c] - input_tensor_info.normalize.mean[c]) * input_tensor_info.normalize.norm[c];
#else
                dst[i * img_channel + c] = (src[i * img_channel + c] / 255.0f - input_tensor_info.normalize.mean[c]) / input_tensor_info.normalize.norm[c];
#endif
            }
        }
    }
}

void InferenceHelper::PreProcessImage(int32_t num_thread, const InputTensorInfo& input_tensor_info, uint8_t* dst)
{
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

void InferenceHelper::PreProcessImage(int32_t num_thread, const InputTensorInfo& input_tensor_info, int8_t* dst)
{
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

template<typename T>
void InferenceHelper::PreProcessBlob(int32_t num_thread, const InputTensorInfo& input_tensor_info, T* dst)
{
    const int32_t img_width = input_tensor_info.GetWidth();
    const int32_t img_height = input_tensor_info.GetHeight();
    const int32_t img_channel = input_tensor_info.GetChannel();
    T* src = static_cast<T*>(input_tensor_info.data);
    if ((input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNchw && input_tensor_info.is_nchw) || (input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNhwc && !input_tensor_info.is_nchw)) {
        std::copy(src, src + input_tensor_info.GetElementNum(), dst);
    } else if (input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNchw) {
        /* NCHW -> NHWC */
#pragma omp parallel for num_threads(num_thread)
        for (int32_t i = 0; i < img_width * img_height; i++) {
            for (int32_t c = 0; c < img_channel; c++) {
                dst[i * img_channel + c] = src[c * (img_width * img_height) + i];
            }
        }
    } else if (input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNhwc) {
        /* NHWC -> NCHW */
#pragma omp parallel for num_threads(num_thread)
        for (int32_t i = 0; i < img_width * img_height; i++) {
            for (int32_t c = 0; c < img_channel; c++) {
                dst[c * (img_width * img_height) + i] = src[i * img_channel + c];
            }
        }
    }
}

template void InferenceHelper::PreProcessBlob<float>(int32_t num_thread, const InputTensorInfo& input_tensor_info, float* dst);
template void InferenceHelper::PreProcessBlob<int32_t>(int32_t num_thread, const InputTensorInfo& input_tensor_info, int32_t* dst);
template void InferenceHelper::PreProcessBlob<int64_t>(int32_t num_thread, const InputTensorInfo& input_tensor_info, int64_t* dst);
template void InferenceHelper::PreProcessBlob<uint8_t>(int32_t num_thread, const InputTensorInfo& input_tensor_info, uint8_t* dst);
template void InferenceHelper::PreProcessBlob<int8_t>(int32_t num_thread, const InputTensorInfo& input_tensor_info, int8_t* dst);
