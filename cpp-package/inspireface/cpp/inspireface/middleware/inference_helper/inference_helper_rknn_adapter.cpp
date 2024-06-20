//
// Created by tunm on 2023/9/24.
//

#ifdef INFERENCE_HELPER_ENABLE_RKNN

#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include "inference_helper_rknn_adapter.h"
#include "inference_helper_log.h"
#include "log.h"


/*** Macro ***/
#define TAG "InferenceHelperRknn"
#define PRINT(...)   INFERENCE_HELPER_LOG_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) INFERENCE_HELPER_LOG_PRINT_E(TAG, __VA_ARGS__)


InferenceHelperRknnAdapter::InferenceHelperRknnAdapter() {
    num_threads_ = 1;
}

InferenceHelperRknnAdapter::~InferenceHelperRknnAdapter() {
}

int32_t InferenceHelperRknnAdapter::SetNumThreads(const int32_t num_threads) {
    num_threads_ = num_threads;
    return kRetOk;
}

int32_t InferenceHelperRknnAdapter::SetCustomOps(const std::vector<std::pair<const char *, const void *>> &custom_ops) {
    PRINT("[WARNING] This method is not supported\n")
    return kRetOk;
}

int32_t InferenceHelperRknnAdapter::ParameterInitialization(std::vector<InputTensorInfo> &input_tensor_info_list,
                                                            std::vector<OutputTensorInfo> &output_tensor_info_list) {

    return kRetOk;
}

int32_t InferenceHelperRknnAdapter::Process(std::vector<OutputTensorInfo> &output_tensor_info_list) {
    if (output_tensor_info_list[0].tensor_type == TensorInfo::kTensorTypeFp32) {
        net_->setOutputsWantFloat(1);
//        INSPIRE_LOGD("WANT FLOAT!");
    }

    auto ret = net_->RunModel();
    if (ret != 0) {
        INSPIRE_LOGE("Run model error.");
        return kRetErr;
    }
    auto outputs_size = net_->GetOutputsNum();

//    INSPIRE_LOGD("==================");
//    auto f = net_->GetOutputData(0);
//    for (int i = 0; i < 512; ++i) {
//        std::cout << f[i] << ", ";
//    }
//    std::cout << std::endl;
//
//    INSPIRE_LOGD("==================");

    assert(outputs_size == output_tensor_info_list.size());

    for (int index = 0; index < outputs_size; ++index) {
        auto &output_tensor = output_tensor_info_list[index];
        output_tensor.data = net_->GetOutputFlow(index);

        auto dim = net_->GetOutputTensorSize(index);
        output_tensor.tensor_dims.clear();
        for (int i = 0; i < dim.size(); ++i) {
            output_tensor.tensor_dims.push_back((int)dim[i]);
//            INSPIRE_LOGE("dim: %d", dim[i]);
        }

    }

    net_->ReleaseOutputs();

    return kRetOk;
}

int32_t InferenceHelperRknnAdapter::PreProcess(const std::vector<InputTensorInfo> &input_tensor_info_list) {
    for (int i = 0; i < input_tensor_info_list.size(); ++i) {
        auto &input_tensor_info = input_tensor_info_list[i];
//        cv::Mat mat(input_tensor_info.GetHeight(), input_tensor_info.GetWidth(), CV_8UC3, input_tensor_info.data);
//        INSPIRE_LOGD("decode : %d", input_tensor_info.GetHeight());
//        INSPIRE_LOGD("decode : %d", input_tensor_info.GetWidth());
//        cv::Mat bgr;
//        cv::cvtColor(mat, bgr, cv::COLOR_RGB2BGR);
//        cv::imwrite("dec.jpg", bgr);
        rknn_tensor_format fmt = RKNN_TENSOR_NHWC;
        if (input_tensor_info.is_nchw) {
            fmt = RKNN_TENSOR_NCHW;
        } else {
            fmt = RKNN_TENSOR_NHWC;
//            INSPIRE_LOGD("NHWC!");
        }
        rknn_tensor_type type = RKNN_TENSOR_UINT8;
        if (input_tensor_info.tensor_type == InputTensorInfo::TensorInfo::kTensorTypeFp32) {
            type = RKNN_TENSOR_FLOAT32;
        } else if (input_tensor_info.tensor_type == InputTensorInfo::TensorInfo::kTensorTypeUint8) {
            type = RKNN_TENSOR_UINT8;
//            INSPIRE_LOGD("UINT8!");
        }
        auto ret = net_->SetInputData(i, input_tensor_info.data, input_tensor_info.GetWidth(), input_tensor_info.GetHeight(), input_tensor_info.GetChannel(), type, fmt);
        if (ret != 0) {
            INSPIRE_LOGE("Set data error.");
            return ret;
        }
    }
    return kRetOk;
}

int32_t
InferenceHelperRknnAdapter::Initialize(const std::string &model_filename, std::vector<InputTensorInfo> &input_tensor_info_list,
                                       std::vector<OutputTensorInfo> &output_tensor_info_list) {
    INSPIRE_LOGE("NOT IMPL");

    return 0;
}

int32_t InferenceHelperRknnAdapter::Initialize(char *model_buffer, int model_size,
                                               std::vector<InputTensorInfo> &input_tensor_info_list,
                                               std::vector<OutputTensorInfo> &output_tensor_info_list) {
    net_ = std::make_shared<RKNNAdapter>();
    auto ret = net_->Initialize((unsigned char* )model_buffer, model_size);
    if (ret != 0) {
        INSPIRE_LOGE("Rknn init error.");
        return kRetErr;
    }
    return ParameterInitialization(input_tensor_info_list, output_tensor_info_list);
}

int32_t InferenceHelperRknnAdapter::Finalize(void) {
    if (net_ != nullptr) {
        net_->Release();
    }
    return kRetOk;
}

std::vector<std::string> InferenceHelperRknnAdapter::GetInputNames() {
    return std::vector<std::string>();
}

int32_t InferenceHelperRknnAdapter::ResizeInput(const std::vector<InputTensorInfo>& input_tensor_info_list) {
    // The function is not supported
    return 0;
}

#endif  // INFERENCE_HELPER_ENABLE_RKNN