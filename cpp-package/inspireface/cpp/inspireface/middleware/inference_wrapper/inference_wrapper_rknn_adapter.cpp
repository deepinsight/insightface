/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#ifdef INFERENCE_WRAPPER_ENABLE_RKNN

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
#include <cassert>
#include "inference_wrapper_rknn_adapter.h"
#include "inference_wrapper_log.h"
#include "log.h"
#include <cassert>

#define TAG "InferenceWrapperRKNNAdapter"
#define PRINT(...) INFERENCE_WRAPPER_LOG_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) INFERENCE_WRAPPER_LOG_PRINT_E(TAG, __VA_ARGS__)

InferenceWrapperRKNNAdapter::InferenceWrapperRKNNAdapter() {
    num_threads_ = 1;
}

InferenceWrapperRKNNAdapter::~InferenceWrapperRKNNAdapter() {}

int32_t InferenceWrapperRKNNAdapter::SetNumThreads(const int32_t num_threads) {
    num_threads_ = num_threads;
    return WrapperOk;
}

int32_t InferenceWrapperRKNNAdapter::ParameterInitialization(std::vector<InputTensorInfo> &input_tensor_info_list,
                                                             std::vector<OutputTensorInfo> &output_tensor_info_list) {
    return WrapperOk;
}

int32_t InferenceWrapperRKNNAdapter::Process(std::vector<OutputTensorInfo> &output_tensor_info_list) {
    if (output_tensor_info_list[0].tensor_type == TensorInfo::TensorTypeFp32) {
        net_->setOutputsWantFloat(1);
        //        INSPIRE_LOGD("WANT FLOAT!");
    }

    auto ret = net_->RunModel();
    if (ret != 0) {
        INSPIRE_LOGE("Run model error.");
        return WrapperError;
    }
    auto outputs_size = net_->GetOutputsNum();

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

    return WrapperOk;
}

int32_t InferenceWrapperRKNNAdapter::PreProcess(const std::vector<InputTensorInfo> &input_tensor_info_list) {
    for (int i = 0; i < input_tensor_info_list.size(); ++i) {
        auto &input_tensor_info = input_tensor_info_list[i];

        rknn_tensor_format fmt = RKNN_TENSOR_NHWC;
        if (input_tensor_info.is_nchw) {
            fmt = RKNN_TENSOR_NCHW;
        } else {
            fmt = RKNN_TENSOR_NHWC;
            //            INSPIRE_LOGD("NHWC!");
        }
        rknn_tensor_type type = RKNN_TENSOR_UINT8;
        if (input_tensor_info.tensor_type == InputTensorInfo::TensorInfo::TensorTypeFp32) {
            type = RKNN_TENSOR_FLOAT32;
        } else if (input_tensor_info.tensor_type == InputTensorInfo::TensorInfo::TensorTypeUint8) {
            type = RKNN_TENSOR_UINT8;
            //            INSPIRE_LOGD("UINT8!");
        }
        auto ret = net_->SetInputData(i, input_tensor_info.data, input_tensor_info.GetWidth(), input_tensor_info.GetHeight(),
                                      input_tensor_info.GetChannel(), type, fmt);
        if (ret != 0) {
            INSPIRE_LOGE("Set data error.");
            return ret;
        }
    }
    return WrapperOk;
}

int32_t InferenceWrapperRKNNAdapter::Initialize(const std::string &model_filename, std::vector<InputTensorInfo> &input_tensor_info_list,
                                                std::vector<OutputTensorInfo> &output_tensor_info_list) {
    INSPIRE_LOGE("NOT IMPL");

    return 0;
}

int32_t InferenceWrapperRKNNAdapter::Initialize(char *model_buffer, int model_size, std::vector<InputTensorInfo> &input_tensor_info_list,
                                                std::vector<OutputTensorInfo> &output_tensor_info_list) {
    net_ = std::make_shared<RKNNAdapter>();
    auto ret = net_->Initialize((unsigned char *)model_buffer, model_size);
    if (ret != 0) {
        INSPIRE_LOGE("Rknn init error.");
        return WrapperError;
    }
    return ParameterInitialization(input_tensor_info_list, output_tensor_info_list);
}

int32_t InferenceWrapperRKNNAdapter::Finalize(void) {
    if (net_ != nullptr) {
        net_->Release();
    }
    return WrapperOk;
}

std::vector<std::string> InferenceWrapperRKNNAdapter::GetInputNames() {
    return std::vector<std::string>();
}

int32_t InferenceWrapperRKNNAdapter::ResizeInput(const std::vector<InputTensorInfo> &input_tensor_info_list) {
    // The function is not supported
    return 0;
}

#endif  // INFERENCE_WRAPPER_ENABLE_RKNN