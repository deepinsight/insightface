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
#include <limits>
#include "inference_wrapper_rknn_adapter.h"
#include "inference_wrapper_log.h"
#include "log.h"

#define TAG "InferenceWrapperRKNNAdapter"
#define PRINT(...) INFERENCE_WRAPPER_LOG_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) INFERENCE_WRAPPER_LOG_PRINT_E(TAG, __VA_ARGS__)

InferenceWrapperRKNNAdapter::InferenceWrapperRKNNAdapter() {
    num_threads_ = 1;
}

InferenceWrapperRKNNAdapter::~InferenceWrapperRKNNAdapter() {
    Finalize();
}

int32_t InferenceWrapperRKNNAdapter::SetNumThreads(const int32_t num_threads) {
    num_threads_ = num_threads;
    return WrapperOk;
}

int32_t InferenceWrapperRKNNAdapter::ParameterInitialization(std::vector<InputTensorInfo> &input_tensor_info_list,
                                                             std::vector<OutputTensorInfo> &output_tensor_info_list) {
    return WrapperOk;
}

int32_t InferenceWrapperRKNNAdapter::Process(std::vector<OutputTensorInfo> &output_tensor_info_list) {
    for (auto &output_tensor_info : output_tensor_info_list) {
        output_tensor_info.data = nullptr;
    }
    if (net_ == nullptr || output_tensor_info_list.empty()) {
        INSPIRE_LOGE("RKNN runtime or output metadata is not initialized.");
        return WrapperError;
    }

    std::vector<void *> output_views(output_tensor_info_list.size(), nullptr);
    std::vector<std::vector<int32_t>> output_dimensions(output_tensor_info_list.size());
    std::vector<float> output_scales(output_tensor_info_list.size(), 1.0f);
    std::vector<int32_t> output_zero_points(output_tensor_info_list.size(), 0);
    const bool success = inference_wrapper_detail::RunWithDeferredOutputRelease(
      output_release_,
      [this]() { return net_->ReleaseOutputs(); },
      [this, &output_tensor_info_list]() {
          net_->setOutputsWantFloat(output_tensor_info_list[0].tensor_type == TensorInfo::TensorTypeFp32 ? 1 : 0);
          return net_->RunModel();
      },
      [this, &output_tensor_info_list, &output_views, &output_dimensions, &output_scales, &output_zero_points]() {
          const size_t outputs_size = net_->GetOutputsNum();
          if (outputs_size != output_tensor_info_list.size()) {
              INSPIRE_LOGE("RKNN output count mismatch: runtime=%zu, expected=%zu", outputs_size, output_tensor_info_list.size());
              return false;
          }

          const auto &attributes = net_->GetOutputTensorAttr();
          if (attributes.size() != outputs_size) {
              return false;
          }
          for (size_t index = 0; index < outputs_size; ++index) {
              void *output_data = net_->GetOutputFlow(static_cast<int>(index));
              const auto dimensions = net_->GetOutputTensorSize(static_cast<int>(index));
              if (output_data == nullptr || dimensions.empty()) {
                  INSPIRE_LOGE("RKNN output %zu has invalid data or dimensions.", index);
                  return false;
              }

              output_dimensions[index].reserve(dimensions.size());
              for (const auto dimension : dimensions) {
                  if (dimension == 0 || dimension > static_cast<unsigned long>(std::numeric_limits<int32_t>::max())) {
                      INSPIRE_LOGE("RKNN output %zu contains an invalid dimension.", index);
                      return false;
                  }
                  output_dimensions[index].push_back(static_cast<int32_t>(dimension));
              }
              TensorInfo checked;
              checked.tensor_dims = output_dimensions[index];
              if (checked.GetElementNum() <= 0 || attributes[index].zp > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
                  !std::isfinite(attributes[index].scale)) {
                  INSPIRE_LOGE("RKNN output %zu element count is invalid.", index);
                  return false;
              }
              output_views[index] = output_data;
              output_scales[index] = attributes[index].scale;
              output_zero_points[index] = static_cast<int32_t>(attributes[index].zp);
          }
          return true;
      });

    if (!success) {
        for (auto &output_tensor_info : output_tensor_info_list) {
            output_tensor_info.data = nullptr;
        }
        INSPIRE_LOGE("RKNN inference or output lifecycle handling failed.");
        return WrapperError;
    }
    for (size_t index = 0; index < output_tensor_info_list.size(); ++index) {
        output_tensor_info_list[index].data = output_views[index];
        output_tensor_info_list[index].tensor_dims = std::move(output_dimensions[index]);
        output_tensor_info_list[index].quant.scale = output_scales[index];
        output_tensor_info_list[index].quant.zero_point = output_zero_points[index];
    }
    return WrapperOk;
}

int32_t InferenceWrapperRKNNAdapter::PreProcess(const std::vector<InputTensorInfo> &input_tensor_info_list) {
    if (net_ == nullptr || input_tensor_info_list.empty()) {
        return WrapperError;
    }
    for (size_t i = 0; i < input_tensor_info_list.size(); ++i) {
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
        } else if (input_tensor_info.tensor_type == InputTensorInfo::TensorInfo::TensorTypeInt8) {
            type = RKNN_TENSOR_INT8;
        } else {
            return WrapperError;
        }
        const int width = input_tensor_info.GetWidth();
        const int height = input_tensor_info.GetHeight();
        const int channel = input_tensor_info.GetChannel();
        if (input_tensor_info.data == nullptr || width <= 0 || height <= 0 || channel <= 0) {
            return WrapperError;
        }
        auto ret = net_->SetInputData(static_cast<int>(i), input_tensor_info.data, width, height,
                                      channel, type, fmt);
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

    return WrapperError;
}

int32_t InferenceWrapperRKNNAdapter::Initialize(char *model_buffer, int model_size, std::vector<InputTensorInfo> &input_tensor_info_list,
                                                std::vector<OutputTensorInfo> &output_tensor_info_list) {
    Finalize();
    if (model_buffer == nullptr || model_size <= 0 || output_tensor_info_list.empty()) {
        return WrapperError;
    }
    net_ = std::make_shared<RKNNAdapter>();
    auto ret = net_->Initialize((unsigned char *)model_buffer, model_size);
    if (ret != 0) {
        INSPIRE_LOGE("Rknn init error.");
        net_.reset();
        return WrapperError;
    }
    ret = ParameterInitialization(input_tensor_info_list, output_tensor_info_list);
    if (ret != WrapperOk) {
        Finalize();
    }
    return ret;
}

int32_t InferenceWrapperRKNNAdapter::Finalize(void) {
    int32_t status = WrapperOk;
    if (net_ != nullptr) {
        if (!output_release_.Release([this]() { return net_->ReleaseOutputs(); })) {
            INSPIRE_LOGE("Failed to release RKNN outputs.");
            status = WrapperError;
        }
        net_->Release();
        net_.reset();
    }
    output_release_.ResetAfterRuntimeDestroyed();
    return status;
}

std::vector<std::string> InferenceWrapperRKNNAdapter::GetInputNames() {
    return std::vector<std::string>();
}

int32_t InferenceWrapperRKNNAdapter::ResizeInput(const std::vector<InputTensorInfo> &input_tensor_info_list) {
    // The function is not supported
    return WrapperError;
}

#endif  // INFERENCE_WRAPPER_ENABLE_RKNN
