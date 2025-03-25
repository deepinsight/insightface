/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#ifndef INSPIREFACE_INFERENCE_WRAPPER_RKNN_ADAPTER_NANO_H
#define INSPIREFACE_INFERENCE_WRAPPER_RKNN_ADAPTER_NANO_H

#ifdef INFERENCE_WRAPPER_ENABLE_RKNN2

#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include "inference_wrapper.h"
#include "customized/rknn_adapter_nano.h"

class InferenceWrapperRKNNAdapter : public InferenceWrapper {
public:
    InferenceWrapperRKNNAdapter();
    ~InferenceWrapperRKNNAdapter() override;
    int32_t SetNumThreads(const int32_t num_threads) override;
    int32_t Initialize(const std::string& model_filename, std::vector<InputTensorInfo>& input_tensor_info_list,
                       std::vector<OutputTensorInfo>& output_tensor_info_list) override;
    int32_t Initialize(char* model_buffer, int model_size, std::vector<InputTensorInfo>& input_tensor_info_list,
                       std::vector<OutputTensorInfo>& output_tensor_info_list) override;
    int32_t Finalize(void) override;
    int32_t PreProcess(const std::vector<InputTensorInfo>& input_tensor_info_list) override;
    int32_t Process(std::vector<OutputTensorInfo>& output_tensor_info_list) override;
    int32_t ParameterInitialization(std::vector<InputTensorInfo>& input_tensor_info_list,
                                    std::vector<OutputTensorInfo>& output_tensor_info_list) override;
    std::vector<std::string> GetInputNames() override;

    int32_t ResizeInput(const std::vector<InputTensorInfo>& input_tensor_info_list) override;

private:
    std::shared_ptr<RKNNAdapterNano> net_;
    int32_t num_threads_;
};

#endif  // INFERENCE_WRAPPER_ENABLE_RKNN2

#endif  // INSPIREFACE_INFERENCE_WRAPPER_RKNN_ADAPTER_NANO_H
