//
// Created by tunm on 2023/9/24.
//

#ifndef HYPERFACEREPO_INFERENCE_HELPER_RKNN_ADAPTER_H
#define HYPERFACEREPO_INFERENCE_HELPER_RKNN_ADAPTER_H

#ifdef INFERENCE_HELPER_ENABLE_RKNN

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <memory>
/* for My modules */
#include "inference_helper.h"

#include "customized/rknn_adapter.h"

class InferenceHelperRknnAdapter: public InferenceHelper {
public:
    InferenceHelperRknnAdapter();
    ~InferenceHelperRknnAdapter() override;
    int32_t SetNumThreads(const int32_t num_threads) override;
    int32_t SetCustomOps(const std::vector<std::pair<const char*, const void*>>& custom_ops) override;
    int32_t Initialize(const std::string& model_filename, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list) override;
    int32_t Initialize(char* model_buffer, int model_size, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list) override;
    int32_t Finalize(void) override;
    int32_t PreProcess(const std::vector<InputTensorInfo>& input_tensor_info_list) override;
    int32_t Process(std::vector<OutputTensorInfo>& output_tensor_info_list) override;
    int32_t ParameterInitialization(std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list) override;
    std::vector<std::string> GetInputNames() override;

    int32_t ResizeInput(const std::vector<InputTensorInfo>& input_tensor_info_list) override;

private:
    std::shared_ptr<RKNNAdapter> net_;
    int32_t num_threads_;

};

#endif  // INFERENCE_HELPER_ENABLE_RKNN

#endif //HYPERFACEREPO_INFERENCE_HELPER_RKNN_ADAPTER_H
