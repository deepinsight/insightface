//
// Created by tunm on 2023/2/5.
//

#ifndef RKNPU_PTOCR_INFERENCE_HELPER_RKNN_H
#define RKNPU_PTOCR_INFERENCE_HELPER_RKNN_H
#ifdef INFERENCE_HELPER_ENABLE_RKNN

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <memory>

/* for RKNN */
#include <rknn_api.h>

/* for My modules */
#include "inference_helper.h"

class InferenceHelperRKNN: public InferenceHelper {
public:
    InferenceHelperRKNN();
    ~InferenceHelperRKNN() override;
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
    rknn_context net_;
    int32_t num_threads_;
    std::vector<rknn_output> output_tensors_;
    rknn_input_output_num rk_io_num_;
    std::vector<std::string> input_names_;

private:


};

#endif // INFERENCE_HELPER_ENABLE_RKNN
#endif //RKNPU_PTOCR_INFERENCE_HELPER_RKNN_H
