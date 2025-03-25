#ifndef INFERENCE_WRAPPER_MNN_
#define INFERENCE_WRAPPER_MNN_

#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/AutoTime.hpp>
#include "inference_wrapper.h"

class InferenceWrapperMNN : public InferenceWrapper {
public:
    InferenceWrapperMNN();
    ~InferenceWrapperMNN() override;
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

    int32_t ResizeInput(const std::vector<InputTensorInfo>& input_tensor_info_list) override;

    std::vector<std::string> GetInputNames() override;

private:
    std::unique_ptr<MNN::Interpreter> net_;
    MNN::Session* session_;
    std::vector<std::unique_ptr<MNN::Tensor>> out_mat_list_;
    int32_t num_threads_;

    std::vector<std::string> input_names_;
};

#endif
