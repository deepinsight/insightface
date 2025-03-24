#ifndef INFERENCE_WRAPPER_COREML_
#define INFERENCE_WRAPPER_COREML_

#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include "coreml/CoreMLAdapter.h"
#include <MNN/ImageProcess.hpp>
#include "inference_wrapper.h"

class InferenceWrapperCoreML : public InferenceWrapper {
public:
    InferenceWrapperCoreML();
    ~InferenceWrapperCoreML() override;
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
    std::unique_ptr<CoreMLAdapter> net_;
    int32_t num_threads_;

    std::vector<std::string> input_names_;

    /** Using MNN imageprocess to do Image Preprocessing */
    std::unique_ptr<MNN::Tensor> input_tensor_;
};

#endif  // INFERENCE_WRAPPER_COREML_
