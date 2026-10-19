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

    InferenceCacheStatistics GetCacheStatistics() const override;
    void ResetCacheStatistics() override;
#ifdef ISF_ENABLE_MNN_CACHE_GUARD
    void SetCacheEnabledForTesting(bool enabled) override;
#endif

private:
    struct TensorSignature {
        bool Matches(const MNN::Tensor* candidate) const {
            if (candidate == nullptr || tensor != candidate || dimension_type != candidate->getDimensionType()) {
                return false;
            }
            const auto candidate_type = candidate->getType();
            if (type.code != candidate_type.code || type.bits != candidate_type.bits || type.lanes != candidate_type.lanes ||
                dimensions.size() != static_cast<size_t>(candidate->dimensions())) {
                return false;
            }
            for (size_t index = 0; index < dimensions.size(); ++index) {
                if (dimensions[index] != candidate->length(static_cast<int>(index))) {
                    return false;
                }
            }
            return true;
        }

        void Capture(MNN::Tensor* candidate) {
            tensor = candidate;
            dimension_type = candidate->getDimensionType();
            type = candidate->getType();
            dimensions.clear();
            dimensions.reserve(static_cast<size_t>(candidate->dimensions()));
            for (int index = 0; index < candidate->dimensions(); ++index) {
                dimensions.push_back(candidate->length(index));
            }
        }

        MNN::Tensor* tensor = nullptr;
        MNN::Tensor::DimensionType dimension_type = MNN::Tensor::CAFFE;
        halide_type_t type{};
        std::vector<int> dimensions;
    };

    struct ImageProcessSignature {
        bool Matches(const InputTensorInfo& input, int source, int destination) const {
            if (!valid || source_format != source || destination_format != destination || width != input.image_info.width ||
                height != input.image_info.height || channel != input.image_info.channel || crop_x != input.image_info.crop_x ||
                crop_y != input.image_info.crop_y || crop_width != input.image_info.crop_width ||
                crop_height != input.image_info.crop_height || target_width != input.GetWidth() ||
                target_height != input.GetHeight() || target_channel != input.GetChannel()) {
                return false;
            }
            for (size_t index = 0; index < mean.size(); ++index) {
                if (mean[index] != input.normalize.mean[index] || normal[index] != input.normalize.norm[index]) {
                    return false;
                }
            }
            return true;
        }

        void Capture(const InputTensorInfo& input, int source, int destination) {
            valid = true;
            source_format = source;
            destination_format = destination;
            width = input.image_info.width;
            height = input.image_info.height;
            channel = input.image_info.channel;
            crop_x = input.image_info.crop_x;
            crop_y = input.image_info.crop_y;
            crop_width = input.image_info.crop_width;
            crop_height = input.image_info.crop_height;
            target_width = input.GetWidth();
            target_height = input.GetHeight();
            target_channel = input.GetChannel();
            for (size_t index = 0; index < mean.size(); ++index) {
                mean[index] = input.normalize.mean[index];
                normal[index] = input.normalize.norm[index];
            }
        }

        bool valid = false;
        int source_format = 0;
        int destination_format = 0;
        int width = 0;
        int height = 0;
        int channel = 0;
        int crop_x = 0;
        int crop_y = 0;
        int crop_width = 0;
        int crop_height = 0;
        int target_width = 0;
        int target_height = 0;
        int target_channel = 0;
        std::array<float, 3> mean{};
        std::array<float, 3> normal{};
    };

    struct InputCacheEntry {
        std::string name;
        MNN::Tensor* input_tensor = nullptr;
        ImageProcessSignature image_signature;
        std::unique_ptr<MNN::CV::ImageProcess> image_process;
        TensorSignature blob_signature;
        MNN::Tensor::DimensionType blob_dimension_type = MNN::Tensor::CAFFE;
        std::unique_ptr<MNN::Tensor> blob_host_tensor;
    };

    struct OutputCacheEntry {
        std::string name;
        MNN::Tensor* output_tensor = nullptr;
        TensorSignature signature;
        std::unique_ptr<MNN::Tensor> host_tensor;
    };

    void ClearRuntimeCaches();
#ifdef ISF_ENABLE_MNN_CACHE_GUARD
    int32_t PreProcessWithoutCache(const std::vector<InputTensorInfo>& input_tensor_info_list);
    int32_t ProcessWithoutCache(std::vector<OutputTensorInfo>& output_tensor_info_list);
#endif

    std::unique_ptr<MNN::Interpreter> net_;
    MNN::Session* session_;
    std::vector<InputCacheEntry> input_cache_;
    std::vector<OutputCacheEntry> output_cache_;
#ifdef ISF_ENABLE_MNN_CACHE_GUARD
    bool cache_enabled_for_testing_ = true;
    std::vector<std::unique_ptr<MNN::Tensor>> uncached_output_tensors_;
#endif
    int32_t num_threads_;

    std::vector<std::string> input_names_;
    InferenceCacheStatistics cache_statistics_;
};

#endif
