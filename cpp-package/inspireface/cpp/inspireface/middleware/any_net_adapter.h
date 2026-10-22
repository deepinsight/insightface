/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#pragma once
#ifndef INSPIREFACE_ANYNETADAPTER_H
#define INSPIREFACE_ANYNETADAPTER_H

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <utility>
#include <inspirecv/inspirecv.h>
#include "data_type.h"
#include "inference_wrapper/inference_wrapper.h"
#include "configurable.h"
#include "log.h"
#include "model_archive/inspire_archive.h"
#include "image_process/nexus_processor/image_processor.h"
#include <launch.h>
#include "system.h"

namespace inspire {

using AnyTensorOutputs = std::vector<std::pair<std::string, std::vector<float>>>;

struct AnyTensorView {
    AnyTensorView(const std::string *tensor_name, const float *tensor_data, size_t tensor_size)
    : name(tensor_name), data(tensor_data), size(tensor_size) {}

    const std::string *name;
    const float *data;
    size_t size;
};

using AnyTensorViews = std::vector<AnyTensorView>;

/**
 * @class AnyNet
 * @brief Generic neural network class for various inference tasks.
 *
 * This class provides a general interface for different types of neural networks,
 * facilitating loading parameters, initializing models, and executing forward passes.
 */
class INSPIRE_API AnyNetAdapter {
    CONFIGURABLE_SUPPORT

public:
    /**
     * @brief Constructor for AnyNet.
     * @param name Name of the neural network.
     */
    explicit AnyNetAdapter(std::string name) : m_name_(std::move(name)) {
        m_processor_ = nexus::ImageProcessor::Create(INSPIREFACE_CONTEXT->GetImageProcessingBackend());
#if defined(ISF_ENABLE_RKNN)
        m_processor_->SetAlignedWidth(INSPIREFACE_CONTEXT->GetImageProcessAlignedWidth());
#endif
    }

    ~AnyNetAdapter() {
        if (m_nn_inference_ != nullptr) {
            m_nn_inference_->Finalize();
        }
    }

    /**
     * @brief Loads parameters and initializes the model for inference.
     * @param param Parameters for network configuration.
     * @param model Pointer to the model.
     * @param type Type of the inference helper (default: INFER_MNN).
     * @return int32_t Status of the loading and initialization process.
     */
    int32_t LoadData(InspireModel &model, InferenceWrapper::EngineType type = InferenceWrapper::INFER_MNN, bool dynamic = false) {
        m_ready_ = false;
        m_infer_type_ = type;
        try {
            // must
            pushData<int>(model.Config(), "model_index", 0);
            pushData<std::string>(model.Config(), "input_layer", "");
            pushData<std::vector<std::string>>(model.Config(), "outputs_layers",
                                               {
                                                 "",
                                               });
            pushData<std::vector<int>>(model.Config(), "input_size", {320, 320});
            pushData<std::vector<float>>(model.Config(), "mean", {127.5f, 127.5f, 127.5f});
            pushData<std::vector<float>>(model.Config(), "norm", {0.0078125f, 0.0078125f, 0.0078125f});
            // rarely
            pushData<int>(model.Config(), "input_channel", 3);
            pushData<int>(model.Config(), "input_image_channel", 3);
            pushData<bool>(model.Config(), "nchw", true);
            pushData<bool>(model.Config(), "swap_color", false);
            pushData<int>(model.Config(), "data_type", InputTensorInfo::InputTensorInfo::DataTypeImage);
            pushData<int>(model.Config(), "input_tensor_type", InputTensorInfo::TensorInfo::TensorTypeFp32);
            pushData<int>(model.Config(), "output_tensor_type", InputTensorInfo::TensorInfo::TensorTypeFp32);
            pushData<int>(model.Config(), "infer_backend", 0);
            pushData<int>(model.Config(), "threads", 1);
        } catch (const std::exception &error) {
            INSPIRE_LOGE("Invalid model configuration for %s: %s", m_name_.c_str(), error.what());
            return InferenceWrapper::WrapperError;
        }

        if (m_nn_inference_ != nullptr) {
            m_nn_inference_->Finalize();
        }
        m_nn_inference_.reset();
        m_input_tensor_info_list_.clear();
        m_output_tensor_info_list_.clear();

        m_nn_inference_.reset(InferenceWrapper::Create(m_infer_type_));
        if (m_nn_inference_ == nullptr) {
            INSPIRE_LOGE("Unsupported inference engine: %d", static_cast<int>(m_infer_type_));
            return InferenceWrapper::WrapperError;
        }
        m_nn_inference_->SetNumThreads(getData<int>("threads"));

        if (m_infer_type_ == InferenceWrapper::INFER_TENSORRT) {
            m_nn_inference_->SetDevice(INSPIREFACE_CONTEXT->GetCudaDeviceId());
        }

#if defined(ISF_GLOBAL_INFERENCE_BACKEND_USE_MNN_CUDA) && !defined(ISF_ENABLE_RKNN)
        INSPIRE_LOGW("You have forced the global use of MNN_CUDA as the neural network inference backend");
        m_nn_inference_->SetSpecialBackend(InferenceWrapper::MMM_CUDA);
#endif

#if defined(ISF_ENABLE_APPLE_EXTENSION)
        if (INSPIREFACE_CONTEXT->GetGlobalCoreMLInferenceMode() == InferenceWrapper::COREML_CPU) {
            m_nn_inference_->SetSpecialBackend(InferenceWrapper::COREML_CPU);
        } else if (INSPIREFACE_CONTEXT->GetGlobalCoreMLInferenceMode() == InferenceWrapper::COREML_GPU) {
            m_nn_inference_->SetSpecialBackend(InferenceWrapper::COREML_GPU);
        } else if (INSPIREFACE_CONTEXT->GetGlobalCoreMLInferenceMode() == InferenceWrapper::COREML_ANE) {
            m_nn_inference_->SetSpecialBackend(InferenceWrapper::COREML_ANE);
        }
#endif

        std::vector<std::string> outputs_layers = getData<std::vector<std::string>>("outputs_layers");
        std::vector<int> input_size = getData<std::vector<int>>("input_size");
        std::vector<float> mean = getData<std::vector<float>>("mean");
        std::vector<float> norm = getData<std::vector<float>>("norm");
        const int data_type = getData<int>("data_type");
        const int channel = getData<int>("input_channel");
        const int image_channel = getData<int>("input_image_channel");
        const int threads = getData<int>("threads");
        bool normalization_valid = mean.size() >= 3 && norm.size() >= 3;
        for (size_t index = 0; normalization_valid && index < 3; ++index) {
            normalization_valid = std::isfinite(mean[index]) && std::isfinite(norm[index]) && norm[index] != 0.0f;
            if (normalization_valid && m_infer_type_ == InferenceWrapper::INFER_MNN) {
                const float converted_mean = mean[index] * 255.0f;
                const float converted_norm = 1.0f / (norm[index] * 255.0f);
                normalization_valid = std::isfinite(converted_mean) && std::isfinite(converted_norm);
            }
        }
        bool shape_valid = input_size.size() >= 2 && input_size[0] > 0 && input_size[1] > 0 && channel > 0;
        if (shape_valid) {
            const int64_t element_count = static_cast<int64_t>(input_size[0]) * input_size[1] * channel;
            shape_valid = element_count > 0 && element_count <= std::numeric_limits<int32_t>::max();
        }
        const bool channel_valid = data_type == InputTensorInfo::DataTypeImage
                                     ? (channel == 1 || channel == 3) && (image_channel == 1 || image_channel == 3) &&
                                         (m_infer_type_ == InferenceWrapper::INFER_MNN || image_channel == channel)
                                     : image_channel > 0;
        if (outputs_layers.empty() || !shape_valid || !channel_valid || !normalization_valid || threads <= 0 ||
            (!model.loadFilePath && (model.buffer == nullptr || model.bufferSize == 0 ||
                                     model.bufferSize > static_cast<size_t>(std::numeric_limits<int>::max())))) {
            INSPIRE_LOGE("Invalid model configuration for %s", m_name_.c_str());
            return InferenceWrapper::WrapperError;
        }
        int tensor_type = getData<int>("input_tensor_type");
        int out_tensor_type = getData<int>("output_tensor_type");
        for (auto &name : outputs_layers) {
            m_output_tensor_info_list_.push_back(OutputTensorInfo(name, out_tensor_type));
        }
        int32_t ret;
        if (model.loadFilePath) {
            auto extensionPath = INSPIREFACE_CONTEXT->GetExtensionPath();
            if (extensionPath.empty()) {
                INSPIRE_LOGE("Extension path is empty");
                return InferenceWrapper::WrapperError;
            }
            std::string filePath = os::PathJoin(extensionPath, model.fullname);
            ret = m_nn_inference_->Initialize(filePath, m_input_tensor_info_list_, m_output_tensor_info_list_);
        } else {
            ret = m_nn_inference_->Initialize(model.buffer, model.bufferSize, m_input_tensor_info_list_, m_output_tensor_info_list_);
        }
        if (ret != InferenceWrapper::WrapperOk) {
            INSPIRE_LOGE("NN Initialize fail");
            return ret;
        }

        InputTensorInfo input_tensor_info(getData<std::string>("input_layer"), tensor_type, getData<bool>("nchw"));
        int width = input_size[0];
        int height = input_size[1];
        m_input_image_size_ = {width, height};
        if (getData<bool>("nchw")) {
            input_tensor_info.tensor_dims = {1, channel, m_input_image_size_.GetHeight(), m_input_image_size_.GetWidth()};
        } else {
            input_tensor_info.tensor_dims = {1, m_input_image_size_.GetHeight(), m_input_image_size_.GetWidth(), channel};
        }

        input_tensor_info.data_type = data_type;
        input_tensor_info.image_info.channel = image_channel;

        input_tensor_info.normalize.mean[0] = mean[0];
        input_tensor_info.normalize.mean[1] = mean[1];
        input_tensor_info.normalize.mean[2] = mean[2];
        input_tensor_info.normalize.norm[0] = norm[0];
        input_tensor_info.normalize.norm[1] = norm[1];
        input_tensor_info.normalize.norm[2] = norm[2];

        input_tensor_info.image_info.width = width;
        input_tensor_info.image_info.height = height;
        input_tensor_info.image_info.channel = image_channel;
        input_tensor_info.image_info.crop_x = 0;
        input_tensor_info.image_info.crop_y = 0;
        input_tensor_info.image_info.crop_width = width;
        input_tensor_info.image_info.crop_height = height;
        input_tensor_info.image_info.is_bgr = getData<bool>("nchw");
        input_tensor_info.image_info.swap_color = getData<bool>("swap_color");

        m_input_tensor_info_list_.push_back(input_tensor_info);

        if (dynamic) {
            ret = m_nn_inference_->ResizeInput(m_input_tensor_info_list_);
            if (ret != InferenceWrapper::WrapperOk) {
                INSPIRE_LOGE("NN ResizeInput fail");
                return ret;
            }
        }

        m_ready_ = true;
        return InferenceWrapper::WrapperOk;
    }

    int32_t Forward(const inspirecv::Image &image, AnyTensorOutputs &outputs) {
        outputs.clear();
        if (!ValidateImageInput(image)) {
            return InferenceWrapper::WrapperError;
        }
        InputTensorInfo &input_tensor_info = getMInputTensorInfoList()[0];
        if (m_infer_type_ == InferenceWrapper::INFER_RKNN) {
            if (getData<bool>("swap_color")) {
                m_cache_ = image.SwapRB();
                input_tensor_info.data = (uint8_t *)m_cache_.Data();
            } else {
                input_tensor_info.data = (uint8_t *)image.Data();
            }
        } else {
            input_tensor_info.data = (uint8_t *)image.Data();
        }
        return Forward(outputs);
    }

    int32_t ForwardViews(const inspirecv::Image &image, AnyTensorViews &outputs) {
        outputs.clear();
        if (!ValidateImageInput(image)) {
            return InferenceWrapper::WrapperError;
        }
        InputTensorInfo &input_tensor_info = getMInputTensorInfoList()[0];
        if (m_infer_type_ == InferenceWrapper::INFER_RKNN && getData<bool>("swap_color")) {
            m_cache_ = image.SwapRB();
            input_tensor_info.data = const_cast<uint8_t *>(m_cache_.Data());
        } else {
            input_tensor_info.data = const_cast<uint8_t *>(image.Data());
        }
        return ForwardViews(outputs);
    }

    /**
     * @brief Performs a forward pass of the network.
     * @param outputs Outputs of the network (tensor outputs).
     */
    int32_t ForwardViews(AnyTensorViews &outputs) {
        outputs.clear();
        if (!m_ready_ || m_nn_inference_ == nullptr || m_input_tensor_info_list_.size() != 1 ||
            m_input_tensor_info_list_.front().data == nullptr || m_output_tensor_info_list_.empty()) {
            INSPIRE_LOGE("%s is not ready for inference", m_name_.c_str());
            return InferenceWrapper::WrapperError;
        }
        if (m_nn_inference_->PreProcess(m_input_tensor_info_list_) != InferenceWrapper::WrapperOk) {
            INSPIRE_LOGE("%s preprocessing failed", m_name_.c_str());
            return InferenceWrapper::WrapperError;
        }
        if (m_nn_inference_->Process(m_output_tensor_info_list_) != InferenceWrapper::WrapperOk) {
            INSPIRE_LOGE("%s inference failed", m_name_.c_str());
            return InferenceWrapper::WrapperError;
        }
        outputs.reserve(m_output_tensor_info_list_.size());
        for (auto &tensor : m_output_tensor_info_list_) {
            const int32_t element_count = tensor.GetElementNum();
            const float *data = tensor.GetDataAsFloat();
            if (data == nullptr || element_count <= 0) {
                outputs.clear();
                INSPIRE_LOGE("%s output tensor '%s' is invalid", m_name_.c_str(), tensor.name.c_str());
                return InferenceWrapper::WrapperError;
            }
            outputs.emplace_back(&tensor.name, data, static_cast<size_t>(element_count));
        }
        return InferenceWrapper::WrapperOk;
    }

    int32_t Forward(AnyTensorOutputs &outputs) {
        AnyTensorViews views;
        const int32_t status = ForwardViews(views);
        outputs.clear();
        if (status != InferenceWrapper::WrapperOk) {
            return status;
        }
        outputs.reserve(views.size());
        for (const auto &view : views) {
            outputs.emplace_back(*view.name, std::vector<float>(view.data, view.data + view.size));
        }
        return InferenceWrapper::WrapperOk;
    }

public:
    /**
     * @brief Gets a reference to the input tensor information list.
     * @return Reference to the vector of input tensor information.
     */
    std::vector<InputTensorInfo> &getMInputTensorInfoList() {
        return m_input_tensor_info_list_;
    }

    /**
     * @brief Gets a reference to the output tensor information list.
     * @return Reference to the vector of output tensor information.
     */
    std::vector<OutputTensorInfo> &getMOutputTensorInfoList() {
        return m_output_tensor_info_list_;
    }

    InferenceCacheStatistics GetInferenceCacheStatistics() const {
        return m_nn_inference_ == nullptr ? InferenceCacheStatistics{} : m_nn_inference_->GetCacheStatistics();
    }

    void ResetInferenceCacheStatistics() {
        if (m_nn_inference_ != nullptr) {
            m_nn_inference_->ResetCacheStatistics();
        }
    }

#ifdef ISF_ENABLE_MNN_CACHE_GUARD
    void SetInferenceCacheEnabledForTesting(bool enabled) {
        if (m_nn_inference_ != nullptr) {
            m_nn_inference_->SetCacheEnabledForTesting(enabled);
        }
    }

    int32_t ResizeInferenceInputForTesting() {
        return m_nn_inference_ == nullptr ? InferenceWrapper::WrapperError
                                         : m_nn_inference_->ResizeInput(m_input_tensor_info_list_);
    }
#endif

    /**
     * @brief Gets the size of the input image.
     * @return Size of the input image.
     */
    inspirecv::Size<int> &getMInputImageSize() {
        return m_input_image_size_;
    }

    /**
     * @brief Softmax function.
     *
     * @param input The input vector.
     * @return The softmax result.
     */
    static std::vector<float> Softmax(const std::vector<float> &input) {
        if (input.empty()) {
            return {};
        }

        float maximum = input.front();
        for (float value : input) {
            if (!std::isfinite(value)) {
                return {};
            }
            maximum = std::max(maximum, value);
        }

        std::vector<float> result(input.size());
        float sum = 0.0f;
        for (size_t index = 0; index < input.size(); ++index) {
            result[index] = std::exp(input[index] - maximum);
            sum += result[index];
        }
        if (!std::isfinite(sum) || sum <= 0.0f) {
            return {};
        }

        for (float &value : result) {
            value /= sum;
        }

        return result;
    }

protected:
    bool ValidateImageInput(const inspirecv::Image &image) const {
        if (!m_ready_ || m_nn_inference_ == nullptr || m_input_tensor_info_list_.size() != 1 || image.Empty() ||
            image.Data() == nullptr) {
            INSPIRE_LOGE("Invalid image input for %s", m_name_.c_str());
            return false;
        }
        const auto &image_info = m_input_tensor_info_list_.front().image_info;
        if (image.Width() != image_info.width || image.Height() != image_info.height ||
            image.Channels() != image_info.channel) {
            INSPIRE_LOGE("Image shape mismatch for %s: got %dx%dx%d, expected %dx%dx%d", m_name_.c_str(),
                         image.Width(), image.Height(), image.Channels(), image_info.width, image_info.height,
                         image_info.channel);
            return false;
        }
        return true;
    }

    int32_t ResizeImageForInference(const inspirecv::Image &source, int target_width, int target_height,
                                    inspirecv::Image &output) {
        output = {};
        if (m_processor_ == nullptr || source.Empty() || target_width <= 0 || target_height <= 0) {
            INSPIRE_LOGE("Invalid resize request for %s", m_name_.c_str());
            return InferenceWrapper::WrapperError;
        }
        if (source.Width() == target_width && source.Height() == target_height) {
            output = inspirecv::Image::Create(source.Width(), source.Height(), source.Channels(), source.Data(), false);
            return InferenceWrapper::WrapperOk;
        }

        uint8_t *resized_data = nullptr;
        const int32_t status = m_processor_->Resize(source.Data(), source.Width(), source.Height(), source.Channels(),
                                                    &resized_data, target_width, target_height);
        if (status != 0 || resized_data == nullptr) {
            INSPIRE_LOGE("Image resize failed for %s: %d", m_name_.c_str(), status);
            return InferenceWrapper::WrapperError;
        }
        output = inspirecv::Image::Create(target_width, target_height, source.Channels(), resized_data, false);
        if (output.Empty()) {
            INSPIRE_LOGE("Image resize returned an invalid buffer for %s", m_name_.c_str());
            return InferenceWrapper::WrapperError;
        }
        return InferenceWrapper::WrapperOk;
    }

    int32_t ResizeAndPadImageForInference(const inspirecv::Image &source, int target_width, int target_height,
                                          inspirecv::Image &output, float &scale) {
        output = {};
        scale = 0.0f;
        if (m_processor_ == nullptr || source.Empty() || target_width <= 0 || target_height <= 0) {
            INSPIRE_LOGE("Invalid resize-and-pad request for %s", m_name_.c_str());
            return InferenceWrapper::WrapperError;
        }
        if (source.Width() == target_width && source.Height() == target_height) {
            output = inspirecv::Image::Create(source.Width(), source.Height(), source.Channels(), source.Data(), false);
            scale = 1.0f;
            return InferenceWrapper::WrapperOk;
        }

        uint8_t *resized_data = nullptr;
        const int32_t status = m_processor_->ResizeAndPadding(
          source.Data(), source.Width(), source.Height(), source.Channels(), target_width, target_height, &resized_data, scale);
        if (status != 0 || resized_data == nullptr || !std::isfinite(scale) || scale <= 0.0f) {
            INSPIRE_LOGE("Image resize-and-pad failed for %s: %d", m_name_.c_str(), status);
            scale = 0.0f;
            return InferenceWrapper::WrapperError;
        }
        output = inspirecv::Image::Create(target_width, target_height, source.Channels(), resized_data, false);
        if (output.Empty()) {
            INSPIRE_LOGE("Image resize-and-pad returned an invalid buffer for %s", m_name_.c_str());
            scale = 0.0f;
            return InferenceWrapper::WrapperError;
        }
        return InferenceWrapper::WrapperOk;
    }

    std::string m_name_;  ///< Name of the neural network.

    std::unique_ptr<nexus::ImageProcessor> m_processor_;  ///< Assign a nexus processor to each anynet object

private:
    InferenceWrapper::EngineType m_infer_type_ = InferenceWrapper::INFER_MNN;  ///< Inference engine type
    bool m_ready_ = false;                                     ///< Whether initialization completed successfully.
    std::shared_ptr<InferenceWrapper> m_nn_inference_;         ///< Shared pointer to the inference helper.
    std::vector<InputTensorInfo> m_input_tensor_info_list_;    ///< List of input tensor information.
    std::vector<OutputTensorInfo> m_output_tensor_info_list_;  ///< List of output tensor information.
    inspirecv::Size<int> m_input_image_size_{};                ///< Size of the input image.
    inspirecv::Image m_cache_;                                 ///< Cached matrix for image data.
};

template <typename ImageT, typename TensorT>
AnyTensorOutputs ForwardService(std::shared_ptr<AnyNetAdapter> net, const ImageT &input, std::function<void(const ImageT &, TensorT &)> transform) {
    if (net == nullptr || net->getMInputTensorInfoList().size() != 1 || !transform) {
        return {};
    }
    InputTensorInfo &input_tensor_info = net->getMInputTensorInfoList()[0];
    TensorT transform_tensor;
    transform(input, transform_tensor);
    input_tensor_info.data = transform_tensor.data;  // input tensor only support cv2::Mat

    AnyTensorOutputs outputs;
    if (net->Forward(outputs) != InferenceWrapper::WrapperOk) {
        outputs.clear();
    }

    return outputs;
}

/**
 * @brief Executes a forward pass through the neural network for a given input, with preprocessing.
 * @tparam ImageT Type of the input image.
 * @tparam TensorT Type of the transformed tensor.
 * @tparam PreprocessCallbackT Type of the preprocessing callback function.
 * @param net Shared pointer to the AnyNet neural network object.
 * @param input The input image to be processed.
 * @param callback Preprocessing callback function to be applied to the input.
 * @param transform Transformation function to convert the input image to a tensor.
 * @return AnyTensorOutputs Outputs of the network (tensor outputs).
 *
 * This template function handles the preprocessing of the input image, transformation to tensor,
 * and then passes it through the neural network to get the output. The function is generic and
 * can work with different types of images and tensors, as specified by the template parameters.
 */
template <typename ImageT, typename TensorT, typename PreprocessCallbackT>
AnyTensorOutputs ForwardService(std::shared_ptr<AnyNetAdapter> net, const ImageT &input, PreprocessCallbackT &callback,
                                std::function<void(const ImageT &, TensorT &, PreprocessCallbackT &)> transform) {
    if (net == nullptr || net->getMInputTensorInfoList().size() != 1 || !transform) {
        return {};
    }
    InputTensorInfo &input_tensor_info = net->getMInputTensorInfoList()[0];
    TensorT transform_tensor;
    transform(input, transform_tensor, callback);
    input_tensor_info.data = transform_tensor.data;  // input tensor only support cv2::Mat

    AnyTensorOutputs outputs;
    if (net->Forward(outputs) != InferenceWrapper::WrapperOk) {
        outputs.clear();
    }

    return outputs;
}

}  // namespace inspire

#endif  // INSPIREFACE_ANYNETADAPTER_H
