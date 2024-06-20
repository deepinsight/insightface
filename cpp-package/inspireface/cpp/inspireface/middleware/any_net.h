//
// Created by tunm on 2023/5/6.
//
#pragma once
#ifndef BIGGUYSMAIN_ANYNET_H
#define BIGGUYSMAIN_ANYNET_H

#include <utility>

#include "../data_type.h"
#include "inference_helper/inference_helper.h"
#include "configurable.h"
#include "opencv2/opencv.hpp"
#include "../log.h"
#include "model_archive/inspire_archive.h"

namespace inspire {

using AnyTensorOutputs = std::vector<std::pair<std::string, std::vector<float>>>;

/**
* @class AnyNet
* @brief Generic neural network class for various inference tasks.
*
* This class provides a general interface for different types of neural networks,
* facilitating loading parameters, initializing models, and executing forward passes.
*/
class INSPIRE_API AnyNet {
CONFIGURABLE_SUPPORT

public:
    /**
     * @brief Constructor for AnyNet.
     * @param name Name of the neural network.
     */
    explicit AnyNet(std::string name) : m_name_(std::move(name)) {}

    ~AnyNet() {
        m_nn_inference_->Finalize();
    }

    /**
    * @brief Loads parameters and initializes the model for inference.
    * @param param Parameters for network configuration.
    * @param model Pointer to the model.
    * @param type Type of the inference helper (default: kMnn).
    * @return int32_t Status of the loading and initialization process.
    */
    int32_t
    loadData(InspireModel &model, InferenceHelper::HelperType type = InferenceHelper::kMnn, bool dynamic = false) {
        m_infer_type_ = type;
        // must
        pushData<int>(model.Config(), "model_index", 0);
        pushData<std::string>(model.Config(), "input_layer", "");
        pushData<std::vector<std::string>>(model.Config(), "outputs_layers", {"",});
        pushData<std::vector<int>>(model.Config(), "input_size", {320, 320});
        pushData<std::vector<float>>(model.Config(), "mean", {127.5f, 127.5f, 127.5f});
        pushData<std::vector<float>>(model.Config(), "norm", {0.0078125f, 0.0078125f, 0.0078125f});
        // rarely
        pushData<int>(model.Config(), "input_channel", 3);
        pushData<int>(model.Config(), "input_image_channel", 3);
        pushData<bool>(model.Config(), "nchw", true);
        pushData<bool>(model.Config(), "swap_color", false);
        pushData<int>(model.Config(), "data_type", InputTensorInfo::InputTensorInfo::kDataTypeImage);
        pushData<int>(model.Config(), "input_tensor_type", InputTensorInfo::TensorInfo::kTensorTypeFp32);
        pushData<int>(model.Config(), "output_tensor_type", InputTensorInfo::TensorInfo::kTensorTypeFp32);
        pushData<int>(model.Config(), "threads", 1);

        m_nn_inference_.reset(InferenceHelper::Create(m_infer_type_));
        m_nn_inference_->SetNumThreads(getData<int>("threads"));
#if defined(ISF_GLOBAL_INFERENCE_BACKEND_USE_MNN_CUDA) && !defined(ISF_ENABLE_RKNN)
        INSPIRE_LOGW("You have forced the global use of MNN_CUDA as the neural network inference backend");
        m_nn_inference_->SetSpecialBackend(InferenceHelper::kMnnCuda);
#endif
        m_output_tensor_info_list_.clear();
        std::vector<std::string> outputs_layers = getData<std::vector<std::string>>("outputs_layers");
        int tensor_type = getData<int>("input_tensor_type");
        int out_tensor_type = getData<int>("output_tensor_type");
        for (auto &name: outputs_layers) {
            m_output_tensor_info_list_.push_back(OutputTensorInfo(name, out_tensor_type));
        }
        auto ret = m_nn_inference_->Initialize(model.buffer, model.bufferSize, m_input_tensor_info_list_,
                                               m_output_tensor_info_list_);
        if (ret != InferenceHelper::kRetOk) {
            INSPIRE_LOGE("NN Initialize fail");
            return ret;
        }

        m_input_tensor_info_list_.clear();
        InputTensorInfo input_tensor_info(getData<std::string>("input_layer"), tensor_type, getData<bool>("nchw"));
        std::vector<int> input_size = getData<std::vector<int>>("input_size");
        int width = input_size[0];
        int height = input_size[1];
        m_input_image_size_ = {width, height};
        int channel = getData<int>("input_channel");
        if (getData<bool>("nchw")) {
            input_tensor_info.tensor_dims = {1, channel, m_input_image_size_.height, m_input_image_size_.width};
        } else {
            input_tensor_info.tensor_dims = {1, m_input_image_size_.height, m_input_image_size_.width, channel};
        }

        input_tensor_info.data_type = getData<int>("data_type");
        int image_channel = getData<int>("input_image_channel");
        input_tensor_info.image_info.channel = image_channel;

        std::vector<float> mean = getData<std::vector<float>>("mean");
        std::vector<float> norm = getData<std::vector<float>>("norm");
        input_tensor_info.normalize.mean[0] = mean[0];
        input_tensor_info.normalize.mean[1] = mean[1];
        input_tensor_info.normalize.mean[2] = mean[2];
        input_tensor_info.normalize.norm[0] = norm[0];
        input_tensor_info.normalize.norm[1] = norm[1];
        input_tensor_info.normalize.norm[2] = norm[2];

        input_tensor_info.image_info.width = width;
        input_tensor_info.image_info.height = height;
        input_tensor_info.image_info.channel = channel;
        input_tensor_info.image_info.crop_x = 0;
        input_tensor_info.image_info.crop_y = 0;
        input_tensor_info.image_info.crop_width = width;
        input_tensor_info.image_info.crop_height = height;
        input_tensor_info.image_info.is_bgr = getData<bool>("nchw");
        input_tensor_info.image_info.swap_color = getData<bool>("swap_color");

        m_input_tensor_info_list_.push_back(input_tensor_info);

        if (dynamic) {
            m_nn_inference_->ResizeInput(m_input_tensor_info_list_);
        }



        return 0;
    }

    /**
     * @brief Performs a forward pass of the network with given input data.
     * @param data The input matrix (image) to process.
     * @param outputs Outputs of the network (tensor outputs).
     */
    void Forward(const Matrix &data, AnyTensorOutputs &outputs) {
        InputTensorInfo &input_tensor_info = getMInputTensorInfoList()[0];
        if (m_infer_type_ == InferenceHelper::kRknn) {
            // Start by simply implementing a temporary color shift on the outside
            if (getData<bool>("swap_color")) {
                cv::cvtColor(data, m_cache_, cv::COLOR_BGR2RGB);
                input_tensor_info.data = m_cache_.data;
            } else {
                input_tensor_info.data = data.data;
            }
        } else {
            input_tensor_info.data = data.data;
        }
        Forward(outputs);
    }

    /**
     * @brief Performs a forward pass of the network.
     * @param outputs Outputs of the network (tensor outputs).
     */
    void Forward(AnyTensorOutputs &outputs) {

//        LOGD("ppPreProcess");
        if (m_nn_inference_->PreProcess(m_input_tensor_info_list_) != InferenceHelper::kRetOk) {
            INSPIRE_LOGD("PreProcess error");
        }
//        LOGD("PreProcess");
        if (m_nn_inference_->Process(m_output_tensor_info_list_) != InferenceHelper::kRetOk) {
            INSPIRE_LOGD("Process error");
        }
//        LOGD("Process");
        for (int i = 0; i < m_output_tensor_info_list_.size(); ++i) {
            std::vector<float> output_score_raw_list(m_output_tensor_info_list_[i].GetDataAsFloat(),
                                                     m_output_tensor_info_list_[i].GetDataAsFloat() +
                                                     m_output_tensor_info_list_[i].GetElementNum());
//            LOGE("m_output_tensor_info_list_[i].GetElementNum(): %d",m_output_tensor_info_list_[i].GetElementNum());
            outputs.push_back(std::make_pair(m_output_tensor_info_list_[i].name, output_score_raw_list));
        }

        m_cache_.release();
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

    /**
     * @brief Gets the size of the input image.
     * @return Size of the input image.
     */
    cv::Size &getMInputImageSize() {
        return m_input_image_size_;
    }

protected:
    std::string m_name_;                                                ///< Name of the neural network.

private:
    InferenceHelper::HelperType m_infer_type_;                          ///< Inference engine type
    std::shared_ptr<InferenceHelper> m_nn_inference_;                   ///< Shared pointer to the inference helper.
    std::vector<InputTensorInfo> m_input_tensor_info_list_;             ///< List of input tensor information.
    std::vector<OutputTensorInfo> m_output_tensor_info_list_;           ///< List of output tensor information.
    cv::Size m_input_image_size_{};                                     ///< Size of the input image.
    cv::Mat m_cache_;                                                   ///< Cached matrix for image data.

};

template<typename ImageT, typename TensorT>
AnyTensorOutputs ForwardService(
        std::shared_ptr<AnyNet> net,
        const ImageT &input,
        std::function<void(const ImageT &, TensorT &)> transform) {
    InputTensorInfo &input_tensor_info = net->getMInputTensorInfoList()[0];
    TensorT transform_tensor;
    transform(input, transform_tensor);
    input_tensor_info.data = transform_tensor.data;      // input tensor only support cv2::Mat

    AnyTensorOutputs outputs;
    net->Forward(outputs);

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
template<typename ImageT, typename TensorT, typename PreprocessCallbackT>
AnyTensorOutputs ForwardService(
        std::shared_ptr<AnyNet> net,
        const ImageT &input,
        PreprocessCallbackT &callback,
        std::function<void(const ImageT &, TensorT &, PreprocessCallbackT &)> transform) {
    InputTensorInfo &input_tensor_info = net->getMInputTensorInfoList()[0];
    TensorT transform_tensor;
    transform(input, transform_tensor, callback);
    input_tensor_info.data = transform_tensor.data;      // input tensor only support cv2::Mat

    AnyTensorOutputs outputs;
    net->Forward(outputs);

    return outputs;
}


} // namespace

#endif //BIGGUYSMAIN_ANYNET_H
