/* Copyright 2021 iwatake2222

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/*** Include ***/
/* for general */
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>

/* for MNN */
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/AutoTime.hpp>

/* for My modules */
#include "inference_helper_log.h"
#include "inference_helper_mnn.h"
#include "log.h"
/*** Macro ***/
#define TAG "InferenceHelperMnn"
#define PRINT(...)   INFERENCE_HELPER_LOG_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) INFERENCE_HELPER_LOG_PRINT_E(TAG, __VA_ARGS__)

using namespace inspire;

/*** Function ***/
InferenceHelperMnn::InferenceHelperMnn()
{
    num_threads_ = 1;
}

InferenceHelperMnn::~InferenceHelperMnn()
{
}

int32_t InferenceHelperMnn::SetNumThreads(const int32_t num_threads)
{
    num_threads_ = num_threads;
    return kRetOk;
}

int32_t InferenceHelperMnn::SetCustomOps(const std::vector<std::pair<const char*, const void*>>& custom_ops)
{
    PRINT("[WARNING] This method is not supported\n");
    return kRetOk;
}

int32_t InferenceHelperMnn::ParameterInitialization(std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list) {
    /* Check tensor info fits the info from model */
    for (auto& input_tensor_info : input_tensor_info_list) {
        auto input_tensor = net_->getSessionInput(session_, input_tensor_info.name.c_str());
        if (input_tensor == nullptr) {
            PRINT_E("Invalid input name (%s)\n", input_tensor_info.name.c_str());
//            LOGD("Invalid input name (%s)\n", input_tensor_info.name.c_str());
            return kRetErr;
        }
        if ((input_tensor->getType().code == halide_type_float) && (input_tensor_info.tensor_type == TensorInfo::kTensorTypeFp32)) {
            /* OK */
        } else if ((input_tensor->getType().code == halide_type_uint) && (input_tensor_info.tensor_type == TensorInfo::kTensorTypeUint8)) {
            /* OK */
        } else {
            PRINT_E("Incorrect input tensor type (%d, %d)\n", input_tensor->getType().code, input_tensor_info.tensor_type);
            return kRetErr;
        }
        if ((input_tensor->channel() != -1) && (input_tensor->height() != -1) && (input_tensor->width() != -1)) {
            if (input_tensor_info.GetChannel() != -1) {
                if ((input_tensor->channel() == input_tensor_info.GetChannel()) && (input_tensor->height() == input_tensor_info.GetHeight()) && (input_tensor->width() == input_tensor_info.GetWidth())) {
                    /* OK */
                } else {
                    INSPIRE_LOGW("W: %d != %d", input_tensor->width() , input_tensor_info.GetWidth());
                    INSPIRE_LOGW("H: %d != %d", input_tensor->height() , input_tensor_info.GetHeight());
                    INSPIRE_LOGW("C: %d != %d", input_tensor->channel() , input_tensor_info.GetChannel());
                    INSPIRE_LOGW("There may be some risk of input that is not used by model default");
                    net_->resizeTensor(input_tensor, { 1, input_tensor_info.GetChannel(), input_tensor_info.GetHeight(), input_tensor_info.GetWidth() });
                    net_->resizeSession(session_);
                    return kRetOk;
                }
            } else {
                PRINT("Input tensor size is set from the model\n");
                input_tensor_info.tensor_dims.clear();
                for (int32_t dim = 0; dim < input_tensor->dimensions(); dim++) {
                    input_tensor_info.tensor_dims.push_back(input_tensor->length(dim));
                }
            }
        } else {
            if (input_tensor_info.GetChannel() != -1) {
                PRINT("Input tensor size is resized\n");
                /* In case the input size  is not fixed */
                net_->resizeTensor(input_tensor, { 1, input_tensor_info.GetChannel(), input_tensor_info.GetHeight(), input_tensor_info.GetWidth() });
                net_->resizeSession(session_);
                INSPIRE_LOGE("GO RESIZE");
            } else {
                PRINT_E("Model input size is not set\n");
                return kRetErr;
            }
        }
    }
    for (const auto& output_tensor_info : output_tensor_info_list) {
        auto output_tensor = net_->getSessionOutput(session_, output_tensor_info.name.c_str());
        if (output_tensor == nullptr) {
            PRINT_E("Invalid output name (%s)\n", output_tensor_info.name.c_str());
            return kRetErr;
        }
        /* Output size is set when run inference later */
    }

    /* Convert normalize parameter to speed up */
    for (auto& input_tensor_info : input_tensor_info_list) {
        ConvertNormalizeParameters(input_tensor_info);
    }


    /* Check if tensor info is set */
    for (const auto& input_tensor_info : input_tensor_info_list) {
        for (const auto& dim : input_tensor_info.tensor_dims) {
            if (dim <= 0) {
                PRINT_E("Invalid tensor size\n");
                return kRetErr;
            }
        }
    }
    //for (const auto& output_tensor_info : output_tensor_info_list) {
    //    for (const auto& dim : output_tensor_info.tensor_dims) {
    //        if (dim <= 0) {
    //            PRINT_E("Invalid tensor size\n");
    //            return kRetErr;
    //        }
    //    }
    //}

    return kRetOk;
}

int32_t InferenceHelperMnn::Initialize(char* model_buffer, int model_size, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list) {
//    PRINT("-Initialize\n");
//    LOGD("-Initialize");
    /*** Create network ***/
//    LOG_INFO("init MNN");
//    PRINT_E("init MNN");
    net_.reset(MNN::Interpreter::createFromBuffer(model_buffer, model_size));
    if (!net_) {
        PRINT_E("Failed to load model model buffer\n");
        return kRetErr;
    }
    MNN::ScheduleConfig scheduleConfig;
    scheduleConfig.numThread = num_threads_;    // it seems, setting 1 has better performance on Android
    MNN::BackendConfig bnconfig;
    bnconfig.power = MNN::BackendConfig::Power_High;
    bnconfig.precision = MNN::BackendConfig::Precision_High;
    if (special_backend_ == kMnnCuda) {
        INSPIRE_LOGD("Enable CUDA");
        scheduleConfig.type = MNN_FORWARD_CUDA;
        bnconfig.power = MNN::BackendConfig::Power_Normal;
        bnconfig.precision = MNN::BackendConfig::Precision_Normal;
    } else {
        scheduleConfig.type = MNN_FORWARD_CPU;
    }
    scheduleConfig.backendConfig = &bnconfig;

    session_ = net_->createSession(scheduleConfig);
//    LOG_INFO("-INPUT: {}", net_->getSessionInputAll(session_).size());
//    PRINT("-INPUT: %lu", net_->getSessionInputAll(session_).size());
//    LOGD("-INPUT: %lu", net_->getSessionInputAll(session_).size());
    for (auto &item: net_->getSessionInputAll(session_)) {
        input_names_.push_back(item.first.c_str());
    }
    if (!session_) {
        PRINT_E("Failed to create session\n");
        return kRetErr;
    }

    return ParameterInitialization(input_tensor_info_list, output_tensor_info_list);
}

int32_t InferenceHelperMnn::Initialize(const std::string& model_filename, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list)
{
//    LOG_INFO("init MNN");
    /*** Create network ***/
    net_.reset(MNN::Interpreter::createFromFile(model_filename.c_str()));
    if (!net_) {
        PRINT_E("Failed to load model file (%s)\n", model_filename.c_str());
        return kRetErr;
    }

    MNN::ScheduleConfig scheduleConfig;
    scheduleConfig.type = MNN_FORWARD_CPU;
    scheduleConfig.numThread = num_threads_;    // it seems, setting 1 has better performance on Android
    // MNN::BackendConfig bnconfig;
    // bnconfig.power = MNN::BackendConfig::Power_High;
    // bnconfig.precision = MNN::BackendConfig::Precision_Low;
    // scheduleConfig.backendConfig = &bnconfig;
    session_ = net_->createSession(scheduleConfig);
    if (!session_) {
        PRINT_E("Failed to create session\n");
        return kRetErr;
    }

    return ParameterInitialization(input_tensor_info_list, output_tensor_info_list);

};


int32_t InferenceHelperMnn::Finalize(void)
{
    net_->releaseSession(session_);
    net_->releaseModel();
    net_.reset();
    out_mat_list_.clear();
    return kRetOk;
}

int32_t InferenceHelperMnn::PreProcess(const std::vector<InputTensorInfo>& input_tensor_info_list)
{
//    for (auto &item: net_->getSessionInputAll(session_)) {
//        PRINT("sss: %s", item.first.c_str());
//    }
    for (const auto& input_tensor_info : input_tensor_info_list) {
        auto input_tensor = net_->getSessionInput(session_, input_tensor_info.name.c_str());
        if (input_tensor == nullptr) {
            PRINT_E("Invalid input name (%s)\n", input_tensor_info.name.c_str());
            INSPIRE_LOGE("Invalid input name (%s)\n", input_tensor_info.name.c_str());
            return kRetErr;
        }
        if (input_tensor_info.data_type == InputTensorInfo::kDataTypeImage) {
            /* Crop */
            if ((input_tensor_info.image_info.width != input_tensor_info.image_info.crop_width) || (input_tensor_info.image_info.height != input_tensor_info.image_info.crop_height)) {
                PRINT_E("Crop is not supported\n");
                return kRetErr;
            }

            MNN::CV::ImageProcess::Config image_processconfig;
            /* Convert color type */
//            LOGD("input_tensor_info.image_info.channel: %d", input_tensor_info.image_info.channel);
//            LOGD("input_tensor_info.GetChannel(): %d", input_tensor_info.GetChannel());

            // !!!!!! BUG !!!!!!!!!
            // When initializing, setting the image channel to 3 and the tensor channel to 1, 
            // and configuring the processing to convert the color image to grayscale may cause some bugs. 
            // For example, the image channel might automatically change to 1. 
            // This issue has not been fully investigated, 
            // so it's necessary to manually convert the image to grayscale before input.
            // !!!!!! BUG !!!!!!!!!

            if ((input_tensor_info.image_info.channel == 3) && (input_tensor_info.GetChannel() == 3)) {
                image_processconfig.sourceFormat = (input_tensor_info.image_info.is_bgr) ? MNN::CV::BGR : MNN::CV::RGB;
                if (input_tensor_info.image_info.swap_color) {
                    image_processconfig.destFormat = (input_tensor_info.image_info.is_bgr) ? MNN::CV::RGB : MNN::CV::BGR;
                } else {
                    image_processconfig.destFormat = (input_tensor_info.image_info.is_bgr) ? MNN::CV::BGR : MNN::CV::RGB;
                }
            } else if ((input_tensor_info.image_info.channel == 1) && (input_tensor_info.GetChannel() == 1)) {
                image_processconfig.sourceFormat = MNN::CV::GRAY;
                image_processconfig.destFormat = MNN::CV::GRAY;
            } else if ((input_tensor_info.image_info.channel == 3) && (input_tensor_info.GetChannel() == 1)) {
                image_processconfig.sourceFormat = (input_tensor_info.image_info.is_bgr) ? MNN::CV::BGR : MNN::CV::RGB;
                image_processconfig.destFormat = MNN::CV::GRAY;
//                LOGD("2gray");
            } else if ((input_tensor_info.image_info.channel == 1) && (input_tensor_info.GetChannel() == 3)) {
                image_processconfig.sourceFormat = MNN::CV::GRAY;
                image_processconfig.destFormat = MNN::CV::BGR;
            } else {
                PRINT_E("Unsupported color conversion (%d, %d)\n", input_tensor_info.image_info.channel, input_tensor_info.GetChannel());
                return kRetErr;
            }

            /* Normalize image */
            std::memcpy(image_processconfig.mean, input_tensor_info.normalize.mean, sizeof(image_processconfig.mean));
            std::memcpy(image_processconfig.normal, input_tensor_info.normalize.norm, sizeof(image_processconfig.normal));
            
            /* Resize image */
            image_processconfig.filterType = MNN::CV::BILINEAR;
            MNN::CV::Matrix trans;
            trans.setScale(static_cast<float>(input_tensor_info.image_info.crop_width) / input_tensor_info.GetWidth(), static_cast<float>(input_tensor_info.image_info.crop_height) / input_tensor_info.GetHeight());

            /* Do pre-process */
            std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(image_processconfig));
            pretreat->setMatrix(trans);
//            LOGD("k1");
            pretreat->convert(static_cast<uint8_t*>(input_tensor_info.data), input_tensor_info.image_info.crop_width, input_tensor_info.image_info.crop_height, 0, input_tensor);

        } else if ( (input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNhwc) || (input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNchw) ) {
            std::unique_ptr<MNN::Tensor> tensor;
            if (input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNhwc) {
                tensor.reset(new MNN::Tensor(input_tensor, MNN::Tensor::TENSORFLOW));
            } else {
                tensor.reset(new MNN::Tensor(input_tensor, MNN::Tensor::CAFFE));
            }
            if (tensor->getType().code == halide_type_float) {
                for (int32_t i = 0; i < input_tensor_info.GetWidth() * input_tensor_info.GetHeight() * input_tensor_info.GetChannel(); i++) {
                    tensor->host<float>()[i] = static_cast<float*>(input_tensor_info.data)[i];
                }
            } else {
                for (int32_t i = 0; i < input_tensor_info.GetWidth() * input_tensor_info.GetHeight() * input_tensor_info.GetChannel(); i++) {
                    tensor->host<uint8_t>()[i] = static_cast<uint8_t*>(input_tensor_info.data)[i];
                }
            }
            input_tensor->copyFromHostTensor(tensor.get());
        } else {
            PRINT_E("Unsupported data type (%d)\n", input_tensor_info.data_type);
            return kRetErr;
        }
    }
    return kRetOk;
}

int32_t InferenceHelperMnn::Process(std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    net_->runSession(session_);

    out_mat_list_.clear();
    for (auto& output_tensor_info : output_tensor_info_list) {
        auto output_tensor = net_->getSessionOutput(session_, output_tensor_info.name.c_str());
        if (output_tensor == nullptr) {
            PRINT_E("Invalid output name (%s)\n", output_tensor_info.name.c_str());
            return kRetErr;
        }

        auto dimType = output_tensor->getDimensionType();
        std::unique_ptr<MNN::Tensor> outputUser(new MNN::Tensor(output_tensor, dimType));
        output_tensor->copyToHostTensor(outputUser.get());
        auto type = outputUser->getType();
        if (type.code == halide_type_float) {
            output_tensor_info.tensor_type = TensorInfo::kTensorTypeFp32;
            output_tensor_info.data = outputUser->host<float>();
        } else if (type.code == halide_type_uint && type.bytes() == 1) {
            output_tensor_info.tensor_type = TensorInfo::kTensorTypeUint8;
            output_tensor_info.data = outputUser->host<uint8_t>();
        } else {
            PRINT_E("Unexpected data type\n");
            return kRetErr;
        }
        
        output_tensor_info.tensor_dims.clear();
        for (int32_t dim = 0; dim < outputUser->dimensions(); dim++) {
            output_tensor_info.tensor_dims.push_back(outputUser->length(dim));
        }

        out_mat_list_.push_back(std::move(outputUser));	// store data in member variable so that data keep exist
    }

    return kRetOk;
}

std::vector<std::string> InferenceHelperMnn::GetInputNames() {
    return input_names_;
}

int32_t InferenceHelperMnn::ResizeInput(const std::vector<InputTensorInfo>& input_tensor_info_list) {
    for (const auto& input_tensor_info : input_tensor_info_list) {
        auto input_tensor = net_->getSessionInput(session_, input_tensor_info.name.c_str());
        net_->resizeTensor(input_tensor, { 1, input_tensor_info.GetChannel(), input_tensor_info.GetHeight(), input_tensor_info.GetWidth() });
        net_->resizeSession(session_);
    }
    return 0;
}
