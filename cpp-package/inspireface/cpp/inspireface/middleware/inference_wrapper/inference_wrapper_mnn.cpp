#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/AutoTime.hpp>
#include "inference_wrapper_log.h"
#include "inference_wrapper_mnn.h"
#include "log.h"
#include "launch.h"
#define TAG "InferenceWrapperMNN"
#define PRINT(...) INFERENCE_WRAPPER_LOG_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) INFERENCE_WRAPPER_LOG_PRINT_E(TAG, __VA_ARGS__)

using namespace inspire;

namespace {

MNN::BackendConfig::PowerMode GetConfiguredCPUEnginePowerMode() {
    switch (INSPIREFACE_CONTEXT->GetGlobalCPUEnginePowerMode()) {
        case Launch::CPU_ENGINE_POWER_HIGH:
            return MNN::BackendConfig::Power_High;
        case Launch::CPU_ENGINE_POWER_LOW:
            return MNN::BackendConfig::Power_Low;
        case Launch::CPU_ENGINE_POWER_NORMAL:
        default:
            return MNN::BackendConfig::Power_Normal;
    }
}

}  // namespace

InferenceWrapperMNN::InferenceWrapperMNN() {
    num_threads_ = 1;
    session_ = nullptr;
}

InferenceWrapperMNN::~InferenceWrapperMNN() {}

void InferenceWrapperMNN::ClearRuntimeCaches() {
    input_cache_.clear();
    output_cache_.clear();
#ifdef ISF_ENABLE_MNN_CACHE_GUARD
    uncached_output_tensors_.clear();
#endif
}

#ifdef ISF_ENABLE_MNN_CACHE_GUARD
void InferenceWrapperMNN::SetCacheEnabledForTesting(bool enabled) {
    if (cache_enabled_for_testing_ != enabled) {
        cache_enabled_for_testing_ = enabled;
        ClearRuntimeCaches();
    }
}
#endif

int32_t InferenceWrapperMNN::SetNumThreads(const int32_t num_threads) {
    num_threads_ = num_threads;
    return WrapperOk;
}

int32_t InferenceWrapperMNN::ParameterInitialization(std::vector<InputTensorInfo>& input_tensor_info_list,
                                                     std::vector<OutputTensorInfo>& output_tensor_info_list) {
    /* Check tensor info fits the info from model */
    for (auto& input_tensor_info : input_tensor_info_list) {
        auto input_tensor = net_->getSessionInput(session_, input_tensor_info.name.c_str());
        if (input_tensor == nullptr) {
            PRINT_E("Invalid input name (%s)\n", input_tensor_info.name.c_str());
            //            LOGD("Invalid input name (%s)\n", input_tensor_info.name.c_str());
            return WrapperError;
        }
        if ((input_tensor->getType().code == halide_type_float) && (input_tensor_info.tensor_type == TensorInfo::TensorTypeFp32)) {
            /* OK */
        } else if ((input_tensor->getType().code == halide_type_uint) && (input_tensor_info.tensor_type == TensorInfo::TensorTypeUint8)) {
            /* OK */
        } else {
            PRINT_E("Incorrect input tensor type (%d, %d)\n", input_tensor->getType().code, input_tensor_info.tensor_type);
            return WrapperError;
        }
        if ((input_tensor->channel() != -1) && (input_tensor->height() != -1) && (input_tensor->width() != -1)) {
            if (input_tensor_info.GetChannel() != -1) {
                if ((input_tensor->channel() == input_tensor_info.GetChannel()) && (input_tensor->height() == input_tensor_info.GetHeight()) &&
                    (input_tensor->width() == input_tensor_info.GetWidth())) {
                    /* OK */
                } else {
                    INSPIRE_LOGW("W: %d != %d", input_tensor->width(), input_tensor_info.GetWidth());
                    INSPIRE_LOGW("H: %d != %d", input_tensor->height(), input_tensor_info.GetHeight());
                    INSPIRE_LOGW("C: %d != %d", input_tensor->channel(), input_tensor_info.GetChannel());
                    INSPIRE_LOGW("There may be some risk of input that is not used by model default");
                    net_->resizeTensor(input_tensor,
                                       {1, input_tensor_info.GetChannel(), input_tensor_info.GetHeight(), input_tensor_info.GetWidth()});
                    net_->resizeSession(session_);
                    ClearRuntimeCaches();
                    return WrapperOk;
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
                net_->resizeTensor(input_tensor, {1, input_tensor_info.GetChannel(), input_tensor_info.GetHeight(), input_tensor_info.GetWidth()});
                net_->resizeSession(session_);
                ClearRuntimeCaches();
                INSPIRE_LOGE("GO RESIZE");
            } else {
                PRINT_E("Model input size is not set\n");
                return WrapperError;
            }
        }
    }
    for (const auto& output_tensor_info : output_tensor_info_list) {
        auto output_tensor = net_->getSessionOutput(session_, output_tensor_info.name.c_str());
        if (output_tensor == nullptr) {
            PRINT_E("Invalid output name (%s)\n", output_tensor_info.name.c_str());
            return WrapperError;
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
                return WrapperError;
            }
        }
    }

    return WrapperOk;
}

int32_t InferenceWrapperMNN::Initialize(char* model_buffer, int model_size, std::vector<InputTensorInfo>& input_tensor_info_list,
                                        std::vector<OutputTensorInfo>& output_tensor_info_list) {
    ClearRuntimeCaches();
    ResetCacheStatistics();
    input_names_.clear();
    net_.reset(MNN::Interpreter::createFromBuffer(model_buffer, model_size));
    if (!net_) {
        PRINT_E("Failed to load model model buffer\n");
        return WrapperError;
    }
    MNN::ScheduleConfig scheduleConfig;
    scheduleConfig.numThread = num_threads_;  // it seems, setting 1 has better performance on Android
    MNN::BackendConfig bnconfig;
    bnconfig.power = GetConfiguredCPUEnginePowerMode();
    bnconfig.precision = MNN::BackendConfig::Precision_Normal;
    if (special_backend_ == MMM_CUDA) {
        INSPIRE_LOGD("Enable CUDA");
        scheduleConfig.type = MNN_FORWARD_CUDA;
        bnconfig.power = MNN::BackendConfig::Power_Normal;
        bnconfig.precision = MNN::BackendConfig::Precision_Normal;
    } else {
        scheduleConfig.type = MNN_FORWARD_CPU;
    }
    scheduleConfig.backendConfig = &bnconfig;

    session_ = net_->createSession(scheduleConfig);
    if (!session_) {
        PRINT_E("Failed to create session\n");
        return WrapperError;
    }
    for (auto& item : net_->getSessionInputAll(session_)) {
        input_names_.push_back(item.first.c_str());
    }

    return ParameterInitialization(input_tensor_info_list, output_tensor_info_list);
}

int32_t InferenceWrapperMNN::Initialize(const std::string& model_filename, std::vector<InputTensorInfo>& input_tensor_info_list,
                                        std::vector<OutputTensorInfo>& output_tensor_info_list) {
    ClearRuntimeCaches();
    ResetCacheStatistics();
    input_names_.clear();
    net_.reset(MNN::Interpreter::createFromFile(model_filename.c_str()));
    if (!net_) {
        PRINT_E("Failed to load model file (%s)\n", model_filename.c_str());
        return WrapperError;
    }

    MNN::ScheduleConfig scheduleConfig;
    scheduleConfig.type = MNN_FORWARD_CPU;
    scheduleConfig.numThread = num_threads_;  // it seems, setting 1 has better performance on Android
    MNN::BackendConfig bnconfig;
    bnconfig.power = GetConfiguredCPUEnginePowerMode();
    bnconfig.precision = MNN::BackendConfig::Precision_Normal;
    scheduleConfig.backendConfig = &bnconfig;
    session_ = net_->createSession(scheduleConfig);
    if (!session_) {
        PRINT_E("Failed to create session\n");
        return WrapperError;
    }

    return ParameterInitialization(input_tensor_info_list, output_tensor_info_list);
};

int32_t InferenceWrapperMNN::Finalize(void) {
    ClearRuntimeCaches();
    if (net_ != nullptr) {
        if (session_ != nullptr) {
            net_->releaseSession(session_);
            session_ = nullptr;
        }
        net_->releaseModel();
        net_.reset();
    }
    input_names_.clear();
    return WrapperOk;
}

#ifdef ISF_ENABLE_MNN_CACHE_GUARD
int32_t InferenceWrapperMNN::PreProcessWithoutCache(const std::vector<InputTensorInfo>& input_tensor_info_list) {
    if (net_ == nullptr || session_ == nullptr) {
        return WrapperError;
    }
    for (const auto& input_tensor_info : input_tensor_info_list) {
        auto* input_tensor = net_->getSessionInput(session_, input_tensor_info.name.c_str());
        if (input_tensor == nullptr || input_tensor_info.data == nullptr) {
            return WrapperError;
        }
        if (input_tensor_info.data_type == InputTensorInfo::DataTypeImage) {
            if (input_tensor_info.image_info.width != input_tensor_info.image_info.crop_width ||
                input_tensor_info.image_info.height != input_tensor_info.image_info.crop_height) {
                return WrapperError;
            }
            MNN::CV::ImageProcess::Config config;
            if (input_tensor_info.image_info.channel == 3 && input_tensor_info.GetChannel() == 3) {
                config.sourceFormat = input_tensor_info.image_info.is_bgr ? MNN::CV::BGR : MNN::CV::RGB;
                if (input_tensor_info.image_info.swap_color) {
                    config.destFormat = input_tensor_info.image_info.is_bgr ? MNN::CV::RGB : MNN::CV::BGR;
                } else {
                    config.destFormat = input_tensor_info.image_info.is_bgr ? MNN::CV::BGR : MNN::CV::RGB;
                }
            } else if (input_tensor_info.image_info.channel == 1 && input_tensor_info.GetChannel() == 1) {
                config.sourceFormat = MNN::CV::GRAY;
                config.destFormat = MNN::CV::GRAY;
            } else if (input_tensor_info.image_info.channel == 3 && input_tensor_info.GetChannel() == 1) {
                config.sourceFormat = input_tensor_info.image_info.is_bgr ? MNN::CV::BGR : MNN::CV::RGB;
                config.destFormat = MNN::CV::GRAY;
            } else if (input_tensor_info.image_info.channel == 1 && input_tensor_info.GetChannel() == 3) {
                config.sourceFormat = MNN::CV::GRAY;
                config.destFormat = MNN::CV::BGR;
            } else {
                return WrapperError;
            }
            std::memcpy(config.mean, input_tensor_info.normalize.mean, sizeof(config.mean));
            std::memcpy(config.normal, input_tensor_info.normalize.norm, sizeof(config.normal));
            config.filterType = MNN::CV::BILINEAR;
            MNN::CV::Matrix transform;
            transform.setScale(static_cast<float>(input_tensor_info.image_info.crop_width) / input_tensor_info.GetWidth(),
                               static_cast<float>(input_tensor_info.image_info.crop_height) / input_tensor_info.GetHeight());
            std::unique_ptr<MNN::CV::ImageProcess> image_process(MNN::CV::ImageProcess::create(config));
            if (image_process == nullptr) {
                return WrapperError;
            }
            ++cache_statistics_.image_process_creations;
            image_process->setMatrix(transform);
            if (image_process->convert(static_cast<uint8_t*>(input_tensor_info.data), input_tensor_info.image_info.crop_width,
                                       input_tensor_info.image_info.crop_height, 0, input_tensor) != MNN::NO_ERROR) {
                return WrapperError;
            }
        } else if (input_tensor_info.data_type == InputTensorInfo::DataTypeBlobNhwc ||
                   input_tensor_info.data_type == InputTensorInfo::DataTypeBlobNchw) {
            const auto dimension_type = input_tensor_info.data_type == InputTensorInfo::DataTypeBlobNhwc ? MNN::Tensor::TENSORFLOW
                                                                                                          : MNN::Tensor::CAFFE;
            std::unique_ptr<MNN::Tensor> host_tensor(new MNN::Tensor(input_tensor, dimension_type));
            ++cache_statistics_.blob_host_tensor_creations;
            const size_t element_count = static_cast<size_t>(input_tensor_info.GetWidth()) *
                                         static_cast<size_t>(input_tensor_info.GetHeight()) *
                                         static_cast<size_t>(input_tensor_info.GetChannel());
            if (element_count != static_cast<size_t>(host_tensor->elementSize())) {
                return WrapperError;
            }
            const auto tensor_type = host_tensor->getType();
            if (tensor_type.code == halide_type_float && tensor_type.bytes() == sizeof(float)) {
                std::memcpy(host_tensor->host<float>(), input_tensor_info.data, element_count * sizeof(float));
            } else if (tensor_type.code == halide_type_uint && tensor_type.bytes() == sizeof(uint8_t)) {
                std::memcpy(host_tensor->host<uint8_t>(), input_tensor_info.data, element_count * sizeof(uint8_t));
            } else {
                return WrapperError;
            }
            if (!input_tensor->copyFromHostTensor(host_tensor.get())) {
                return WrapperError;
            }
        } else {
            return WrapperError;
        }
    }
    return WrapperOk;
}

int32_t InferenceWrapperMNN::ProcessWithoutCache(std::vector<OutputTensorInfo>& output_tensor_info_list) {
    if (net_ == nullptr || session_ == nullptr || net_->runSession(session_) != MNN::NO_ERROR) {
        return WrapperError;
    }
    uncached_output_tensors_.clear();
    uncached_output_tensors_.reserve(output_tensor_info_list.size());
    for (auto& output_tensor_info : output_tensor_info_list) {
        auto* output_tensor = net_->getSessionOutput(session_, output_tensor_info.name.c_str());
        if (output_tensor == nullptr) {
            return WrapperError;
        }
        std::unique_ptr<MNN::Tensor> host_tensor(new MNN::Tensor(output_tensor, output_tensor->getDimensionType()));
        ++cache_statistics_.output_host_tensor_creations;
        if (!output_tensor->copyToHostTensor(host_tensor.get())) {
            return WrapperError;
        }
        const auto type = host_tensor->getType();
        if (type.code == halide_type_float && type.bytes() == sizeof(float)) {
            output_tensor_info.tensor_type = TensorInfo::TensorTypeFp32;
            output_tensor_info.data = host_tensor->host<float>();
        } else if (type.code == halide_type_uint && type.bytes() == 1) {
            output_tensor_info.tensor_type = TensorInfo::TensorTypeUint8;
            output_tensor_info.data = host_tensor->host<uint8_t>();
        } else {
            return WrapperError;
        }
        output_tensor_info.tensor_dims.clear();
        for (int dimension = 0; dimension < host_tensor->dimensions(); ++dimension) {
            output_tensor_info.tensor_dims.push_back(host_tensor->length(dimension));
        }
        uncached_output_tensors_.push_back(std::move(host_tensor));
    }
    return WrapperOk;
}
#endif

int32_t InferenceWrapperMNN::PreProcess(const std::vector<InputTensorInfo>& input_tensor_info_list) {
#ifdef ISF_ENABLE_MNN_CACHE_GUARD
    if (!cache_enabled_for_testing_) {
        return PreProcessWithoutCache(input_tensor_info_list);
    }
#endif
    if (net_ == nullptr || session_ == nullptr) {
        return WrapperError;
    }
    if (input_cache_.size() != input_tensor_info_list.size()) {
        input_cache_.clear();
        input_cache_.resize(input_tensor_info_list.size());
    }

    for (size_t input_index = 0; input_index < input_tensor_info_list.size(); ++input_index) {
        const auto& input_tensor_info = input_tensor_info_list[input_index];
        auto& cache = input_cache_[input_index];
        if (cache.name != input_tensor_info.name) {
            cache = InputCacheEntry{};
            cache.name = input_tensor_info.name;
        }
        if (cache.input_tensor == nullptr) {
            cache.input_tensor = net_->getSessionInput(session_, input_tensor_info.name.c_str());
            if (cache.input_tensor == nullptr) {
                PRINT_E("Invalid input name (%s)\n", input_tensor_info.name.c_str());
                INSPIRE_LOGE("Invalid input name (%s)\n", input_tensor_info.name.c_str());
                return WrapperError;
            }
        }
        auto* input_tensor = cache.input_tensor;
        if (input_tensor_info.data == nullptr) {
            PRINT_E("Input data is null (%s)\n", input_tensor_info.name.c_str());
            return WrapperError;
        }
        if (input_tensor_info.data_type == InputTensorInfo::DataTypeImage) {
            /* Crop */
            if ((input_tensor_info.image_info.width != input_tensor_info.image_info.crop_width) ||
                (input_tensor_info.image_info.height != input_tensor_info.image_info.crop_height)) {
                PRINT_E("Crop is not supported\n");
                return WrapperError;
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
                return WrapperError;
            }

            /* Normalize image */
            std::memcpy(image_processconfig.mean, input_tensor_info.normalize.mean, sizeof(image_processconfig.mean));
            std::memcpy(image_processconfig.normal, input_tensor_info.normalize.norm, sizeof(image_processconfig.normal));

            /* Resize image */
            image_processconfig.filterType = MNN::CV::BILINEAR;
            const int source_format = static_cast<int>(image_processconfig.sourceFormat);
            const int destination_format = static_cast<int>(image_processconfig.destFormat);
            if (!cache.image_signature.Matches(input_tensor_info, source_format, destination_format) || cache.image_process == nullptr) {
                std::unique_ptr<MNN::CV::ImageProcess> image_process(MNN::CV::ImageProcess::create(image_processconfig));
                if (image_process == nullptr) {
                    PRINT_E("Unable to create image process (%s)\n", input_tensor_info.name.c_str());
                    return WrapperError;
                }
                MNN::CV::Matrix transform;
                transform.setScale(static_cast<float>(input_tensor_info.image_info.crop_width) / input_tensor_info.GetWidth(),
                                   static_cast<float>(input_tensor_info.image_info.crop_height) / input_tensor_info.GetHeight());
                image_process->setMatrix(transform);
                cache.image_process = std::move(image_process);
                cache.image_signature.Capture(input_tensor_info, source_format, destination_format);
                ++cache_statistics_.image_process_creations;
            }
            if (cache.image_process->convert(static_cast<uint8_t*>(input_tensor_info.data), input_tensor_info.image_info.crop_width,
                                             input_tensor_info.image_info.crop_height, 0, input_tensor) != MNN::NO_ERROR) {
                PRINT_E("Image preprocessing failed (%s)\n", input_tensor_info.name.c_str());
                return WrapperError;
            }

        } else if ((input_tensor_info.data_type == InputTensorInfo::DataTypeBlobNhwc) ||
                   (input_tensor_info.data_type == InputTensorInfo::DataTypeBlobNchw)) {
            const auto dimension_type = input_tensor_info.data_type == InputTensorInfo::DataTypeBlobNhwc ? MNN::Tensor::TENSORFLOW
                                                                                                          : MNN::Tensor::CAFFE;
            if (cache.blob_host_tensor == nullptr || cache.blob_dimension_type != dimension_type ||
                !cache.blob_signature.Matches(input_tensor)) {
                std::unique_ptr<MNN::Tensor> blob_host_tensor(new MNN::Tensor(input_tensor, dimension_type));
                cache.blob_host_tensor = std::move(blob_host_tensor);
                cache.blob_dimension_type = dimension_type;
                cache.blob_signature.Capture(input_tensor);
                ++cache_statistics_.blob_host_tensor_creations;
            }
            const size_t element_count = static_cast<size_t>(input_tensor_info.GetWidth()) *
                                         static_cast<size_t>(input_tensor_info.GetHeight()) *
                                         static_cast<size_t>(input_tensor_info.GetChannel());
            if (element_count != static_cast<size_t>(cache.blob_host_tensor->elementSize())) {
                PRINT_E("Blob input size mismatch (%zu != %d)\n", element_count, cache.blob_host_tensor->elementSize());
                return WrapperError;
            }
            const auto tensor_type = cache.blob_host_tensor->getType();
            if (tensor_type.code == halide_type_float && tensor_type.bytes() == sizeof(float)) {
                std::memcpy(cache.blob_host_tensor->host<float>(), input_tensor_info.data, element_count * sizeof(float));
            } else if (tensor_type.code == halide_type_uint && tensor_type.bytes() == sizeof(uint8_t)) {
                std::memcpy(cache.blob_host_tensor->host<uint8_t>(), input_tensor_info.data, element_count * sizeof(uint8_t));
            } else {
                PRINT_E("Unsupported blob tensor type (%d, %d)\n", tensor_type.code, tensor_type.bytes());
                return WrapperError;
            }
            if (!input_tensor->copyFromHostTensor(cache.blob_host_tensor.get())) {
                PRINT_E("Blob input copy failed (%s)\n", input_tensor_info.name.c_str());
                return WrapperError;
            }
        } else {
            PRINT_E("Unsupported data type (%d)\n", input_tensor_info.data_type);
            return WrapperError;
        }
    }
    return WrapperOk;
}

int32_t InferenceWrapperMNN::Process(std::vector<OutputTensorInfo>& output_tensor_info_list) {
#ifdef ISF_ENABLE_MNN_CACHE_GUARD
    if (!cache_enabled_for_testing_) {
        return ProcessWithoutCache(output_tensor_info_list);
    }
#endif
    if (net_ == nullptr || session_ == nullptr || net_->runSession(session_) != MNN::NO_ERROR) {
        return WrapperError;
    }
    if (output_cache_.size() != output_tensor_info_list.size()) {
        output_cache_.clear();
        output_cache_.resize(output_tensor_info_list.size());
    }

    for (size_t output_index = 0; output_index < output_tensor_info_list.size(); ++output_index) {
        auto& output_tensor_info = output_tensor_info_list[output_index];
        auto& cache = output_cache_[output_index];
        if (cache.name != output_tensor_info.name) {
            cache = OutputCacheEntry{};
            cache.name = output_tensor_info.name;
        }
        auto* output_tensor = net_->getSessionOutput(session_, output_tensor_info.name.c_str());
        if (cache.output_tensor != output_tensor) {
            cache.output_tensor = output_tensor;
            cache.host_tensor.reset();
            cache.signature = TensorSignature{};
        }
        if (cache.output_tensor == nullptr) {
            PRINT_E("Invalid output name (%s)\n", output_tensor_info.name.c_str());
            return WrapperError;
        }
        if (cache.host_tensor == nullptr || !cache.signature.Matches(cache.output_tensor)) {
            std::unique_ptr<MNN::Tensor> host_tensor(
              new MNN::Tensor(cache.output_tensor, cache.output_tensor->getDimensionType()));
            cache.host_tensor = std::move(host_tensor);
            cache.signature.Capture(cache.output_tensor);
            ++cache_statistics_.output_host_tensor_creations;
        }
        if (!cache.output_tensor->copyToHostTensor(cache.host_tensor.get())) {
            PRINT_E("Output tensor copy failed (%s)\n", output_tensor_info.name.c_str());
            return WrapperError;
        }
        auto type = cache.host_tensor->getType();
        if (type.code == halide_type_float && type.bytes() == sizeof(float)) {
            output_tensor_info.tensor_type = TensorInfo::TensorTypeFp32;
            output_tensor_info.data = cache.host_tensor->host<float>();
        } else if (type.code == halide_type_uint && type.bytes() == 1) {
            output_tensor_info.tensor_type = TensorInfo::TensorTypeUint8;
            output_tensor_info.data = cache.host_tensor->host<uint8_t>();
        } else {
            PRINT_E("Unexpected data type\n");
            return WrapperError;
        }

        output_tensor_info.tensor_dims.clear();
        for (int32_t dim = 0; dim < cache.host_tensor->dimensions(); dim++) {
            output_tensor_info.tensor_dims.push_back(cache.host_tensor->length(dim));
        }
    }

    return WrapperOk;
}

std::vector<std::string> InferenceWrapperMNN::GetInputNames() {
    return input_names_;
}

InferenceCacheStatistics InferenceWrapperMNN::GetCacheStatistics() const {
    return cache_statistics_;
}

void InferenceWrapperMNN::ResetCacheStatistics() {
    cache_statistics_ = {};
}

int32_t InferenceWrapperMNN::ResizeInput(const std::vector<InputTensorInfo>& input_tensor_info_list) {
    if (net_ == nullptr || session_ == nullptr) {
        return WrapperError;
    }
    for (const auto& input_tensor_info : input_tensor_info_list) {
        auto input_tensor = net_->getSessionInput(session_, input_tensor_info.name.c_str());
        if (input_tensor == nullptr || input_tensor_info.GetChannel() <= 0 || input_tensor_info.GetHeight() <= 0 ||
            input_tensor_info.GetWidth() <= 0) {
            return WrapperError;
        }
        net_->resizeTensor(input_tensor, {1, input_tensor_info.GetChannel(), input_tensor_info.GetHeight(), input_tensor_info.GetWidth()});
    }
    net_->resizeSession(session_);
    ClearRuntimeCaches();
    return WrapperOk;
}
