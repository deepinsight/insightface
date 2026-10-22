/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#ifndef MODELLOADERTAR_INSPIRE_MODEL_H
#define MODELLOADERTAR_INSPIRE_MODEL_H

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include "yaml-cpp/yaml.h"
#include "middleware/configurable.h"
#include "log.h"
#include "middleware/inference_wrapper/inference_wrapper.h"

namespace inspire {

typedef enum {
    InspireInferBackendAuto = 10,
    InspireInferBackendCPU = 0,
    InspireInferBackendRKNPU = 1,
    InspireInferBackendCUDA = 2,
} InspireInferBackend;

typedef enum {
    InspireInferEngineMNN = 0,
    InspireInferEngineRKNN = 1,
    InspireInferEngineCoreML = 2,
    InspireInferEngineTensorRT = 3,
} InspireInferEngine;

class INSPIRE_API InspireModel {
    CONFIGURABLE_SUPPORT

public:
    explicit InspireModel(const YAML::Node &node) {
        Reset(node);
    }

    InspireModel() = default;

    // Parse into temporary state first. Invalid metadata must not leave a
    // half-updated model or stale enum/buffer values behind.
    int32_t Reset(const YAML::Node &node) {
        try {
            if (!node || !node.IsMap()) {
                return -1;
            }

            std::string next_name;
            std::string next_fullname;
            std::string next_version;
            InferenceWrapper::EngineType next_model_type = InferenceWrapper::INFER_MNN;
            int next_infer_engine = InferenceWrapper::INFER_MNN;
            int next_infer_device = InspireInferEngineMNN;
            int next_infer_backend = InspireInferBackendCPU;
            int next_load_file_path = 0;
            Configurable next_configuration;

            if (node["name"]) {
                next_name = node["name"].as<std::string>();
            }
            if (node["fullname"]) {
                next_fullname = node["fullname"].as<std::string>();
            }
            if (node["version"]) {
                next_version = node["version"].as<std::string>();
            }
            if (node["model_type"]) {
                if (!DecodeEngine(node["model_type"].as<std::string>(), next_model_type)) {
                    return -1;
                }
                if (next_model_type == InferenceWrapper::INFER_COREML) {
                    // CoreML consumes a bundle path rather than an archive buffer.
                    next_load_file_path = 1;
                }
            }
            if (node["infer_engine"]) {
                InferenceWrapper::EngineType engine = InferenceWrapper::INFER_MNN;
                if (!DecodeEngine(node["infer_engine"].as<std::string>(), engine)) {
                    return -1;
                }
                next_infer_engine = engine;
            }
            if (node["infer_device"]) {
                const auto type = node["infer_device"].as<std::string>();
                if (type == "MNN") {
                    next_infer_device = InspireInferEngineMNN;
                } else if (type == "RKNPU") {
                    next_infer_device = InspireInferEngineRKNN;
                } else if (type == "COREML") {
                    next_infer_device = InspireInferEngineCoreML;
                } else if (type == "CUDA") {
                    next_infer_device = InspireInferEngineTensorRT;
                } else {
                    return -1;
                }
            }
            if (node["infer_backend"]) {
                const auto type = node["infer_backend"].as<std::string>();
                if (type == "CPU") {
                    next_infer_backend = InspireInferBackendCPU;
                } else if (type == "RKNPU") {
                    next_infer_backend = InspireInferBackendRKNPU;
                } else if (type == "AUTO") {
                    next_infer_backend = InspireInferBackendAuto;
                } else if (type == "CUDA") {
                    next_infer_backend = InspireInferBackendCUDA;
                } else {
                    return -1;
                }
            }
            if (DecodeConfiguration(node, next_configuration) != 0) {
                return -1;
            }

            name.swap(next_name);
            fullname.swap(next_fullname);
            version.swap(next_version);
            modelType = next_model_type;
            inferEngine = next_infer_engine;
            inferDevice = next_infer_device;
            inferBackend = next_infer_backend;
            loadFilePath = next_load_file_path;
            m_configuration = next_configuration;
            m_buffer_owner_.reset();
            buffer = nullptr;
            bufferSize = 0;
            return 0;
        } catch (const std::exception &error) {
            INSPIRE_LOGE("An error occurred parsing a model config: %s", error.what());
            return -1;
        }
    }

    void print() {
        INSPIRE_LOGD("%s", m_configuration.toString().c_str());
    }

    // Compatibility overload for callers that explicitly own the vector.
    void SetBuffer(std::vector<char> &modelBuffer, size_t size) {
        m_buffer_owner_.reset();
        buffer = modelBuffer.empty() ? nullptr : modelBuffer.data();
        bufferSize = std::min(size, modelBuffer.size());
    }

    // Archive-loaded models retain their immutable backing storage, so a
    // reload/close cannot invalidate the pointer passed to an inference engine.
    void SetBuffer(const std::shared_ptr<const std::vector<char>>& modelBuffer) {
        m_buffer_owner_ = modelBuffer;
        buffer = modelBuffer && !modelBuffer->empty() ? const_cast<char*>(modelBuffer->data()) : nullptr;
        bufferSize = modelBuffer ? modelBuffer->size() : 0;
    }

    Configurable &Config() {
        return m_configuration;
    }

private:
    static bool DecodeEngine(const std::string& type, InferenceWrapper::EngineType& engine) {
        if (type == "MNN") {
            engine = InferenceWrapper::INFER_MNN;
        } else if (type == "RKNN") {
            engine = InferenceWrapper::INFER_RKNN;
        } else if (type == "COREML") {
            engine = InferenceWrapper::INFER_COREML;
        } else if (type == "TensorRT") {
            engine = InferenceWrapper::INFER_TENSORRT;
        } else {
            return false;
        }
        return true;
    }

    static bool DecodeTensorType(const std::string& type, int& tensor_type) {
        if (type == "none") {
            tensor_type = InputTensorInfo::TensorInfo::TensorTypeNone;
        } else if (type == "uint8") {
            tensor_type = InputTensorInfo::TensorInfo::TensorTypeUint8;
        } else if (type == "int8") {
            tensor_type = InputTensorInfo::TensorInfo::TensorTypeInt8;
        } else if (type == "float32") {
            tensor_type = InputTensorInfo::TensorInfo::TensorTypeFp32;
        } else if (type == "int32") {
            tensor_type = InputTensorInfo::TensorInfo::TensorTypeInt32;
        } else if (type == "int64") {
            tensor_type = InputTensorInfo::TensorInfo::TensorTypeInt64;
        } else {
            return false;
        }
        return true;
    }

    static int32_t DecodeConfiguration(const YAML::Node &node, Configurable& configuration) {
        try {
            if (node["input_channel"]) {
                configuration.set<int>("input_channel", node["input_channel"].as<int>());
            }
            if (node["input_image_channel"]) {
                configuration.set<int>("input_image_channel", node["input_image_channel"].as<int>());
            }
            if (node["nchw"]) {
                configuration.set<bool>("nchw", node["nchw"].as<bool>());
            }
            if (node["swap_color"]) {
                configuration.set<bool>("swap_color", node["swap_color"].as<bool>());
            }
            if (node["data_type"]) {
                const auto type = node["data_type"].as<std::string>();
                int data_type = InputTensorInfo::InputTensorInfo::DataTypeImage;
                if (type == "data_nhwc") {
                    data_type = InputTensorInfo::InputTensorInfo::DataTypeBlobNhwc;
                } else if (type == "data_nchw") {
                    data_type = InputTensorInfo::InputTensorInfo::DataTypeBlobNchw;
                } else if (type != "image") {
                    return -1;
                }
                configuration.set<int>("data_type", data_type);
            }
            if (node["input_tensor_type"]) {
                int tensor_type = InputTensorInfo::TensorInfo::TensorTypeNone;
                if (!DecodeTensorType(node["input_tensor_type"].as<std::string>(), tensor_type)) {
                    return -1;
                }
                configuration.set<int>("input_tensor_type", tensor_type);
            }
            if (node["output_tensor_type"]) {
                int tensor_type = InputTensorInfo::TensorInfo::TensorTypeNone;
                if (!DecodeTensorType(node["output_tensor_type"].as<std::string>(), tensor_type)) {
                    return -1;
                }
                configuration.set<int>("output_tensor_type", tensor_type);
            }
            if (node["threads"]) {
                configuration.set<int>("threads", node["threads"].as<int>());
            }
            if (node["input_layer"]) {
                configuration.set<std::string>("input_layer", node["input_layer"].as<std::string>());
            }
            if (node["outputs_layers"]) {
                const auto values = node["outputs_layers"];
                std::vector<std::string> names;
                names.reserve(values.size());
                for (std::size_t i = 0; i < values.size(); ++i) {
                    names.push_back(values[i].as<std::string>());
                }
                configuration.set<std::vector<std::string>>("outputs_layers", names);
            }
            if (node["input_size"]) {
                const auto values = node["input_size"];
                std::vector<int> size;
                size.reserve(values.size());
                for (std::size_t i = 0; i < values.size(); ++i) {
                    size.push_back(values[i].as<int>());
                }
                configuration.set<std::vector<int>>("input_size", size);
            }
            if (node["mean"]) {
                const auto values = node["mean"];
                std::vector<float> mean;
                mean.reserve(values.size());
                for (std::size_t i = 0; i < values.size(); ++i) {
                    mean.push_back(values[i].as<float>());
                }
                configuration.set<std::vector<float>>("mean", mean);
            }
            if (node["norm"]) {
                const auto values = node["norm"];
                std::vector<float> norm;
                norm.reserve(values.size());
                for (std::size_t i = 0; i < values.size(); ++i) {
                    norm.push_back(values[i].as<float>());
                }
                configuration.set<std::vector<float>>("norm", norm);
            }
        } catch (const YAML::Exception &error) {
            INSPIRE_LOGE("An error occurred parsing the interpretation file in archive: %s", error.what());
            return -1;
        }
        return 0;
    }

public:
    std::string name;
    std::string fullname;
    std::string version;
    InferenceWrapper::EngineType modelType{InferenceWrapper::INFER_MNN};
    int inferEngine{InferenceWrapper::INFER_MNN};
    int inferDevice{InspireInferEngineMNN};
    int inferBackend{InspireInferBackendCPU};
    int loadFilePath{0};

    char *buffer{nullptr};
    size_t bufferSize{0};

private:
    std::shared_ptr<const std::vector<char>> m_buffer_owner_;
};

}  // namespace inspire

#endif  // MODELLOADERTAR_INSPIRE_MODEL_H
