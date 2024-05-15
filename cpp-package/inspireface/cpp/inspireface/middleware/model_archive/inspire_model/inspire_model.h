//
// Created by tunm on 2024/4/2.
//

#ifndef MODELLOADERTAR_INSPIRE_MODEL_H
#define MODELLOADERTAR_INSPIRE_MODEL_H
#include <iostream>
#include "vector"
#include "yaml-cpp/yaml.h"
#include "middleware/configurable.h"
#include "log.h"
#include "middleware/inference_helper/inference_helper.h"

namespace inspire {

class INSPIRE_API InspireModel {
CONFIGURABLE_SUPPORT
public:
    explicit InspireModel(const YAML::Node &node) {
        Reset(node);
    }

    explicit InspireModel() {};

    int32_t Reset(const YAML::Node &node) {
        buffer = nullptr;
        bufferSize = 0;
        if (node["name"]) {
            name = node["name"].as<std::string>();
        }
        if (node["fullname"]) {
            fullname = node["fullname"].as<std::string>();
        }
        if (node["version"]) {
            version = node["version"].as<std::string>();
        }
        if (node["model_type"]) {
            auto type = node["model_type"].as<std::string>();
            if (type == "MNN") {
                modelType = InferenceHelper::kMnn;
            } else if (type == "RKNN") {
                modelType = InferenceHelper::kRknn;
            }
        }
        if (node["infer_engine"]) {
            auto type = node["infer_engine"].as<std::string>();
            if (type == "MNN") {
                inferEngine = InferenceHelper::kMnn;
            } else if (type == "RKNN") {
                inferEngine = InferenceHelper::kRknn;
            }
        }
        if (node["infer_device"]) {
            auto type = node["infer_device"].as<std::string>();
            if (type == "MNN") {
                inferDevice = 0;
            } else if (type == "RKNPU") {
                inferDevice = 1;
            }
        }
        if (node["infer_backend"]) {
            auto type = node["infer_backend"].as<std::string>();
            if (type == "CPU") {
                inferBackend = 0;
            } else if (type == "RKNPU") {
                inferBackend = 1;
            }
        }
        return decode(node);
    }

    void print() {
        INSPIRE_LOGD("%s", m_configuration.toString().c_str());
    }

    void SetBuffer(std::vector<char>& modelBuffer, size_t size) {
        buffer = modelBuffer.data();
        bufferSize = size;
    }

    Configurable& Config() {
        return m_configuration;
    }

private:

    int32_t decode(const YAML::Node &node) {
        try {
            if (node["input_channel"]) {
                setData<int>("input_channel", node["input_channel"].as<int>());
            }
            if (node["input_image_channel"]) {
                setData<int>("input_image_channel", node["input_image_channel"].as<int>());
            }
            if (node["nchw"]) {
                setData<bool>("nchw", node["nchw"].as<bool>());
            }
            if (node["swap_color"]) {
                setData<bool>("swap_color", node["swap_color"].as<bool>());
            }
            if (node["data_type"]) {
                auto type = node["data_type"].as<std::string>();
                if (type == "image") {
                    setData<int>("data_type", InputTensorInfo::InputTensorInfo::kDataTypeImage);
                } else if(type == "data_nhwc") {
                    setData<int>("data_type", InputTensorInfo::InputTensorInfo::kDataTypeBlobNhwc);
                } else if(type == "data_nchw") {
                    setData<int>("data_type", InputTensorInfo::InputTensorInfo::kDataTypeBlobNchw);
                }
            }
            if (node["input_tensor_type"]) {
                auto type = node["input_tensor_type"].as<std::string>();
                if (type == "none") {
                    setData<int>("input_tensor_type", InputTensorInfo::TensorInfo::kTensorTypeNone);
                } else if(type == "uint8") {
                    setData<int>("input_tensor_type", InputTensorInfo::TensorInfo::kTensorTypeUint8);
                } else if(type == "int8") {
                    setData<int>("input_tensor_type", InputTensorInfo::TensorInfo::kTensorTypeInt8);
                } else if(type == "float32") {
                    setData<int>("input_tensor_type", InputTensorInfo::TensorInfo::kTensorTypeFp32);
                } else if(type == "int32") {
                    setData<int>("input_tensor_type", InputTensorInfo::TensorInfo::kTensorTypeInt32);
                } else if(type == "int64") {
                    setData<int>("input_tensor_type", InputTensorInfo::TensorInfo::kTensorTypeInt64);
                }
            }
            if (node["output_tensor_type"]) {
                auto type = node["output_tensor_type"].as<std::string>();
                if (type == "none") {
                    setData<int>("output_tensor_type", InputTensorInfo::TensorInfo::kTensorTypeNone);
                } else if(type == "uint8") {
                    setData<int>("output_tensor_type", InputTensorInfo::TensorInfo::kTensorTypeUint8);
                } else if(type == "int8") {
                    setData<int>("output_tensor_type", InputTensorInfo::TensorInfo::kTensorTypeInt8);
                } else if(type == "float32") {
                    setData<int>("output_tensor_type", InputTensorInfo::TensorInfo::kTensorTypeFp32);
                } else if(type == "int32") {
                    setData<int>("output_tensor_type", InputTensorInfo::TensorInfo::kTensorTypeInt32);
                } else if(type == "int64") {
                    setData<int>("output_tensor_type", InputTensorInfo::TensorInfo::kTensorTypeInt64);
                }
            }
            if (node["threads"]) {
                setData<int>("threads", node["threads"].as<int>());
            }
            if (node["input_layer"]) {
                setData<std::string>("input_layer", node["input_layer"].as<std::string>());
            }
            if (node["outputs_layers"]) {
                auto n = node["outputs_layers"];
                std::vector<std::string> names;
                for (std::size_t i = 0; i < n.size(); ++i) {
                    names.push_back(n[i].as<std::string>());
                }
                setData<std::vector<std::string>>("outputs_layers", names);
            }
            if (node["input_size"]) {
                auto n = node["input_size"];
                std::vector<int> size;
                for (std::size_t i = 0; i < n.size(); ++i) {
                    size.push_back(n[i].as<int>());
                }
                setData<std::vector<int>>("input_size", size);
            }
            if (node["mean"]) {
                auto n = node["mean"];
                std::vector<float> mean;
                for (std::size_t i = 0; i < n.size(); ++i) {
                    mean.push_back(n[i].as<float>());
                }
                setData<std::vector<float>>("mean", mean);
            }
            if (node["norm"]) {
                auto n = node["norm"];
                std::vector<float> norm;
                for (std::size_t i = 0; i < n.size(); ++i) {
                    norm.push_back(n[i].as<float>());
                }
                setData<std::vector<float>>("norm", norm);
            }
        } catch (const YAML::Exception& e) {
            INSPIRE_LOGE("An error occurred parsing the interpretation file in archive: %s", e.what());
            return -1;
        }
        return 0;
    }

public:
    std::string name;
    std::string fullname;
    std::string version;
    InferenceHelper::HelperType modelType;
    int inferEngine;
    int inferDevice;
    int inferBackend;

    char *buffer;
    size_t bufferSize;

};

};   // namespace inspire

#endif //MODELLOADERTAR_INSPIRE_MODEL_H
