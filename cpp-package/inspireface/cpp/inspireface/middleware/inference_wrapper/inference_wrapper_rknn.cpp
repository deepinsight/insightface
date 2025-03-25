/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#ifdef INFERENCE_WRAPPER_ENABLE_RKNN
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include "inference_wrapper_rknn.h"
#include "inference_wrapper_log.h"
#include "log.h"

static unsigned char* load_data(FILE* fp, size_t ofst, size_t sz) {
    unsigned char* data;
    int ret;

    data = NULL;

    if (NULL == fp) {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0) {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char*)malloc(sz);
    if (data == NULL) {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char* load_model(const char* filename, int* model_size) {
    FILE* fp;
    unsigned char* data;

    fp = fopen(filename, "rb");
    if (NULL == fp) {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

/*** Macro ***/
#define TAG "InferenceWrapperRKNN"
#define PRINT(...) INFERENCE_WRAPPER_LOG_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) INFERENCE_WRAPPER_LOG_PRINT_E(TAG, __VA_ARGS__)

InferenceWrapperRKNN::InferenceWrapperRKNN() {
    num_threads_ = 1;
}

InferenceWrapperRKNN::~InferenceWrapperRKNN() {}

int32_t InferenceWrapperRKNN::SetNumThreads(const int32_t num_threads) {
    num_threads_ = num_threads;
    return WrapperOk;
}

int32_t InferenceWrapperRKNN::ParameterInitialization(std::vector<InputTensorInfo>& input_tensor_info_list,
                                                      std::vector<OutputTensorInfo>& output_tensor_info_list) {
    auto ret = rknn_query(net_, RKNN_QUERY_IN_OUT_NUM, &rk_io_num_, sizeof(rk_io_num_));

    if (ret != RKNN_SUCC) {
        PRINT_E("rknn_query ctx fail! ret=%d\n", ret)
        return WrapperError;
    }

    std::vector<rknn_tensor_attr> output_attrs_;
    output_attrs_.resize(rk_io_num_.n_output);
    for (int i = 0; i < rk_io_num_.n_output; ++i) {
        //        memset(&output_attrs_[i], 0, sizeof(output_attrs_[i]));
        //        memset(&output_tensors_[i], 0, sizeof(output_tensors_[i]));
        output_attrs_[i].index = i;
        ret = rknn_query(net_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]), sizeof(rknn_tensor_attr));
        auto& output = output_tensor_info_list[i];
        output.tensor_dims.clear();
        for (int j = 0; j < output_attrs_[i].n_dims; ++j) {
            output.tensor_dims.push_back(output_attrs_[i].dims[j]);
            std::cout << "dim: " << output_attrs_[i].dims[j] << std::endl;
        }
        //        std::cout << output_attrs_[i].n_dims << std::endl;
    }

    return WrapperOk;
}

int32_t InferenceWrapperRKNN::Initialize(const std::string& model_filename, std::vector<InputTensorInfo>& input_tensor_info_list,
                                         std::vector<OutputTensorInfo>& output_tensor_info_list) {
    int model_data_size = 0;
    unsigned char* model_data = load_model(model_filename.c_str(), &model_data_size);
    int ret = rknn_init(&net_, model_data, model_data_size, 0);
    if (ret < 0) {
        PRINT_E("Failed to load model file (%s)\n", model_filename.c_str())
        return WrapperError;
    }
    rknn_sdk_version version;
    ret = rknn_query(net_, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0) {
        PRINT_E("rknn_init error ret=%d\n", ret)
        return WrapperError;
    }

    return ParameterInitialization(input_tensor_info_list, output_tensor_info_list);
}

int32_t InferenceWrapperRKNN::Initialize(char* model_buffer, int model_size, std::vector<InputTensorInfo>& input_tensor_info_list,
                                         std::vector<OutputTensorInfo>& output_tensor_info_list) {
    int ret = rknn_init(&net_, model_buffer, model_size, 0);
    if (ret < 0) {
        PRINT_E("rknn_init error ret=%d\n", ret)
        return WrapperError;
    }
    rknn_sdk_version version;
    ret = rknn_query(net_, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0) {
        PRINT_E("rknn_init error ret=%d\n", ret)
        return WrapperError;
    }
    return ParameterInitialization(input_tensor_info_list, output_tensor_info_list);
}

int32_t InferenceWrapperRKNN::Finalize(void) {
    rknn_destroy(net_);
    return WrapperOk;
}

int32_t InferenceWrapperRKNN::PreProcess(const std::vector<InputTensorInfo>& input_tensor_info_list) {
    /* Check tensor info fits the info from model */
    if (input_tensor_info_list.size() != rk_io_num_.n_input) {
        PRINT_E("The inputs quantity is inconsistent: i: %d, m: %d", input_tensor_info_list.size(), rk_io_num_.n_input);
        return WrapperError;
    }

    std::vector<rknn_input> input_tensors_;
    for (size_t index = 0; index < input_tensor_info_list.size(); index++) {
        auto& input_tensor_info = input_tensor_info_list[index];
        if (input_tensor_info.data_type == InputTensorInfo::DataTypeImage) {
            /* Crop */
            // Not Implement

            /* Convert color type */
            // Not Implement
            rknn_input input;
            if (input_tensor_info.tensor_type == TensorInfo::TensorTypeUint8) {
                input.type = RKNN_TENSOR_UINT8;
            } else if (input_tensor_info.tensor_type == TensorInfo::TensorTypeFp32) {
                input.type = RKNN_TENSOR_FLOAT32;
            } else {
                PRINT_E("Unsupported input type.")
                return WrapperError;
            }
            if (input_tensor_info.is_nchw) {
                input.fmt = RKNN_TENSOR_NCHW;
            } else {
                input.fmt = RKNN_TENSOR_NHWC;
                std::cout << "NHWC" << std::endl;
            }
            input.index = index;
            input.size = input_tensor_info.GetWidth() * input_tensor_info.GetHeight() * input_tensor_info.GetChannel();
            input.buf = input_tensor_info.data;
            input.pass_through = 0;
            input_tensors_.push_back(input);
        }
    }
    //    INSPIRE_LOGD("Prepare data!");
    int ret = rknn_inputs_set(net_, input_tensor_info_list.size(), input_tensors_.data());
    //    INSPIRE_LOGD("Set data!");
    if (ret < 0) {
        PRINT_E("rknn_run fail! ret=%d", ret)
        return WrapperError;
    }
    return WrapperOk;
}

int32_t InferenceWrapperRKNN::Process(std::vector<OutputTensorInfo>& output_tensor_info_list) {
    if (output_tensor_info_list.size() != rk_io_num_.n_output) {
        PRINT_E("The outputs quantity is inconsistent")
        return WrapperError;
    }
    auto ret = rknn_run(net_, NULL);
    if (ret < 0) {
        PRINT_E("rknn_run fail! ret=%d", ret)
        return WrapperError;
    }

    for (size_t index = 0; index < output_tensor_info_list.size(); index++) {
        auto& output_tensor = output_tensor_info_list[index];
        rknn_output output;
        if (output_tensor.tensor_type == TensorInfo::TensorTypeFp32) {
            output.want_float = 1;
            INSPIRE_LOGD("want_float=1");
        }
        output.is_prealloc = 0;
        output_tensors_.push_back(output);
        //        output.want_float = 1;  // float
    }

    ret = rknn_outputs_get(net_, output_tensor_info_list.size(), output_tensors_.data(), NULL);
    if (ret < 0) {
        PRINT_E("rknn_run fail! ret=%d", ret)
        return WrapperError;
    }
    for (size_t index = 0; index < output_tensor_info_list.size(); index++) {
        auto& output_tensor = output_tensor_info_list[index];
        auto& output = output_tensors_[index];
        output_tensor.data = output.buf;
        float* outBlob = (float*)output.buf;
        //        output_tensor.
    }

    return WrapperOk;
}

std::vector<std::string> InferenceWrapperRKNN::GetInputNames() {
    return input_names_;
}

int32_t InferenceWrapperRKNN::ResizeInput(const std::vector<InputTensorInfo>& input_tensor_info_list) {
    // The function is not supported
    return 0;
}

#endif  // INFERENCE_WRAPPER_ENABLE_RKNN
