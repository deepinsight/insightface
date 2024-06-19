//
// Created by tunm on 2023/2/5.
//
/*** Include ***/
/* for general */

#ifdef INFERENCE_HELPER_ENABLE_RKNN

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

#include "inference_helper_rknn.h"
#include "inference_helper_log.h"
#include "log.h"

static unsigned char* load_data(FILE* fp, size_t ofst, size_t sz)
{
    unsigned char* data;
    int            ret;

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

static unsigned char* load_model(const char* filename, int* model_size)
{
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
#define TAG "InferenceHelperRknn"
#define PRINT(...)   INFERENCE_HELPER_LOG_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) INFERENCE_HELPER_LOG_PRINT_E(TAG, __VA_ARGS__)

InferenceHelperRKNN::InferenceHelperRKNN() {
    num_threads_ = 1;
}

InferenceHelperRKNN::~InferenceHelperRKNN() {
}

int32_t InferenceHelperRKNN::SetNumThreads(const int32_t num_threads) {
    num_threads_ = num_threads;
    return kRetOk;
}

int32_t InferenceHelperRKNN::SetCustomOps(const std::vector<std::pair<const char *, const void *>> &custom_ops) {
    PRINT("[WARNING] This method is not supported\n");
    return kRetOk;
}

int32_t InferenceHelperRKNN::ParameterInitialization(std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list) {
    auto ret = rknn_query(net_, RKNN_QUERY_IN_OUT_NUM, &rk_io_num_,
                     sizeof(rk_io_num_));

    if (ret != RKNN_SUCC) {
        PRINT_E("rknn_query ctx fail! ret=%d\n", ret)
        return kRetErr;
    }


//    for (size_t index = 0; index < input_tensor_info_list.size(); ++index) {
//        auto &input_tensor_info = input_tensor_info_list[index];
//    }
    std::vector<rknn_tensor_attr> output_attrs_;
    output_attrs_.resize(rk_io_num_.n_output);
    for (int i = 0; i < rk_io_num_.n_output; ++i) {
//        memset(&output_attrs_[i], 0, sizeof(output_attrs_[i]));
//        memset(&output_tensors_[i], 0, sizeof(output_tensors_[i]));
        output_attrs_[i].index = i;
        ret = rknn_query(net_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]),
                         sizeof(rknn_tensor_attr));
        auto &output = output_tensor_info_list[i];
        output.tensor_dims.clear();
        for (int j = 0; j < output_attrs_[i].n_dims; ++j) {
            output.tensor_dims.push_back(output_attrs_[i].dims[j]);
            std::cout << "dim: " << output_attrs_[i].dims[j] << std::endl;
        }
//        std::cout << output_attrs_[i].n_dims << std::endl;
    }

    return kRetOk;
}

int32_t
InferenceHelperRKNN::Initialize(const std::string &model_filename, std::vector<InputTensorInfo> &input_tensor_info_list,
                                std::vector<OutputTensorInfo> &output_tensor_info_list) {
    int model_data_size = 0;
    unsigned char* model_data = load_model(model_filename.c_str(), &model_data_size);
    int ret = rknn_init(&net_, model_data, model_data_size, 0);
    if (ret < 0) {
        PRINT_E("Failed to load model file (%s)\n", model_filename.c_str())
        return kRetErr;
    }
    rknn_sdk_version version;
    ret = rknn_query(net_, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0) {
        PRINT_E("rknn_init error ret=%d\n", ret)
        return kRetErr;
    }

    return ParameterInitialization(input_tensor_info_list, output_tensor_info_list);
}

int32_t InferenceHelperRKNN::Initialize(char* model_buffer, int model_size, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list) {
    int ret = rknn_init(&net_, model_buffer, model_size, 0);
    if (ret < 0) {
        PRINT_E("rknn_init error ret=%d\n", ret)
        return kRetErr;
    }
    rknn_sdk_version version;
    ret = rknn_query(net_, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0) {
        PRINT_E("rknn_init error ret=%d\n", ret)
        return kRetErr;
    }
    return ParameterInitialization(input_tensor_info_list, output_tensor_info_list);
}

int32_t InferenceHelperRKNN::Finalize(void) {
    rknn_destroy(net_);
    return kRetOk;
}

int32_t InferenceHelperRKNN::PreProcess(const std::vector<InputTensorInfo> &input_tensor_info_list) {

    /* Check tensor info fits the info from model */
    if (input_tensor_info_list.size() != rk_io_num_.n_input) {
        PRINT_E("The inputs quantity is inconsistent: i: %d, m: %d", input_tensor_info_list.size(), rk_io_num_.n_input);
        return kRetErr;
    }

    std::vector<rknn_input> input_tensors_;
    for (size_t index = 0; index < input_tensor_info_list.size(); index++) {
        auto &input_tensor_info = input_tensor_info_list[index];
        if (input_tensor_info.data_type == InputTensorInfo::kDataTypeImage) {
            /* Crop */
            // Not Implement

            /* Convert color type */
            // Not Implement
            rknn_input input;
            if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeUint8) {
                input.type = RKNN_TENSOR_UINT8;
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeFp32) {
                input.type = RKNN_TENSOR_FLOAT32;
            } else {
                PRINT_E("Unsupported input type.")
                return kRetErr;
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
    if (ret < 0){
        PRINT_E("rknn_run fail! ret=%d", ret)
        return kRetErr;
    }
    return kRetOk;
}

int32_t InferenceHelperRKNN::Process(std::vector<OutputTensorInfo> &output_tensor_info_list) {

    if (output_tensor_info_list.size() != rk_io_num_.n_output) {
        PRINT_E("The outputs quantity is inconsistent")
        return kRetErr;
    }
    auto ret = rknn_run(net_, NULL);
    if (ret < 0){
        PRINT_E("rknn_run fail! ret=%d", ret)
        return kRetErr;
    }

    for (size_t index = 0; index < output_tensor_info_list.size(); index++) {
        auto &output_tensor = output_tensor_info_list[index];
        rknn_output output;
        if (output_tensor.tensor_type == TensorInfo::kTensorTypeFp32) {
            output.want_float = 1;
            INSPIRE_LOGD("want_float=1");
        }
        output.is_prealloc = 0;
        output_tensors_.push_back(output);
//        output.want_float = 1;  // float
    }

    ret = rknn_outputs_get(net_, output_tensor_info_list.size(), output_tensors_.data(), NULL);
    if (ret < 0){
        PRINT_E("rknn_run fail! ret=%d", ret)
        return kRetErr;
    }
    for (size_t index = 0; index < output_tensor_info_list.size(); index++) {
        auto &output_tensor = output_tensor_info_list[index];
        auto &output = output_tensors_[index];
        output_tensor.data = output.buf;
        float* outBlob = (float* )output.buf;
//        output_tensor.
    }


    return kRetOk;
}

std::vector<std::string> InferenceHelperRKNN::GetInputNames() {
    return input_names_;
}

int32_t InferenceHelperRKNN::ResizeInput(const std::vector<InputTensorInfo>& input_tensor_info_list) {
    // The function is not supported
    return 0;
}


#endif // INFERENCE_HELPER_ENABLE_RKNN
