//
// Created by Tunm-Air13 on 2022/10/10.
//

#ifndef MAGIC_GESTURES_RKNN_ADAPTER_H
#define MAGIC_GESTURES_RKNN_ADAPTER_H

#include <iostream>
#include "opencv2/opencv.hpp"
#include "rknn_api.h"
#include "data_type.h"
#include "log.h"


/**
 * @brief Function to get RKNN data type string.
 * @param type Data type
 * @return String representation of the data type
 */
inline const char *get_type_string_(rknn_tensor_type type) {
    switch (type) {
        case RKNN_TENSOR_FLOAT32:
            return "FP32";
        case RKNN_TENSOR_FLOAT16:
            return "FP16";
        case RKNN_TENSOR_INT8:
            return "INT8";
        case RKNN_TENSOR_UINT8:
            return "UINT8";
        case RKNN_TENSOR_INT16:
            return "INT16";
        default:
            return "UNKNOW";
    }
}

/**
 * @brief Function to get quantization type string.
 * @param type Quantization type
 * @return String representation of the quantization type
 */
inline const char *get_qnt_type_string_(rknn_tensor_qnt_type type) {
    switch (type) {
        case RKNN_TENSOR_QNT_NONE:
            return "NONE";
        case RKNN_TENSOR_QNT_DFP:
            return "DFP";
        case RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC:
            return "AFFINE";
        default:
            return "UNKNOW";
    }
}

/**
 * @brief Function to print tensor attributes.
 * @param attr Tensor attributes
 */
inline void print_tensor_attr_(const rknn_tensor_attr &attr) {
    printf("  n_dims:%d \n", attr.n_dims);
    printf("  [ ");
    for (int i = 0; i < attr.n_dims; i++) {
        printf(" %d ", attr.dims[i]);
    }
    printf("] \n");
    printf("  size:%d \n", attr.size);
    printf("  n_elems:%d \n", attr.n_elems);
    printf("  scale:%f \n", attr.scale);
    printf("  name:%s \n", attr.name);
}

/**
 * @brief Function to load data from a file.
 * @param fp File pointer
 * @param ofst Offset
 * @param sz Size
 * @return Pointer to loaded data
 */
inline unsigned char *load_data_(FILE *fp, size_t ofst, size_t sz) {
    unsigned char *data;
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

    data = (unsigned char *) malloc(sz);
    if (data == NULL) {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

/**
 * @brief Function to load a model from a file.
 * @param filename Model file name
 * @param model_size Pointer to store model size
 * @return Pointer to loaded model data
 */
inline unsigned char *load_model_(const char *filename, int *model_size) {
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp) {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data_(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

/**
 * @brief Status of RKNN inference execution
 * @ingroup NeuralNetwork
 */
enum Status {
    SUCCESS = 0,                ///< Executed successfully
    ERROR_SHAPE_MATCH = 1,      ///< Execution error. tensor shapes don't match
    ERROR_DATA_ORDER = 2        ///< Execution error, tensor data sorting error
};

/**
 * @brief RKNN Neural Network Inference Adapter
 * @details Customizable general-purpose RKNN inference adapter wrapper class, suitable for use in tasks requiring neural network inference with RKNN
 * @ingroup NeuralNetwork
 */
class RKNNAdapter {
public:

    RKNNAdapter(const RKNNAdapter &) = delete;
    RKNNAdapter &operator=(const RKNNAdapter &) = delete;
    RKNNAdapter() = default;

    /**
     * @brief Manually initialize
     * @details Initialize the RKNN model and allocate memory for creating the inference engine session
     * @param model_path Path to the RKNN model
     * @return Initialization result
     */
    int Initialize(const char *model_path) {
        /* Create the neural network */
        int model_data_size = 0;
        model_data = load_model_(model_path, &model_data_size);
        load_ = true;
        int ret = rknn_init(&rk_ctx_, model_data, model_data_size, 0);
//        INSPIRE_LOG_INFO("RKNN Init ok.");
        if (ret < 0) {
            INSPIRE_LOGE("rknn_init fail! ret=%d", ret);
            return -1;
        }
        run_ = true;

        return init_();
    }


    /**
     * @brief Manually initialize
     * @details Initialize the RKNN model using model data and its size, and allocate memory for creating the inference engine session
     * @param model_data Pointer to the model data
     * @param model_size Size of the model data
     * @return Initialization result
     */
    int Initialize(const unsigned char* model_data, const unsigned int model_size) {
        /* Create the neural network */
        INSPIRE_LOGD("The neural network is being initialized...");
        int ret = rknn_init(&rk_ctx_, (void *) model_data, model_size, 0);

        if (ret < 0) {
            INSPIRE_LOGE("rknn_init fail! ret=%d", ret);
            return -1;
        }
        run_ = true;

        return init_();
    }

    /**
     * @brief Get the size of the input image Tensor
     * @param index Index of the input layer
     * @return Dimensions information composed of various sizes
     */
    std::vector<int> GetInputTensorSize(const int &index) {
        std::vector<int> dims(input_attrs_[index].dims,
                              input_attrs_[index].dims +
                              input_attrs_[index].n_dims);
        return dims;
    }

    /**
     * @brief Get the size of the output image Tensor
     * @param index Index of the output layer
     * @return Dimensions information composed of various sizes
     */
    std::vector<unsigned long> GetOutputTensorSize(const int &index) {
//        std::cout << "output_attrs_[index].n_dims:" << output_attrs_[index].n_dims << std::endl;
        std::vector<unsigned long> dims(output_attrs_[index].dims,
                                        output_attrs_[index].dims +
                                        output_attrs_[index].n_dims);
        return dims;
    }

    /**
     * @brief Get the length of the output Tensor
     * @param index Index of the output layer
     * @return Length information
     */
    int GetOutputTensorLen(const int &index) {
        std::vector<unsigned long> tensor_size_out = GetOutputTensorSize(index);
        int size = 1;
        for (auto &one: tensor_size_out) size *= one;
        return size;
    }

    /**
     * @brief Set the data stream for the input layer
     * @param index Index of the input layer
     * @param data Image data in the form of an OpenCV Mat
     * @return Input status
     */
    Status SetInputData(const int index, const cv::Mat &data) {
        if (data.type() != CV_8UC3) {
            INSPIRE_LOGE("error: input data required CV_8UC3");
        }
        if (index < input_tensors_.size()) {
            input_tensors_[index].index = 0;
            input_tensors_[index].type = RKNN_TENSOR_UINT8;
            input_tensors_[index].size = data.cols * data.rows * data.channels();
            input_tensors_[index].fmt = RKNN_TENSOR_NHWC;
            input_tensors_[index].buf = data.data;
            input_tensors_[index].pass_through = 0;
        } else {
            INSPIRE_LOGE("error: assert index < len");
        }
        return SUCCESS;
    }

    /**
     * @brief Set the data stream for the input layer
     * @param index Index of the input layer
     * @param data Pointer to the input data
     * @param width Width of the input data
     * @param height Height of the input data
     * @param channels Number of channels in the input data
     * @param type Type of the input data (default: RKNN_TENSOR_UINT8)
     * @param format Format of the input data (default: RKNN_TENSOR_NHWC)
     * @return Input status
     */
    Status SetInputData(const int index, void* data, int width, int height, int channels,
                        rknn_tensor_type type = RKNN_TENSOR_UINT8,
                        rknn_tensor_format format = RKNN_TENSOR_NHWC) {
        if (index < input_tensors_.size()) {
            input_tensors_[index].index = 0;
            input_tensors_[index].type = type;
            input_tensors_[index].size = width * height * channels;
            input_tensors_[index].fmt = format;
            input_tensors_[index].buf = data;
            input_tensors_[index].pass_through = 0;
        } else {
            INSPIRE_LOGE("error: assert index < len");
        }
        return SUCCESS;
    }

    /**
     * @brief Execute neural network inference
     * @details This step should be executed after input data has been provided to the input layer. This step is time-consuming.
     * @return Inference status result
     */
    int RunModel() {
//        INSPIRE_LOGD("set input");
        int ret = rknn_inputs_set(rk_ctx_, rk_io_num_.n_input, input_tensors_.data());
        if (ret < 0)
            INSPIRE_LOGE("rknn_input fail! ret=%d", ret);

        for (int i = 0; i < rk_io_num_.n_output; i++) {
            output_tensors_[i].want_float = outputs_want_float_;
        }

//        INSPIRE_LOGD("rknn_run");
        ret = rknn_run(rk_ctx_, nullptr);
        if (ret < 0) {
            INSPIRE_LOGE("rknn_run fail! ret=%d", ret);
            return -1;
        }

        ret = rknn_outputs_get(rk_ctx_, rk_io_num_.n_output, output_tensors_.data(), NULL);
        if (ret < 0) {
            INSPIRE_LOGE("rknn_init fail! ret=%d", ret);
            exit(0);
        }
        return ret;
    }

    /**
     * @brief Get the data from the output layer
     * @param index Index of the output layer
     * @return Pointer to the output data
     */
    const float *GetOutputData(const int index) {
        return (float *) (output_tensors_[index].buf);
    }

    /**
     * @brief Get the output data buffer for a specific output layer
     * @param index Index of the output layer
     * @return Pointer to the output data buffer
     */
    void *GetOutputFlow(const int index) {
        return output_tensors_[index].buf;
    }

    /**
     * @brief Get the output layer data (UINT8)
     * @details Returns the output layer UInt8 format data after inference, which can only be obtained by inference first
     * @param index Output level index
     * @return Returns a pointer to the output data
     */
    u_int8_t *GetOutputDataU8(const int index) {
        return (uint8_t *) (output_tensors_[index].buf);
    }

    int32_t ReleaseOutputs() {
        auto ret = rknn_outputs_release(rk_ctx_, rk_io_num_.n_output, output_tensors_.data());
        return ret;
    }

    /**
     * @brief Resize the input tensor with a specified name to a new shape
     * @details This function is currently not implemented.
     * @param index_name Name of the input tensor
     * @param shape New shape for the input tensor
     */
    void ResizeInputTensor(const std::string &index_name,
                           const std::vector<int> &shape) {
        // No implementation
    }

    /**
     * @brief Check the size (Placeholder)
     * @details This function is currently not implemented.
     */
    void CheckSize() {
        // No implementation
    }

    /**
     * @brief Get the number of output layers
     * @details This function is typically used in multi-task or multi-output neural networks.
     * @return The number of output layers.
     */
    size_t GetOutputsNum() const {
        return rk_io_num_.n_output;
    }

    /**
     * @brief Get a reference to the vector of output tensors
     * @return A reference to the vector of output tensors.
    */
    std::vector<rknn_output> &GetOutputTensors() {
        return output_tensors_;
    }

    /**
     * @brief Get a reference to a vector containing information about the attributes of the output tensors
     * @details The attributes include output sizes, data types, and other relevant information.
     * @return A reference to the vector containing output tensor attributes.
     */
    std::vector<rknn_tensor_attr> &GetOutputTensorAttr() {
        return output_attrs_;
    }

    ~RKNNAdapter() {
        Release();
    }

    /**
     * @brief Release resources
     * @details Release all resources in memory, typically called in the destructor
     */
    void Release() {
        if (run_){
            rknn_destroy(rk_ctx_);
            if (load_) {
                free(model_data);
            }
            run_ = false;
        }
    }

    /**
     * @brief Set the output mode to support float format
     * @details Depending on the encoding style, some post-processing may use UInt8 format for encoding and decoding
     * @param outputsWantFloat 0 or 1
     */
    void setOutputsWantFloat(int outputsWantFloat) {
        outputs_want_float_ = outputsWantFloat;
    }

private:
    /**
     * initial
     * @return
     */
    int init_() {
        rknn_sdk_version version;
        int ret = rknn_query(rk_ctx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
        if (ret < 0) {
            INSPIRE_LOGE("rknn_init fail! ret=%d", ret);
            return -1;
        }
        INSPIRE_LOGD("sdk version: %s driver version: %s", version.api_version, version.drv_version);

        ret = rknn_query(rk_ctx_, RKNN_QUERY_IN_OUT_NUM, &rk_io_num_,
                         sizeof(rk_io_num_));

        if (ret != RKNN_SUCC) {
            INSPIRE_LOGE("rknn_query ctx fail! ret=%d", ret);
            return -1;
        }

        INSPIRE_LOGD("models input num: %d, output num: %d", rk_io_num_.n_input, rk_io_num_.n_output);


//        spdlog::trace("input tensors: ");
        input_attrs_.resize(rk_io_num_.n_input);
        output_attrs_.resize(rk_io_num_.n_output);
        input_tensors_.resize(rk_io_num_.n_input);
        output_tensors_.resize(rk_io_num_.n_output);

        for (int i = 0; i < rk_io_num_.n_input; ++i) {
            memset(&input_attrs_[i], 0, sizeof(input_attrs_[i]));
            memset(&input_tensors_[i], 0, sizeof(input_tensors_[i]));
            input_attrs_[i].index = i;
            ret = rknn_query(rk_ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs_[i]),
                             sizeof(rknn_tensor_attr));

            INSPIRE_LOGD("input node index %d", i);
            int channel = 3;
            int width = 0;
            int height = 0;
            if (input_attrs_[i].fmt == RKNN_TENSOR_NCHW) {
                INSPIRE_LOGD("models is NCHW input fmt");
                width = input_attrs_[i].dims[0];
                height = input_attrs_[i].dims[1];
            } else {
                INSPIRE_LOGD("models is NHWC input fmt");
                width = input_attrs_[i].dims[1];
                height = input_attrs_[i].dims[2];
            }
            INSPIRE_LOGD("models input height=%d, width=%d, channel=%d", height, width, channel);
//            print_tensor_attr_(input_attrs_);
            if (ret != RKNN_SUCC) {
                INSPIRE_LOGE("rknn_query fail! ret=%d", ret);
                return -1;
            }
        }

//        printf("[debug]models input num: %d, output num: %d\n", rk_io_num_.n_input, rk_io_num_.n_output);
        for (int i = 0; i < rk_io_num_.n_output; ++i) {
            memset(&output_attrs_[i], 0, sizeof(output_attrs_[i]));
            memset(&output_tensors_[i], 0, sizeof(output_tensors_[i]));
            output_attrs_[i].index = i;
            ret = rknn_query(rk_ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]),
                             sizeof(rknn_tensor_attr));

            if (output_attrs_[i].qnt_type != RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC ||
                output_attrs_[i].type != RKNN_TENSOR_UINT8) {
                INSPIRE_LOGW("The Demo required for a Affine asymmetric u8 quantized rknn models, but output quant type is %s, output "
                        "data type is %s",
                        get_qnt_type_string_(output_attrs_[i].qnt_type), get_type_string_(output_attrs_[i].type));
//                return -1;
            }
//            print_tensor_attr_(output_attrs_[i]);


//            rknn_tensor_attr rknn_attr;
//            memset(&rknn_attr, 0, sizeof(rknn_tensor_attr));
//
//            ret = rknn_query(rk_ctx_, RKNN_QUERY_OUTPUT_ATTR, &rknn_attr,
//                             sizeof(rknn_tensor_attr));
//            printf("output node index %d \n", i);
//            print_tensor_attr_(rknn_attr);


            if (ret != RKNN_SUCC) {
                INSPIRE_LOGE("rknn_query fail! ret=%d", ret);
                return -1;
            }
        }

        return ret;
    }

private:
    rknn_context rk_ctx_;               ///< The context manager for RKNN.
    rknn_input_output_num rk_io_num_;   ///< The number of input and output streams in RKNN.

    std::vector<rknn_tensor_attr> input_attrs_;   ///< Attributes of input tensors.
    std::vector<rknn_tensor_attr> output_attrs_;  ///< Attributes of output tensors.
    std::vector<rknn_input> input_tensors_;       ///< Input data for the neural network.
    std::vector<rknn_output> output_tensors_;     ///< Output data from the neural network.

    int outputs_want_float_ = 0;        ///< Flag to indicate support for floating-point output.

    std::vector<int> tensor_shape_;     ///< The shape of input tensors.
    int width_;                         ///< The width of input data (typically for images).
    int height_;                        ///< The height of input data (typically for images).
    bool run_status_;                   ///< Flag to indicate the execution status of the neural network.

    unsigned char *model_data;          ///< Pointer to the model's data stream.
    bool load_;
    bool run_;
};


#endif //MAGIC_GESTURES_RKNN_ADAPTER_H
