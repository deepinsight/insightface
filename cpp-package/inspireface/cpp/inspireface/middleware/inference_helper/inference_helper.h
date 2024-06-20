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
#ifndef INFERENCE_HELPER_
#define INFERENCE_HELPER_

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <memory>

class TensorInfo {
public:
    enum {
        kTensorTypeNone,
        kTensorTypeUint8,
        kTensorTypeInt8,
        kTensorTypeFp32,
        kTensorTypeInt32,
        kTensorTypeInt64,
    };

public:
    TensorInfo()
        : name("")
        , id(-1)
        , tensor_type(kTensorTypeNone)
        , is_nchw(true)
    {}
    ~TensorInfo() {}

    int32_t GetElementNum() const
    {
        int32_t element_num = 1;
        for (const auto& dim : tensor_dims) {
            element_num *= dim;
        }
        return element_num;
    }

    int32_t GetBatch() const
    {
        if (tensor_dims.size() <= 0) return -1;
        return tensor_dims[0];
    }
    
    int32_t GetChannel() const
    {
        if (is_nchw) {
            if (tensor_dims.size() <= 1) return -1;
            return tensor_dims[1];
        } else {
            if (tensor_dims.size() <= 3) return -1;
            return tensor_dims[3];
        }
    }

    int32_t GetHeight() const
    {
        if (is_nchw) {
            if (tensor_dims.size() <= 2) return -1;
            return tensor_dims[2];
        } else {
            if (tensor_dims.size() <= 1) return -1;
            return tensor_dims[1];
        }
    }

    int32_t GetWidth() const
    {
        if (is_nchw) {
            if (tensor_dims.size() <= 3) return -1;
            return tensor_dims[3];
        } else {
            if (tensor_dims.size() <= 2) return -1;
            return tensor_dims[2];
        }
    }

public:
    std::string          name;           // [In] Set the name_ of tensor
    int32_t              id;             // [Out] Do not modify (Used in InferenceHelper)
    int32_t              tensor_type;    // [In] The type of tensor (e.g. kTensorTypeFp32)
    std::vector<int32_t> tensor_dims;    // InputTensorInfo:   [In] The dimentions of tensor. (If empty at initialize, the size is updated from model info.)
                                         // OutputTensorInfo: [Out] The dimentions of tensor is set from model information
    bool                 is_nchw;        // [IN] NCHW or NHWC
};

class InputTensorInfo : public TensorInfo {
public:
    enum {
        kDataTypeImage,
        kDataTypeBlobNhwc,  // data_ which already finished preprocess(color conversion, resize, normalize_, etc.)
        kDataTypeBlobNchw,
    };

public:
    InputTensorInfo()
        : data(nullptr)
        , data_type(kDataTypeImage)
        , image_info({ -1, -1, -1, -1, -1, -1, -1, true, false })
        , normalize({ 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f })
    {}

    InputTensorInfo(std::string name_, int32_t tensor_type_, bool is_nchw_ = true)
        : InputTensorInfo()
    {
        name = name_;
        tensor_type = tensor_type_;
        is_nchw = is_nchw_;
    }

    ~InputTensorInfo() {}

public:
    void*   data;      // [In] Set the pointer to image/blob
    int32_t data_type; // [In] Set the type of data_ (e.g. kDataTypeImage)

    struct {
        int32_t width;
        int32_t height;
        int32_t channel;
        int32_t crop_x;
        int32_t crop_y;
        int32_t crop_width;
        int32_t crop_height;
        bool    is_bgr;        // used when channel == 3 (true: BGR, false: RGB)
        bool    swap_color;
    } image_info;              // [In] used when data_type_ == kDataTypeImage

    struct {
        float mean[3];
        float norm[3];
    } normalize;              // [In] used when data_type_ == kDataTypeImage
};


class OutputTensorInfo : public TensorInfo {
public:
    OutputTensorInfo()
        : data(nullptr)
        , quant({ 1.0f, 0 })
        , data_fp32_(nullptr)
    {}

    OutputTensorInfo(std::string name_, int32_t tensor_type_, bool is_nchw_ = true)
        : OutputTensorInfo()
    {
        name = name_;
        tensor_type = tensor_type_;
        is_nchw = is_nchw_;
    }
    
    ~OutputTensorInfo() {
        if (data_fp32_ != nullptr) {
            delete[] data_fp32_;
        }
    }

    float* GetDataAsFloat() {       /* Returned pointer should be with const, but returning pointer without const is convenient to create cv::Mat */
        if (tensor_type == kTensorTypeUint8 || tensor_type == kTensorTypeInt8) {
            if (data_fp32_ == nullptr) {
                data_fp32_ = new float[GetElementNum()];
            }
            if (tensor_type == kTensorTypeUint8) {
#pragma omp parallel
                for (int32_t i = 0; i < GetElementNum(); i++) {
                    const uint8_t* val_uint8 = static_cast<const uint8_t*>(data);
                    float val_float = (val_uint8[i] - quant.zero_point) * quant.scale;
                    data_fp32_[i] = val_float;
                }
            } else {
#pragma omp parallel
                for (int32_t i = 0; i < GetElementNum(); i++) {
                    const int8_t* val_int8 = static_cast<const int8_t*>(data);
                    float val_float = (val_int8[i] - quant.zero_point) * quant.scale;
                    data_fp32_[i] = val_float;
                }
            }
            return data_fp32_;
        } else if (tensor_type == kTensorTypeFp32) {
            return static_cast<float*>(data);
        } else {
            return nullptr;
        }
    }

public:
    void* data;     // [Out] Pointer to the output data_
    struct {
        float   scale;
        int32_t zero_point;
    } quant;        // [Out] Parameters for dequantization (convert uint8 to float)

private:
    float* data_fp32_;
};


namespace cv {
    class Mat;
};

class InferenceHelper {
public:
    enum {
        kRetOk = 0,
        kRetErr = -1,
    };

    typedef enum {
        kDefaultCPU,
        kMnnCuda,
    } SpecialBackend;

    typedef enum {
        kOpencv,
        kOpencvGpu,
        kTensorflowLite,
        kTensorflowLiteXnnpack,
        kTensorflowLiteGpu,
        kTensorflowLiteEdgetpu,
        kTensorflowLiteNnapi,
        kTensorrt,
        kNcnn,
        kNcnnVulkan,
        kMnn,
        kSnpe,
        kArmnn,
        kNnabla,
        kNnablaCuda,
        kOnnxRuntime,
        kOnnxRuntimeCuda,
        kLibtorch,
        kLibtorchCuda,
        kTensorflow,
        kTensorflowGpu,
        kSample,
        kRknn,
    } HelperType;

public:
    static InferenceHelper* Create(const HelperType helper_type);
    static void PreProcessByOpenCV(const InputTensorInfo& input_tensor_info, bool is_nchw, cv::Mat& img_blob);   // use this if the selected inference engine doesn't support pre-process

public:
    virtual ~InferenceHelper() {}
    virtual int32_t SetNumThreads(const int32_t num_threads) = 0;
    virtual int32_t SetCustomOps(const std::vector<std::pair<const char*, const void*>>& custom_ops) = 0;
    virtual int32_t Initialize(const std::string& model_filename, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list) = 0;
    virtual int32_t Initialize(char* model_buffer, int model_size, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list) = 0;
    virtual int32_t Finalize(void) = 0;
    virtual int32_t PreProcess(const std::vector<InputTensorInfo>& input_tensor_info_list) = 0;
    virtual int32_t Process(std::vector<OutputTensorInfo>& output_tensor_info_list) = 0;
    virtual int32_t ParameterInitialization(std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list) = 0;

    virtual int32_t SetSpecialBackend(SpecialBackend backend) {
        special_backend_ = backend;
        return kRetOk;
    };

    virtual int32_t ResizeInput(const std::vector<InputTensorInfo>& input_tensor_info_list) = 0;

    virtual std::vector<std::string> GetInputNames() = 0;

protected:
    void ConvertNormalizeParameters(InputTensorInfo& tensor_info);

    void PreProcessImage(int32_t num_thread, const InputTensorInfo& input_tensor_info, float* dst);
    void PreProcessImage(int32_t num_thread, const InputTensorInfo& input_tensor_info, uint8_t* dst);
    void PreProcessImage(int32_t num_thread, const InputTensorInfo& input_tensor_info, int8_t* dst);

    template<typename T>
    void PreProcessBlob(int32_t num_thread, const InputTensorInfo& input_tensor_info, T *dst);

protected:
    HelperType helper_type_;
    SpecialBackend special_backend_ = kDefaultCPU;
};

#endif
