#ifndef INFERENCE_WRAPPER_
#define INFERENCE_WRAPPER_

#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <limits>
#include <memory>

class TensorInfo {
public:
    enum {
        TensorTypeNone,
        TensorTypeUint8,
        TensorTypeInt8,
        TensorTypeFp32,
        TensorTypeInt32,
        TensorTypeInt64,
    };

public:
    TensorInfo() : name(""), id(-1), tensor_type(TensorTypeNone), is_nchw(true) {}
    TensorInfo(const TensorInfo&) = default;
    TensorInfo(TensorInfo&&) noexcept = default;
    TensorInfo& operator=(const TensorInfo&) = default;
    TensorInfo& operator=(TensorInfo&&) noexcept = default;
    ~TensorInfo() = default;

    int32_t GetElementNum() const {
        if (tensor_dims.empty()) {
            return -1;
        }
        int32_t element_num = 1;
        for (const auto& dim : tensor_dims) {
            if (dim <= 0 || element_num > std::numeric_limits<int32_t>::max() / dim) {
                return -1;
            }
            element_num *= dim;
        }
        return element_num;
    }

    int32_t GetBatch() const {
        if (tensor_dims.size() <= 0)
            return -1;
        return tensor_dims[0];
    }

    int32_t GetChannel() const {
        if (is_nchw) {
            if (tensor_dims.size() <= 1)
                return -1;
            return tensor_dims[1];
        } else {
            if (tensor_dims.size() <= 3)
                return -1;
            return tensor_dims[3];
        }
    }

    int32_t GetHeight() const {
        if (is_nchw) {
            if (tensor_dims.size() <= 2)
                return -1;
            return tensor_dims[2];
        } else {
            if (tensor_dims.size() <= 1)
                return -1;
            return tensor_dims[1];
        }
    }

    int32_t GetWidth() const {
        if (is_nchw) {
            if (tensor_dims.size() <= 3)
                return -1;
            return tensor_dims[3];
        } else {
            if (tensor_dims.size() <= 2)
                return -1;
            return tensor_dims[2];
        }
    }

public:
    std::string name;
    int32_t id;
    int32_t tensor_type;
    std::vector<int32_t> tensor_dims;
    bool is_nchw;
};

class InputTensorInfo : public TensorInfo {
public:
    enum {
        DataTypeImage,
        DataTypeBlobNhwc,  // data_ which already finished preprocess(color conversion, resize, normalize_, etc.)
        DataTypeBlobNchw,
    };

public:
    InputTensorInfo()
    : data(nullptr),
      data_type(DataTypeImage),
      image_info({-1, -1, -1, -1, -1, -1, -1, true, false}),
      normalize({0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f}) {}

    InputTensorInfo(std::string name_, int32_t tensor_type_, bool is_nchw_ = true) : InputTensorInfo() {
        name = name_;
        tensor_type = tensor_type_;
        is_nchw = is_nchw_;
    }

    ~InputTensorInfo() {}

public:
    void* data;
    int32_t data_type;

    struct {
        int32_t width;
        int32_t height;
        int32_t channel;
        int32_t crop_x;
        int32_t crop_y;
        int32_t crop_width;
        int32_t crop_height;
        bool is_bgr;  // used when channel == 3 (true: BGR, false: RGB)
        bool swap_color;
    } image_info;

    struct {
        float mean[3];
        float norm[3];
    } normalize;
};

class OutputTensorInfo : public TensorInfo {
public:
    OutputTensorInfo() : data(nullptr), quant({1.0f, 0}) {}

    OutputTensorInfo(std::string name_, int32_t tensor_type_, bool is_nchw_ = true) : OutputTensorInfo() {
        name = name_;
        tensor_type = tensor_type_;
        is_nchw = is_nchw_;
    }

    OutputTensorInfo(const OutputTensorInfo&) = default;
    OutputTensorInfo(OutputTensorInfo&&) noexcept = default;
    OutputTensorInfo& operator=(const OutputTensorInfo&) = default;
    OutputTensorInfo& operator=(OutputTensorInfo&&) noexcept = default;
    ~OutputTensorInfo() = default;

    float* GetDataAsFloat() {
        const int32_t element_num = GetElementNum();
        if (data == nullptr || element_num <= 0) {
            return nullptr;
        }
        if (tensor_type == TensorTypeUint8 || tensor_type == TensorTypeInt8) {
            data_fp32_.resize(static_cast<size_t>(element_num));
            if (tensor_type == TensorTypeUint8) {
                const auto* values = static_cast<const uint8_t*>(data);
#pragma omp parallel for
                for (int32_t i = 0; i < element_num; i++) {
                    data_fp32_[static_cast<size_t>(i)] =
                      (static_cast<int32_t>(values[i]) - quant.zero_point) * quant.scale;
                }
            } else {
                const auto* values = static_cast<const int8_t*>(data);
#pragma omp parallel for
                for (int32_t i = 0; i < element_num; i++) {
                    data_fp32_[static_cast<size_t>(i)] =
                      (static_cast<int32_t>(values[i]) - quant.zero_point) * quant.scale;
                }
            }
            return data_fp32_.data();
        } else if (tensor_type == TensorTypeFp32) {
            return static_cast<float*>(data);
        } else {
            return nullptr;
        }
    }

public:
    void* data;
    struct {
        float scale;
        int32_t zero_point;
    } quant;

private:
    std::vector<float> data_fp32_;
};

struct InferenceCacheStatistics {
    uint64_t image_process_creations = 0;
    uint64_t blob_host_tensor_creations = 0;
    uint64_t output_host_tensor_creations = 0;
};

namespace cv {
class Mat;
};

class InferenceWrapper {
public:
    enum {
        WrapperOk = 0,
        WrapperError = -1,
    };

    typedef enum {
        DEFAULT_CPU,
        MMM_CUDA,
        COREML_CPU,
        COREML_GPU,
        COREML_ANE,
        TENSORRT_CUDA,
    } SpecialBackend;

    typedef enum {
        INFER_MNN,
        INFER_RKNN,
        INFER_COREML,
        INFER_TENSORRT,
    } EngineType;

public:
    static InferenceWrapper* Create(const EngineType helper_type);

public:
    virtual ~InferenceWrapper() {}
    virtual int32_t SetNumThreads(const int32_t num_threads) = 0;
    virtual int32_t Initialize(const std::string& model_filename, std::vector<InputTensorInfo>& input_tensor_info_list,
                               std::vector<OutputTensorInfo>& output_tensor_info_list) = 0;
    virtual int32_t Initialize(char* model_buffer, int model_size, std::vector<InputTensorInfo>& input_tensor_info_list,
                               std::vector<OutputTensorInfo>& output_tensor_info_list) = 0;
    virtual int32_t Finalize(void) = 0;
    virtual int32_t PreProcess(const std::vector<InputTensorInfo>& input_tensor_info_list) = 0;
    virtual int32_t Process(std::vector<OutputTensorInfo>& output_tensor_info_list) = 0;
    virtual int32_t ParameterInitialization(std::vector<InputTensorInfo>& input_tensor_info_list,
                                            std::vector<OutputTensorInfo>& output_tensor_info_list) = 0;
    
#ifdef BATCH_FORWARD_IMPLEMENTED
    virtual int32_t PreProcessBatch(const std::vector<std::vector<InputTensorInfo>>& input_tensor_info_list) = 0;
    virtual int32_t ProcessBatch(std::vector<std::vector<OutputTensorInfo>>& output_tensor_info_list) = 0;
    virtual int32_t PostProcessBatch(std::vector<std::vector<OutputTensorInfo>>& output_tensor_info_list) = 0;
#endif

    virtual int32_t SetSpecialBackend(SpecialBackend backend) {
        special_backend_ = backend;
        return WrapperOk;
    };

    virtual int32_t SetDevice(int32_t device_id) {
        device_id_ = device_id;
        return WrapperOk;
    };

    virtual int32_t ResizeInput(const std::vector<InputTensorInfo>& input_tensor_info_list) = 0;

    virtual std::vector<std::string> GetInputNames() = 0;

    virtual InferenceCacheStatistics GetCacheStatistics() const {
        return {};
    }

    virtual void ResetCacheStatistics() {}

#ifdef ISF_ENABLE_MNN_CACHE_GUARD
    virtual void SetCacheEnabledForTesting(bool) {}
#endif

protected:
    void ConvertNormalizeParameters(InputTensorInfo& tensor_info);

    void PreProcessImage(int32_t num_thread, const InputTensorInfo& input_tensor_info, float* dst);
    void PreProcessImage(int32_t num_thread, const InputTensorInfo& input_tensor_info, uint8_t* dst);
    void PreProcessImage(int32_t num_thread, const InputTensorInfo& input_tensor_info, int8_t* dst);

    template <typename T>
    void PreProcessBlob(int32_t num_thread, const InputTensorInfo& input_tensor_info, T* dst);

protected:
    EngineType helper_type_;
    SpecialBackend special_backend_ = DEFAULT_CPU;
    int32_t device_id_ = 0;
};

#endif
