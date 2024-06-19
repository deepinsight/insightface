#ifndef INSPIREFACE_OMNI_INFERENACE__
#define INSPIREFACE_OMNI_INFERENACE__
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <memory>

class XOutputData {
public:

    XOutputData() : size(0), data(nullptr) {}

    std::vector<float> CopyToFloatArray() {
        if (!buffer.empty()) {
            return buffer;
        }
        
        std::vector<float> floatArray;
        floatArray.resize(size);
        std::memcpy(floatArray.data(), data, size * sizeof(float));
        return floatArray;
    }

public:
    size_t size;
    float *data;                // Use pointer
    std::vector<float> buffer;  // Use copy
};

typedef std::vector<XOutputData> XOutputDataList;

class XTransform {
public:

    XTransform() : swap_color(false) {
        std::fill(std::begin(normalize.mean), std::end(normalize.mean), 0.0f);
        std::fill(std::begin(normalize.norm), std::end(normalize.norm), 1.0f);
    }

    bool swap_color;

    struct {
        float mean[3];
        float norm[3];
    } normalize;
};

class XInputData {
public:
    XInputData() : nchw(false), bgr(false), height(0), width(0), channel(0), data(nullptr) {}

public:
    bool nchw;
    bool bgr;
    int32_t height;
    int32_t width;
    int32_t channel;
    uint8_t *data;
};


typedef enum {
    xEngineMNN,
    xEngineRKNN,
} EngineType;

class InferenceAdapter {
public:
    enum {
        xRetOk = 0,
        xRetErr = -1,
    };

    typedef enum {
        xDefaultCPU,
        xMNNCuda,
    } SpecialBackend;

public:
    virtual ~InferenceAdapter() {};
    virtual int32_t SetNumThreads(const int32_t num_threads) = 0;
    virtual int32_t Initialize(const std::string& model_filename, const XTransform& transform, const std::string& input_name, const std::vector<std::string> &outputs_name) = 0;
    virtual int32_t Initialize(char* model_buffer, int model_size, const std::string& input_name, const XTransform& transform, const std::vector<std::string> &outputs_name) = 0;
    virtual int32_t Finalize(void) = 0;
    virtual int32_t SetInputsData(const std::vector<XInputData>& batch, ) = 0;
    virtual int32_t Forward(std::vector<XOutputDataList> &batch_outputs) = 0;

    virtual int32_t ResizeInputs() = 0;

    virtual int32_t SetSpecialBackend(SpecialBackend backend) {
        special_backend_ = backend;
        return xRetOk;
    };

protected:
    EngineType engine_type_;
    SpecialBackend special_backend_ = xDefaultCPU;
    
};



#endif  // INSPIREFACE_OMNI_INFERENACE__