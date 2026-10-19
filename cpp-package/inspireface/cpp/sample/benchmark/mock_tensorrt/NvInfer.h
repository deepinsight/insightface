#ifndef INSPIREFACE_SAMPLE_FAKE_NVINFER_H
#define INSPIREFACE_SAMPLE_FAKE_NVINFER_H

#include <cstddef>
#include <cstring>
#include <map>
#include <string>

#include "cuda_runtime_api.h"

namespace fake_tensorrt {

extern bool dynamic_shapes;
extern bool fail_runtime;
extern bool fail_engine;
extern bool fail_context;
extern bool fail_enqueue;

}  // namespace fake_tensorrt

namespace nvinfer1 {

class ILogger {
public:
    enum class Severity { kINTERNAL_ERROR = 0, kERROR = 1, kWARNING = 2, kINFO = 3, kVERBOSE = 4 };
    virtual ~ILogger() = default;
    virtual void log(Severity severity, const char *message) noexcept = 0;
};

enum class DataType { kFLOAT, kHALF, kINT8, kINT32, kBOOL };
enum class TensorIOMode { kINPUT, kOUTPUT };

struct Dims {
    int nbDims{0};
    int d[8]{};
};

inline Dims makeDims(int batch) {
    Dims dims;
    dims.nbDims = 4;
    dims.d[0] = batch;
    dims.d[1] = 1;
    dims.d[2] = 1;
    dims.d[3] = 4;
    return dims;
}

class IExecutionContext {
public:
    virtual ~IExecutionContext() = default;
    virtual bool setInputShape(const char *name, Dims dimensions) = 0;
    virtual bool setTensorAddress(const char *name, void *data) = 0;
    virtual bool enqueueV3(cudaStream_t stream) = 0;
    virtual Dims getTensorShape(const char *name) const = 0;
};

class ICudaEngine {
public:
    virtual ~ICudaEngine() = default;
    virtual int getNbIOTensors() const = 0;
    virtual const char *getIOTensorName(int index) const = 0;
    virtual TensorIOMode getTensorIOMode(const char *name) const = 0;
    virtual Dims getTensorShape(const char *name) const = 0;
    virtual DataType getTensorDataType(const char *name) const = 0;
    virtual IExecutionContext *createExecutionContext() = 0;
    virtual int getNbLayers() const = 0;
};

class IRuntime {
public:
    virtual ~IRuntime() = default;
    virtual ICudaEngine *deserializeCudaEngine(const void *data, size_t size) = 0;
};

class FakeExecutionContext final : public IExecutionContext {
public:
    explicit FakeExecutionContext(bool dynamic) : dynamic_(dynamic), batch_(dynamic ? -1 : 1) {}

    bool setInputShape(const char *name, Dims dimensions) override {
        if (!name || std::strcmp(name, "input") != 0 || dimensions.nbDims != 4 || dimensions.d[0] <= 0) {
            return false;
        }
        batch_ = dimensions.d[0];
        return true;
    }

    bool setTensorAddress(const char *name, void *data) override {
        if (!name || !data) {
            return false;
        }
        addresses_[name] = data;
        return true;
    }

    bool enqueueV3(cudaStream_t) override {
        if (fake_tensorrt::fail_enqueue || batch_ <= 0 || !addresses_.count("input") || !addresses_.count("output")) {
            return false;
        }
        const float *input = static_cast<const float *>(addresses_["input"]);
        float *output = static_cast<float *>(addresses_["output"]);
        for (int index = 0; index < batch_ * 4; ++index) {
            output[index] = input[index] * 2.0f + 1.0f;
        }
        return true;
    }

    Dims getTensorShape(const char *) const override {
        return makeDims(batch_);
    }

private:
    bool dynamic_;
    int batch_;
    std::map<std::string, void *> addresses_;
};

class FakeCudaEngine final : public ICudaEngine {
public:
    explicit FakeCudaEngine(bool dynamic) : dynamic_(dynamic) {}

    int getNbIOTensors() const override {
        return 2;
    }

    const char *getIOTensorName(int index) const override {
        return index == 0 ? "input" : "output";
    }

    TensorIOMode getTensorIOMode(const char *name) const override {
        return std::strcmp(name, "input") == 0 ? TensorIOMode::kINPUT : TensorIOMode::kOUTPUT;
    }

    Dims getTensorShape(const char *) const override {
        return makeDims(dynamic_ ? -1 : 1);
    }

    DataType getTensorDataType(const char *) const override {
        return DataType::kFLOAT;
    }

    IExecutionContext *createExecutionContext() override {
        return fake_tensorrt::fail_context ? nullptr : new FakeExecutionContext(dynamic_);
    }

    int getNbLayers() const override {
        return 1;
    }

private:
    bool dynamic_;
};

class FakeRuntime final : public IRuntime {
public:
    ICudaEngine *deserializeCudaEngine(const void *, size_t size) override {
        if (fake_tensorrt::fail_engine || size == 0) {
            return nullptr;
        }
        return new FakeCudaEngine(fake_tensorrt::dynamic_shapes);
    }
};

inline IRuntime *createInferRuntime(ILogger &) {
    return fake_tensorrt::fail_runtime ? nullptr : new FakeRuntime();
}

}  // namespace nvinfer1

#endif
