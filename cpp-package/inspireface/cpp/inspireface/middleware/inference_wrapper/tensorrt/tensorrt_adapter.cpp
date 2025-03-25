/**
 * Created by Jingyu Yan
 * @date 2025-03-16
 */
#if ISF_ENABLE_TENSORRT
#include "tensorrt_adapter.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <memory>
#include <cstring>
#include <cuda_fp16.h>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <log.h>
#include <isf_check.h>

// define specific deleters for TensorRT objects
struct TRTRuntimeDeleter {
    void operator()(nvinfer1::IRuntime *runtime) const {
        if (runtime)
            delete runtime;
    }
};

struct TRTEngineDeleter {
    void operator()(nvinfer1::ICudaEngine *engine) const {
        if (engine)
            delete engine;
    }
};

struct TRTContextDeleter {
    void operator()(nvinfer1::IExecutionContext *context) const {
        if (context)
            delete context;
    }
};

// custom deleter for CUDA stream
struct CUDAStreamDeleter {
    void operator()(cudaStream_t *stream) const {
        if (stream) {
            cudaStreamDestroy(*stream);
            delete stream;
        }
    }
};

// custom Logger class, inherit from TensorRT's ILogger
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            INSPIRE_LOGI("[TensorRT] %s", msg);
        }
    }
};

// read model file to memory
static std::vector<char> readModelFile(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        INSPIRE_LOGE("failed to open model file: %s", filename.c_str());
        return {};
    }

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        INSPIRE_LOGE("failed to read model file: %s", filename.c_str());
        return {};
    }

    return buffer;
}

// CUDA error check macro
#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t error = call;                                       \
        if (error != cudaSuccess) {                                     \
            INSPIRE_LOGE("[CUDA error] %s", cudaGetErrorString(error)); \
            return TENSORRT_HFAIL;                                      \
        }                                                               \
    } while (0)

// TensorRT adapter implementation class
class TensorRTAdapter::Impl {
public:
    Impl() : m_ownStream(false), m_inferenceMode(TensorRTAdapter::InferenceMode::FP32), m_deviceId(0) {
        // create Logger with smart pointer
        m_logger = std::make_unique<TRTLogger>();
    }

    int32_t initDevice() {
        cudaError_t error = cudaSetDevice(m_deviceId);
        if (error != cudaSuccess) {
            INSPIRE_LOGE("[CUDA error] The device fails to use cuda:%d, %s", m_deviceId, cudaGetErrorString(error));
            return TENSORRT_HFAIL;
        }
        return TENSORRT_HSUCCEED;
    }

    void setDevice(int32_t deviceId) {
        m_deviceId = deviceId;
    }

    ~Impl() {
        // release resources - device memory needs to be released manually
        for (auto &pair : m_deviceBuffers) {
            if (pair.second) {
                cudaFree(pair.second);
            }
        }
        m_deviceBuffers.clear();

        // smart pointers will handle the release of other resources
    }

    int32_t readFromFile(const std::string &enginePath) {
        // read serialized engine file
        std::vector<char> modelData = readModelFile(enginePath);
        if (modelData.empty()) {
            return TENSORRT_HFAIL;
        }

        return deserializeEngine(modelData);
    }

    int32_t readFromBin(const std::vector<char> &model_data) {
        if (model_data.empty()) {
            return TENSORRT_HFAIL;
        }

        return deserializeEngine(model_data);
    }

    int32_t readFromBin(void *model_data, unsigned int model_size) {
        if (!model_data || model_size == 0) {
            INSPIRE_LOGE("[TensorRT error] invalid model data or size");
            return TENSORRT_HFAIL;
        }

        // convert memory data to vector to reuse the existing deserializeEngine method
        std::vector<char> modelBuffer(static_cast<char *>(model_data), static_cast<char *>(model_data) + model_size);

        return deserializeEngine(modelBuffer);
    }

    // create and deserialize engine
    int32_t deserializeEngine(const std::vector<char> &modelData) {
        // init device
        initDevice();
        // create runtime
        m_runtime.reset(nvinfer1::createInferRuntime(*m_logger));
        if (!m_runtime) {
            INSPIRE_LOGE("[TensorRT error] failed to create TensorRT runtime");
            return TENSORRT_HFAIL;
        }

        // deserialize engine
        m_engine.reset(m_runtime->deserializeCudaEngine(modelData.data(), modelData.size()));
        if (!m_engine) {
            INSPIRE_LOGE("[TensorRT error] failed to deserialize engine");
            return TENSORRT_HFAIL;
        }

        // create execution context
        m_context.reset(m_engine->createExecutionContext());
        if (!m_context) {
            INSPIRE_LOGE("[TensorRT error] failed to create execution context");
            return TENSORRT_HFAIL;
        }

        // get all input and output tensor names
        int numIoTensors = m_engine->getNbIOTensors();
        for (int i = 0; i < numIoTensors; ++i) {
            const char *name = m_engine->getIOTensorName(i);
            nvinfer1::TensorIOMode mode = m_engine->getTensorIOMode(name);
            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                m_inputNames.push_back(name);
            } else {
                m_outputNames.push_back(name);
            }
        }

        // initialize CUDA stream
        if (!m_stream) {
            cudaStream_t *stream = new cudaStream_t;
            CHECK_CUDA(cudaStreamCreate(stream));
            m_stream.reset(stream);
            m_ownStream = true;
        }

        // pre-allocate device memory
        return allocateDeviceMemory();
    }

    // allocate device memory
    int32_t allocateDeviceMemory() {
        // allocate device memory for each input and output tensor
        for (const auto &name : m_inputNames) {
            nvinfer1::Dims dims = m_engine->getTensorShape(name.c_str());
            nvinfer1::DataType dtype = m_engine->getTensorDataType(name.c_str());
            size_t size = getMemorySize(dims, dtype);

            void *buffer = nullptr;
            CHECK_CUDA(cudaMalloc(&buffer, size));
            m_deviceBuffers[name] = buffer;

            // store shape information
            m_inputShapes[name] = dimsToVector(dims);
        }

        for (const auto &name : m_outputNames) {
            nvinfer1::Dims dims = m_engine->getTensorShape(name.c_str());
            nvinfer1::DataType dtype = m_engine->getTensorDataType(name.c_str());
            size_t size = getMemorySize(dims, dtype);

            void *buffer = nullptr;
            CHECK_CUDA(cudaMalloc(&buffer, size));
            m_deviceBuffers[name] = buffer;

            // Save shape information
            m_outputShapes[name] = dimsToVector(dims);
        }

        return TENSORRT_HSUCCEED;
    }

    // set input data
    void setInput(const char *inputName, const void *data) {
        auto it = m_deviceBuffers.find(inputName);
        if (it != m_deviceBuffers.end()) {
            nvinfer1::Dims dims = m_engine->getTensorShape(inputName);
            nvinfer1::DataType dtype = m_engine->getTensorDataType(inputName);
            size_t size = getMemorySize(dims, dtype);

            // copy data from host to device
            cudaMemcpyAsync(it->second, data, size, cudaMemcpyHostToDevice, *m_stream.get());
            cudaStreamSynchronize(*m_stream.get());  // add synchronization to ensure data is fully copied
        }
    }

    // set batch size (only for models with dynamic shapes)
    int32_t setBatchSize(int batchSize) {
        if (m_inputNames.empty())
            return TENSORRT_HFAIL;

        for (const auto &name : m_inputNames) {
            nvinfer1::Dims dims = m_engine->getTensorShape(name.c_str());
            if (dims.nbDims > 0) {
                nvinfer1::Dims newDims = dims;
                newDims.d[0] = batchSize;

                if (!m_context->setInputShape(name.c_str(), newDims)) {
                    INSPIRE_LOGE("[TensorRT error] failed to set input shape for %s", name.c_str());
                    return TENSORRT_HFAIL;
                }

                // update shape information
                m_inputShapes[name] = dimsToVector(newDims);
            }
        }

        return TENSORRT_HSUCCEED;
    }

    // forward inference
    int32_t forward() {
        if (!m_context || !m_engine) {
            return TENSORRT_HFAIL;
        }

        // check if all tensors are bound to addresses
        for (const auto &name : m_inputNames) {
            if (!m_context->setTensorAddress(name.c_str(), m_deviceBuffers[name])) {
                INSPIRE_LOGE("[TensorRT error] failed to set input tensor %s address", name.c_str());
                return TENSORRT_FORWARD_FAILED;
            }
        }

        for (const auto &name : m_outputNames) {
            if (!m_context->setTensorAddress(name.c_str(), m_deviceBuffers[name])) {
                INSPIRE_LOGE("[TensorRT error] failed to set output tensor %s address", name.c_str());
                return TENSORRT_FORWARD_FAILED;
            }
        }

        // record start time - use high precision timing
        auto start = std::chrono::high_resolution_clock::now();

        // forward inference
        bool status = m_context->enqueueV3(*m_stream.get());

        // synchronize CUDA stream
        cudaStreamSynchronize(*m_stream.get());

        // record end time
        auto end = std::chrono::high_resolution_clock::now();

        // calculate duration (microseconds) then convert to milliseconds, keep high precision
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        m_inferenceTime = duration_us.count() / 1000.0;

        return status ? TENSORRT_HSUCCEED : TENSORRT_FORWARD_FAILED;
    }

    // get output data
    const void *getOutput(const char *nodeName) {
        auto it = m_deviceBuffers.find(nodeName);
        if (it != m_deviceBuffers.end()) {
            nvinfer1::Dims dims = m_context->getTensorShape(nodeName);
            nvinfer1::DataType dtype = m_engine->getTensorDataType(nodeName);
            size_t size = getMemorySize(dims, dtype);

            // copy output data from device to host
            if (m_hostOutputBuffers.find(nodeName) == m_hostOutputBuffers.end()) {
                m_hostOutputBuffers[nodeName].resize(size);
            }

            cudaMemcpyAsync(m_hostOutputBuffers[nodeName].data(), it->second, size, cudaMemcpyDeviceToHost, *m_stream.get());
            cudaStreamSynchronize(*m_stream.get());

            return m_hostOutputBuffers[nodeName].data();
        }
        return nullptr;
    }

    // get output data and convert to float type vector
    std::vector<float> getOutputAsFloat(const char *nodeName) {
        std::vector<float> result;
        auto it = m_deviceBuffers.find(nodeName);
        if (it != m_deviceBuffers.end()) {
            nvinfer1::Dims dims = m_context->getTensorShape(nodeName);
            nvinfer1::DataType dtype = m_engine->getTensorDataType(nodeName);

            // calculate total number of elements
            size_t numElements = 1;
            for (int i = 0; i < dims.nbDims; ++i) {
                numElements *= dims.d[i];
            }

            // allocate buffer of appropriate size based on data type
            size_t elementSize = 0;
            switch (dtype) {
                case nvinfer1::DataType::kFLOAT:
                    elementSize = sizeof(float);
                    break;
                case nvinfer1::DataType::kHALF:
                    elementSize = sizeof(half);
                    break;
                case nvinfer1::DataType::kINT8:
                    elementSize = sizeof(int8_t);
                    break;
                case nvinfer1::DataType::kINT32:
                    elementSize = sizeof(int32_t);
                    break;
                default:
                    return result;
            }

            // allocate temporary buffer
            std::vector<unsigned char> buffer(numElements * elementSize);

            // copy data from device memory to host memory
            cudaMemcpyAsync(buffer.data(), it->second, buffer.size(), cudaMemcpyDeviceToHost, *m_stream.get());
            cudaStreamSynchronize(*m_stream.get());

            // convert to float based on data type
            result.resize(numElements);
            switch (dtype) {
                case nvinfer1::DataType::kFLOAT:
                    std::memcpy(result.data(), buffer.data(), buffer.size());
                    break;
                case nvinfer1::DataType::kHALF: {
                    const half *halfData = reinterpret_cast<const half *>(buffer.data());
                    for (size_t i = 0; i < numElements; ++i) {
                        result[i] = __half2float(halfData[i]);
                    }
                    break;
                }
                case nvinfer1::DataType::kINT8: {
                    const int8_t *int8Data = reinterpret_cast<const int8_t *>(buffer.data());
                    for (size_t i = 0; i < numElements; ++i) {
                        result[i] = static_cast<float>(int8Data[i]);
                    }
                    break;
                }
                case nvinfer1::DataType::kINT32: {
                    const int32_t *int32Data = reinterpret_cast<const int32_t *>(buffer.data());
                    for (size_t i = 0; i < numElements; ++i) {
                        result[i] = static_cast<float>(int32Data[i]);
                    }
                    break;
                }
            }
        }
        return result;
    }

    // set inference mode
    void setInferenceMode(TensorRTAdapter::InferenceMode mode) {
        m_inferenceMode = mode;
        // apply this setting during actual inference
    }

    // set CUDA stream
    void setCudaStream(void *streamPtr) {
        if (m_ownStream) {
            m_stream.reset();
            m_ownStream = false;
        }

        // create a new smart pointer instead of using reset + lambda
        cudaStream_t *streamPointer = static_cast<cudaStream_t *>(streamPtr);
        // use empty deleter, because this stream is managed by external code
        m_stream =
          std::unique_ptr<cudaStream_t, CUDAStreamDeleter>(streamPointer,
                                                           CUDAStreamDeleter()  // use default deleter, but not actually delete the external stream
          );
    }

    // print model info
    void printModelInfo() const {
        INSPIRE_LOGI("================================================");
        if (!m_engine) {
            INSPIRE_LOGE("[TensorRT error] engine not initialized");
            return;
        }
        INSPIRE_LOGI("\nengine info:");
        INSPIRE_LOGI("engine layers: %d", m_engine->getNbLayers());
        INSPIRE_LOGI("input/output tensors: %d", m_engine->getNbIOTensors());

        INSPIRE_LOGI("\ninput tensors:");
        for (const auto &name : m_inputNames) {
            nvinfer1::Dims dims = m_engine->getTensorShape(name.c_str());
            nvinfer1::DataType dtype = m_engine->getTensorDataType(name.c_str());

            INSPIRE_LOGI("name: %s, shape: (", name.c_str());
            for (int d = 0; d < dims.nbDims; ++d) {
                INSPIRE_LOGI("%d", dims.d[d]);
                if (d < dims.nbDims - 1)
                    INSPIRE_LOGI(", ");
            }
            INSPIRE_LOGI("), type: %s", getDataTypeString(dtype).c_str());
        }

        INSPIRE_LOGI("\noutput tensors:");
        for (const auto &name : m_outputNames) {
            nvinfer1::Dims dims = m_engine->getTensorShape(name.c_str());
            nvinfer1::DataType dtype = m_engine->getTensorDataType(name.c_str());

            INSPIRE_LOGI("name: %s, shape: (", name.c_str());
            for (int d = 0; d < dims.nbDims; ++d) {
                INSPIRE_LOGI("%d", dims.d[d]);
                if (d < dims.nbDims - 1)
                    INSPIRE_LOGI(", ");
            }
            INSPIRE_LOGI("), type: %s", getDataTypeString(dtype).c_str());
        }
        INSPIRE_LOGI("================================================");
    }

    // get input tensor names list
    const std::vector<std::string> &getInputNames() const {
        return m_inputNames;
    }

    // get output tensor names list
    const std::vector<std::string> &getOutputNames() const {
        return m_outputNames;
    }

    // get input tensor shape by name
    const std::vector<int> &getInputShapeByName(const std::string &name) const {
        static std::vector<int> emptyShape;
        auto it = m_inputShapes.find(name);
        return (it != m_inputShapes.end()) ? it->second : emptyShape;
    }

    // get output tensor shape by name
    const std::vector<int> &getOutputShapeByName(const std::string &name) const {
        static std::vector<int> emptyShape;
        auto it = m_outputShapes.find(name);
        return (it != m_outputShapes.end()) ? it->second : emptyShape;
    }

    // get inference time
    double getInferenceTime() const {
        return m_inferenceTime;
    }

private:
    // helper function: convert TensorRT's Dims to standard vector
    std::vector<int> dimsToVector(const nvinfer1::Dims &dims) const {
        std::vector<int> shape;
        for (int i = 0; i < dims.nbDims; ++i) {
            shape.push_back(dims.d[i]);
        }
        return shape;
    }

    // helper function: calculate memory size
    size_t getMemorySize(const nvinfer1::Dims &dims, nvinfer1::DataType dtype) const {
        size_t size = 1;
        for (int i = 0; i < dims.nbDims; ++i) {
            size *= dims.d[i];
        }

        switch (dtype) {
            case nvinfer1::DataType::kFLOAT:
                return size * 4;
            case nvinfer1::DataType::kHALF:
                return size * 2;
            case nvinfer1::DataType::kINT8:
                return size;
            case nvinfer1::DataType::kINT32:
                return size * 4;
            case nvinfer1::DataType::kBOOL:
                return size;
            default:
                return size;
        }
    }

    // helper function: get data type string representation
    std::string getDataTypeString(nvinfer1::DataType dtype) const {
        switch (dtype) {
            case nvinfer1::DataType::kFLOAT:
                return "FLOAT";
            case nvinfer1::DataType::kHALF:
                return "HALF";
            case nvinfer1::DataType::kINT8:
                return "INT8";
            case nvinfer1::DataType::kINT32:
                return "INT32";
            case nvinfer1::DataType::kBOOL:
                return "BOOL";
            default:
                return "UNKNOWN";
        }
    }

    // member variables - using smart pointers
    std::unique_ptr<TRTLogger> m_logger;
    std::unique_ptr<nvinfer1::IRuntime, TRTRuntimeDeleter> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine, TRTEngineDeleter> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext, TRTContextDeleter> m_context;

    bool m_ownStream;
    std::unique_ptr<cudaStream_t, CUDAStreamDeleter> m_stream;

    int32_t m_deviceId{0};

    std::vector<std::string> m_inputNames;
    std::vector<std::string> m_outputNames;
    std::map<std::string, void *> m_deviceBuffers;
    std::map<std::string, std::vector<unsigned char>> m_hostOutputBuffers;

    std::map<std::string, std::vector<int>> m_inputShapes;
    std::map<std::string, std::vector<int>> m_outputShapes;

    TensorRTAdapter::InferenceMode m_inferenceMode;
    double m_inferenceTime;
};

// implement TensorRTAdapter methods
TensorRTAdapter::TensorRTAdapter() : pImpl(new Impl()) {}

TensorRTAdapter::~TensorRTAdapter() {
    if (pImpl) {
        delete pImpl;
        pImpl = nullptr;
    }
}

int32_t TensorRTAdapter::readFromFile(const std::string &enginePath) {
    return pImpl->readFromFile(enginePath);
}

int32_t TensorRTAdapter::readFromBin(void *model_data, unsigned int model_size) {
    return pImpl->readFromBin(model_data, model_size);
}

TensorRTAdapter TensorRTAdapter::readNetFrom(const std::string &enginePath) {
    TensorRTAdapter adapter;
    adapter.readFromFile(enginePath);
    return adapter;
}

TensorRTAdapter TensorRTAdapter::readNetFromBin(const std::vector<char> &model_data) {
    TensorRTAdapter adapter;
    adapter.pImpl->readFromBin(model_data);
    return adapter;
}

std::vector<std::string> TensorRTAdapter::getInputNames() const {
    return pImpl->getInputNames();
}

std::vector<std::string> TensorRTAdapter::getOutputNames() const {
    return pImpl->getOutputNames();
}

std::vector<int> TensorRTAdapter::getInputShapeByName(const std::string &name) {
    return pImpl->getInputShapeByName(name);
}

std::vector<int> TensorRTAdapter::getOutputShapeByName(const std::string &name) {
    return pImpl->getOutputShapeByName(name);
}

void TensorRTAdapter::setInput(const char *inputName, const void *data) {
    pImpl->setInput(inputName, data);
}

int32_t TensorRTAdapter::setBatchSize(int batchSize) {
    return pImpl->setBatchSize(batchSize);
}

int32_t TensorRTAdapter::forward() {
    return pImpl->forward();
}

const void *TensorRTAdapter::getOutput(const char *nodeName) {
    return pImpl->getOutput(nodeName);
}

std::vector<float> TensorRTAdapter::getOutputAsFloat(const char *nodeName) {
    return pImpl->getOutputAsFloat(nodeName);
}

double TensorRTAdapter::getInferenceTime() const {
    return pImpl->getInferenceTime();
}

void TensorRTAdapter::setInferenceMode(InferenceMode mode) {
    pImpl->setInferenceMode(mode);
}

void TensorRTAdapter::setCudaStream(void *streamPtr) {
    pImpl->setCudaStream(streamPtr);
}

void TensorRTAdapter::printModelInfo() const {
    pImpl->printModelInfo();
}

void TensorRTAdapter::setDevice(int32_t deviceId) {
    pImpl->setDevice(deviceId);
}
#endif  // ISF_ENABLE_TENSORRT