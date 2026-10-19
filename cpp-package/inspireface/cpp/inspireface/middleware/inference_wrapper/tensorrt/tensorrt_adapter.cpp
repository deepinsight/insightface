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
#include <limits>
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

// TensorRT adapter implementation class
class TensorRTAdapter::Impl {
public:
    Impl() : m_inferenceMode(TensorRTAdapter::InferenceMode::FP32), m_deviceId(0), m_inferenceTime(0.0) {
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
        resetNetwork();
        releaseOwnedStream();
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
        resetNetwork();
        if (initDevice() != TENSORRT_HSUCCEED) {
            return TENSORRT_HFAIL;
        }
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
        if (!m_streamConfigured) {
            cudaStream_t stream = nullptr;
            if (!checkCuda(cudaStreamCreate(&stream), "failed to create CUDA stream")) {
                resetNetwork();
                return TENSORRT_HFAIL;
            }
            m_stream = stream;
            m_ownStream = true;
            m_streamConfigured = true;
        }

        // pre-allocate device memory
        if (allocateDeviceMemory() != TENSORRT_HSUCCEED) {
            resetNetwork();
            return TENSORRT_HFAIL;
        }
        return TENSORRT_HSUCCEED;
    }

    // allocate device memory
    int32_t allocateDeviceMemory() {
        // allocate device memory for each input and output tensor
        for (const auto &name : m_inputNames) {
            nvinfer1::Dims dims = m_engine->getTensorShape(name.c_str());
            nvinfer1::DataType dtype = m_engine->getTensorDataType(name.c_str());
            size_t size = getMemorySize(dims, dtype);

            if (size == 0 && hasDynamicDimension(dims)) {
                m_inputShapes[name] = dimsToVector(dims);
                continue;
            }
            if (size == 0 || ensureDeviceBuffer(name, size) != TENSORRT_HSUCCEED) {
                INSPIRE_LOGE("[TensorRT error] invalid or unresolved input shape for %s", name.c_str());
                return TENSORRT_HFAIL;
            }

            // store shape information
            m_inputShapes[name] = dimsToVector(dims);
        }

        for (const auto &name : m_outputNames) {
            nvinfer1::Dims dims = m_engine->getTensorShape(name.c_str());
            nvinfer1::DataType dtype = m_engine->getTensorDataType(name.c_str());
            size_t size = getMemorySize(dims, dtype);

            if (size == 0 && hasDynamicDimension(dims)) {
                m_outputShapes[name] = dimsToVector(dims);
                continue;
            }
            if (size == 0 || ensureDeviceBuffer(name, size) != TENSORRT_HSUCCEED) {
                INSPIRE_LOGE("[TensorRT error] invalid or unresolved output shape for %s", name.c_str());
                return TENSORRT_HFAIL;
            }

            // Save shape information
            m_outputShapes[name] = dimsToVector(dims);
        }

        return TENSORRT_HSUCCEED;
    }

    // set input data
    int32_t setInput(const char *inputName, const void *data) {
        if (!inputName || !data || !m_engine || !m_streamConfigured) {
            INSPIRE_LOGE("[TensorRT error] invalid input or uninitialized adapter");
            return TENSORRT_HFAIL;
        }
        auto it = m_deviceBuffers.find(inputName);
        if (it == m_deviceBuffers.end()) {
            INSPIRE_LOGE("[TensorRT error] invalid input name: %s", inputName);
            return TENSORRT_HFAIL;
        }

        nvinfer1::Dims dims = m_context ? m_context->getTensorShape(inputName) : m_engine->getTensorShape(inputName);
        nvinfer1::DataType dtype = m_engine->getTensorDataType(inputName);
        size_t size = getMemorySize(dims, dtype);
        if (size == 0 || size > m_deviceBufferSizes[inputName]) {
            INSPIRE_LOGE("[TensorRT error] invalid input buffer size for %s", inputName);
            return TENSORRT_HFAIL;
        }

        // Copies and inference use the same stream, so enqueue ordering guarantees
        // correctness without a synchronization point for every input tensor.
        return checkCuda(cudaMemcpyAsync(it->second, data, size, cudaMemcpyHostToDevice, m_stream),
                         "failed to copy input to device")
                 ? TENSORRT_HSUCCEED
                 : TENSORRT_HFAIL;
    }

    // set batch size (only for models with dynamic shapes)
    int32_t setBatchSize(int batchSize) {
        if (batchSize <= 0 || m_inputNames.empty() || !m_context || !m_engine)
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

                nvinfer1::Dims resolvedDims = m_context->getTensorShape(name.c_str());
                size_t size = getMemorySize(resolvedDims, m_engine->getTensorDataType(name.c_str()));
                if (size == 0 || ensureDeviceBuffer(name, size) != TENSORRT_HSUCCEED) {
                    INSPIRE_LOGE("[TensorRT error] failed to resize input buffer for %s", name.c_str());
                    return TENSORRT_HFAIL;
                }
            }
        }

        for (const auto &name : m_outputNames) {
            nvinfer1::Dims resolvedDims = m_context->getTensorShape(name.c_str());
            size_t size = getMemorySize(resolvedDims, m_engine->getTensorDataType(name.c_str()));
            if (size == 0 || ensureDeviceBuffer(name, size) != TENSORRT_HSUCCEED) {
                INSPIRE_LOGE("[TensorRT error] failed to resize output buffer for %s", name.c_str());
                return TENSORRT_HFAIL;
            }
            m_outputShapes[name] = dimsToVector(resolvedDims);
        }

        return TENSORRT_HSUCCEED;
    }

    // forward inference
    int32_t forward() {
        if (!m_context || !m_engine || !m_streamConfigured) {
            return TENSORRT_HFAIL;
        }

        // check if all tensors are bound to addresses
        for (const auto &name : m_inputNames) {
            auto buffer = m_deviceBuffers.find(name);
            if (buffer == m_deviceBuffers.end() || !buffer->second || !m_context->setTensorAddress(name.c_str(), buffer->second)) {
                INSPIRE_LOGE("[TensorRT error] failed to set input tensor %s address", name.c_str());
                return TENSORRT_FORWARD_FAILED;
            }
        }

        for (const auto &name : m_outputNames) {
            auto buffer = m_deviceBuffers.find(name);
            if (buffer == m_deviceBuffers.end() || !buffer->second || !m_context->setTensorAddress(name.c_str(), buffer->second)) {
                INSPIRE_LOGE("[TensorRT error] failed to set output tensor %s address", name.c_str());
                return TENSORRT_FORWARD_FAILED;
            }
        }

        // record start time - use high precision timing
        auto start = std::chrono::high_resolution_clock::now();

        // forward inference
        bool status = m_context->enqueueV3(m_stream);
        if (!status) {
            return TENSORRT_FORWARD_FAILED;
        }

        // synchronize CUDA stream
        if (!checkCuda(cudaStreamSynchronize(m_stream), "failed to synchronize inference stream")) {
            return TENSORRT_FORWARD_FAILED;
        }

        // record end time
        auto end = std::chrono::high_resolution_clock::now();

        // calculate duration (microseconds) then convert to milliseconds, keep high precision
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        m_inferenceTime = duration_us.count() / 1000.0;

        return TENSORRT_HSUCCEED;
    }

    // get output data
    const void *getOutput(const char *nodeName) {
        if (!nodeName || !m_context || !m_engine || !m_streamConfigured) {
            return nullptr;
        }
        auto it = m_deviceBuffers.find(nodeName);
        if (it != m_deviceBuffers.end()) {
            nvinfer1::Dims dims = m_context->getTensorShape(nodeName);
            nvinfer1::DataType dtype = m_engine->getTensorDataType(nodeName);
            size_t size = getMemorySize(dims, dtype);
            if (size == 0 || size > m_deviceBufferSizes[nodeName]) {
                INSPIRE_LOGE("[TensorRT error] invalid output buffer size for %s", nodeName);
                return nullptr;
            }

            // copy output data from device to host
            m_hostOutputBuffers[nodeName].resize(size);

            if (!checkCuda(cudaMemcpyAsync(m_hostOutputBuffers[nodeName].data(), it->second, size, cudaMemcpyDeviceToHost, m_stream),
                           "failed to copy output to host") ||
                !checkCuda(cudaStreamSynchronize(m_stream), "failed to synchronize output copy")) {
                return nullptr;
            }

            m_outputShapes[nodeName] = dimsToVector(dims);

            return m_hostOutputBuffers[nodeName].data();
        }
        return nullptr;
    }

    // get output data and convert to float type vector
    std::vector<float> getOutputAsFloat(const char *nodeName) {
        std::vector<float> result;
        if (!nodeName || !m_context || !m_engine || !m_streamConfigured) {
            return result;
        }
        auto it = m_deviceBuffers.find(nodeName);
        if (it != m_deviceBuffers.end()) {
            nvinfer1::Dims dims = m_context->getTensorShape(nodeName);
            nvinfer1::DataType dtype = m_engine->getTensorDataType(nodeName);

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

            size_t byteSize = getMemorySize(dims, dtype);
            if (byteSize == 0 || byteSize > m_deviceBufferSizes[nodeName]) {
                return result;
            }
            size_t numElements = byteSize / elementSize;

            // allocate temporary buffer
            std::vector<unsigned char> buffer(byteSize);

            // copy data from device memory to host memory
            if (!checkCuda(cudaMemcpyAsync(buffer.data(), it->second, buffer.size(), cudaMemcpyDeviceToHost, m_stream),
                           "failed to copy output to host") ||
                !checkCuda(cudaStreamSynchronize(m_stream), "failed to synchronize output copy")) {
                return {};
            }

            m_outputShapes[nodeName] = dimsToVector(dims);

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
                default:
                    return {};
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
    int32_t setCudaStream(void *streamPtr) {
        if (m_streamConfigured && !checkCuda(cudaStreamSynchronize(m_stream), "failed to synchronize previous CUDA stream")) {
            return TENSORRT_HFAIL;
        }
        if (releaseOwnedStream() != TENSORRT_HSUCCEED) {
            return TENSORRT_HFAIL;
        }

        // Copy the handle value. Never retain or destroy the caller's handle
        // storage, and never destroy the borrowed CUDA stream itself.
        m_stream = streamPtr ? *static_cast<cudaStream_t *>(streamPtr) : nullptr;
        m_ownStream = false;
        m_streamConfigured = true;
        return TENSORRT_HSUCCEED;
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
    bool checkCuda(cudaError_t error, const char *operation) const {
        if (error == cudaSuccess) {
            return true;
        }
        INSPIRE_LOGE("[CUDA error] %s: %s", operation, cudaGetErrorString(error));
        return false;
    }

    int32_t ensureDeviceBuffer(const std::string &name, size_t size) {
        auto sizeIt = m_deviceBufferSizes.find(name);
        if (sizeIt != m_deviceBufferSizes.end() && sizeIt->second >= size) {
            return TENSORRT_HSUCCEED;
        }

        void *newBuffer = nullptr;
        if (!checkCuda(cudaMalloc(&newBuffer, size), "failed to allocate device buffer")) {
            return TENSORRT_HFAIL;
        }

        auto bufferIt = m_deviceBuffers.find(name);
        if (bufferIt != m_deviceBuffers.end() && bufferIt->second) {
            if (!checkCuda(cudaFree(bufferIt->second), "failed to release old device buffer")) {
                cudaFree(newBuffer);
                return TENSORRT_HFAIL;
            }
        }
        m_deviceBuffers[name] = newBuffer;
        m_deviceBufferSizes[name] = size;
        return TENSORRT_HSUCCEED;
    }

    void resetNetwork() {
        if (m_streamConfigured) {
            checkCuda(cudaStreamSynchronize(m_stream), "failed to synchronize stream during cleanup");
        }

        m_context.reset();
        for (auto &pair : m_deviceBuffers) {
            if (pair.second) {
                checkCuda(cudaFree(pair.second), "failed to release device buffer");
            }
        }
        m_deviceBuffers.clear();
        m_deviceBufferSizes.clear();
        m_hostOutputBuffers.clear();
        m_engine.reset();
        m_runtime.reset();
        m_inputNames.clear();
        m_outputNames.clear();
        m_inputShapes.clear();
        m_outputShapes.clear();
        m_inferenceTime = 0.0;
    }

    int32_t releaseOwnedStream() {
        if (!m_ownStream) {
            return TENSORRT_HSUCCEED;
        }

        cudaStream_t stream = m_stream;
        m_stream = nullptr;
        m_ownStream = false;
        m_streamConfigured = false;
        return checkCuda(cudaStreamDestroy(stream), "failed to destroy owned CUDA stream") ? TENSORRT_HSUCCEED : TENSORRT_HFAIL;
    }

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
        size_t elementSize = 0;
        switch (dtype) {
            case nvinfer1::DataType::kFLOAT:
                elementSize = 4;
                break;
            case nvinfer1::DataType::kHALF:
                elementSize = 2;
                break;
            case nvinfer1::DataType::kINT8:
                elementSize = 1;
                break;
            case nvinfer1::DataType::kINT32:
                elementSize = 4;
                break;
            case nvinfer1::DataType::kBOOL:
                elementSize = 1;
                break;
            default:
                return 0;
        }

        size_t elements = 1;
        for (int i = 0; i < dims.nbDims; ++i) {
            if (dims.d[i] <= 0 || elements > std::numeric_limits<size_t>::max() / static_cast<size_t>(dims.d[i])) {
                return 0;
            }
            elements *= static_cast<size_t>(dims.d[i]);
        }
        if (elements > std::numeric_limits<size_t>::max() / elementSize) {
            return 0;
        }
        return elements * elementSize;
    }

    bool hasDynamicDimension(const nvinfer1::Dims &dims) const {
        for (int i = 0; i < dims.nbDims; ++i) {
            if (dims.d[i] < 0) {
                return true;
            }
        }
        return false;
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

    cudaStream_t m_stream{nullptr};
    bool m_ownStream{false};
    bool m_streamConfigured{false};

    int32_t m_deviceId{0};

    std::vector<std::string> m_inputNames;
    std::vector<std::string> m_outputNames;
    std::map<std::string, void *> m_deviceBuffers;
    std::map<std::string, size_t> m_deviceBufferSizes;
    std::map<std::string, std::vector<unsigned char>> m_hostOutputBuffers;

    std::map<std::string, std::vector<int>> m_inputShapes;
    std::map<std::string, std::vector<int>> m_outputShapes;

    TensorRTAdapter::InferenceMode m_inferenceMode;
    double m_inferenceTime;
};

// implement TensorRTAdapter methods
TensorRTAdapter::TensorRTAdapter() : pImpl(new Impl()) {}

TensorRTAdapter::TensorRTAdapter(TensorRTAdapter &&other) noexcept : pImpl(other.pImpl) {
    other.pImpl = nullptr;
}

TensorRTAdapter &TensorRTAdapter::operator=(TensorRTAdapter &&other) noexcept {
    if (this != &other) {
        delete pImpl;
        pImpl = other.pImpl;
        other.pImpl = nullptr;
    }
    return *this;
}

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

int32_t TensorRTAdapter::setInput(const char *inputName, const void *data) {
    return pImpl ? pImpl->setInput(inputName, data) : TENSORRT_HFAIL;
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

int32_t TensorRTAdapter::setCudaStream(void *streamPtr) {
    return pImpl ? pImpl->setCudaStream(streamPtr) : TENSORRT_HFAIL;
}

void TensorRTAdapter::printModelInfo() const {
    pImpl->printModelInfo();
}

void TensorRTAdapter::setDevice(int32_t deviceId) {
    pImpl->setDevice(deviceId);
}
#endif  // ISF_ENABLE_TENSORRT
