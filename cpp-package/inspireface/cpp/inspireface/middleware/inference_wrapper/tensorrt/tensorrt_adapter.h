/**
 * Created by Jingyu Yan
 * @date 2025-03-16
 */
#if ISF_ENABLE_TENSORRT
#ifndef INSPIRE_TENSORRT_ADAPTER_H
#define INSPIRE_TENSORRT_ADAPTER_H

#include <string>
#include <vector>
#include <map>

#define TENSORRT_HSUCCEED 0
#define TENSORRT_HFAIL -1
#define TENSORRT_FORWARD_FAILED -2

/**
 * @brief The TensorRT adapter is used for inference
 */
class TensorRTAdapter {
public:
    /**
     * @brief inference mode (abeyance)
     */
    enum class InferenceMode {
        FP32,  ///< FP32 precision inference
        FP16,  ///< FP16 precision inference
        INT8   ///< INT8 precision inference (requires calibration)
    };

    /**
     * @brief output shape mapping
     */
    typedef std::map<std::string, std::vector<int>> TensorShapesMap;

    /**
     * @brief default constructor
     */
    TensorRTAdapter();

    /**
     * @brief destructor
     */
    ~TensorRTAdapter();

    /**
     * @brief read model from file
     * @param enginePath engine file path
     * @return 0 means success, -1 means failure
     */
    int32_t readFromFile(const std::string &enginePath);

    /**
     * @brief read model from binary data
     * @param model_data binary model data
     * @param model_size model data size
     * @return 0 means success, -1 means failure
     */
    int32_t readFromBin(void *model_data, unsigned int model_size);

    /**
     * @brief read model from file static method
     * @param enginePath engine file path
     * @return TensorRTAdapter instance
     */
    static TensorRTAdapter readNetFrom(const std::string &enginePath);

    /**
     * @brief read model from binary data static method
     * @param model_data binary model data
     * @return TensorRTAdapter instance
     */
    static TensorRTAdapter readNetFromBin(const std::vector<char> &model_data);

    /**
     * @brief get all input tensor names
     * @return input tensor names list
     */
    std::vector<std::string> getInputNames() const;

    /**
     * @brief get all output tensor names
     * @return output tensor names list
     */
    std::vector<std::string> getOutputNames() const;

    /**
     * @brief get input tensor shape by name
     * @param name tensor name
     * @return shape vector
     */
    std::vector<int> getInputShapeByName(const std::string &name);

    /**
     * @brief get output tensor shape by name
     * @param name tensor name
     * @return shape vector
     */
    std::vector<int> getOutputShapeByName(const std::string &name);

    /**
     * @brief set input data
     * @param inputName input tensor name
     * @param data input data
     */
    void setInput(const char *inputName, const void *data);

    /**
     * @brief set dynamic batch size
     * @param batchSize batch size
     * @return 0 means success, -1 means failure
     */
    int32_t setBatchSize(int batchSize);

    /**
     * @brief forward inference
     * @return 0 means success, -1 means failure
     */
    int32_t forward();

    /**
     * @brief get output data
     * @param nodeName output tensor name
     * @return output data pointer
     */
    const void *getOutput(const char *nodeName);

    /**
     * @brief get output data and convert to float type
     * @param nodeName output tensor name
     * @return output data pointer
     */
    std::vector<float> getOutputAsFloat(const char *nodeName);

    /**
     * @brief get inference time (ms)
     * @return inference time
     */
    double getInferenceTime() const;

    /**
     * @brief set inference mode
     * @param mode inference mode
     */
    void setInferenceMode(InferenceMode mode);

    /**
     * @brief set CUDA stream
     * @param stream CUDA stream
     */
    void setCudaStream(void *streamPtr);

    /**
     * @brief print model info
     */
    void printModelInfo() const;

    /**
     * @brief set CUDA device
     * @param deviceId CUDA device id
     */
    void setDevice(int32_t deviceId);

private:
    // use PIMPL pattern to hide implementation details
    class Impl;
    Impl *pImpl;

    // output shape cache
    TensorShapesMap m_inputShapes;
    TensorShapesMap m_outputShapes;
};

#endif  // INSPIRE_TENSORRT_ADAPTER_H
#endif  // ISF_ENABLE_TENSORRT