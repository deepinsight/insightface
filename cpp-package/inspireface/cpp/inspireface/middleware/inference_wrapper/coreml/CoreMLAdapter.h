/**
 * Created by Jingyu Yan
 * @date 2025-02-01
 */

#ifndef INSPIRE_FACE_COREML_ADAPTER_H
#define INSPIRE_FACE_COREML_ADAPTER_H
#include <string>
#include <vector>
#include <map>
#ifdef __OBJC__
@class MLModel;
@protocol MLFeatureProvider;
#else
typedef void MLModel;
typedef void MLFeatureProvider;
#endif

#define COREML_HSUCCEED 0
#define COREML_HFAIL -1
#define COREML_FORWARD_FAILED -2

/**
 * @brief CoreML adapter for inference
 */
class CoreMLAdapter {
public:
    /**
     * @brief Inference mode
     */
    enum class InferenceMode {
        CPU,  ///< CPU inference
        GPU,  ///< GPU inference
        ANE   ///< Automatic selection, ANE first
    };

    /**
     * @brief Output shapes map
     */
    typedef std::map<std::string, std::vector<int>> OutputShapesMap;

    CoreMLAdapter();
    ~CoreMLAdapter();

    /**
     * @brief Read model from file
     * @param modelPath model path
     * @return 0 if success, -1 if failed
     */
    int32_t readFromFile(const std::string &modelPath);

    static CoreMLAdapter readNetFrom(const std::string &modelPath);

    /**
     * @brief Not implemented
     */
    static CoreMLAdapter readNetFromBin(const std::vector<char> &model_data);

    std::vector<std::string> getInputNames() const;
    std::vector<std::string> getOutputNames() const;
    std::vector<int> getInputShapeByName(const std::string &name);
    std::vector<int> getOutputShapeByName(const std::string &name);

    void setInput(const char *inputName, const char *data);
    int32_t forward();
    const char *getOutput(const char *nodeName);
    const std::vector<int> &getOutputShapeByName(const std::string &name) const;

    /**
     * @brief Set inference mode
     * @param mode inference mode
     */
    void setInferenceMode(InferenceMode mode);

    void printModelInfo() const;

private:
    class Impl;
    Impl *pImpl;

    OutputShapesMap m_outputShapes;
};

#endif  // INSPIRE_FACE_COREML_ADAPTER_H
