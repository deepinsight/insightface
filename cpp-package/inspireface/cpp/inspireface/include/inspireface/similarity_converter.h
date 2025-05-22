#ifndef SIMILARITY_CONVERTER_H
#define SIMILARITY_CONVERTER_H

#include <iostream>
#include <cmath>
#include <mutex>
#include "data_type.h"

#define SIMILARITY_CONVERTER_UPDATE_CONFIG(config) inspire::SimilarityConverter::getInstance().updateConfig(config)
#define SIMILARITY_CONVERTER_RUN(cosine) inspire::SimilarityConverter::getInstance().convert(cosine)
#define SIMILARITY_CONVERTER_GET_CONFIG() inspire::SimilarityConverter::getInstance().getConfig()
#define SIMILARITY_CONVERTER_GET_RECOMMENDED_COSINE_THRESHOLD() inspire::SimilarityConverter::getInstance().getRecommendedCosineThreshold()
#define SIMILARITY_CONVERTER_SET_RECOMMENDED_COSINE_THRESHOLD(threshold) \
    inspire::SimilarityConverter::getInstance().setRecommendedCosineThreshold(threshold)

namespace inspire {

struct SimilarityConverterConfig {
    double threshold = 0.48;   // Similarity threshold (e.g. 0.48 or 0.32)
    double middleScore = 0.6;  // Target score at threshold (e.g. 0.6)
    double steepness = 8.0;    // Steepness of the curve
    double outputMin = 0.01;   // Minimum value of output range
    double outputMax = 1.0;    // Maximum value of output range
};

class INSPIRE_API_EXPORT SimilarityConverter {
private:
    SimilarityConverterConfig config;
    double outputScale;              // Scale of output range
    double bias;                     // Sigmoid bias
    mutable std::mutex configMutex;  // Mutex for protecting config updates

    // Recommended cosine threshold
    float recommendedCosineThreshold = 0.48;

    static SimilarityConverter* instance;
    static std::mutex instanceMutex;

    // Update internal calculation parameters
    void updateParameters() {
        outputScale = config.outputMax - config.outputMin;
        bias = -std::log((config.outputMax - config.middleScore) / (config.middleScore - config.outputMin));
    }

public:
    // Get global singleton instance
    static SimilarityConverter& getInstance() {
        std::lock_guard<std::mutex> lock(instanceMutex);
        if (instance == nullptr) {
            instance = new SimilarityConverter();
        }
        return *instance;
    }

    // Allow external creation of new instances
    explicit SimilarityConverter(const SimilarityConverterConfig& config = SimilarityConverterConfig()) : config(config) {
        updateParameters();
    }

    // Prevent copying
    SimilarityConverter(const SimilarityConverter&) = delete;
    SimilarityConverter& operator=(const SimilarityConverter&) = delete;

    // Update configuration (thread-safe)
    void updateConfig(const SimilarityConverterConfig& newConfig) {
        std::lock_guard<std::mutex> lock(configMutex);
        config = newConfig;
        updateParameters();
    }

    // Get current configuration (thread-safe)
    SimilarityConverterConfig getConfig() const {
        std::lock_guard<std::mutex> lock(configMutex);
        return config;
    }

    // Convert similarity (thread-safe)
    template <typename T>
    double convert(T cosine) {
        std::lock_guard<std::mutex> lock(configMutex);
        // Calculate shifted input
        double shiftedInput = config.steepness * (static_cast<double>(cosine) - config.threshold);
        // Apply sigmoid function
        double sigmoid = 1.0 / (1.0 + std::exp(-shiftedInput - bias));
        // Map to output range
        return sigmoid * outputScale + config.outputMin;
    }

    // Clean up singleton instance
    static void destroyInstance() {
        std::lock_guard<std::mutex> lock(instanceMutex);
        if (instance != nullptr) {
            delete instance;
            instance = nullptr;
        }
    }

    // Get recommended cosine threshold
    float getRecommendedCosineThreshold() const {
        return recommendedCosineThreshold;
    }

    // Set recommended cosine threshold
    void setRecommendedCosineThreshold(float threshold) {
        recommendedCosineThreshold = threshold;
    }

    ~SimilarityConverter() = default;
};  // class SimilarityConverter

}  // namespace inspire

#endif  // SIMILARITY_CONVERTER_H