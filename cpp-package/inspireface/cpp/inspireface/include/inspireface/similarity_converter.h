#ifndef SIMILARITY_CONVERTER_H
#define SIMILARITY_CONVERTER_H

#include <cmath>
#include <mutex>

#include "data_type.h"

#define ISF_SIMILARITY_CONVERTER_COHERENT_STATE 1

#define SIMILARITY_CONVERTER_UPDATE_CONFIG(config) inspire::SimilarityConverter::getInstance().updateConfig(config)
#define SIMILARITY_CONVERTER_RUN(cosine) inspire::SimilarityConverter::getInstance().convert(cosine)
#define SIMILARITY_CONVERTER_GET_CONFIG() inspire::SimilarityConverter::getInstance().getConfig()
#define SIMILARITY_CONVERTER_GET_RECOMMENDED_COSINE_THRESHOLD() inspire::SimilarityConverter::getInstance().getRecommendedCosineThreshold()
#define SIMILARITY_CONVERTER_SET_RECOMMENDED_COSINE_THRESHOLD(threshold) \
    inspire::SimilarityConverter::getInstance().setRecommendedCosineThreshold(threshold)

namespace inspire {

struct SimilarityConverterConfig {
    double threshold = 0.48;
    double middleScore = 0.6;
    double steepness = 8.0;
    double outputMin = 0.01;
    double outputMax = 1.0;
};

struct SimilarityConverterState {
    SimilarityConverterConfig config;
    double outputScale = 0.99;
    double bias = 0.0;
    float recommendedCosineThreshold = 0.48f;
};

class INSPIRE_API_EXPORT SimilarityConverter {
private:
    // Keep the original field order: SimilarityConverter is an exported C++
    // type, and changing this layout would break existing binary consumers.
    SimilarityConverterConfig config;
    double outputScale;
    double bias;
    mutable std::mutex configMutex;
    float recommendedCosineThreshold = 0.48f;

    // Retained for binary compatibility with clients built against the
    // previous inline singleton implementation.
    static SimilarityConverter *instance;
    static std::mutex instanceMutex;

    void updateParameters() {
        outputScale = config.outputMax - config.outputMin;
        bias = -std::log((config.outputMax - config.middleScore) / (config.middleScore - config.outputMin));
    }

    static bool CalculateParameters(const SimilarityConverterConfig &candidate, double &output_scale, double &candidate_bias) {
        if (!IsConfigValid(candidate)) {
            return false;
        }
        output_scale = candidate.outputMax - candidate.outputMin;
        candidate_bias = -std::log((candidate.outputMax - candidate.middleScore) / (candidate.middleScore - candidate.outputMin));
        return std::isfinite(output_scale) && std::isfinite(candidate_bias);
    }

public:
    static SimilarityConverter &getInstance() {
        std::lock_guard<std::mutex> lock(instanceMutex);
        if (instance == nullptr) {
            instance = new SimilarityConverter();
        }
        return *instance;
    }

    explicit SimilarityConverter(const SimilarityConverterConfig &initial_config = SimilarityConverterConfig())
    : config(IsConfigValid(initial_config) ? initial_config : SimilarityConverterConfig()) {
        updateParameters();
    }

    SimilarityConverter(const SimilarityConverter &) = delete;
    SimilarityConverter &operator=(const SimilarityConverter &) = delete;

    static bool IsConfigValid(const SimilarityConverterConfig &candidate) {
        return std::isfinite(candidate.threshold) && std::isfinite(candidate.middleScore) && std::isfinite(candidate.steepness) &&
               std::isfinite(candidate.outputMin) && std::isfinite(candidate.outputMax) && candidate.steepness > 0.0 &&
               candidate.outputMin < candidate.middleScore && candidate.middleScore < candidate.outputMax;
    }

    bool updateConfig(const SimilarityConverterConfig &new_config) {
        double new_output_scale = 0.0;
        double new_bias = 0.0;
        if (!CalculateParameters(new_config, new_output_scale, new_bias)) {
            return false;
        }
        std::lock_guard<std::mutex> lock(configMutex);
        config = new_config;
        outputScale = new_output_scale;
        bias = new_bias;
        return true;
    }

    bool updateConfigAndRecommendedThreshold(const SimilarityConverterConfig &new_config, float recommended_threshold) {
        double new_output_scale = 0.0;
        double new_bias = 0.0;
        if (!std::isfinite(recommended_threshold) || !CalculateParameters(new_config, new_output_scale, new_bias)) {
            return false;
        }
        std::lock_guard<std::mutex> lock(configMutex);
        config = new_config;
        outputScale = new_output_scale;
        bias = new_bias;
        recommendedCosineThreshold = recommended_threshold;
        return true;
    }

    SimilarityConverterState getState() const {
        std::lock_guard<std::mutex> lock(configMutex);
        SimilarityConverterState state;
        state.config = config;
        state.outputScale = outputScale;
        state.bias = bias;
        state.recommendedCosineThreshold = recommendedCosineThreshold;
        return state;
    }

    SimilarityConverterConfig getConfig() const {
        return getState().config;
    }

    template <typename T>
    double convert(T cosine) const {
        const SimilarityConverterState state = getState();
        const double shifted_input = state.config.steepness * (static_cast<double>(cosine) - state.config.threshold);
        const double sigmoid = 1.0 / (1.0 + std::exp(-shifted_input - state.bias));
        return sigmoid * state.outputScale + state.config.outputMin;
    }

    static void destroyInstance() {
        std::lock_guard<std::mutex> lock(instanceMutex);
        if (instance != nullptr) {
            delete instance;
            instance = nullptr;
        }
    }

    float getRecommendedCosineThreshold() const {
        return getState().recommendedCosineThreshold;
    }

    bool setRecommendedCosineThreshold(float threshold) {
        if (!std::isfinite(threshold)) {
            return false;
        }
        std::lock_guard<std::mutex> lock(configMutex);
        recommendedCosineThreshold = threshold;
        return true;
    }

    ~SimilarityConverter() = default;
};

}  // namespace inspire

#endif  // SIMILARITY_CONVERTER_H
