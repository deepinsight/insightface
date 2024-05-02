//
// Created by Tunm-Air13 on 2023/9/11.
//
#pragma once
#ifndef HYPERFACEREPO_FEATUREBLOCKOPENCV_H
#define HYPERFACEREPO_FEATUREBLOCKOPENCV_H
#include "feature_hub/features_block/feature_block.h"

namespace inspire {

/**
 * @class FeatureBlockOpenCV
 * @brief Class derived from FeatureBlock for managing facial features using OpenCV.
 *
 * This class provides an implementation of FeatureBlock using OpenCV's Mat data structure
 * for storing and manipulating facial features.
 */
class INSPIRE_API FeatureBlockOpenCV : public FeatureBlock{
public:

    /**
     * @brief Constructor for FeatureBlockOpenCV.
     * @param features_max Maximum number of features that can be stored.
     * @param feature_length Length of each feature vector.
     */
    explicit FeatureBlockOpenCV(int32_t features_max = 512, int32_t feature_length = 512);

    /**
     * @brief Searches for the nearest feature in the block to a given query feature.
     * @param queryFeature Query feature vector.
     * @param searchResult SearchResult structure to store the search results.
     * @return int32_t Status of the search operation.
     */
    int32_t SearchNearest(const std::vector<float>& queryFeature, SearchResult &searchResult) override;

    /**
     * @brief Search the first k features in a block that are closest to a given query feature.
     * @param topK Maximum number of similarities
     * @param searchResults outputs
     * */
    int32_t SearchTopKNearest(const std::vector<float>& queryFeature, size_t topK, std::vector<SearchResult> &searchResults) override;

    /**
     * @brief Retrieves a feature from the feature block.
     * @param row Index of the feature to retrieve.
     * @param feature Vector to store the retrieved feature.
     * @return int32_t Status of the retrieval operation.
     */
    int32_t GetFeature(int row, std::vector<float> &feature) override;

protected:
    /**
     * @brief Adds a feature to the feature block without thread safety.
     *        This method should be overridden in derived classes.
     * @param feature Vector of floats representing the feature.
     * @param tag String tag associated with the feature.
     * @param customId Custom identifier for the feature.
     * @return int32_t Status of the feature addition.
     */
    int32_t UnsafeAddFeature(const std::vector<float> &feature, const std::string &tag, int32_t customId) override;

    /**
     * @brief Registers a feature at a specific index in the feature block without thread safety.
     *        This method should be overridden in derived classes.
     * @param rowToUpdate Index at which to register the new feature.
     * @param feature Feature vector to be registered.
     * @param tag Tag associated with the feature.
     * @param customId Custom identifier for the feature.
     * @return int32_t Status of the feature registration.
     */
    int32_t UnsafeDeleteFeature(int rowToDelete) override;

    /**
     * @brief Deletes a feature from the feature block without thread safety.
     *        This method should be overridden in derived classes.
     * @param rowToDelete Index of the feature to be deleted.
     * @return int32_t Status of the feature deletion.
     */
    int32_t UnsafeUpdateFeature(int rowToUpdate, const std::vector<float> &newFeature, const std::string &tag, int32_t customId) override;

    /**
     * @brief Updates a feature in the feature block without thread safety.
     *        This method should be overridden in derived classes.
     * @param rowToUpdate Index of the feature to be updated.
     * @param newFeature New feature vector to replace the old one.
     * @param tag New tag for the updated feature.
     * @param customId Custom identifier for the updated feature.
     * @return int32_t Status of the feature update.
     */
    int32_t UnsafeRegisterFeature(int rowToUpdate, const std::vector<float> &feature, const std::string &tag, int32_t customId) override;


public:
    /**
     * @brief Prints the size of the feature matrix.
     */
    void PrintMatrixSize() override;

    /**
     * @brief Prints the entire feature matrix.
     */
    void PrintMatrix() override;

private:

    cv::Mat m_feature_matrix_;      ///< Matrix for storing feature vectors.

};

}   // namespace hyper


#endif //HYPERFACEREPO_FEATUREBLOCKOPENCV_H
