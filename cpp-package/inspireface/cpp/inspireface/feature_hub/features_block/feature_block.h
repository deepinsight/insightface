//
// Created by Tunm-Air13 on 2023/9/11.
//
#pragma once
#ifndef HYPERFACEREPO_FEATUREBLOCK_H
#define HYPERFACEREPO_FEATUREBLOCK_H
#include <mutex>
#include <iostream>
#include <algorithm>
#include "data_type.h"


namespace inspire {

/**
 * @enum MatrixCore
 * @brief Enumeration for different types of matrix cores used in feature extraction.
 */
typedef enum {
    MC_NONE,           ///< C/C++ Native matrix core.
    MC_OPENCV,         ///< OpenCV Mat based matrix core.
    MC_EIGEN,          ///< Eigen3 Mat based matrix core.
} MatrixCore;

/**
 * @enum FEATURE_STATE
 * @brief Enumeration for states of feature slots in the feature block.
 */
typedef enum {
    IDLE = 0,          ///< Slot is idle.
    USED,              ///< Slot is used.
} FEATURE_STATE;


/**
 * @struct SearchResult
 * @brief Structure to store the results of a feature search.
 */
typedef struct SearchResult {
    float score = -1.0f;      ///< Score of the search result.
    int32_t index = -1;       ///< Index of the result in the feature block.
    std::string tag = "None"; ///< Tag associated with the feature.
    int32_t customId = -1;    ///< Custom identifier for the feature.
} SearchResult;

/**
 * @class FeatureBlock
 * @brief Class for managing and operating on a block of facial features.
 *
 * This class provides methods to add, delete, update, and search facial features
 * in a feature block, with thread safety using mutexes.
 */
class INSPIRE_API FeatureBlock {
public:
    static FeatureBlock* Create(const MatrixCore crop_type, int32_t features_max = 512, int32_t feature_length = 512);

public:
    /**
     * @brief Destructor for the FeatureBlock class.
     */
    virtual ~FeatureBlock() {}

    /**
     * @brief Adds a feature to the feature block.
     * @param feature Vector of floats representing the feature.
     * @param tag String tag associated with the feature.
     * @param customId Custom identifier for the feature.
     * @return int32_t Status of the feature addition.
     */
    virtual int32_t AddFeature(const std::vector<float>& feature, const std::string &tag, int32_t customId) {
        std::lock_guard<std::mutex> lock(m_mtx_);  // Use mutex to protect shared data
        return UnsafeAddFeature(feature, tag, customId);
    }

    /**
     * @brief Deletes a feature from the feature block.
     * @param rowToDelete Index of the feature to be deleted.
     * @return int32_t Status of the feature deletion.
     */
    virtual int32_t DeleteFeature(int rowToDelete) {
        std::lock_guard<std::mutex> lock(m_mtx_);
        return UnsafeDeleteFeature(rowToDelete);
    }

    /**
     * @brief Updates a feature in the feature block.
     * @param rowToUpdate Index of the feature to be updated.
     * @param newFeature New feature vector to replace the old one.
     * @param tag New tag for the updated feature.
     * @param customId Custom identifier for the updated feature.
     * @return int32_t Status of the feature update.
     */
    virtual int32_t UpdateFeature(int rowToUpdate, const std::vector<float>& newFeature, const std::string &tag, int32_t customId) {
        std::lock_guard<std::mutex> lock(m_mtx_);
        return UnsafeUpdateFeature(rowToUpdate, newFeature, tag, customId);
    }

    /**
     * @brief Registers a feature at a specific index in the feature block.
     * @param rowToUpdate Index at which to register the new feature.
     * @param feature Feature vector to be registered.
     * @param tag Tag associated with the feature.
     * @param customId Custom identifier for the feature.
     * @return int32_t Status of the feature registration.
     */
    virtual int32_t RegisterFeature(int rowToUpdate, const std::vector<float>& feature, const std::string &tag, int32_t customId) {
        std::lock_guard<std::mutex> lock(m_mtx_);
        return UnsafeRegisterFeature(rowToUpdate, feature, tag, customId);
    }

    /**
     * @brief Searches for the nearest feature in the block to a given query feature.
     * @param queryFeature Query feature vector.
     * @param searchResult SearchResult structure to store the search results.
     * @return int32_t Status of the search operation.
     */
    virtual int32_t SearchNearest(const std::vector<float>& queryFeature, SearchResult &searchResult) = 0;

    /**
     * @brief Search the first k features in a block that are closest to a given query feature.
     * @param topK Maximum number of similarities
     * @param searchResults outputs
     * */
    virtual int32_t SearchTopKNearest(const std::vector<float>& queryFeature, size_t topK, std::vector<SearchResult> &searchResults) = 0;

    /**
     * @brief Retrieves a feature from the feature block.
     * @param row Index of the feature to retrieve.
     * @param feature Vector to store the retrieved feature.
     * @return int32_t Status of the retrieval operation.
     */
    virtual int32_t GetFeature(int row, std::vector<float>& feature) = 0;

    /**
     * @brief Prints the size of the feature matrix.
     */
    virtual void PrintMatrixSize() = 0;

    /**
     * @brief Prints the entire feature matrix.
     */
    virtual void PrintMatrix() = 0;

public:

    /**
     * @brief Retrieves the tag associated with a feature at a given row index.
     * @param row Index of the feature to retrieve the tag for.
     * @return std::string Tag associated with the feature at the given row, or an empty string if the row is invalid.
     */
    std::string GetTagFromRow(int row) {
        std::lock_guard<std::mutex> lock(m_mtx_);  // Ensure thread safety
        if (row >= 0 && row < m_tag_list_.size() && m_feature_state_[row] == FEATURE_STATE::USED) {
            return m_tag_list_[row];
        } else {
            return "";  // Return an empty string for invalid row or unused slot
        }
    }

    /**
     * @brief Retrieves the state of a feature slot at a given row index.
     * @param row Index of the feature slot to retrieve the state for.
     * @return FEATURE_STATE State of the feature slot at the given row, or IDLE if the row is invalid.
     */
    FEATURE_STATE GetStateFromRow(int row) {
        std::lock_guard<std::mutex> lock(m_mtx_);  // Ensure thread safety
        if (row >= 0 && row < m_feature_state_.size()) {
            return m_feature_state_[row];
        } else {
            return FEATURE_STATE::IDLE;  // Treat invalid rows as IDLE
        }
    }

    /**
     * @brief Finds the index of the first idle (unused) feature slot.
     * @return int Index of the first idle slot, or -1 if no idle slot is found.
     */
    int FindFirstIdleIndex() const {
        for (int i = 0; i < m_feature_state_.size(); ++i) {
            if (m_feature_state_[i] == FEATURE_STATE::IDLE) {
                return i; // Find the first IDLE index
            }
        }
        return -1; // No IDLE found
    }

    /**
     * @brief Finds the index of the first used feature slot.
     * @return int Index of the first used slot, or -1 if no used slot is found.
     */
    int FindFirstUsedIndex() const {
        for (int i = 0; i < m_feature_state_.size(); ++i) {
            if (m_feature_state_[i] == FEATURE_STATE::USED) {
                return i; // Find the first USED index
            }
        }
        return -1; // not fond USED
    }

    /**
     * @brief Counts the number of used feature slots.
     * @return int Count of used feature slots.
     */
    int GetUsedCount() const {
        int usedCount = 0;
        for (const FEATURE_STATE& state : m_feature_state_) {
            if (state == FEATURE_STATE::USED) {
                usedCount++;
            }
        }
        return usedCount;
    }

    /**
     * @brief Checks if all feature slots are used.
     * @return bool True if all slots are used, false otherwise.
     */
    bool IsUsedFull() const {
        int usedCount = GetUsedCount();
        return usedCount >= m_features_max_;
    }

    /**
     * @brief Finds the index of a feature slot by its custom ID.
     * @param customId The custom ID to search for.
     * @return size_t Index of the slot with the given custom ID, or -1 if not found.
     */
    size_t FindIndexByCustomId(int32_t customId) {
        auto it = std::find(m_custom_id_list_.begin(), m_custom_id_list_.end(), customId);
        if (it != m_custom_id_list_.end()) {
            return std::distance(m_custom_id_list_.begin(), it);  // return index
        }
        return -1;
    }

protected:
    /**
     * @brief Adds a feature to the feature block without thread safety.
     *        This method should be overridden in derived classes.
     * @param feature Vector of floats representing the feature.
     * @param tag String tag associated with the feature.
     * @param customId Custom identifier for the feature.
     * @return int32_t Status of the feature addition.
     */
    virtual int32_t UnsafeAddFeature(const std::vector<float>& feature, const std::string &tag, int32_t customId) = 0;

    /**
     * @brief Registers a feature at a specific index in the feature block without thread safety.
     *        This method should be overridden in derived classes.
     * @param rowToUpdate Index at which to register the new feature.
     * @param feature Feature vector to be registered.
     * @param tag Tag associated with the feature.
     * @param customId Custom identifier for the feature.
     * @return int32_t Status of the feature registration.
     */
    virtual int32_t UnsafeRegisterFeature(int rowToUpdate, const std::vector<float>& feature, const std::string &tag, int32_t customId) = 0;

    /**
     * @brief Deletes a feature from the feature block without thread safety.
     *        This method should be overridden in derived classes.
     * @param rowToDelete Index of the feature to be deleted.
     * @return int32_t Status of the feature deletion.
     */
    virtual int32_t UnsafeDeleteFeature(int rowToDelete) = 0;

    /**
     * @brief Updates a feature in the feature block without thread safety.
     *        This method should be overridden in derived classes.
     * @param rowToUpdate Index of the feature to be updated.
     * @param newFeature New feature vector to replace the old one.
     * @param tag New tag for the updated feature.
     * @param customId Custom identifier for the updated feature.
     * @return int32_t Status of the feature update.
     */
    virtual int32_t UnsafeUpdateFeature(int rowToUpdate, const std::vector<float>& newFeature, const std::string &tag, int32_t customId) = 0;

protected:
    MatrixCore m_matrix_core_;              ///< Type of matrix core used.
    int32_t m_features_max_;                ///< Maximum number of features in the block.
    int32_t m_feature_length_;              ///< Length of each feature vector.
    std::mutex m_mtx_;                      ///< Mutex for thread safety.
    std::vector<FEATURE_STATE> m_feature_state_; ///< State of each feature slot.
    std::vector<String> m_tag_list_;        ///< List of tags associated with each feature.
    std::vector<int32_t> m_custom_id_list_; ///< List of custom IDs associated with each feature.

};

}   // namespace hyper

#endif //HYPERFACEREPO_FEATUREBLOCK_H
