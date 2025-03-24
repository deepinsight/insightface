/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#pragma once
#ifndef INSPIRE_FEATURE_HUB_DB_H
#define INSPIRE_FEATURE_HUB_DB_H

#include <mutex>
#include <vector>
#include <string>
#include <memory>
#include "data_type.h"
#include "feature_hub/embedding_db/embedding_db.h"
#include "log.h"

// Default database file name used in the FaceContext.
#define DB_FILE_NAME ".feature_hub_db_v0"

#define FEATURE_HUB_DB FeatureHubDB::GetInstance()

namespace inspire {

// Comparator function object to sort SearchResult by score (descending order)
struct CompareByScore {
    bool operator()(const FaceSearchResult& a, const FaceSearchResult& b) const {
        return a.similarity > b.similarity;
    }
};

typedef enum SearchMode {
    SEARCH_MODE_EAGER = 0,   // Eager mode: Stops when a vector meets the threshold.
    SEARCH_MODE_EXHAUSTIVE,  // Exhaustive mode: Searches until the best match is found.
} SearchMode;

typedef enum PrimaryKeyMode {
    AUTO_INCREMENT = 0,  // Auto-increment primary key
    MANUAL_INPUT,        // Manual input primary key
} PrimaryKeyMode;

/**
 * @struct DatabaseConfiguration
 * @brief Structure to configure database settings for FaceRecognition.
 */
using DatabaseConfiguration = struct DatabaseConfiguration {
    PrimaryKeyMode primary_key_mode = PrimaryKeyMode::AUTO_INCREMENT;  ///<
    bool enable_persistence = false;                                   ///< Whether to enable data persistence.
    std::string persistence_db_path;                                   ///< Path to the database file.
    float recognition_threshold = 0.48f;                               ///< Face search threshold
    SearchMode search_mode = SEARCH_MODE_EAGER;                        ///< Search mode (!!Temporarily unavailable!!)
};

/**
 * @class FeatureHub
 * @brief Service for internal feature vector storage.
 *
 * This class provides methods for face feature extraction, registration, update, search, and more.
 */
class INSPIRE_API FeatureHubDB {
private:
    static std::mutex mutex_;                        ///< Mutex lock
    static std::shared_ptr<FeatureHubDB> instance_;  ///< FeatureHub Instance

    FeatureHubDB(const FeatureHubDB&) = delete;
    FeatureHubDB& operator=(const FeatureHubDB&) = delete;

public:
    /**
     * @brief Enables the feature hub with the specified configuration and matrix core.
     *
     * This function initializes and configures the feature hub based on the provided database
     * configuration and the specified matrix processing core. It prepares the hub for operation,
     * setting up necessary resources such as database connections and data processing pipelines.
     *
     * @param configuration The database configuration settings used to configure the hub.
     * @param core The matrix core used for processing, defaulting to OpenCV if not specified.
     * @return int32_t Returns a status code indicating success (0) or failure (non-zero).
     */
    int32_t EnableHub(const DatabaseConfiguration& configuration);

    /**
     * @brief Disables the feature hub, freeing all associated resources.
     *
     * This function stops all operations within the hub, releases all occupied resources,
     * such as database connections and internal data structures. It is used to safely
     * shutdown the hub when it is no longer needed or before the application exits, ensuring
     * that all resources are properly cleaned up.
     *
     * @return int32_t Returns a status code indicating success (0) or failure (non-zero).
     */
    int32_t DisableHub();

    /**
     * @brief Get all ids in the database.
     * @param ids Output parameter to store the ids.
     * @return int32_t Status code of the operation.
     */
    int32_t GetAllIds();

    static std::shared_ptr<FeatureHubDB> GetInstance();

    /**
     * @brief Searches for a face feature within stored data.
     * @param queryFeature Embedded feature to search for.
     * @param searchResult SearchResult object to store search results.
     * @return int32_t Status code of the search operation.
     */
    int32_t SearchFaceFeature(const Embedded& queryFeature, FaceSearchResult& searchResult, bool returnFeature = true);

    /**
     * @brief Search the stored data for the top k facial features that are most similar.
     * @param topK Maximum search
     * @return int32_t Status code of the search operation.
     */
    int32_t SearchFaceFeatureTopKCache(const Embedded& queryFeature, size_t topK);

    /**
     * @brief Search the stored data for the top k facial features that are most similar.
     * @param topK Maximum search
     * @return int32_t Status code of the search operation.
     */
    int32_t SearchFaceFeatureTopK(const Embedded& queryFeature, std::vector<FaceSearchResult>& searchResult, size_t topK, bool returnFeature = false);

    /**
     * @brief Inserts a face feature with a custom ID.
     * @param feature Vector of floats representing the face feature.
     * @param tag String tag associated with the feature.
     * @param customId Custom ID for the feature.
     * @return int32_t Status code of the insertion operation.
     */
    int32_t FaceFeatureInsert(const std::vector<float>& feature, int32_t id, int64_t& result_id);

    /**
     * @brief Removes a face feature by its custom ID.
     * @param customId Custom ID of the feature to remove.
     * @return int32_t Status code of the removal operation.
     */
    int32_t FaceFeatureRemove(int32_t id);

    /**
     * @brief Updates a face feature by its custom ID.
     * @param feature Vector of floats representing the new face feature.
     * @param tag String tag associated with the feature.
     * @param customId Custom ID of the feature to update.
     * @return int32_t Status code of the update operation.
     */
    int32_t FaceFeatureUpdate(const std::vector<float>& feature, int32_t customId);

    /**
     * @brief Retrieves a face feature by its custom ID.
     * @param customId Custom ID of the feature to retrieve.
     * @return int32_t Status code of the retrieval operation.
     */
    int32_t GetFaceFeature(int32_t id);

    /**
     * @brief Retrieves a face feature by its custom ID.
     * @param customId Custom ID of the feature to retrieve.
     * @param feature Vector of floats representing the face feature.
     * @return int32_t Status code of the retrieval operation.
     */
    int32_t GetFaceFeature(int32_t id, std::vector<float>& feature);

    /**
     * @brief Views the database table containing face data.
     * @return int32_t Status code of the operation.
     */
    int32_t ViewDBTable();

    /**
     * @brief Sets the recognition threshold for face recognition.
     * @param threshold Float value of the new threshold.
     */
    void SetRecognitionThreshold(float threshold);

    /**
     * @brief Sets the search mode for face recognition.
     * @param mode Search mode.
     */
    void SetRecognitionSearchMode(SearchMode mode);

    /**
     * @brief Computes the cosine similarity between two feature vectors.
     *
     * @param v1 First feature vector.
     * @param v2 Second feature vector.
     * @param res Output parameter to store the cosine similarity result.
     * @return int32_t Status code indicating success (0) or failure.
     */
    static int32_t CosineSimilarity(const std::vector<float>& v1, const std::vector<float>& v2, float& res, bool normalize = false);

    /**
     * @brief Computes the cosine similarity between two feature vectors.
     *
     * @param v1 Pointer to the first feature vector.
     * @param v2 Pointer to the second feature vector.
     * @param size Size of the feature vectors.
     * @param res Output parameter to store the cosine similarity result.
     * @return int32_t Status code indicating success (0) or failure.
     */
    static int32_t CosineSimilarity(const float* v1, const float* v2, int32_t size, float& res, bool normalize = true);

public:
    // Getter Function

    /**
     * @brief Gets the cache used for search operations in face feature data.
     * @return A const reference to the Embedded object containing face feature data for search.
     */
    const Embedded& GetSearchFaceFeatureCache() const;

    /**
     * @brief Gets the cache of face feature pointers.
     * @return A shared pointer to the cache of face feature pointers.
     */
    const std::shared_ptr<FaceFeaturePtr>& GetFaceFeaturePtrCache() const;

    /**
     * @brief Retrieves the total number of facial features stored in the feature block.
     *
     * @return int32_t Total number of facial features.
     */
    int32_t GetFaceFeatureCount();

    /**
     * @brief Retrieves the confidence scores for the top k facial features.
     * @return A reference to the vector of confidence scores.
     */
    std::vector<float>& GetTopKConfidence();

    /**
     * @brief Retrieves the custom IDs for the top k facial features.
     * @return A reference to the vector of custom IDs.
     */
    std::vector<int64_t>& GetTopKCustomIdsCache();

    /**
     * @brief Retrieves the existing ids in the database.
     * @return A reference to the vector of existing ids.
     */
    std::vector<int64_t>& GetExistingIds();

    /**
     * @brief Constructor for FeatureHub class.
     */
    FeatureHubDB();

    /**
     * @brief Prints information about the feature matrix.
     */
    void PrintFeatureMatrixInfo();

private:
    Embedded m_search_face_feature_cache_;                      ///< Cache for face feature data used in search operations
    Embedded m_getter_face_feature_cache_;                      ///< Cache for face feature data used in search operations
    std::shared_ptr<FaceFeaturePtr> m_face_feature_ptr_cache_;  ///< Shared pointer to cache of face feature pointers

    std::vector<FaceSearchResult> m_search_top_k_cache_;  ///< Cache for top k search results
    std::vector<float> m_top_k_confidence_;               ///< Cache for top k confidence scores
    std::vector<int64_t> m_top_k_custom_ids_cache_;       ///< Cache for top k custom ids

    std::vector<int64_t> m_all_ids_;  ///< Cache for all ids

private:
    DatabaseConfiguration m_db_configuration_;     ///< Configuration settings for the database
    float m_recognition_threshold_{0.48f};         ///< Threshold value for face recognition
    SearchMode m_search_mode_{SEARCH_MODE_EAGER};  ///< Flag to determine if the search should find the most similar feature

    bool m_enable_{false};  ///< Running status

    std::mutex m_res_mtx_;  ///< Mutex for thread safety.
};

}  // namespace inspire

#endif  // INSPIRE_FEATURE_HUB_DB_H
