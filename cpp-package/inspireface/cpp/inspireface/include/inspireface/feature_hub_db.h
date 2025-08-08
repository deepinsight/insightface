/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#pragma once
#ifndef INSPIRE_FEATURE_HUB_DB_H
#define INSPIRE_FEATURE_HUB_DB_H

#include <memory>
#include <vector>
#include <string>
#include "data_type.h"
#include <mutex>

#define INSPIREFACE_FEATURE_HUB inspire::FeatureHubDB::GetInstance()
#define INSPIRE_INVALID_ID -1

namespace inspire {

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
    PrimaryKeyMode primary_key_mode = PrimaryKeyMode::AUTO_INCREMENT;  ///< Primary key mode
    bool enable_persistence = false;                                   ///< Whether to enable data persistence.
    std::string persistence_db_path;                                   ///< Path to the database file.
    float recognition_threshold = 0.48f;                               ///< Face search threshold
    SearchMode search_mode = SEARCH_MODE_EAGER;                        ///< Search mode (!!Temporarily unavailable!!)
};

/**
 * @class FeatureHubDB
 * @brief Service for internal feature vector storage.
 *
 * This class provides methods for face feature extraction, registration, update, search, and more.
 * It uses the PIMPL (Pointer to Implementation) pattern to hide implementation details.
 */
class INSPIRE_API_EXPORT FeatureHubDB {
public:
    /**
     * @brief Constructor for FeatureHubDB class.
     */
    FeatureHubDB();

    /**
     * @brief Destructor for FeatureHubDB class.
     */
    ~FeatureHubDB();

    FeatureHubDB(const FeatureHubDB&) = delete;
    FeatureHubDB& operator=(const FeatureHubDB&) = delete;

    /**
     * @brief Gets the singleton instance of FeatureHubDB.
     * @return Shared pointer to the FeatureHubDB instance.
     */
    static std::shared_ptr<FeatureHubDB> GetInstance();

    /**
     * @brief Enables the feature hub with the specified configuration.
     * @param configuration The database configuration settings.
     * @return int32_t Returns a status code indicating success (0) or failure (non-zero).
     */
    int32_t EnableHub(const DatabaseConfiguration& configuration);

    /**
     * @brief Disables the feature hub, freeing all associated resources.
     * @return int32_t Returns a status code indicating success (0) or failure (non-zero).
     */
    int32_t DisableHub();

    /**
     * @brief Get all ids in the database.
     * @return int32_t Status code of the operation.
     */
    int32_t GetAllIds();

    /**
     * @brief Searches for a face feature within stored data.
     * @param queryFeature Embedded feature to search for.
     * @param searchResult SearchResult object to store search results.
     * @param returnFeature Whether to return the feature data.
     * @return int32_t Status code of the search operation.
     */
    int32_t SearchFaceFeature(const Embedded& queryFeature, FaceSearchResult& searchResult, bool returnFeature = true);

    /**
     * @brief Search the stored data for the top k facial features that are most similar.
     * @param queryFeature Embedded feature to search for.
     * @param topK Maximum number of results to return.
     * @return int32_t Status code of the search operation.
     */
    int32_t SearchFaceFeatureTopKCache(const Embedded& queryFeature, size_t topK);

    /**
     * @brief Search the stored data for the top k facial features that are most similar.
     * @param queryFeature Embedded feature to search for.
     * @param searchResult Vector to store search results.
     * @param topK Maximum number of results to return.
     * @param returnFeature Whether to return the feature data.
     * @return int32_t Status code of the search operation.
     */
    int32_t SearchFaceFeatureTopK(const Embedded& queryFeature, std::vector<FaceSearchResult>& searchResult, size_t topK, bool returnFeature = false);

    /**
     * @brief Inserts a face feature with a custom ID.
     * @param feature Vector of floats representing the face feature.
     * @param id ID for the feature.
     * @param result_id Output parameter to store the resulting ID.
     * @return int32_t Status code of the insertion operation.
     */
    int32_t FaceFeatureInsert(const std::vector<float>& feature, int32_t id, int64_t& result_id);

    /**
     * @brief Removes a face feature by its ID.
     * @param id ID of the feature to remove.
     * @return int32_t Status code of the removal operation.
     */
    int32_t FaceFeatureRemove(int32_t id);

    /**
     * @brief Updates a face feature by its ID.
     * @param feature Vector of floats representing the new face feature.
     * @param customId ID of the feature to update.
     * @return int32_t Status code of the update operation.
     */
    int32_t FaceFeatureUpdate(const std::vector<float>& feature, int32_t customId);

    /**
     * @brief Retrieves a face feature by its ID.
     * @param id ID of the feature to retrieve.
     * @return int32_t Status code of the retrieval operation.
     */
    int32_t GetFaceFeature(int32_t id);

    /**
     * @brief Retrieves a face feature by its ID.
     * @param id ID of the feature to retrieve.
     * @param feature Vector to store the retrieved feature.
     * @return int32_t Status code of the retrieval operation.
     */
    int32_t GetFaceFeature(int32_t id, std::vector<float>& feature);

    /**
     * @brief Retrieves a face feature by its ID.
     * @param id ID of the feature to retrieve.
     * @param feature Vector to store the retrieved feature.
     * @return int32_t Status code of the retrieval operation.
     */
    int32_t GetFaceFeature(int32_t id, FaceEmbedding& feature);

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
     * @param v1 First feature vector.
     * @param v2 Second feature vector.
     * @param res Output parameter to store the cosine similarity result.
     * @param normalize Whether to normalize the vectors before computing similarity.
     * @return int32_t Status code indicating success (0) or failure.
     */
    static int32_t CosineSimilarity(const std::vector<float>& v1, const std::vector<float>& v2, float& res, bool normalize = false);

    /**
     * @brief Computes the cosine similarity between two feature vectors.
     * @param v1 Pointer to the first feature vector.
     * @param v2 Pointer to the second feature vector.
     * @param size Size of the feature vectors.
     * @param res Output parameter to store the cosine similarity result.
     * @param normalize Whether to normalize the vectors before computing similarity.
     * @return int32_t Status code indicating success (0) or failure.
     */
    static int32_t CosineSimilarity(const float* v1, const float* v2, int32_t size, float& res, bool normalize = true);

    // Getter methods

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
     * @brief Retrieves the total number of facial features stored.
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

private:
    class Impl;

    std::unique_ptr<Impl> pImpl;

    static std::mutex mutex_;
    static std::shared_ptr<FeatureHubDB> instance_;
};

}  // namespace inspire

#endif  // INSPIRE_FEATURE_HUB_DB_H