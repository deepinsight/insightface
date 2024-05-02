//
// Created by tunm on 2023/9/8.
//
#pragma once
#ifndef HYPERFACEREPO_FACERECOGNITION_H
#define HYPERFACEREPO_FACERECOGNITION_H

#include <mutex>
#include "common/face_info/face_object.h"
#include "common/face_data/data_tools.h"
#include "middleware/camera_stream/camera_stream.h"
#include "feature_hub/features_block/feature_block.h"
#include "feature_hub/persistence/sqlite_faces_manage.h"
#include "middleware/model_archive/inspire_archive.h"

/**
* @def DB_FILE_NAME
* @brief Default database file name used in the FaceContext.
*/
#define DB_FILE_NAME ".E63520A95DD5B3892C56DA38C3B28E551D8173FD"

#define FEATURE_HUB FeatureHub::GetInstance()

namespace inspire {

// Comparator function object to sort SearchResult by score (descending order)
struct CompareByScore {
    bool operator()(const SearchResult& a, const SearchResult& b) const {
        return a.score > b.score;
    }
};

typedef enum SearchMode {
    SEARCH_MODE_EAGER = 0,     // Eager mode: Stops when a vector meets the threshold.
    SEARCH_MODE_EXHAUSTIVE,    // Exhaustive mode: Searches until the best match is found.
} SearchMode;


/**
 * @struct DatabaseConfiguration
 * @brief Structure to configure database settings for FaceRecognition.
 */
using DatabaseConfiguration = struct DatabaseConfiguration {
    int feature_block_num = 20;
    bool enable_use_db = false;                    ///< Whether to enable data persistence.
    std::string db_path;                           ///< Path to the database file.
    float recognition_threshold = 0.48f;           ///< Face search threshold
    SearchMode search_mode = SEARCH_MODE_EAGER;    ///< Search mode
};

/**
 * @class FeatureHub
 * @brief Service for internal feature vector storage.
 *
 * This class provides methods for face feature extraction, registration, update, search, and more.
 */
class INSPIRE_API FeatureHub {
private:
    static std::mutex mutex_;                         ///< Mutex lock
    static std::shared_ptr<FeatureHub> instance_;     ///< FeatureHub Instance
    const int32_t NUM_OF_FEATURES_IN_BLOCK = 512;     ///< Number of features in each feature block.

    FeatureHub(const FeatureHub&) = delete;
    FeatureHub& operator=(const FeatureHub&) = delete;

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
    int32_t EnableHub(const DatabaseConfiguration& configuration, MatrixCore core = MC_OPENCV);

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


    static std::shared_ptr<FeatureHub> GetInstance();

    /**
     * @brief Searches for a face feature within stored data.
     * @param queryFeature Embedded feature to search for.
     * @param searchResult SearchResult object to store search results.
     * @return int32_t Status code of the search operation.
     */
    int32_t SearchFaceFeature(const Embedded& queryFeature, SearchResult &searchResult);

    /**
     * @brief Search the stored data for the top k facial features that are most similar.
     * @param topK Maximum search
     * @return int32_t Status code of the search operation.
     */
    int32_t SearchFaceFeatureTopK(const Embedded& queryFeature, size_t topK);

    /**
     * @brief Inserts a face feature with a custom ID.
     * @param feature Vector of floats representing the face feature.
     * @param tag String tag associated with the feature.
     * @param customId Custom ID for the feature.
     * @return int32_t Status code of the insertion operation.
     */
    int32_t FaceFeatureInsertFromCustomId(const std::vector<float>& feature, const std::string &tag, int32_t customId);

    /**
     * @brief Removes a face feature by its custom ID.
     * @param customId Custom ID of the feature to remove.
     * @return int32_t Status code of the removal operation.
     */
    int32_t FaceFeatureRemoveFromCustomId(int32_t customId);

    /**
     * @brief Updates a face feature by its custom ID.
     * @param feature Vector of floats representing the new face feature.
     * @param tag String tag associated with the feature.
     * @param customId Custom ID of the feature to update.
     * @return int32_t Status code of the update operation.
     */
    int32_t FaceFeatureUpdateFromCustomId(const std::vector<float>& feature, const std::string &tag, int32_t customId);

    /**
     * @brief Retrieves a face feature by its custom ID.
     * @param customId Custom ID of the feature to retrieve.
     * @return int32_t Status code of the retrieval operation.
     */
    int32_t GetFaceFeatureFromCustomId(int32_t customId);

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
    static int32_t CosineSimilarity(const std::vector<float>& v1, const std::vector<float>& v2, float &res);

    /**
     * @brief Computes the cosine similarity between two feature vectors.
     *
     * @param v1 Pointer to the first feature vector.
     * @param v2 Pointer to the second feature vector.
     * @param size Size of the feature vectors.
     * @param res Output parameter to store the cosine similarity result.
     * @return int32_t Status code indicating success (0) or failure.
     */
    static int32_t CosineSimilarity(const float* v1, const float *v2, int32_t size, float &res);

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
     * @brief Gets the cache for temporary string storage.
     * @return A pointer to the character array used as a string cache.
     */
    char* GetStringCache();


    /**
     * @brief Gets the number of features in the feature block.
     *
     * @return int32_t Number of features.
     */
    int32_t GetFeatureNum() const;


    /**
     * @brief Retrieves the total number of facial features stored in the feature block.
     *
     * @return int32_t Total number of facial features.
     */
    int32_t GetFaceFeatureCount();

    std::vector<float> &GetTopKConfidence();

    std::vector<int32_t> &GetTopKCustomIdsCache();

public:

    /**
     * @brief Constructor for FeatureHub class.
     */
    FeatureHub();

    /**
     * @brief Registers a facial feature in the feature block.
     *
     * @param feature Vector of floats representing the feature.
     * @param featureIndex Index of the feature in the block.
     * @param tag String tag associated with the feature.
     * @param customId Custom identifier for the feature.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t RegisterFaceFeature(const std::vector<float>& feature, int featureIndex, const std::string &tag, int32_t customId);

    /**
     * @brief Updates a facial feature in the feature block.
     *
     * @param feature Vector of floats representing the updated feature.
     * @param featureIndex Index of the feature in the block.
     * @param tag New string tag for the feature.
     * @param customId Custom identifier for the feature.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t UpdateFaceFeature(const std::vector<float>& feature, int featureIndex, const std::string &tag, int32_t customId);

    /**
     * @brief Searches for the nearest facial feature in the feature block to a given query feature.
     *
     * @param queryFeature Query feature vector.
     * @param searchResult SearchResult structure to store the search results.
     * @param threshold Threshold for considering a match.
     * @param mostSimilar Whether to find the most similar feature.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t SearchFaceFeature(const std::vector<float>& queryFeature, SearchResult &searchResult, float threshold, bool mostSimilar=true);

    /**
     * Search for the top K face features that are most similar to a given query feature.
     * @param queryFeature A vector of floats representing the feature to query against.
     * @param searchResultList A reference to a vector where the top K search results will be stored.
     * @param maxTopK The maximum number of top results to return.
     * @param threshold A float representing the minimum similarity score threshold.
     * @return int32_t Returns a status code (0 for success, non-zero for any errors).
     */
    int32_t SearchFaceFeatureTopK(const std::vector<float>& queryFeature, std::vector<SearchResult> &searchResultList, size_t maxTopK, float threshold);

    /**
     * @brief Inserts a facial feature into the feature block.
     *
     * @param feature Vector of floats representing the feature.
     * @param tag String tag associated with the feature.
     * @param customId Custom identifier for the feature.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t InsertFaceFeature(const std::vector<float>& feature, const std::string &tag, int32_t customId);

    /**
     * @brief Deletes a facial feature from the feature block.
     *
     * @param featureIndex Index of the feature to delete.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t DeleteFaceFeature(int featureIndex);

    /**
     * @brief Retrieves a facial feature from the feature block.
     *
     * @param featureIndex Index of the feature to retrieve.
     * @param feature Output parameter to store the retrieved feature.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t GetFaceFeature(int featureIndex, Embedded &feature);

    /**
     * @brief Retrieves a facial entity from the feature block.
     *
     * @param featureIndex Index of the feature to retrieve.
     * @param result Output parameter to store the retrieved entity.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t GetFaceEntity(int featureIndex, Embedded &feature, std::string& tag, FEATURE_STATE& status);

    /**
     * @brief Finds the index of a feature by its custom ID.
     *
     * @param customId Custom identifier to search for.
     * @return int32_t Index of the feature with the given custom ID, or -1 if not found.
     */
    int32_t FindFeatureIndexByCustomId(int32_t customId);

    /**
     * @brief Prints information about the feature matrix.
     */
    void PrintFeatureMatrixInfo();

private:

    Embedded m_search_face_feature_cache_;                         ///< Cache for face feature data used in search operations
    Embedded m_getter_face_feature_cache_;                         ///< Cache for face feature data used in search operations
    std::shared_ptr<FaceFeaturePtr> m_face_feature_ptr_cache_;     ///< Shared pointer to cache of face feature pointers
    char m_string_cache_[256];                                     ///< Cache for temporary string storage

    std::vector<SearchResult> m_search_top_k_cache_;               ///<
    std::vector<float> m_top_k_confidence_;
    std::vector<int32_t> m_top_k_custom_ids_cache_;

private:
    std::vector<std::shared_ptr<FeatureBlock>> m_feature_matrix_list_; ///< List of feature blocks.

    DatabaseConfiguration m_db_configuration_;                     ///< Configuration settings for the database
    float m_recognition_threshold_{0.48f};                          ///< Threshold value for face recognition
    SearchMode m_search_mode_{SEARCH_MODE_EAGER};                  ///< Flag to determine if the search should find the most similar feature

    std::shared_ptr<SQLiteFaceManage> m_db_;                       ///< Shared pointer to the SQLiteFaceManage object

    bool m_enable_{false};                                         ///< Running status

    std::mutex m_res_mtx_;                                         ///< Mutex for thread safety.

};


}   // namespace inspire

#endif //HYPERFACEREPO_FACERECOGNITION_H
