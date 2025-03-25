/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "feature_hub_db.h"
#include "simd.h"
#include "herror.h"
#include <thread>
#include "middleware/utils.h"
#include "middleware/system.h"

namespace inspire {

std::mutex FeatureHubDB::mutex_;
std::shared_ptr<FeatureHubDB> FeatureHubDB::instance_ = nullptr;

FeatureHubDB::FeatureHubDB() {}

std::shared_ptr<FeatureHubDB> FeatureHubDB::GetInstance() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!instance_) {
        instance_ = std::shared_ptr<FeatureHubDB>(new FeatureHubDB());
    }
    return instance_;
}

int32_t FeatureHubDB::DisableHub() {
    if (!m_enable_) {
        INSPIRE_LOGW("FeatureHub is already disabled.");
        return HSUCCEED;
    }
    // Close the database if it starts
    if (EMBEDDING_DB::GetInstance().IsInitialized()) {
        EMBEDDING_DB::Deinit();
        // if (ret != HSUCCEED) {
        //     INSPIRE_LOGE("Failed to close the database: %d", ret);
        //     return ret;
        // }
        // m_db_.reset();
    }

    m_search_face_feature_cache_.clear();

    m_db_configuration_ = DatabaseConfiguration();  // Reset using the default constructor
    m_recognition_threshold_ = 0.0f;
    m_search_mode_ = SEARCH_MODE_EAGER;

    m_face_feature_ptr_cache_.reset();
    m_enable_ = false;

    return HSUCCEED;
}

int32_t FeatureHubDB::GetAllIds() {
    if (!m_enable_) {
        INSPIRE_LOGE("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }
    m_all_ids_ = EMBEDDING_DB::GetInstance().GetAllIds();
    return HSUCCEED;
}

int32_t FeatureHubDB::EnableHub(const DatabaseConfiguration &configuration) {
    int32_t ret;
    if (m_enable_) {
        INSPIRE_LOGW("You have enabled the FeatureHub feature. It is not valid to do so again");
        return HSUCCEED;
    }
    // Config
    m_db_configuration_ = configuration;
    m_recognition_threshold_ = m_db_configuration_.recognition_threshold;
    if (m_recognition_threshold_ < -1.0f || m_recognition_threshold_ > 1.0f) {
        INSPIRE_LOGW("The search threshold entered does not fit the required range (-1.0f, 1.0f) and has been set to 0.5 by default");
        m_recognition_threshold_ = 0.5f;
    }
    std::string dbFile = ":memory:";
    if (m_db_configuration_.enable_persistence) {
        if (IsDirectory(m_db_configuration_.persistence_db_path)) {
            dbFile = os::PathJoin(m_db_configuration_.persistence_db_path, DB_FILE_NAME);
        } else {
            dbFile = m_db_configuration_.persistence_db_path;
        }
    }

    EMBEDDING_DB::Init(dbFile, 512, IdMode(configuration.primary_key_mode));
    m_enable_ = true;
    m_face_feature_ptr_cache_ = std::make_shared<FaceFeatureEntity>();

    return HSUCCEED;
}

int32_t FeatureHubDB::CosineSimilarity(const std::vector<float> &v1, const std::vector<float> &v2, float &res, bool normalize) {
    if (v1.size() != v2.size() || v1.empty()) {
        return HERR_SESS_REC_CONTRAST_FEAT_ERR;  // The similarity cannot be calculated if the vector lengths are not equal
    }
    if (normalize) {
        std::vector<float> v1_norm = v1;
        std::vector<float> v2_norm = v2;
        float mse1 = 0.0f;
        float mse2 = 0.0f;
        for (const auto &one : v1_norm) {
            mse1 += one * one;
        }
        mse1 = sqrt(mse1);
        for (float &one : v1_norm) {
            one /= mse1;
        }

        for (const auto &one : v2_norm) {
            mse2 += one * one;
        }
        mse2 = sqrt(mse2);
        for (float &one : v2_norm) {
            one /= mse2;
        }
        res = simd_dot(v1_norm.data(), v2_norm.data(), v1_norm.size());
    } else {
        // Calculate the cosine similarity
        res = simd_dot(v1.data(), v2.data(), v1.size());
    }

    return HSUCCEED;
}

int32_t FeatureHubDB::CosineSimilarity(const float *v1, const float *v2, int32_t size, float &res, bool normalize) {
    if (normalize) {
        std::vector<float> v1_norm(v1, v1 + size);
        std::vector<float> v2_norm(v2, v2 + size);
        float mse1 = 0.0f;
        float mse2 = 0.0f;
        for (const auto &one : v1_norm) {
            mse1 += one * one;
        }
        mse1 = sqrt(mse1);
        for (float &one : v1_norm) {
            one /= mse1;
        }
        for (const auto &one : v2_norm) {
            mse2 += one * one;
        }
        mse2 = sqrt(mse2);
        for (float &one : v2_norm) {
            one /= mse2;
        }
        res = simd_dot(v1_norm.data(), v2_norm.data(), v1_norm.size());
    } else {
        res = simd_dot(v1, v2, size);
    }

    return HSUCCEED;
}

int32_t FeatureHubDB::GetFaceFeatureCount() {
    if (!m_enable_) {
        INSPIRE_LOGW("FeatureHub is disabled, please enable it before it can be served");
        return 0;
    }
    int totalFeatureCount = 0;

    // Iterate over all FeatureBlocks and add up the number of feature vectors used
    totalFeatureCount = EMBEDDING_DB::GetInstance().GetVectorCount();

    return totalFeatureCount;
}

int32_t FeatureHubDB::SearchFaceFeature(const Embedded &queryFeature, FaceSearchResult &searchResult, bool returnFeature) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!m_enable_) {
        INSPIRE_LOGE("FeatureHub is disabled, please enable it before it can be served");
        return HSUCCEED;
    }
    m_search_face_feature_cache_.clear();
    auto results = EMBEDDING_DB::GetInstance().SearchSimilarVectors(queryFeature, 1, m_recognition_threshold_, returnFeature);
    searchResult.id = -1;
    if (!results.empty()) {
        auto &searched = results[0];
        searchResult.similarity = searched.similarity;
        searchResult.id = searched.id;
        if (returnFeature) {
            searchResult.feature = searched.feature;
            // copy feature to cache
            m_search_face_feature_cache_ = searched.feature;
            m_face_feature_ptr_cache_->data = m_search_face_feature_cache_.data();
            m_face_feature_ptr_cache_->dataSize = m_search_face_feature_cache_.size();
        }
    }

    return HSUCCEED;
}

int32_t FeatureHubDB::SearchFaceFeatureTopKCache(const Embedded &queryFeature, size_t topK) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!m_enable_) {
        INSPIRE_LOGE("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }
    m_top_k_confidence_.clear();
    m_top_k_custom_ids_cache_.clear();
    auto results = EMBEDDING_DB::GetInstance().SearchSimilarVectors(queryFeature, topK, m_recognition_threshold_, false);
    for (size_t i = 0; i < results.size(); i++) {
        m_top_k_custom_ids_cache_.push_back(results[i].id);
        m_top_k_confidence_.push_back(results[i].similarity);
    }

    return HSUCCEED;
}

int32_t FeatureHubDB::SearchFaceFeatureTopK(const Embedded &queryFeature, std::vector<FaceSearchResult> &searchResult, size_t topK,
                                            bool returnFeature) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!m_enable_) {
        INSPIRE_LOGW("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }
    searchResult = EMBEDDING_DB::GetInstance().SearchSimilarVectors(queryFeature, topK, m_recognition_threshold_, returnFeature);
    return HSUCCEED;
}

int32_t FeatureHubDB::FaceFeatureInsert(const std::vector<float> &feature, int32_t id, int64_t &result_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!m_enable_) {
        INSPIRE_LOGE("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }

    bool ret = EMBEDDING_DB::GetInstance().InsertVector(id, feature, result_id);
    if (!ret) {
        result_id = -1;
        return HERR_FT_HUB_INSERT_FAILURE;
    }

    return HSUCCEED;
}

int32_t FeatureHubDB::FaceFeatureRemove(int32_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!m_enable_) {
        INSPIRE_LOGE("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }
    EMBEDDING_DB::GetInstance().DeleteVector(id);

    return HSUCCEED;
}

int32_t FeatureHubDB::FaceFeatureUpdate(const std::vector<float> &feature, int32_t customId) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!m_enable_) {
        INSPIRE_LOGE("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }
    try {
        EMBEDDING_DB::GetInstance().UpdateVector(customId, feature);
    } catch (const std::exception &e) {
        INSPIRE_LOGW("Failed to update face feature, id: %d", customId);
        return HERR_FT_HUB_NOT_FOUND_FEATURE;
    }

    return HSUCCEED;
}

int32_t FeatureHubDB::GetFaceFeature(int32_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!m_enable_) {
        INSPIRE_LOGE("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }
    auto vec = EMBEDDING_DB::GetInstance().GetVector(id);
    if (vec.empty()) {
        return HERR_FT_HUB_NOT_FOUND_FEATURE;
    }
    m_getter_face_feature_cache_ = vec;
    m_face_feature_ptr_cache_->data = m_getter_face_feature_cache_.data();
    m_face_feature_ptr_cache_->dataSize = m_getter_face_feature_cache_.size();

    return HSUCCEED;
}

int32_t FeatureHubDB::GetFaceFeature(int32_t id, std::vector<float> &feature) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!m_enable_) {
        INSPIRE_LOGW("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }
    try {
        feature = EMBEDDING_DB::GetInstance().GetVector(id);
    } catch (const std::exception &e) {
        INSPIRE_LOGW("Failed to get face feature, id: %d", id);
        return HERR_FT_HUB_NOT_FOUND_FEATURE;
    }
    return HSUCCEED;
}

int32_t FeatureHubDB::ViewDBTable() {
    if (!m_enable_) {
        INSPIRE_LOGE("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }
    EMBEDDING_DB::GetInstance().ShowTable();
    return HSUCCEED;
}

void FeatureHubDB::SetRecognitionThreshold(float threshold) {
    m_recognition_threshold_ = threshold;
}

void FeatureHubDB::SetRecognitionSearchMode(SearchMode mode) {
    m_search_mode_ = mode;
}

// =========== Getter ===========

const Embedded &FeatureHubDB::GetSearchFaceFeatureCache() const {
    return m_search_face_feature_cache_;
}

const std::shared_ptr<FaceFeaturePtr> &FeatureHubDB::GetFaceFeaturePtrCache() const {
    return m_face_feature_ptr_cache_;
}

std::vector<float> &FeatureHubDB::GetTopKConfidence() {
    return m_top_k_confidence_;
}

std::vector<int64_t> &FeatureHubDB::GetTopKCustomIdsCache() {
    return m_top_k_custom_ids_cache_;
}

std::vector<int64_t> &FeatureHubDB::GetExistingIds() {
    return m_all_ids_;
}

}  // namespace inspire
