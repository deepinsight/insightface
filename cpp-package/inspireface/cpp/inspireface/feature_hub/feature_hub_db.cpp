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
#include "log.h"
#include "feature_hub/embedding_db/embedding_db.h"

#define DB_FILE_NAME ".feature_hub_db_v0"

namespace inspire {

class FeatureHubDB::Impl {
public:
    Impl() : m_enable_(false), m_recognition_threshold_(0.48f), m_search_mode_(SEARCH_MODE_EAGER) {}

    Embedded m_search_face_feature_cache_;
    Embedded m_getter_face_feature_cache_;
    std::shared_ptr<FaceFeaturePtr> m_face_feature_ptr_cache_;

    std::vector<FaceSearchResult> m_search_top_k_cache_;
    std::vector<float> m_top_k_confidence_;
    std::vector<int64_t> m_top_k_custom_ids_cache_;

    std::vector<int64_t> m_all_ids_;

    DatabaseConfiguration m_db_configuration_;
    float m_recognition_threshold_;
    SearchMode m_search_mode_;

    bool m_enable_;

    std::mutex m_res_mtx_;
};

std::mutex FeatureHubDB::mutex_;
std::shared_ptr<FeatureHubDB> FeatureHubDB::instance_ = nullptr;

FeatureHubDB::FeatureHubDB() : pImpl(new Impl()) {}

FeatureHubDB::~FeatureHubDB() = default;

std::shared_ptr<FeatureHubDB> FeatureHubDB::GetInstance() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!instance_) {
        instance_ = std::shared_ptr<FeatureHubDB>(new FeatureHubDB());
    }
    return instance_;
}

int32_t FeatureHubDB::DisableHub() {
    if (!pImpl->m_enable_) {
        INSPIRE_LOGW("FeatureHub is already disabled.");
        return HSUCCEED;
    }

    if (EMBEDDING_DB::GetInstance().IsInitialized()) {
        EMBEDDING_DB::Deinit();
    }

    pImpl->m_search_face_feature_cache_.clear();

    pImpl->m_db_configuration_ = DatabaseConfiguration();
    pImpl->m_recognition_threshold_ = 0.0f;
    pImpl->m_search_mode_ = SEARCH_MODE_EAGER;

    pImpl->m_face_feature_ptr_cache_.reset();
    pImpl->m_enable_ = false;

    return HSUCCEED;
}

int32_t FeatureHubDB::GetAllIds() {
    if (!pImpl->m_enable_) {
        INSPIRE_LOGE("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }
    pImpl->m_all_ids_ = EMBEDDING_DB::GetInstance().GetAllIds();
    return HSUCCEED;
}

int32_t FeatureHubDB::EnableHub(const DatabaseConfiguration &configuration) {
    if (pImpl->m_enable_) {
        INSPIRE_LOGW("You have enabled the FeatureHub feature. It is not valid to do so again");
        return HSUCCEED;
    }

    pImpl->m_db_configuration_ = configuration;
    pImpl->m_recognition_threshold_ = pImpl->m_db_configuration_.recognition_threshold;
    if (pImpl->m_recognition_threshold_ < -1.0f || pImpl->m_recognition_threshold_ > 1.0f) {
        INSPIRE_LOGW("The search threshold entered does not fit the required range (-1.0f, 1.0f) and has been set to 0.5 by default");
        pImpl->m_recognition_threshold_ = 0.5f;
    }

    std::string dbFile = ":memory:";
    if (pImpl->m_db_configuration_.enable_persistence) {
        if (IsDirectory(pImpl->m_db_configuration_.persistence_db_path)) {
            dbFile = os::PathJoin(pImpl->m_db_configuration_.persistence_db_path, DB_FILE_NAME);
        } else {
            dbFile = pImpl->m_db_configuration_.persistence_db_path;
        }
    }

    EMBEDDING_DB::Init(dbFile, 512, IdMode(configuration.primary_key_mode));
    pImpl->m_enable_ = true;
    pImpl->m_face_feature_ptr_cache_ = std::make_shared<FaceFeatureEntity>();

    return HSUCCEED;
}

int32_t FeatureHubDB::CosineSimilarity(const std::vector<float> &v1, const std::vector<float> &v2, float &res, bool normalize) {
    if (v1.size() != v2.size() || v1.empty()) {
        return HERR_SESS_REC_CONTRAST_FEAT_ERR;
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
    if (!pImpl->m_enable_) {
        INSPIRE_LOGW("FeatureHub is disabled, please enable it before it can be served");
        return 0;
    }

    return EMBEDDING_DB::GetInstance().GetVectorCount();
}

int32_t FeatureHubDB::SearchFaceFeature(const Embedded &queryFeature, FaceSearchResult &searchResult, bool returnFeature) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pImpl->m_enable_) {
        INSPIRE_LOGE("FeatureHub is disabled, please enable it before it can be served");
        return HSUCCEED;
    }

    pImpl->m_search_face_feature_cache_.clear();
    auto results = EMBEDDING_DB::GetInstance().SearchSimilarVectors(queryFeature, 1, pImpl->m_recognition_threshold_, returnFeature);
    searchResult.id = -1;

    if (!results.empty()) {
        auto &searched = results[0];
        searchResult.similarity = searched.similarity;
        searchResult.id = searched.id;
        if (returnFeature) {
            searchResult.feature = searched.feature;
            pImpl->m_search_face_feature_cache_ = searched.feature;
            pImpl->m_face_feature_ptr_cache_->data = pImpl->m_search_face_feature_cache_.data();
            pImpl->m_face_feature_ptr_cache_->dataSize = pImpl->m_search_face_feature_cache_.size();
        }
    }

    return HSUCCEED;
}

int32_t FeatureHubDB::SearchFaceFeatureTopKCache(const Embedded &queryFeature, size_t topK) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pImpl->m_enable_) {
        INSPIRE_LOGE("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }

    pImpl->m_top_k_confidence_.clear();
    pImpl->m_top_k_custom_ids_cache_.clear();
    auto results = EMBEDDING_DB::GetInstance().SearchSimilarVectors(queryFeature, topK, pImpl->m_recognition_threshold_, false);

    for (size_t i = 0; i < results.size(); i++) {
        pImpl->m_top_k_custom_ids_cache_.push_back(results[i].id);
        pImpl->m_top_k_confidence_.push_back(results[i].similarity);
    }

    return HSUCCEED;
}

int32_t FeatureHubDB::SearchFaceFeatureTopK(const Embedded &queryFeature, std::vector<FaceSearchResult> &searchResult, size_t topK,
                                            bool returnFeature) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pImpl->m_enable_) {
        INSPIRE_LOGW("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }

    searchResult = EMBEDDING_DB::GetInstance().SearchSimilarVectors(queryFeature, topK, pImpl->m_recognition_threshold_, returnFeature);
    return HSUCCEED;
}

int32_t FeatureHubDB::FaceFeatureInsert(const std::vector<float> &feature, int32_t id, int64_t &result_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pImpl->m_enable_) {
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
    if (!pImpl->m_enable_) {
        INSPIRE_LOGE("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }

    EMBEDDING_DB::GetInstance().DeleteVector(id);
    return HSUCCEED;
}

int32_t FeatureHubDB::FaceFeatureUpdate(const std::vector<float> &feature, int32_t customId) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pImpl->m_enable_) {
        INSPIRE_LOGE("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }

    EMBEDDING_DB::GetInstance().UpdateVector(customId, feature);
    return HSUCCEED;
}

int32_t FeatureHubDB::GetFaceFeature(int32_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pImpl->m_enable_) {
        INSPIRE_LOGE("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }

    auto vec = EMBEDDING_DB::GetInstance().GetVector(id);
    if (vec.empty()) {
        return HERR_FT_HUB_NOT_FOUND_FEATURE;
    }

    pImpl->m_getter_face_feature_cache_ = vec;
    pImpl->m_face_feature_ptr_cache_->data = pImpl->m_getter_face_feature_cache_.data();
    pImpl->m_face_feature_ptr_cache_->dataSize = pImpl->m_getter_face_feature_cache_.size();

    return HSUCCEED;
}

int32_t FeatureHubDB::GetFaceFeature(int32_t id, std::vector<float> &feature) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pImpl->m_enable_) {
        INSPIRE_LOGW("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }

    feature = EMBEDDING_DB::GetInstance().GetVector(id);
    if (feature.empty()) {
        return HERR_FT_HUB_NOT_FOUND_FEATURE;
    }

    return HSUCCEED;
}

int32_t FeatureHubDB::GetFaceFeature(int32_t id, FaceEmbedding& feature) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pImpl->m_enable_) {
        INSPIRE_LOGW("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }

    feature.embedding = EMBEDDING_DB::GetInstance().GetVector(id);
    if (feature.embedding.empty()) {
        return HERR_FT_HUB_NOT_FOUND_FEATURE;
    }

    return HSUCCEED;
}

int32_t FeatureHubDB::ViewDBTable() {
    if (!pImpl->m_enable_) {
        INSPIRE_LOGE("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }
    EMBEDDING_DB::GetInstance().ShowTable();
    return HSUCCEED;
}

void FeatureHubDB::SetRecognitionThreshold(float threshold) {
    pImpl->m_recognition_threshold_ = threshold;
}

void FeatureHubDB::SetRecognitionSearchMode(SearchMode mode) {
    pImpl->m_search_mode_ = mode;
}

// =========== Getter ===========

const Embedded &FeatureHubDB::GetSearchFaceFeatureCache() const {
    return pImpl->m_search_face_feature_cache_;
}

const std::shared_ptr<FaceFeaturePtr> &FeatureHubDB::GetFaceFeaturePtrCache() const {
    return pImpl->m_face_feature_ptr_cache_;
}

std::vector<float> &FeatureHubDB::GetTopKConfidence() {
    return pImpl->m_top_k_confidence_;
}

std::vector<int64_t> &FeatureHubDB::GetTopKCustomIdsCache() {
    return pImpl->m_top_k_custom_ids_cache_;
}

std::vector<int64_t> &FeatureHubDB::GetExistingIds() {
    return pImpl->m_all_ids_;
}

}  // namespace inspire
