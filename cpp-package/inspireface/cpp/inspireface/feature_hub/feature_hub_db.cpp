/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "feature_hub_db.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "feature_hub/embedding_db/embedding_db.h"
#include "herror.h"
#include "log.h"
#include "middleware/system.h"
#include "middleware/utils.h"
#include "simd.h"

#define DB_FILE_NAME ".feature_hub_db_v0"

namespace inspire {

namespace {

constexpr size_t kFeatureDimension = 512;

bool IsFiniteFeature(const std::vector<float> &feature, size_t expected_size = kFeatureDimension) {
    if (feature.size() != expected_size) {
        return false;
    }
    for (const float value : feature) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

bool IsValidTopK(size_t top_k) {
    return top_k > 0 && top_k <= static_cast<size_t>(std::numeric_limits<int64_t>::max());
}

}  // namespace

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

    void InvalidateFaceFeaturePointer() {
        if (m_face_feature_ptr_cache_) {
            m_face_feature_ptr_cache_->data = nullptr;
            m_face_feature_ptr_cache_->dataSize = 0;
        }
    }
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
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pImpl->m_enable_) {
        INSPIRE_LOGW("FeatureHub is already disabled.");
        return HSUCCEED;
    }

    pImpl->m_enable_ = false;
    EMBEDDING_DB::Deinit();
    pImpl->m_search_face_feature_cache_.clear();
    pImpl->m_getter_face_feature_cache_.clear();
    pImpl->m_search_top_k_cache_.clear();
    pImpl->m_top_k_confidence_.clear();
    pImpl->m_top_k_custom_ids_cache_.clear();
    pImpl->m_all_ids_.clear();
    pImpl->m_face_feature_ptr_cache_.reset();
    pImpl->m_db_configuration_ = DatabaseConfiguration();
    pImpl->m_recognition_threshold_ = 0.48f;
    pImpl->m_search_mode_ = SEARCH_MODE_EAGER;
    return HSUCCEED;
}

int32_t FeatureHubDB::EnableHub(const DatabaseConfiguration &configuration) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pImpl->m_enable_) {
        INSPIRE_LOGW("You have enabled the FeatureHub feature. It is not valid to do so again");
        return HSUCCEED;
    }
    if ((configuration.primary_key_mode != PrimaryKeyMode::AUTO_INCREMENT && configuration.primary_key_mode != PrimaryKeyMode::MANUAL_INPUT) ||
        (configuration.search_mode != SEARCH_MODE_EAGER && configuration.search_mode != SEARCH_MODE_EXHAUSTIVE)) {
        INSPIRE_LOGE("FeatureHub configuration contains an invalid enum value");
        return HERR_INVALID_PARAM;
    }

    const float threshold = configuration.recognition_threshold;
    if (!std::isfinite(threshold) || threshold < -1.0f || threshold > 1.0f) {
        INSPIRE_LOGE("The search threshold must be finite and within [-1, 1]");
        return HERR_INVALID_PARAM;
    }

    std::string database_file = ":memory:";
    if (configuration.enable_persistence) {
        if (configuration.persistence_db_path.empty()) {
            INSPIRE_LOGE("FeatureHub persistence requires a non-empty database path");
            return HERR_INVALID_PARAM;
        }
        database_file = IsDirectory(configuration.persistence_db_path)
                          ? os::PathJoin(configuration.persistence_db_path, DB_FILE_NAME)
                          : configuration.persistence_db_path;
    }

    EMBEDDING_DB::Deinit();
    if (!EMBEDDING_DB::Init(database_file, kFeatureDimension, IdMode(configuration.primary_key_mode))) {
        INSPIRE_LOGE("Failed to initialize FeatureHub database");
        return HERR_FT_HUB_DATABASE_FAILURE;
    }

    pImpl->m_db_configuration_ = configuration;
    pImpl->m_recognition_threshold_ = threshold;
    pImpl->m_search_mode_ = configuration.search_mode;
    pImpl->m_face_feature_ptr_cache_ = std::make_shared<FaceFeatureEntity>();
    pImpl->m_search_face_feature_cache_.clear();
    pImpl->m_getter_face_feature_cache_.clear();
    pImpl->m_search_top_k_cache_.clear();
    pImpl->m_top_k_confidence_.clear();
    pImpl->m_top_k_custom_ids_cache_.clear();
    pImpl->m_all_ids_.clear();
    pImpl->m_enable_ = true;
    return HSUCCEED;
}

int32_t FeatureHubDB::GetAllIds() {
    std::lock_guard<std::mutex> lock(mutex_);
    pImpl->m_all_ids_.clear();
    if (!pImpl->m_enable_) {
        INSPIRE_LOGE("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }
    const auto database = EMBEDDING_DB::AcquireInstance();
    if (!database || !database->GetAllIds(pImpl->m_all_ids_)) {
        return HERR_FT_HUB_DATABASE_FAILURE;
    }
    return HSUCCEED;
}

int32_t FeatureHubDB::GetAllIds(std::vector<int64_t> &ids) {
    std::lock_guard<std::mutex> lock(mutex_);
    ids.clear();
    if (!pImpl->m_enable_) {
        return HERR_FT_HUB_DISABLE;
    }
    const auto database = EMBEDDING_DB::AcquireInstance();
    if (!database || !database->GetAllIds(ids)) {
        return HERR_FT_HUB_DATABASE_FAILURE;
    }
    return HSUCCEED;
}

int32_t FeatureHubDB::CosineSimilarity(const std::vector<float> &v1, const std::vector<float> &v2, float &res, bool normalize) {
    res = 0.0f;
    if (v1.size() != v2.size() || v1.empty() || v1.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        !IsFiniteFeature(v1, v1.size()) || !IsFiniteFeature(v2, v2.size())) {
        return HERR_SESS_REC_CONTRAST_FEAT_ERR;
    }
    return CosineSimilarity(v1.data(), v2.data(), static_cast<int32_t>(v1.size()), res, normalize);
}

int32_t FeatureHubDB::CosineSimilarity(const float *v1, const float *v2, int32_t size, float &res, bool normalize) {
    res = 0.0f;
    if (!v1 || !v2 || size <= 0) {
        return HERR_SESS_REC_CONTRAST_FEAT_ERR;
    }

    if (!normalize) {
        res = simd_dot(v1, v2, static_cast<size_t>(size));
        if (!std::isfinite(res)) {
            res = 0.0f;
            return HERR_SESS_REC_CONTRAST_FEAT_ERR;
        }
        return HSUCCEED;
    }

    double left_squared_norm = 0.0;
    double right_squared_norm = 0.0;
    for (int32_t i = 0; i < size; ++i) {
        if (!std::isfinite(v1[i]) || !std::isfinite(v2[i])) {
            return HERR_SESS_REC_CONTRAST_FEAT_ERR;
        }
        left_squared_norm += static_cast<double>(v1[i]) * v1[i];
        right_squared_norm += static_cast<double>(v2[i]) * v2[i];
    }
    if (!(left_squared_norm > 0.0) || !(right_squared_norm > 0.0) || !std::isfinite(left_squared_norm) ||
        !std::isfinite(right_squared_norm)) {
        return HERR_SESS_REC_CONTRAST_FEAT_ERR;
    }

    double dot = simd_dot(v1, v2, static_cast<size_t>(size));
    if (!std::isfinite(dot) || dot == 0.0) {
        dot = 0.0;
        for (int32_t i = 0; i < size; ++i) {
            dot += static_cast<double>(v1[i]) * v2[i];
        }
    }
    const double denominator = std::sqrt(left_squared_norm) * std::sqrt(right_squared_norm);
    const double similarity = dot / denominator;
    if (!std::isfinite(similarity)) {
        return HERR_SESS_REC_CONTRAST_FEAT_ERR;
    }
    res = static_cast<float>(std::max(-1.0, std::min(1.0, similarity)));
    return HSUCCEED;
}

int32_t FeatureHubDB::GetFaceFeatureCount() {
    int32_t count = 0;
    return GetFaceFeatureCount(count) == HSUCCEED ? count : 0;
}

int32_t FeatureHubDB::GetFaceFeatureCount(int32_t &count) {
    std::lock_guard<std::mutex> lock(mutex_);
    count = 0;
    if (!pImpl->m_enable_) {
        INSPIRE_LOGW("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }
    const auto database = EMBEDDING_DB::AcquireInstance();
    if (!database) {
        return HERR_FT_HUB_DATABASE_FAILURE;
    }
    const int64_t database_count = database->GetVectorCount();
    if (database_count < 0 || database_count > std::numeric_limits<int32_t>::max()) {
        return HERR_FT_HUB_DATABASE_FAILURE;
    }
    count = static_cast<int32_t>(database_count);
    return HSUCCEED;
}

int32_t FeatureHubDB::SearchFaceFeature(const Embedded &queryFeature, FaceSearchResult &searchResult, bool returnFeature) {
    bool found = false;
    return SearchFaceFeatureV2(queryFeature, searchResult, found, returnFeature);
}

int32_t FeatureHubDB::SearchFaceFeatureV2(const Embedded &queryFeature, FaceSearchResult &searchResult, bool &found, bool returnFeature) {
    std::lock_guard<std::mutex> lock(mutex_);
    found = false;
    searchResult.id = -1;
    searchResult.similarity = -1.0;
    searchResult.feature.clear();
    pImpl->InvalidateFaceFeaturePointer();
    pImpl->m_search_face_feature_cache_.clear();
    if (!pImpl->m_enable_) {
        INSPIRE_LOGE("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }
    if (!IsFiniteFeature(queryFeature)) {
        return HERR_FT_HUB_INVALID_FEATURE;
    }

    const auto database = EMBEDDING_DB::AcquireInstance();
    std::vector<FaceSearchResult> results;
    if (!database || !database->SearchSimilarVectors(queryFeature, results, 1, pImpl->m_recognition_threshold_, returnFeature)) {
        return HERR_FT_HUB_DATABASE_FAILURE;
    }
    if (results.empty()) {
        return HSUCCEED;
    }

    found = true;
    searchResult = std::move(results.front());
    if (returnFeature) {
        pImpl->m_search_face_feature_cache_ = searchResult.feature;
        if (pImpl->m_face_feature_ptr_cache_) {
            pImpl->m_face_feature_ptr_cache_->data = pImpl->m_search_face_feature_cache_.data();
            pImpl->m_face_feature_ptr_cache_->dataSize = static_cast<int32_t>(pImpl->m_search_face_feature_cache_.size());
        }
    }
    return HSUCCEED;
}

int32_t FeatureHubDB::SearchFaceFeatureTopKCache(const Embedded &queryFeature, size_t topK) {
    std::lock_guard<std::mutex> lock(mutex_);
    pImpl->m_top_k_confidence_.clear();
    pImpl->m_top_k_custom_ids_cache_.clear();
    if (!pImpl->m_enable_) {
        return HERR_FT_HUB_DISABLE;
    }
    if (!IsValidTopK(topK) || !IsFiniteFeature(queryFeature)) {
        return HERR_FT_HUB_INVALID_FEATURE;
    }

    const auto database = EMBEDDING_DB::AcquireInstance();
    std::vector<FaceSearchResult> results;
    if (!database || !database->SearchSimilarVectors(queryFeature, results, topK, pImpl->m_recognition_threshold_, false)) {
        return HERR_FT_HUB_DATABASE_FAILURE;
    }
    pImpl->m_top_k_confidence_.reserve(results.size());
    pImpl->m_top_k_custom_ids_cache_.reserve(results.size());
    for (const FaceSearchResult &result : results) {
        pImpl->m_top_k_custom_ids_cache_.push_back(result.id);
        pImpl->m_top_k_confidence_.push_back(static_cast<float>(result.similarity));
    }
    return HSUCCEED;
}

int32_t FeatureHubDB::SearchFaceFeatureTopK(const Embedded &queryFeature, std::vector<FaceSearchResult> &searchResult, size_t topK,
                                            bool returnFeature) {
    std::lock_guard<std::mutex> lock(mutex_);
    searchResult.clear();
    if (!pImpl->m_enable_) {
        return HERR_FT_HUB_DISABLE;
    }
    if (!IsValidTopK(topK) || !IsFiniteFeature(queryFeature)) {
        return HERR_FT_HUB_INVALID_FEATURE;
    }
    const auto database = EMBEDDING_DB::AcquireInstance();
    if (!database || !database->SearchSimilarVectors(queryFeature, searchResult, topK, pImpl->m_recognition_threshold_, returnFeature)) {
        return HERR_FT_HUB_DATABASE_FAILURE;
    }
    return HSUCCEED;
}

int32_t FeatureHubDB::FaceFeatureInsert(const std::vector<float> &feature, int32_t id, int64_t &result_id) {
    return FaceFeatureInsert(feature, static_cast<int64_t>(id), result_id);
}

int32_t FeatureHubDB::FaceFeatureInsert(const std::vector<float> &feature, int64_t id, int64_t &result_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    result_id = -1;
    if (!pImpl->m_enable_) {
        return HERR_FT_HUB_DISABLE;
    }
    if (!IsFiniteFeature(feature)) {
        return HERR_FT_HUB_INVALID_FEATURE;
    }
    if (pImpl->m_db_configuration_.primary_key_mode == PrimaryKeyMode::MANUAL_INPUT && id == INSPIRE_INVALID_ID) {
        return HERR_INVALID_PARAM;
    }
    const auto database = EMBEDDING_DB::AcquireInstance();
    if (!database || !database->InsertVector(id, feature, result_id)) {
        result_id = -1;
        return HERR_FT_HUB_INSERT_FAILURE;
    }
    return HSUCCEED;
}

int32_t FeatureHubDB::FaceFeatureRemove(int32_t id) {
    return FaceFeatureRemove(static_cast<int64_t>(id));
}

int32_t FeatureHubDB::FaceFeatureRemove(int64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pImpl->m_enable_) {
        return HERR_FT_HUB_DISABLE;
    }
    const auto database = EMBEDDING_DB::AcquireInstance();
    if (!database) {
        return HERR_FT_HUB_DATABASE_FAILURE;
    }
    return database->DeleteVector(id) ? HSUCCEED : HERR_FT_HUB_NOT_FOUND_FEATURE;
}

int32_t FeatureHubDB::FaceFeatureUpdate(const std::vector<float> &feature, int32_t customId) {
    return FaceFeatureUpdate(feature, static_cast<int64_t>(customId));
}

int32_t FeatureHubDB::FaceFeatureUpdate(const std::vector<float> &feature, int64_t customId) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pImpl->m_enable_) {
        return HERR_FT_HUB_DISABLE;
    }
    if (!IsFiniteFeature(feature)) {
        return HERR_FT_HUB_INVALID_FEATURE;
    }
    const auto database = EMBEDDING_DB::AcquireInstance();
    if (!database) {
        return HERR_FT_HUB_DATABASE_FAILURE;
    }
    return database->UpdateVector(customId, feature) ? HSUCCEED : HERR_FT_HUB_NOT_FOUND_FEATURE;
}

int32_t FeatureHubDB::GetFaceFeature(int32_t id) {
    return GetFaceFeature(static_cast<int64_t>(id));
}

int32_t FeatureHubDB::GetFaceFeature(int64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    pImpl->InvalidateFaceFeaturePointer();
    pImpl->m_getter_face_feature_cache_.clear();
    if (!pImpl->m_enable_) {
        return HERR_FT_HUB_DISABLE;
    }
    const auto database = EMBEDDING_DB::AcquireInstance();
    if (!database || !database->GetVector(id, pImpl->m_getter_face_feature_cache_)) {
        return database ? HERR_FT_HUB_NOT_FOUND_FEATURE : HERR_FT_HUB_DATABASE_FAILURE;
    }
    if (pImpl->m_face_feature_ptr_cache_) {
        pImpl->m_face_feature_ptr_cache_->data = pImpl->m_getter_face_feature_cache_.data();
        pImpl->m_face_feature_ptr_cache_->dataSize = static_cast<int32_t>(pImpl->m_getter_face_feature_cache_.size());
    }
    return HSUCCEED;
}

int32_t FeatureHubDB::GetFaceFeature(int32_t id, std::vector<float> &feature) {
    return GetFaceFeature(static_cast<int64_t>(id), feature);
}

int32_t FeatureHubDB::GetFaceFeature(int64_t id, std::vector<float> &feature) {
    std::lock_guard<std::mutex> lock(mutex_);
    feature.clear();
    if (!pImpl->m_enable_) {
        return HERR_FT_HUB_DISABLE;
    }
    const auto database = EMBEDDING_DB::AcquireInstance();
    if (!database || !database->GetVector(id, feature)) {
        return database ? HERR_FT_HUB_NOT_FOUND_FEATURE : HERR_FT_HUB_DATABASE_FAILURE;
    }
    return HSUCCEED;
}

int32_t FeatureHubDB::GetFaceFeature(int32_t id, FaceEmbedding &feature) {
    return GetFaceFeature(static_cast<int64_t>(id), feature);
}

int32_t FeatureHubDB::GetFaceFeature(int64_t id, FaceEmbedding &feature) {
    return GetFaceFeature(id, feature.embedding);
}

int32_t FeatureHubDB::ViewDBTable() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pImpl->m_enable_) {
        return HERR_FT_HUB_DISABLE;
    }
    const auto database = EMBEDDING_DB::AcquireInstance();
    return database && database->ShowTable() ? HSUCCEED : HERR_FT_HUB_DATABASE_FAILURE;
}

void FeatureHubDB::SetRecognitionThreshold(float threshold) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!std::isfinite(threshold) || threshold < -1.0f || threshold > 1.0f) {
        INSPIRE_LOGW("Ignoring invalid FeatureHub threshold");
        return;
    }
    pImpl->m_recognition_threshold_ = threshold;
}

void FeatureHubDB::SetRecognitionSearchMode(SearchMode mode) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode != SEARCH_MODE_EAGER && mode != SEARCH_MODE_EXHAUSTIVE) {
        INSPIRE_LOGW("Ignoring invalid FeatureHub search mode");
        return;
    }
    pImpl->m_search_mode_ = mode;
}

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
