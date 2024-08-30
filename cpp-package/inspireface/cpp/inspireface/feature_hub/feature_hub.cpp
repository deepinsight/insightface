//
// Created by tunm on 2023/9/8.
//

#include "feature_hub.h"
#include "simd.h"
#include "herror.h"
#include <thread>


namespace inspire {

std::mutex FeatureHub::mutex_;
std::shared_ptr<FeatureHub> FeatureHub::instance_ = nullptr;

FeatureHub::FeatureHub(){}

std::shared_ptr<FeatureHub> FeatureHub::GetInstance() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!instance_) {
        instance_ = std::shared_ptr<FeatureHub>(new FeatureHub());
    }
    return instance_;
}

int32_t FeatureHub::DisableHub() {
    if (!m_enable_) {
        INSPIRE_LOGW("FeatureHub is already disabled.");
        return HERR_FT_HUB_DISABLE_REPETITION;
    }
    // Close the database if it starts
    if (m_db_) {
        int ret = m_db_->CloseDatabase();
        if (ret != HSUCCEED) {
            INSPIRE_LOGE("Failed to close the database: %d", ret);
            return ret;
        }
        m_db_.reset();
    }

    m_feature_matrix_list_.clear();
    m_search_face_feature_cache_.clear();

    m_db_configuration_ = DatabaseConfiguration();  // Reset using the default constructor
    m_recognition_threshold_ = 0.0f;
    m_search_mode_ = SEARCH_MODE_EAGER;

    m_face_feature_ptr_cache_.reset();
    m_enable_ = false;

    return HSUCCEED;
}

int32_t FeatureHub::EnableHub(const DatabaseConfiguration &configuration, MatrixCore core) {
    int32_t ret;
    if (m_enable_) {
        INSPIRE_LOGW("You have enabled the FeatureHub feature. It is not valid to do so again");
        return HERR_FT_HUB_ENABLE_REPETITION;
    }
    // Config
    m_db_configuration_ = configuration;
    m_recognition_threshold_ = m_db_configuration_.recognition_threshold;
    if (m_recognition_threshold_ < -1.0f || m_recognition_threshold_ > 1.0f) {
        INSPIRE_LOGW("The search threshold entered does not fit the required range (-1.0f, 1.0f) and has been set to 0.5 by default");
        m_recognition_threshold_ = 0.5f;
    }
    m_search_mode_ = m_db_configuration_.search_mode;
    if (m_db_configuration_.feature_block_num <= 0) {
        m_db_configuration_.feature_block_num = 10;
        INSPIRE_LOGW("The number of feature blocks cannot be 0, but has been set to the default number of 10, that is, the maximum number of stored faces is supported: 5120");
    } else if (m_db_configuration_.feature_block_num > 25) {
        m_db_configuration_.feature_block_num = 25;
        INSPIRE_LOGW("The number of feature blocks cannot exceed 25, which has been set to the maximum value, that is, the maximum number of stored faces supported: 12800");
    }
    // Allocate memory for the feature matrix
    for (int i = 0; i < m_db_configuration_.feature_block_num; ++i) {
        std::shared_ptr<FeatureBlock> block;
        block.reset(FeatureBlock::Create(core, 512, 512));
        m_feature_matrix_list_.push_back(block);
    }
    if (m_db_configuration_.enable_use_db) {
        m_db_ = std::make_shared<SQLiteFaceManage>();
        if (IsDirectory(m_db_configuration_.db_path)){
            std::string dbFile = m_db_configuration_.db_path + "/" + DB_FILE_NAME;
            ret = m_db_->OpenDatabase(dbFile);
        } else {
            ret = m_db_->OpenDatabase(m_db_configuration_.db_path);
        }
        if (ret != HSUCCEED) {
            INSPIRE_LOGE("An error occurred while opening the database: %d", ret);
            return ret;
        }

        std::vector<FaceFeatureInfo> infos;
        ret = m_db_->GetTotalFeatures(infos);
        if (ret == HSUCCEED) {
            if (!infos.empty()) {
                for (auto const &info: infos) {
                    ret = InsertFaceFeature(info.feature, info.tag, info.customId);
                    if (ret != HSUCCEED) {
                        INSPIRE_LOGE("ID: %d, Inserting error: %d", info.customId, ret);
                        return ret;
                    }
                }
            }
            m_enable_ = true;
        } else {
            INSPIRE_LOGE("Failed to get the vector from the database.");
            return ret;
        }
    } else {
        m_enable_ = true;
    }

    m_face_feature_ptr_cache_ = std::make_shared<FaceFeatureEntity>();

    return HSUCCEED;
}

int32_t FeatureHub::CosineSimilarity(const std::vector<float>& v1, const std::vector<float>& v2, float &res) {
    if (v1.size() != v2.size() || v1.empty()) {
        return HERR_SESS_REC_CONTRAST_FEAT_ERR; // The similarity cannot be calculated if the vector lengths are not equal
    }
    // Calculate the cosine similarity
    res = simd_dot(v1.data(), v2.data(), v1.size());

    return HSUCCEED;
}


int32_t FeatureHub::CosineSimilarity(const float *v1, const float *v2, int32_t size, float &res) {
    res = simd_dot(v1, v2, size);

    return HSUCCEED;
}


int32_t FeatureHub::RegisterFaceFeature(const std::vector<float>& feature, int featureIndex, const std::string &tag, int32_t customId) {
    if (featureIndex < 0 || featureIndex >= m_feature_matrix_list_.size() * NUM_OF_FEATURES_IN_BLOCK) {
        return HERR_SESS_REC_INVALID_INDEX; // Invalid feature index number
    }

    // Compute which FeatureBlock and which row the feature vector should be stored in
    int blockIndex = featureIndex / NUM_OF_FEATURES_IN_BLOCK; // The FeatureBlock where the computation is located
    int rowIndex = featureIndex % NUM_OF_FEATURES_IN_BLOCK;   // Calculate the line number in the FeatureBlock

    // Call the appropriate FeatureBlock registration function
    int32_t result = m_feature_matrix_list_[blockIndex]->RegisterFeature(rowIndex, feature, tag, customId);

    return result;
}

int32_t FeatureHub::InsertFaceFeature(const std::vector<float>& feature, const std::string &tag, int32_t customId) {
    int32_t ret = HSUCCEED;
    for (int i = 0; i < m_feature_matrix_list_.size(); ++i) {
        auto &block = m_feature_matrix_list_[i];
        ret = block->AddFeature(feature, tag, customId);
        if (ret != HERR_SESS_REC_BLOCK_FULL) {
            break;
        }
    }

    return ret;
}

int32_t FeatureHub::SearchFaceFeature(const std::vector<float>& queryFeature, SearchResult &searchResult, float threshold, bool mostSimilar) {
    if (queryFeature.size() != NUM_OF_FEATURES_IN_BLOCK) {
        return HERR_SESS_REC_FEAT_SIZE_ERR; // Query feature size does not match expectations
    }

    bool found = false; // Whether matching features are found
    float maxScore = -1.0f; // The maximum score is initialized to a negative number
    int maxIndex = -1; // The index corresponding to the maximum score
    std::string tag = "None";
    int maxCid = -1;

    for (int blockIndex = 0; blockIndex < m_feature_matrix_list_.size(); ++blockIndex) {
        if (m_feature_matrix_list_[blockIndex]->GetUsedCount() == 0) {
            // If the FeatureBlock has no used features, skip to the next block
            continue;
        }

        int startIndex = blockIndex * NUM_OF_FEATURES_IN_BLOCK;
        SearchResult tempResult;

        // Call the appropriate FeatureBlock search function
        int32_t result = m_feature_matrix_list_[blockIndex]->SearchNearest(queryFeature, tempResult);

        if (result != HSUCCEED) {
            // Error
            return result;
        }

        // If you find a higher score feature
        if (tempResult.score > maxScore) {
            maxScore = tempResult.score;
            maxIndex = startIndex + tempResult.index;
            tag = tempResult.tag;
            maxCid = tempResult.customId;
            if (maxScore >= threshold) {
                found = true;
                if (!mostSimilar) {
                    // Use Eager-Mode: When the score is greater than or equal to the threshold, stop searching for the next FeatureBlock
                    break;
                }
            }
        }
    }

    if (found) {
        searchResult.score = maxScore;
        searchResult.index = maxIndex;
        searchResult.tag = tag;
        searchResult.customId = maxCid;
    } else {
        searchResult.score = -1.0f;
        searchResult.index = -1;
        searchResult.tag = "None";
        searchResult.customId = -1;
    }

    return HSUCCEED; // No matching feature found but not an error
}



int32_t FeatureHub::SearchFaceFeatureTopK(const std::vector<float>& queryFeature, std::vector<SearchResult> &searchResultList, size_t maxTopK, float threshold) {
    if (queryFeature.size() != NUM_OF_FEATURES_IN_BLOCK) {
        return HERR_SESS_REC_FEAT_SIZE_ERR;
    }

    std::vector<SearchResult> tempResultList;
    searchResultList.clear();

    for (int blockIndex = 0; blockIndex < m_feature_matrix_list_.size(); ++blockIndex) {
        if (m_feature_matrix_list_[blockIndex]->GetUsedCount() == 0) {
            continue;
        }

        tempResultList.clear();
        int32_t result = m_feature_matrix_list_[blockIndex]->SearchTopKNearest(queryFeature, maxTopK, tempResultList);
        if (result != HSUCCEED) {
            return result;
        }

        for (const SearchResult& result : tempResultList) {
            if (result.score >= threshold) {
                searchResultList.push_back(result);
            }
        }
    }

    std::sort(searchResultList.begin(), searchResultList.end(), [](const SearchResult& a, const SearchResult& b) {
        return a.score > b.score;
    });

    if (searchResultList.size() > maxTopK) {
        searchResultList.resize(maxTopK);
    }

    return HSUCCEED;
}

int32_t FeatureHub::DeleteFaceFeature(int featureIndex) {
    if (featureIndex < 0 || featureIndex >= m_feature_matrix_list_.size() * NUM_OF_FEATURES_IN_BLOCK) {
        return HERR_SESS_REC_INVALID_INDEX; // Invalid feature index number
    }

    // Calculate which FeatureBlock and which row the feature vector should be removed in
    int blockIndex = featureIndex / NUM_OF_FEATURES_IN_BLOCK; // The FeatureBlock where the computation is located
    int rowIndex = featureIndex % NUM_OF_FEATURES_IN_BLOCK;   // Calculate the line number in the FeatureBlock

    // Call the appropriate FeatureBlock delete function
    int32_t result = m_feature_matrix_list_[blockIndex]->DeleteFeature(rowIndex);

    return result;
}

int32_t FeatureHub::GetFaceFeature(int featureIndex, Embedded &feature) {
    if (featureIndex < 0 || featureIndex >= m_feature_matrix_list_.size() * NUM_OF_FEATURES_IN_BLOCK) {
        return HERR_SESS_REC_INVALID_INDEX; // Invalid feature index number
    }
    // Calculate which FeatureBlock and which row the feature vector should be removed in
    int blockIndex = featureIndex / NUM_OF_FEATURES_IN_BLOCK; // The FeatureBlock where the computation is located
    int rowIndex = featureIndex % NUM_OF_FEATURES_IN_BLOCK;   // Calculate the line number in the FeatureBlock

    int32_t result = m_feature_matrix_list_[blockIndex]->GetFeature(rowIndex, feature);

    return result;
}

int32_t FeatureHub::GetFaceEntity(int featureIndex, Embedded &feature, std::string& tag, FEATURE_STATE& status) {
    if (featureIndex < 0 || featureIndex >= m_feature_matrix_list_.size() * NUM_OF_FEATURES_IN_BLOCK) {
        return HERR_SESS_REC_INVALID_INDEX; // Invalid feature index number
    }
    // Calculate which FeatureBlock and which row the feature vector should be removed in
    int blockIndex = featureIndex / NUM_OF_FEATURES_IN_BLOCK; // The FeatureBlock where the computation is located
    int rowIndex = featureIndex % NUM_OF_FEATURES_IN_BLOCK;   // Calculate the line number in the FeatureBlock

    int32_t result = m_feature_matrix_list_[blockIndex]->GetFeature(rowIndex, feature);
    tag = m_feature_matrix_list_[blockIndex]->GetTagFromRow(rowIndex);
    status = m_feature_matrix_list_[blockIndex]->GetStateFromRow(rowIndex);


    return result;
}

int32_t FeatureHub::GetFaceFeatureCount() {
    int totalFeatureCount = 0;

    // Iterate over all FeatureBlocks and add up the number of feature vectors used
    for (const auto& block : m_feature_matrix_list_) {
        totalFeatureCount += block->GetUsedCount();
    }

    return totalFeatureCount;
}

int32_t FeatureHub::GetFeatureNum() const {
    return NUM_OF_FEATURES_IN_BLOCK;
}

int32_t FeatureHub::UpdateFaceFeature(const std::vector<float> &feature, int featureIndex, const std::string &tag, int32_t customId) {
    if (featureIndex < 0 || featureIndex >= m_feature_matrix_list_.size() * NUM_OF_FEATURES_IN_BLOCK) {
        return HERR_SESS_REC_INVALID_INDEX; // Invalid feature index number
    }

    // Calculate which FeatureBlock and which row the feature vector should be removed in
    int blockIndex = featureIndex / NUM_OF_FEATURES_IN_BLOCK; // The FeatureBlock where the computation is located
    int rowIndex = featureIndex % NUM_OF_FEATURES_IN_BLOCK;   // Calculate the line number in the FeatureBlock

    // Call the appropriate FeatureBlock registration function
    int32_t result = m_feature_matrix_list_[blockIndex]->UpdateFeature(rowIndex, feature, tag, customId);

    return result;
}

void FeatureHub::PrintFeatureMatrixInfo() {
    m_feature_matrix_list_[0]->PrintMatrix();
}


int32_t FeatureHub::FindFeatureIndexByCustomId(int32_t customId) {
    // Iterate over all FeatureBlocks
    for (int blockIndex = 0; blockIndex < m_feature_matrix_list_.size(); ++blockIndex) {
        int startIndex = blockIndex * NUM_OF_FEATURES_IN_BLOCK;

        // Query the customId from the current FeatureBlock
        int rowIndex = m_feature_matrix_list_[blockIndex]->FindIndexByCustomId(customId);

        if (rowIndex != -1) {
            return startIndex + rowIndex;  // 返回行号
        }
    }

    return -1;  // If none of the featureBlocks is found, -1 is returned
}


int32_t FeatureHub::SearchFaceFeature(const Embedded &queryFeature, SearchResult &searchResult) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!m_enable_) {
        INSPIRE_LOGE("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }
    m_search_face_feature_cache_.clear();
    std::memset(m_string_cache_, 0, sizeof(m_string_cache_)); // Initial Zero
    auto ret = SearchFaceFeature(queryFeature, searchResult, m_recognition_threshold_,
                                 m_search_mode_ == SEARCH_MODE_EXHAUSTIVE);
    if (ret == HSUCCEED) {
        if (searchResult.index != -1) {
            ret = GetFaceFeature(searchResult.index, m_search_face_feature_cache_);
        }
        m_face_feature_ptr_cache_->data = m_search_face_feature_cache_.data();
        m_face_feature_ptr_cache_->dataSize = m_search_face_feature_cache_.size();
        // Ensure that buffer overflows do not occur
        size_t copy_length = std::min(searchResult.tag.size(), sizeof(m_string_cache_) - 1);
        std::strncpy(m_string_cache_, searchResult.tag.c_str(), copy_length);
        // Make sure the string ends with a null character
        m_string_cache_[copy_length] = '\0';
    }

    return ret;
}

int32_t FeatureHub::SearchFaceFeatureTopK(const Embedded& queryFeature, size_t topK) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!m_enable_) {
        INSPIRE_LOGE("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }
    m_top_k_confidence_.clear();
    m_top_k_custom_ids_cache_.clear();
    auto ret = SearchFaceFeatureTopK(queryFeature, m_search_top_k_cache_, topK, m_recognition_threshold_);
    if (ret == HSUCCEED) {
        for (int i = 0; i < m_search_top_k_cache_.size(); ++i) {
            auto &item = m_search_top_k_cache_[i];
            m_top_k_custom_ids_cache_.push_back(item.customId);
            m_top_k_confidence_.push_back(item.score);
        }
    }

    return ret;
}

int32_t FeatureHub::FaceFeatureInsertFromCustomId(const std::vector<float> &feature, const std::string &tag,
                                                   int32_t customId) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!m_enable_) {
        INSPIRE_LOGE("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }
    auto index = FindFeatureIndexByCustomId(customId);
    if (index != -1) {
        return HERR_SESS_REC_ID_ALREADY_EXIST;
    }
    auto ret = InsertFaceFeature(feature, tag, customId);
    if (ret == HSUCCEED && m_db_ != nullptr) {
        // operational database
        FaceFeatureInfo item = {0};
        item.customId = customId;
        item.tag = tag;
        item.feature = feature;
        ret = m_db_->InsertFeature(item);
    }

    return ret;
}

int32_t FeatureHub::FaceFeatureRemoveFromCustomId(int32_t customId) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!m_enable_) {
        INSPIRE_LOGE("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }
    auto index = FindFeatureIndexByCustomId(customId);
    if (index == -1) {
        return HERR_SESS_REC_INVALID_INDEX;
    }
    auto ret = DeleteFaceFeature(index);
    if (ret == HSUCCEED && m_db_ != nullptr) {
        ret = m_db_->DeleteFeature(customId);
    }

    return ret;
}

int32_t FeatureHub::FaceFeatureUpdateFromCustomId(const std::vector<float> &feature, const std::string &tag,
                                                   int32_t customId) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!m_enable_) {
        INSPIRE_LOGE("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }
    auto index = FindFeatureIndexByCustomId(customId);
    if (index == -1) {
        return HERR_SESS_REC_INVALID_INDEX;
    }
    auto ret = UpdateFaceFeature(feature, index, tag, customId);
    if (ret == HSUCCEED && m_db_ != nullptr) {
        FaceFeatureInfo item = {0};
        item.customId = customId;
        item.tag = tag;
        item.feature = feature;
        ret = m_db_->UpdateFeature(item);
    }

    return ret;
}

int32_t FeatureHub::GetFaceFeatureFromCustomId(int32_t customId) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!m_enable_) {
        INSPIRE_LOGE("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }
    auto index = FindFeatureIndexByCustomId(customId);
    if (index == -1) {
        return HERR_SESS_REC_INVALID_INDEX;
    }
    m_getter_face_feature_cache_.clear();
    std::string tag;
    FEATURE_STATE status;
    auto ret = GetFaceEntity(index, m_getter_face_feature_cache_, tag, status);
    m_face_feature_ptr_cache_->data = m_getter_face_feature_cache_.data();
    m_face_feature_ptr_cache_->dataSize = m_getter_face_feature_cache_.size();
    // Ensure that buffer overflows do not occur
    size_t copy_length = std::min(tag.size(), sizeof(m_string_cache_) - 1);
    std::strncpy(m_string_cache_, tag.c_str(), copy_length);
    // Make sure the string ends with a null character
    m_string_cache_[copy_length] = '\0';

    return ret;
}

int32_t FeatureHub::ViewDBTable() {
    if (!m_enable_) {
        INSPIRE_LOGE("FeatureHub is disabled, please enable it before it can be served");
        return HERR_FT_HUB_DISABLE;
    }
    auto ret = m_db_->ViewTotal();
    return ret;
}

void FeatureHub::SetRecognitionThreshold(float threshold) {
    m_recognition_threshold_ = threshold;
}

void FeatureHub::SetRecognitionSearchMode(SearchMode mode) {
    m_search_mode_ = mode;
}

// =========== Getter ===========

const Embedded& FeatureHub::GetSearchFaceFeatureCache() const {
    return m_search_face_feature_cache_;
}

char *FeatureHub::GetStringCache() {
    return m_string_cache_;
}

const std::shared_ptr<FaceFeaturePtr>& FeatureHub::GetFaceFeaturePtrCache() const {
    return m_face_feature_ptr_cache_;
}

std::vector<float> &FeatureHub::GetTopKConfidence() {
    return m_top_k_confidence_;
}

std::vector<int32_t> &FeatureHub::GetTopKCustomIdsCache() {
    return m_top_k_custom_ids_cache_;
}


} // namespace hyper
