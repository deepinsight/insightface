//
// Created by Tunm-Air13 on 2023/9/11.
//

#include "feature_block_opencv.h"
#include "herror.h"
#include "log.h"

namespace inspire {


FeatureBlockOpenCV::FeatureBlockOpenCV(int32_t features_max, int32_t feature_length)
    :m_feature_matrix_(features_max, feature_length, CV_32F, cv::Scalar(0.0f)){

}

int32_t FeatureBlockOpenCV::UnsafeAddFeature(const std::vector<float> &feature, const std::string &tag, int32_t customId) {
    if (feature.empty()) {
        return HERR_SESS_REC_ADD_FEAT_EMPTY; // If the feature is empty, it is not added
    }
    if (feature.size() != m_feature_length_) {
        return HERR_SESS_REC_FEAT_SIZE_ERR;
    }

    if (IsUsedFull()) {
        return HERR_SESS_REC_BLOCK_FULL;
    }

    cv::Mat newFeatureMat(1, feature.size(), CV_32FC1);
    for (int i = 0; i < feature.size(); ++i) {
        newFeatureMat.at<float>(0, i) = feature[i];
    }
    auto idx = FindFirstIdleIndex();    // Find the first free vector position
    if (idx == -1) {
        return HERR_SESS_REC_BLOCK_FULL;
    }
    cv::Mat rowToUpdate = m_feature_matrix_.row(idx);
    newFeatureMat.copyTo(rowToUpdate);

    m_feature_state_[idx] = FEATURE_STATE::USED;    // Set feature vector used
    m_tag_list_[idx] = tag;
    m_custom_id_list_[idx] = customId;

    return HSUCCEED;
}

int32_t FeatureBlockOpenCV::UnsafeDeleteFeature(int rowToDelete) {
    if (m_feature_matrix_.empty() || rowToDelete < 0 || rowToDelete >= m_feature_matrix_.rows) {
        return HERR_SESS_REC_DEL_FAILURE; // Invalid row numbers or matrices are empty and will not be deleted
    }

    cv::Mat rowToUpdate = m_feature_matrix_.row(rowToDelete);
    if (m_feature_state_[rowToDelete] == FEATURE_STATE::IDLE) {
        return HERR_SESS_REC_BLOCK_DEL_FAILURE; // Rows are idle and will not be deleted
    }

    m_feature_state_[rowToDelete] = FEATURE_STATE::IDLE;
    m_custom_id_list_[rowToDelete] = -1;

    return HSUCCEED;
}


int32_t FeatureBlockOpenCV::UnsafeRegisterFeature(int rowToUpdate, const std::vector<float> &feature, const std::string &tag, int32_t customId) {
    if (rowToUpdate < 0 || rowToUpdate >= m_feature_matrix_.rows) {
        return HERR_SESS_REC_FEAT_SIZE_ERR; // Invalid line number, not updated
    }

    if (feature.size() != m_feature_length_) {
        return HERR_SESS_REC_FEAT_SIZE_ERR; // The new feature does not match the expected size and will not be updated
    }
    cv::Mat rowToUpdateMat = m_feature_matrix_.row(rowToUpdate);
    // 将新特征拷贝到指定行
    for (int i = 0; i < feature.size(); ++i) {
        rowToUpdateMat.at<float>(0, i) = feature[i];
    }
    m_feature_state_[rowToUpdate] = USED;
    m_tag_list_[rowToUpdate] = tag;
    m_custom_id_list_[rowToUpdate] = customId;

    return 0;
}

int32_t FeatureBlockOpenCV::UnsafeUpdateFeature(int rowToUpdate, const std::vector<float> &newFeature, const std::string &tag, int32_t customId) {
    if (rowToUpdate < 0 || rowToUpdate >= m_feature_matrix_.rows) {
        return HERR_SESS_REC_FEAT_SIZE_ERR; // Invalid line number, not updated
    }

    if (newFeature.size() != m_feature_length_) {
        return HERR_SESS_REC_FEAT_SIZE_ERR; // The new feature does not match the expected size and will not be updated
    }

    cv::Mat rowToUpdateMat = m_feature_matrix_.row(rowToUpdate);
    if (m_feature_state_[rowToUpdate] == FEATURE_STATE::IDLE) {
        return HERR_SESS_REC_BLOCK_UPDATE_FAILURE; // Rows are idle and not updated
    }

    // Copies the new feature to the specified row
    for (int i = 0; i < newFeature.size(); ++i) {
        rowToUpdateMat.at<float>(0, i) = newFeature[i];
    }
    m_tag_list_[rowToUpdate] = tag;
    m_custom_id_list_[rowToUpdate] = customId;

    return HSUCCEED;
}

int32_t FeatureBlockOpenCV::SearchNearest(const std::vector<float>& queryFeature, SearchResult &searchResult) {
    std::lock_guard<std::mutex> lock(m_mtx_);

    if (queryFeature.size() != m_feature_length_) {
        return HERR_SESS_REC_FEAT_SIZE_ERR;
    }

    if (GetUsedCount() == 0) {
        return HSUCCEED;
    }

    cv::Mat queryMat(queryFeature.size(), 1, CV_32FC1, (void*)queryFeature.data());

    // Calculate the cosine similarity matrix
    cv::Mat cosineSimilarities;
    cv::gemm(m_feature_matrix_, queryMat, 1, cv::Mat(), 0, cosineSimilarities);
    // Asserts that cosineSimilarities are the vector of m_features_max_ x 1
    assert(cosineSimilarities.rows == m_features_max_ && cosineSimilarities.cols == 1);

    // Used to store similarity scores and their indexes
    std::vector<std::pair<float, int>> similarityScores;

    for (int i = 0; i < m_features_max_; ++i) {
        // Check whether the status is IDLE
        if (m_feature_state_[i] == FEATURE_STATE::IDLE) {
            continue; // Skip the eigenvector of IDLE state
        }

        // Gets the similarity score for line i
        float similarityScore = cosineSimilarities.at<float>(i, 0);

        // Adds the similarity score and index to the vector as a pair
        similarityScores.push_back(std::make_pair(similarityScore, i));
    }

    // Find the index of the largest scores in similarityScores
    if (!similarityScores.empty()) {
        auto maxScoreIter = std::max_element(similarityScores.begin(), similarityScores.end());
        float maxScore = maxScoreIter->first;
        int maxScoreIndex = maxScoreIter->second;

        // Sets the value in the searchResult
        searchResult.score = maxScore;
        searchResult.index = maxScoreIndex;
        searchResult.tag = m_tag_list_[maxScoreIndex];
        searchResult.customId = m_custom_id_list_[maxScoreIndex];

        return HSUCCEED; // Indicates that the maximum score is found
    }


    searchResult.score = -1.0f;
    searchResult.index = -1;

    return HSUCCEED;
}


int32_t FeatureBlockOpenCV::SearchTopKNearest(const std::vector<float> &queryFeature, size_t topK, std::vector<SearchResult> &searchResults) {
    std::lock_guard<std::mutex> lock(m_mtx_);

    if (queryFeature.size() != m_feature_length_) {
        return HERR_SESS_REC_FEAT_SIZE_ERR;
    }

    if (GetUsedCount() == 0) {
        return HSUCCEED;
    }

    cv::Mat queryMat(queryFeature.size(), 1, CV_32FC1, (void*)queryFeature.data());

    // Calculate the cosine similarity matrix
    cv::Mat cosineSimilarities;
    cv::gemm(m_feature_matrix_, queryMat, 1, cv::Mat(), 0, cosineSimilarities);
    // Asserts that cosineSimilarities are the vector of m_features_max_ x 1
    assert(cosineSimilarities.rows == m_features_max_ && cosineSimilarities.cols == 1);

    // Used to store similarity scores and their indexes
    std::vector<std::pair<float, int>> similarityScores;

    for (int i = 0; i < m_features_max_; ++i) {
        // Check whether the status is IDLE
        if (m_feature_state_[i] == FEATURE_STATE::IDLE) {
            continue; // Skip the eigenvector of IDLE state
        }

        // Gets the similarity score for line i
        float similarityScore = cosineSimilarities.at<float>(i, 0);

        // Adds the similarity score and index to the vector as a pair
        similarityScores.push_back(std::make_pair(similarityScore, i));
    }

    searchResults.clear();
    if (similarityScores.size() < topK) {
        topK = similarityScores.size();
    }
    std::partial_sort(similarityScores.begin(), similarityScores.begin() + topK, similarityScores.end(),
                      [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                          return a.first > b.first;
                      });

    for (size_t i = 0; i < topK; i++) {
        SearchResult result;
        result.score = similarityScores[i].first;
        result.index = similarityScores[i].second;
        result.tag = m_tag_list_[result.index];
        result.customId = m_custom_id_list_[result.index];
        searchResults.push_back(result);
    }

    return HSUCCEED;
}

void FeatureBlockOpenCV::PrintMatrixSize() {
    std::cout << m_feature_matrix_.size << std::endl;
}

void FeatureBlockOpenCV::PrintMatrix() {
    INSPIRE_LOGD("Num of Features: %d", m_feature_matrix_.cols);
    INSPIRE_LOGD("Feature length: %d", m_feature_matrix_.rows);
}

int32_t FeatureBlockOpenCV::GetFeature(int row, std::vector<float> &feature) {
    if (row < 0 || row >= m_feature_matrix_.rows) {
        return HERR_SESS_REC_FEAT_SIZE_ERR; // Invalid line number, not updated
    }
    cv::Mat feat = m_feature_matrix_.row(row);
    // Copies the new feature to the specified row
    for (int i = 0; i < m_feature_length_; ++i) {
        feature.push_back(feat.at<float>(0, i));
    }

    return HSUCCEED;
}



}   // namespace hyper
