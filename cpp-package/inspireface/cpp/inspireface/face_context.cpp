//
// Created by Tunm-Air13 on 2023/9/7.
//

#include "face_context.h"
#include "Initialization_module/launch.h"
#include <utility>
#include "log.h"
#include "herror.h"
#include "middleware/utils.h"

namespace inspire {


FaceContext::FaceContext() = default;

int32_t FaceContext::Configuration(DetectMode detect_mode, 
                                    int32_t max_detect_face,
                                    CustomPipelineParameter param, 
                                    int32_t detect_level_px, 
                                    int32_t track_by_detect_mode_fps) {
    m_detect_mode_ = detect_mode;
    m_max_detect_face_ = max_detect_face;
    m_parameter_ = param;
    if (!INSPIRE_LAUNCH->isMLoad()) {
        return HERR_ARCHIVE_NOT_LOAD;
    }
    if (INSPIRE_LAUNCH->getMArchive().QueryStatus() != SARC_SUCCESS) {
        return HERR_ARCHIVE_LOAD_FAILURE;
    }

    m_face_track_ = std::make_shared<FaceTrack>(m_detect_mode_, m_max_detect_face_, 20, 192, detect_level_px, track_by_detect_mode_fps);
    m_face_track_->Configuration(INSPIRE_LAUNCH->getMArchive());
    // SetDetectMode(m_detect_mode_);

    m_face_recognition_ = std::make_shared<FeatureExtraction>(INSPIRE_LAUNCH->getMArchive(), m_parameter_.enable_recognition);
    if (m_face_recognition_->QueryStatus() != HSUCCEED) {
        return m_face_recognition_->QueryStatus();
    }

    m_face_pipeline_ = std::make_shared<FacePipeline>(
            INSPIRE_LAUNCH->getMArchive(),
            param.enable_liveness,
            param.enable_mask_detect,
            param.enable_face_attribute,
            param.enable_interaction_liveness
    );

    return HSUCCEED;
}


int32_t FaceContext::FaceDetectAndTrack(CameraStream &image) {
    std::lock_guard<std::mutex> lock(m_mtx_);
    m_detect_cache_.clear();
    m_face_basic_data_cache_.clear();
    m_face_rects_cache_.clear();
    m_track_id_cache_.clear();
    m_quality_results_cache_.clear();
    m_roll_results_cache_.clear();
    m_yaw_results_cache_.clear();
    m_pitch_results_cache_.clear();
    m_quality_score_results_cache_.clear();
    m_react_left_eye_results_cache_.clear();
    m_react_right_eye_results_cache_.clear();

    m_action_normal_results_cache_.clear();
    m_action_shake_results_cache_.clear();
    m_action_blink_results_cache_.clear();
    m_action_jaw_open_results_cache_.clear();
    m_action_raise_head_results_cache_.clear();

    m_quality_score_results_cache_.clear();
    m_attribute_race_results_cache_.clear();
    m_attribute_gender_results_cache_.clear();
    if (m_face_track_ == nullptr) {
        return HERR_SESS_TRACKER_FAILURE;
    }
    m_face_track_->UpdateStream(image);
    for (int i = 0; i < m_face_track_->trackingFace.size(); ++i) {
        auto &face = m_face_track_->trackingFace[i];
        HyperFaceData data = FaceObjectToHyperFaceData(face, i);
        ByteArray byteArray;
        auto ret = SerializeHyperFaceData(data, byteArray);
        if (ret != HSUCCEED) {
            return HERR_INVALID_SERIALIZATION_FAILED;
        }
        m_detect_cache_.push_back(byteArray);
        m_track_id_cache_.push_back(face.GetTrackingId());
        m_face_rects_cache_.push_back(data.rect);
        m_quality_results_cache_.push_back(face.high_result);
        m_roll_results_cache_.push_back(face.high_result.roll);
        m_yaw_results_cache_.push_back(face.high_result.yaw);
        m_pitch_results_cache_.push_back(face.high_result.pitch);
        // Process quality scores
        float avg = 0.0f;
        for (int j = 0; j < 5; ++j) {
            avg += data.quality[j];
        }
        avg /= 5.0f;
        float quality_score = 1.0f - avg;    // reversal
        m_quality_score_results_cache_.push_back(quality_score);
    }
    // ptr face_basic
    m_face_basic_data_cache_.resize(m_face_track_->trackingFace.size());
    for (int i = 0; i < m_face_basic_data_cache_.size(); ++i) {
        auto &basic = m_face_basic_data_cache_[i];
        basic.dataSize = m_detect_cache_[i].size();
        basic.data = m_detect_cache_[i].data();
    }


//    LOGD("Track COST: %f", m_face_track_->GetTrackTotalUseTime());
    return HSUCCEED;
}

int32_t FaceContext::SetFaceDetectThreshold(float value) {
    m_face_track_->SetDetectThreshold(value);
    return HSUCCEED;
}

FaceObjectList& FaceContext::GetTrackingFaceList() {
    return m_face_track_->trackingFace;
}

const std::shared_ptr<FeatureExtraction>& FaceContext::FaceRecognitionModule() {
    return m_face_recognition_;
}

const std::shared_ptr<FacePipeline>& FaceContext::FacePipelineModule() {
    return m_face_pipeline_;
}


const int32_t FaceContext::GetNumberOfFacesCurrentlyDetected() const {
    return m_face_track_->trackingFace.size();
}

int32_t FaceContext::FacesProcess(CameraStream &image, const std::vector<HyperFaceData> &faces, const CustomPipelineParameter &param) {
    std::lock_guard<std::mutex> lock(m_mtx_);
    m_mask_results_cache_.resize(faces.size(), -1.0f);
    m_rgb_liveness_results_cache_.resize(faces.size(), -1.0f);
    m_react_left_eye_results_cache_.resize(faces.size(), -1.0f);
    m_react_right_eye_results_cache_.resize(faces.size(), -1.0f);
    m_attribute_race_results_cache_.resize(faces.size(), -1);
    m_attribute_gender_results_cache_.resize(faces.size(), -1);
    m_attribute_age_results_cache_.resize(faces.size(), -1);
    m_action_normal_results_cache_.resize(faces.size(), -1);
    m_action_jaw_open_results_cache_.resize(faces.size(), -1);
    m_action_blink_results_cache_.resize(faces.size(), -1);
    m_action_raise_head_results_cache_.resize(faces.size(), -1);
    m_action_shake_results_cache_.resize(faces.size(), -1);
    for (int i = 0; i < faces.size(); ++i) {
        const auto &face = faces[i];
        // RGB Liveness Detect
        if (param.enable_liveness) {
            auto ret = m_face_pipeline_->Process(image, face, PROCESS_RGB_LIVENESS);
            if (ret != HSUCCEED) {
                return ret;
            }
            m_rgb_liveness_results_cache_[i] = m_face_pipeline_->faceLivenessCache;
        }
        // Mask detection
        if (param.enable_mask_detect) {
            auto ret = m_face_pipeline_->Process(image, face, PROCESS_MASK);
            if (ret != HSUCCEED) {
                return ret;
            }
            m_mask_results_cache_[i] = m_face_pipeline_->faceMaskCache;
        }
        // Face attribute prediction
        if (param.enable_face_attribute) {
            auto ret = m_face_pipeline_->Process(image, face, PROCESS_ATTRIBUTE);
            if (ret != HSUCCEED) {
                return ret;
            }
            m_attribute_race_results_cache_[i] = m_face_pipeline_->faceAttributeCache[0];
            m_attribute_gender_results_cache_[i] = m_face_pipeline_->faceAttributeCache[1];
            m_attribute_age_results_cache_[i] = m_face_pipeline_->faceAttributeCache[2];
        }

        // Face interaction
        if (param.enable_interaction_liveness) {
            auto ret = m_face_pipeline_->Process(image, face, PROCESS_INTERACTION);
            if (ret != HSUCCEED) {
                return ret;
            }
            // Get eyes status
            m_react_left_eye_results_cache_[i] = m_face_pipeline_->eyesStatusCache[0];
            m_react_right_eye_results_cache_[i] = m_face_pipeline_->eyesStatusCache[1];
            // Special handling:  ff it is a tracking state, it needs to be filtered
            if (face.trackState > 0)
            {   
                auto idx = face.inGroupIndex;
                if (idx < m_face_track_->trackingFace.size()) {
                    auto& target = m_face_track_->trackingFace[idx];
                    if (target.GetTrackingId() == face.trackId) {
                        auto new_eye_left = EmaFilter(m_face_pipeline_->eyesStatusCache[0], target.left_eye_status_, 8, 0.2f);
                        auto new_eye_right = EmaFilter(m_face_pipeline_->eyesStatusCache[1], target.right_eye_status_, 8, 0.2f);
                        if (face.trackState > 1) {
                            // The filtered value can be obtained only in the tracking state
                            m_react_left_eye_results_cache_[i] = new_eye_left;
                            m_react_right_eye_results_cache_[i] = new_eye_right;
                        }
                        const auto actions = target.UpdateFaceAction();
                        m_action_normal_results_cache_[i] = actions.normal;
                        m_action_jaw_open_results_cache_[i] = actions.jawOpen;
                        m_action_blink_results_cache_[i] = actions.blink;
                        m_action_raise_head_results_cache_[i] = actions.raiseHead;
                        m_action_shake_results_cache_[i] = actions.shake;
                    } else {
                        INSPIRE_LOGD("Serialized objects cannot connect to trace objects in memory, and there may be some problems");
                    }
                } else {
                    INSPIRE_LOGW("The index of the trace object does not match the trace list in memory, and there may be some problems");
                }
            }
        }

    }

    return 0;
}


const std::vector<ByteArray>& FaceContext::GetDetectCache() const {
    return m_detect_cache_;
}

const std::vector<FaceBasicData>& FaceContext::GetFaceBasicDataCache() const {
    return m_face_basic_data_cache_;
}

const std::vector<FaceRect>& FaceContext::GetFaceRectsCache() const {
    return m_face_rects_cache_;
}

const std::vector<int32_t>& FaceContext::GetTrackIDCache() const {
    return m_track_id_cache_;
}

const std::vector<float>& FaceContext::GetRollResultsCache() const {
    return m_roll_results_cache_;
}

const std::vector<float>& FaceContext::GetYawResultsCache() const {
    return m_yaw_results_cache_;
}

const std::vector<float>& FaceContext::GetPitchResultsCache() const {
    return m_pitch_results_cache_;
}

const std::vector<FacePoseQualityResult>& FaceContext::GetQualityResultsCache() const {
    return m_quality_results_cache_;
}

const std::vector<float>& FaceContext::GetMaskResultsCache() const {
    return m_mask_results_cache_;
}

const std::vector<float>& FaceContext::GetRgbLivenessResultsCache() const {
    return m_rgb_liveness_results_cache_;
}

const std::vector<float>& FaceContext::GetFaceQualityScoresResultsCache() const {
    return m_quality_score_results_cache_;
}

const std::vector<float>& FaceContext::GetFaceInteractionLeftEyeStatusCache() const {
    return m_react_left_eye_results_cache_;
}

const std::vector<float>& FaceContext::GetFaceInteractionRightEyeStatusCache() const {
    return m_react_right_eye_results_cache_;
}

const Embedded& FaceContext::GetFaceFeatureCache() const {
    return m_face_feature_cache_;
}

const std::vector<int>& FaceContext::GetFaceRaceResultsCache() const {
    return m_attribute_race_results_cache_;
}

const std::vector<int>& FaceContext::GetFaceGenderResultsCache() const {
    return m_attribute_gender_results_cache_;
}

const std::vector<int>& FaceContext::GetFaceAgeBracketResultsCache() const {
    return m_attribute_age_results_cache_;
}

const std::vector<int>& FaceContext::GetFaceNormalAactionsResultCache() const {
    return m_action_normal_results_cache_;
}

const std::vector<int>& FaceContext::GetFaceJawOpenAactionsResultCache() const {
    return m_action_jaw_open_results_cache_;
}

const std::vector<int>& FaceContext::GetFaceBlinkAactionsResultCache() const {
    return m_action_blink_results_cache_;
}

const std::vector<int>& FaceContext::GetFaceShakeAactionsResultCache() const {
    return m_action_shake_results_cache_;
}

const std::vector<int>& FaceContext::GetFaceRaiseHeadAactionsResultCache() const {
    return m_action_raise_head_results_cache_;
}

int32_t FaceContext::FaceFeatureExtract(CameraStream &image, FaceBasicData& data) {
    std::lock_guard<std::mutex> lock(m_mtx_);
    int32_t ret;
    HyperFaceData face = {0};
    ret = DeserializeHyperFaceData((char* )data.data, data.dataSize, face);
    if (ret != HSUCCEED) {
        return ret;
    }
    m_face_feature_cache_.clear();
    ret = m_face_recognition_->FaceExtract(image, face, m_face_feature_cache_);

    return ret;
}


const CustomPipelineParameter &FaceContext::getMParameter() const {
    return m_parameter_;
}


int32_t FaceContext::FaceQualityDetect(FaceBasicData& data, float &result) {
    int32_t ret;
    HyperFaceData face = {0};
    ret = DeserializeHyperFaceData((char* )data.data, data.dataSize, face);
//    PrintHyperFaceData(face);
    if (ret != HSUCCEED) {
        return ret;
    }
    float avg = 0.0f;
    for (int i = 0; i < 5; ++i) {
        avg += face.quality[i];
    }
    avg /= 5.0f;
    result = 1.0f - avg;    // reversal

    return ret;
}


int32_t FaceContext::SetDetectMode(DetectMode mode) {
    m_detect_mode_ = mode;
    if (m_detect_mode_ == DetectMode::DETECT_MODE_ALWAYS_DETECT) {
        m_always_detect_ = true;
    } else {
        m_always_detect_ = false;
    }
    return HSUCCEED;
}

int32_t FaceContext::SetTrackPreviewSize(const int32_t preview_size) {
    m_face_track_->SetTrackPreviewSize(preview_size);
    return HSUCCEED;
}

int32_t FaceContext::SetTrackFaceMinimumSize(int32_t minSize) {
    m_face_track_->SetMinimumFacePxSize(minSize);
    return HSUCCEED;
}

}   // namespace hyper