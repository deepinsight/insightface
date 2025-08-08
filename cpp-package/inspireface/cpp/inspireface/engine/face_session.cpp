/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "face_session.h"
#include <launch.h>
#include <utility>
#include "log.h"
#include "herror.h"
#include "middleware/utils.h"
#include "recognition_module/dest_const.h"

namespace inspire {

FaceSession::FaceSession() = default;

int32_t FaceSession::Configuration(DetectModuleMode detect_mode, int32_t max_detect_face, CustomPipelineParameter param, int32_t detect_level_px,
                                   int32_t track_by_detect_mode_fps) {
    m_detect_mode_ = detect_mode;
    m_max_detect_face_ = max_detect_face;
    m_parameter_ = param;
    if (!INSPIREFACE_CONTEXT->isMLoad()) {
        return HERR_ARCHIVE_NOT_LOAD;
    }
    if (INSPIREFACE_CONTEXT->getMArchive().QueryStatus() != SARC_SUCCESS) {
        return HERR_ARCHIVE_LOAD_FAILURE;
    }

    m_face_track_ = std::make_shared<FaceTrackModule>(m_detect_mode_, m_max_detect_face_, 20, 192, detect_level_px, track_by_detect_mode_fps, true);
    m_face_track_->Configuration(INSPIREFACE_CONTEXT->getMArchive(), "", m_parameter_.enable_face_pose || m_parameter_.enable_face_quality);
    // SetDetectMode(m_detect_mode_);

    m_face_recognition_ = std::make_shared<FeatureExtractionModule>(INSPIREFACE_CONTEXT->getMArchive(), m_parameter_.enable_recognition);
    if (m_face_recognition_->QueryStatus() != HSUCCEED) {
        return m_face_recognition_->QueryStatus();
    }

    m_face_pipeline_ = std::make_shared<FacePipelineModule>(INSPIREFACE_CONTEXT->getMArchive(), param.enable_liveness, param.enable_mask_detect,
                                                            param.enable_face_attribute, param.enable_interaction_liveness, param.enable_face_emotion);
    m_face_track_cost_ = std::make_shared<inspire::SpendTimer>("FaceTrack");

    return HSUCCEED;
}

int32_t FaceSession::FaceDetectAndTrack(inspirecv::FrameProcess& process) {
    std::lock_guard<std::mutex> lock(m_mtx_);
    if (m_enable_track_cost_spend_) {
        m_face_track_cost_->Start();
    }
    m_detect_cache_.clear();
    m_face_basic_data_cache_.clear();
    m_face_rects_cache_.clear();
    m_track_id_cache_.clear();
    m_track_count_cache_.clear();
    m_quality_results_cache_.clear();
    m_roll_results_cache_.clear();
    m_yaw_results_cache_.clear();
    m_pitch_results_cache_.clear();
    m_quality_score_results_cache_.clear();
    m_react_left_eye_results_cache_.clear();
    m_react_right_eye_results_cache_.clear();
    m_face_emotion_results_cache_.clear();

    m_action_normal_results_cache_.clear();
    m_action_shake_results_cache_.clear();
    m_action_blink_results_cache_.clear();
    m_action_jaw_open_results_cache_.clear();
    m_action_raise_head_results_cache_.clear();

    m_quality_score_results_cache_.clear();
    m_attribute_race_results_cache_.clear();
    m_attribute_gender_results_cache_.clear();
    m_det_confidence_cache_.clear();
    if (m_face_track_ == nullptr) {
        return HERR_SESS_TRACKER_FAILURE;
    }
    m_face_track_->UpdateStream(process);
    for (int i = 0; i < m_face_track_->trackingFace.size(); ++i) {
        auto& face = m_face_track_->trackingFace[i];
        FaceTrackWrap data = FaceObjectInternalToHyperFaceData(face, i);
        ByteArray byteArray;
        auto ret = RunSerializeHyperFaceData(data, byteArray);
        if (ret != HSUCCEED) {
            return HERR_INVALID_SERIALIZATION_FAILED;
        }
        m_det_confidence_cache_.push_back(face.GetConfidence());
        m_detect_cache_.push_back(byteArray);
        m_track_id_cache_.push_back(face.GetTrackingId());
        m_track_count_cache_.push_back(face.GetTrackingCount());
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
        float quality_score = 1.0f - avg;  // reversal
        m_quality_score_results_cache_.push_back(quality_score);
    }
    // ptr face_basic
    m_face_basic_data_cache_.resize(m_face_track_->trackingFace.size());
    for (int i = 0; i < m_face_basic_data_cache_.size(); ++i) {
        auto& basic = m_face_basic_data_cache_[i];
        basic.dataSize = m_detect_cache_[i].size();
        basic.data = m_detect_cache_[i].data();
    }
    if (m_enable_track_cost_spend_) {
        m_face_track_cost_->Stop();
    }
    //    LOGD("Track COST: %f", m_face_track_->GetTrackTotalUseTime());
    return HSUCCEED;
}

int32_t FaceSession::SetLandmarkLoop(int32_t value) {
    // TODO: implement this function
    return HSUCCEED;
}

int32_t FaceSession::SetFaceDetectThreshold(float value) {
    m_face_track_->SetDetectThreshold(value);
    return HSUCCEED;
}

FaceObjectInternalList& FaceSession::GetTrackingFaceList() {
    return m_face_track_->trackingFace;
}

const std::shared_ptr<FeatureExtractionModule>& FaceSession::FaceRecognitionModule() {
    return m_face_recognition_;
}

const std::shared_ptr<FacePipelineModule>& FaceSession::PipelineModule() {
    return m_face_pipeline_;
}

const int32_t FaceSession::GetNumberOfFacesCurrentlyDetected() const {
    return m_face_track_->trackingFace.size();
}

int32_t FaceSession::FacesProcess(inspirecv::FrameProcess& process, const std::vector<FaceTrackWrap>& faces, const CustomPipelineParameter& param) {
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
    m_face_emotion_results_cache_.resize(faces.size(), -1);
    for (int i = 0; i < faces.size(); ++i) {
        const auto& face = faces[i];
        // RGB Liveness Detect
        if (param.enable_liveness) {
            auto ret = m_face_pipeline_->Process(process, face, PROCESS_RGB_LIVENESS);
            if (ret != HSUCCEED) {
                return ret;
            }
            m_rgb_liveness_results_cache_[i] = m_face_pipeline_->faceLivenessCache;
        }
        // Mask detection
        if (param.enable_mask_detect) {
            auto ret = m_face_pipeline_->Process(process, face, PROCESS_MASK);
            if (ret != HSUCCEED) {
                return ret;
            }
            m_mask_results_cache_[i] = m_face_pipeline_->faceMaskCache;
        }
        // Face attribute prediction
        if (param.enable_face_attribute) {
            auto ret = m_face_pipeline_->Process(process, face, PROCESS_ATTRIBUTE);
            if (ret != HSUCCEED) {
                return ret;
            }
            m_attribute_race_results_cache_[i] = m_face_pipeline_->faceAttributeCache[0];
            m_attribute_gender_results_cache_[i] = m_face_pipeline_->faceAttributeCache[1];
            m_attribute_age_results_cache_[i] = m_face_pipeline_->faceAttributeCache[2];
        }

        // Face interaction
        if (param.enable_interaction_liveness) {
            auto ret = m_face_pipeline_->Process(process, face, PROCESS_INTERACTION);
            if (ret != HSUCCEED) {
                return ret;
            }
            // Get eyes status
            m_react_left_eye_results_cache_[i] = m_face_pipeline_->eyesStatusCache[0];
            m_react_right_eye_results_cache_[i] = m_face_pipeline_->eyesStatusCache[1];
            // Special handling:  ff it is a tracking state, it needs to be filtered
            if (face.trackState > 0) {
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
                        const auto actions = target.UpdateFaceAction(INSPIREFACE_CONTEXT->getMArchive().GetLandmarkParam()->semantic_index);
                        m_action_normal_results_cache_[i] = actions.normal;
                        m_action_jaw_open_results_cache_[i] = actions.jawOpen;
                        m_action_blink_results_cache_[i] = actions.blink;
                        m_action_raise_head_results_cache_[i] = actions.raiseHead;
                        m_action_shake_results_cache_[i] = actions.shake;
                    } else {
                        INSPIRE_LOGD(
                          "Serialized objects cannot connect to trace objects in memory, and there may be some "
                          "problems");
                    }
                } else {
                    INSPIRE_LOGW(
                      "The index of the trace object does not match the trace list in memory, and there may be some "
                      "problems");
                }
            }
        }
        // Face emotion recognition
        if (param.enable_face_emotion) {
            auto ret = m_face_pipeline_->Process(process, face, PROCESS_FACE_EMOTION);
            if (ret != HSUCCEED) {
                return ret;
            }
            // Default mode
            m_face_emotion_results_cache_[i] = argmax(m_face_pipeline_->faceEmotionCache.begin(), m_face_pipeline_->faceEmotionCache.end());
            if (face.trackState > 0) {
                // Tracking mode
                auto idx = face.inGroupIndex;
                if (idx < m_face_track_->trackingFace.size()) {
                    auto& target = m_face_track_->trackingFace[idx];
                    if (target.GetTrackingId() == face.trackId) {
                        auto new_emotion = VectorEmaFilter(m_face_pipeline_->faceEmotionCache, target.face_emotion_history_, 8, 0.4f);
                        m_face_emotion_results_cache_[i] = argmax(new_emotion.begin(), new_emotion.end());
                    } else { 
                        INSPIRE_LOGW(
                          "Serialized objects cannot connect to trace objects in memory, and there may be some "
                          "problems");
                    }
                } else {
                    INSPIRE_LOGW(
                      "The index of the trace object does not match the trace list in memory, and there may be some "
                      "problems");
                }
            }
        }
    }

    return 0;
}

const std::vector<ByteArray>& FaceSession::GetDetectCache() const {
    return m_detect_cache_;
}

const std::vector<FaceBasicData>& FaceSession::GetFaceBasicDataCache() const {
    return m_face_basic_data_cache_;
}

const std::vector<FaceRect>& FaceSession::GetFaceRectsCache() const {
    return m_face_rects_cache_;
}

const std::vector<int32_t>& FaceSession::GetTrackIDCache() const {
    return m_track_id_cache_;
}

const std::vector<int32_t>& FaceSession::GetTrackCountCache() const {
    return m_track_count_cache_;
}

const std::vector<float>& FaceSession::GetRollResultsCache() const {
    return m_roll_results_cache_;
}

const std::vector<float>& FaceSession::GetYawResultsCache() const {
    return m_yaw_results_cache_;
}

const std::vector<float>& FaceSession::GetPitchResultsCache() const {
    return m_pitch_results_cache_;
}

const std::vector<FacePoseQualityAdaptResult>& FaceSession::GetQualityResultsCache() const {
    return m_quality_results_cache_;
}

const std::vector<float>& FaceSession::GetMaskResultsCache() const {
    return m_mask_results_cache_;
}

const std::vector<float>& FaceSession::GetRgbLivenessResultsCache() const {
    return m_rgb_liveness_results_cache_;
}

const std::vector<float>& FaceSession::GetFaceQualityScoresResultsCache() const {
    return m_quality_score_results_cache_;
}

const std::vector<float>& FaceSession::GetFaceInteractionLeftEyeStatusCache() const {
    return m_react_left_eye_results_cache_;
}

const std::vector<float>& FaceSession::GetFaceInteractionRightEyeStatusCache() const {
    return m_react_right_eye_results_cache_;
}

const Embedded& FaceSession::GetFaceFeatureCache() const {
    return m_face_feature_cache_;
}

const std::vector<float>& FaceSession::GetDetConfidenceCache() const {
    return m_det_confidence_cache_;
}

const float FaceSession::GetFaceFeatureNormCache() const {
    return m_face_feature_norm_;
}

const std::vector<int>& FaceSession::GetFaceRaceResultsCache() const {
    return m_attribute_race_results_cache_;
}

const std::vector<int>& FaceSession::GetFaceGenderResultsCache() const {
    return m_attribute_gender_results_cache_;
}

const std::vector<int>& FaceSession::GetFaceAgeBracketResultsCache() const {
    return m_attribute_age_results_cache_;
}

const std::vector<int>& FaceSession::GetFaceNormalAactionsResultCache() const {
    return m_action_normal_results_cache_;
}

const std::vector<int>& FaceSession::GetFaceJawOpenAactionsResultCache() const {
    return m_action_jaw_open_results_cache_;
}

const std::vector<int>& FaceSession::GetFaceBlinkAactionsResultCache() const {
    return m_action_blink_results_cache_;
}

const std::vector<int>& FaceSession::GetFaceShakeAactionsResultCache() const {
    return m_action_shake_results_cache_;
}

const std::vector<int>& FaceSession::GetFaceRaiseHeadAactionsResultCache() const {
    return m_action_raise_head_results_cache_;
}

const std::vector<int>& FaceSession::GetFaceEmotionResultsCache() const {
    return m_face_emotion_results_cache_;
}

int32_t FaceSession::FaceFeatureExtract(inspirecv::FrameProcess& process, FaceBasicData& data, bool normalize) {
    std::lock_guard<std::mutex> lock(m_mtx_);
    int32_t ret;
    FaceTrackWrap face = {0};
    ret = RunDeserializeHyperFaceData((char*)data.data, data.dataSize, face);
    if (ret != HSUCCEED) {
        return ret;
    }
    m_face_feature_cache_.clear();
    ret = m_face_recognition_->FaceExtract(process, face, m_face_feature_cache_, m_face_feature_norm_, normalize);

    return ret;
}

int32_t FaceSession::FaceFeatureExtract(inspirecv::FrameProcess& process, FaceTrackWrap& data, bool normalize) {
    std::lock_guard<std::mutex> lock(m_mtx_);
    int32_t ret;
    m_face_feature_cache_.clear();
    ret = m_face_recognition_->FaceExtract(process, data, m_face_feature_cache_, m_face_feature_norm_, normalize);
    if (ret != HSUCCEED) {
        return ret;
    }

    return ret;
}

int32_t FaceSession::FaceFeatureExtractWithAlignmentImage(inspirecv::FrameProcess& process, Embedded& embedding, float& norm, bool normalize) {
    std::lock_guard<std::mutex> lock(m_mtx_);
    int32_t ret;
    m_face_feature_cache_.clear();
    ret = m_face_recognition_->FaceExtractWithAlignmentImage(process, embedding, norm, normalize);

    return ret;
}

int32_t FaceSession::FaceFeatureExtractWithAlignmentImage(const inspirecv::Image& wrapped, FaceEmbedding& embedding, float& norm, bool normalize) {
    std::lock_guard<std::mutex> lock(m_mtx_);
    int32_t ret;
    ret = m_face_recognition_->FaceExtractWithAlignmentImage(wrapped, embedding.embedding, norm, normalize);
    return ret;
}

int32_t FaceSession::FaceGetFaceAlignmentImage(inspirecv::FrameProcess& process, FaceBasicData& data, inspirecv::Image& image) {
    std::lock_guard<std::mutex> lock(m_mtx_);
    int32_t ret;
    FaceTrackWrap face = {0};
    ret = RunDeserializeHyperFaceData((char*)data.data, data.dataSize, face);
    if (ret != HSUCCEED) {
        return ret;
    }
    std::vector<inspirecv::Point2f> pointsFive;
    for (const auto& p : face.keyPoints) {
        pointsFive.push_back(inspirecv::Point2f(p.x, p.y));
    }
    auto trans = inspirecv::SimilarityTransformEstimateUmeyama(SIMILARITY_TRANSFORM_DEST, pointsFive);
    image = process.ExecuteImageAffineProcessing(trans, FACE_CROP_SIZE, FACE_CROP_SIZE);
    return ret;
}

const CustomPipelineParameter& FaceSession::getMParameter() const {
    return m_parameter_;
}

int32_t FaceSession::FaceQualityDetect(FaceBasicData& data, float& result) {
    int32_t ret;
    FaceTrackWrap face = {0};
    ret = RunDeserializeHyperFaceData((char*)data.data, data.dataSize, face);
    //    PrintHyperFaceData(face);
    if (ret != HSUCCEED) {
        return ret;
    }
    float avg = 0.0f;
    for (int i = 0; i < 5; ++i) {
        avg += face.quality[i];
    }
    avg /= 5.0f;
    result = 1.0f - avg;  // reversal

    return ret;
}

int32_t FaceSession::SetDetectMode(DetectModuleMode mode) {
    m_detect_mode_ = mode;
    if (m_detect_mode_ == DetectModuleMode::DETECT_MODE_ALWAYS_DETECT) {
        m_always_detect_ = true;
    } else {
        m_always_detect_ = false;
    }
    return HSUCCEED;
}

bool FaceSession::IsDetectModeLandmark() const {
    return m_face_track_->IsDetectModeLandmark();
}

void FaceSession::ClearTrackingFace() {
    m_face_track_->ClearTrackingFace();
}

void FaceSession::SetTrackLostRecoveryMode(bool value) {
    m_face_track_->SetTrackLostRecoveryMode(value);
}

void FaceSession::SetLightTrackConfidenceThreshold(float value) {
    m_face_track_->SetLightTrackConfidenceThreshold(value);
}

int32_t FaceSession::SetTrackPreviewSize(const int32_t preview_size) {
    m_face_track_->SetTrackPreviewSize(preview_size);
    return HSUCCEED;
}

int32_t FaceSession::GetTrackPreviewSize() const {
    return m_face_track_->GetTrackPreviewSize();
}

int32_t FaceSession::SetTrackFaceMinimumSize(int32_t minSize) {
    m_face_track_->SetMinimumFacePxSize(minSize);
    return HSUCCEED;
}

int32_t FaceSession::SetTrackModeSmoothRatio(float value) {
    m_face_track_->SetTrackModeSmoothRatio(value);
    return HSUCCEED;
}

int32_t FaceSession::SetTrackModeNumSmoothCacheFrame(int value) {
    m_face_track_->SetTrackModeNumSmoothCacheFrame(value);
    return HSUCCEED;
}

int32_t FaceSession::SetTrackModeDetectInterval(int value) {
    m_face_track_->SetTrackModeDetectInterval(value);
    return HSUCCEED;
}

int32_t FaceSession::SetEnableTrackCostSpend(int value) {
    m_enable_track_cost_spend_ = value;
    m_face_track_cost_->Reset();
    return HSUCCEED;
}

void FaceSession::PrintTrackCostSpend() {
    if (m_enable_track_cost_spend_) {
        INSPIRE_LOGI("%s", m_face_track_cost_->Report().c_str());
    }
}

int32_t FaceSession::GetDebugPreviewImageSize() const {
    return m_face_track_->GetDebugPreviewImageSize();
}

}  // namespace inspire
