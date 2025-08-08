#include <memory>
#include "session.h"
#include "engine/face_session.h"
#include "recognition_module/dest_const.h"

namespace inspire {

class Session::Impl {
public:
    Impl() : m_face_session_(std::make_unique<FaceSession>()) {}

    int32_t Configure(DetectModuleMode detect_mode, int32_t max_detect_face, CustomPipelineParameter param, int32_t detect_level_px,
                      int32_t track_by_detect_mode_fps) {
        return m_face_session_->Configuration(detect_mode, max_detect_face, param, detect_level_px, track_by_detect_mode_fps);
    }

    void ClearTrackingFace() {
        m_face_session_->ClearTrackingFace();
    }

    ~Impl() = default;

    void SetTrackPreviewSize(int32_t preview_size) {
        m_face_session_->SetTrackPreviewSize(preview_size);
    }

    void SetFilterMinimumFacePixelSize(int32_t min_face_pixel_size) {
        m_face_session_->SetTrackFaceMinimumSize(min_face_pixel_size);
    }

    void SetFaceDetectThreshold(float threshold) {
        m_face_session_->SetFaceDetectThreshold(threshold);
    }

    void SetTrackModeSmoothRatio(int32_t smooth_ratio) {
        m_face_session_->SetTrackModeSmoothRatio(smooth_ratio);
    }

    void SetTrackModeNumSmoothCacheFrame(int32_t num_smooth_cache_frame) {
        m_face_session_->SetTrackModeNumSmoothCacheFrame(num_smooth_cache_frame);
    }

    void SetTrackModeDetectInterval(int32_t detect_interval) {
        m_face_session_->SetTrackModeDetectInterval(detect_interval);
    }

    void SetTrackLostRecoveryMode(bool value) {
        m_face_session_->SetTrackLostRecoveryMode(value);
    }

    void SetLightTrackConfidenceThreshold(float value) {
        m_face_session_->SetLightTrackConfidenceThreshold(value);
    }

    int32_t FaceDetectAndTrack(inspirecv::FrameProcess& process, std::vector<FaceTrackWrap>& results) {
        int32_t ret = m_face_session_->FaceDetectAndTrack(process);
        if (ret < 0) {
            return ret;
        }
        results.clear();
        const auto& face_data = m_face_session_->GetDetectCache();
        for (const auto& data : face_data) {
            FaceTrackWrap hyper_face_data;
            RunDeserializeHyperFaceData(data, hyper_face_data);
            results.emplace_back(hyper_face_data);
        }

        return ret;
    }

    inspirecv::Rect2i GetFaceBoundingBox(const FaceTrackWrap& face_data) {
        return inspirecv::Rect2i{face_data.rect.x, face_data.rect.y, face_data.rect.width, face_data.rect.height};
    }

    std::vector<inspirecv::Point2f> GetFaceDenseLandmark(const FaceTrackWrap& face_data) {
        std::vector<inspirecv::Point2f> points;
        for (const auto& p : face_data.densityLandmark) {
            points.emplace_back(inspirecv::Point2f(p.x, p.y));
        }
        return points;
    }

    std::vector<inspirecv::Point2f> GetFaceFiveKeyPoints(const FaceTrackWrap& face_data) {
        std::vector<inspirecv::Point2f> points;
        for (const auto& p : face_data.keyPoints) {
            points.emplace_back(inspirecv::Point2f(p.x, p.y));
        }
        return points;
    }

    int32_t FaceFeatureExtract(inspirecv::FrameProcess& process, FaceTrackWrap& data, FaceEmbedding& embedding, bool normalize) {
        int32_t ret = m_face_session_->FaceFeatureExtract(process, data, normalize);
        if (ret < 0) {
            return ret;
        }
        embedding.isNormal = normalize;
        embedding.embedding = m_face_session_->GetFaceFeatureCache();
        embedding.norm = m_face_session_->GetFaceFeatureNormCache();

        return ret;
    }

    int32_t FaceFeatureExtractWithAlignmentImage(inspirecv::FrameProcess& process, FaceEmbedding& embedding, bool normalize) {
        int32_t ret = m_face_session_->FaceFeatureExtractWithAlignmentImage(process, embedding.embedding, embedding.norm, normalize);
        if (ret < 0) {
            return ret;
        }
        embedding.isNormal = normalize;
        embedding.norm = embedding.norm;
        return ret;
    }

    int32_t FaceFeatureExtractWithAlignmentImage(const inspirecv::Image& wrapped, FaceEmbedding& embedding, bool normalize) {
        int32_t ret = m_face_session_->FaceFeatureExtractWithAlignmentImage(wrapped, embedding, embedding.norm, normalize);
        if (ret < 0) {
            return ret;
        }
        embedding.isNormal = normalize;
        embedding.norm = embedding.norm;
        return ret;
    }

    void GetFaceAlignmentImage(inspirecv::FrameProcess& process, FaceTrackWrap& data, inspirecv::Image& wrapped) {
        std::vector<inspirecv::Point2f> pointsFive;
        for (const auto& p : data.keyPoints) {
            pointsFive.push_back(inspirecv::Point2f(p.x, p.y));
        }
        auto trans = inspirecv::SimilarityTransformEstimateUmeyama(SIMILARITY_TRANSFORM_DEST, pointsFive);
        wrapped = process.ExecuteImageAffineProcessing(trans, FACE_CROP_SIZE, FACE_CROP_SIZE);
    }

    int32_t MultipleFacePipelineProcess(inspirecv::FrameProcess& process, const CustomPipelineParameter& param,
                                        const std::vector<FaceTrackWrap>& face_data_list) {
        int32_t ret = m_face_session_->FacesProcess(process, face_data_list, param);
        return ret;
    }

    std::vector<float> GetRGBLivenessConfidence() {
        return m_face_session_->GetDetConfidenceCache();
    }

    std::vector<float> GetFaceMaskConfidence() {
        return m_face_session_->GetMaskResultsCache();
    }

    std::vector<float> GetFaceQualityConfidence() {
        return m_face_session_->GetFaceQualityScoresResultsCache();
    }

    std::vector<FaceInteractionState> GetFaceInteractionState() {
        auto left_eyes_confidence = m_face_session_->GetFaceInteractionLeftEyeStatusCache();
        auto right_eyes_confidence = m_face_session_->GetFaceInteractionRightEyeStatusCache();
        std::vector<FaceInteractionState> face_interaction_state;
        for (size_t i = 0; i < left_eyes_confidence.size(); ++i) {
            face_interaction_state.emplace_back(FaceInteractionState{left_eyes_confidence[i], right_eyes_confidence[i]});
        }
        return face_interaction_state;
    }

    std::vector<FaceInteractionAction> GetFaceInteractionAction() {
        auto num = m_face_session_->GetFaceNormalAactionsResultCache().size();
        std::vector<FaceInteractionAction> face_interaction_action;
        face_interaction_action.resize(num);
        for (size_t i = 0; i < num; ++i) {
            face_interaction_action[i].normal = m_face_session_->GetFaceNormalAactionsResultCache()[i];
            face_interaction_action[i].shake = m_face_session_->GetFaceShakeAactionsResultCache()[i];
            face_interaction_action[i].jawOpen = m_face_session_->GetFaceJawOpenAactionsResultCache()[i];
            face_interaction_action[i].headRaise = m_face_session_->GetFaceRaiseHeadAactionsResultCache()[i];
            face_interaction_action[i].blink = m_face_session_->GetFaceBlinkAactionsResultCache()[i];
        }
        return face_interaction_action;
    }

    std::vector<FaceAttributeResult> GetFaceAttributeResult() {
        auto num = m_face_session_->GetFaceNormalAactionsResultCache().size();
        std::vector<FaceAttributeResult> face_attribute_result;
        face_attribute_result.resize(num);
        for (size_t i = 0; i < num; ++i) {
            face_attribute_result[i].race = m_face_session_->GetFaceRaceResultsCache()[i];
            face_attribute_result[i].gender = m_face_session_->GetFaceGenderResultsCache()[i];
            face_attribute_result[i].ageBracket = m_face_session_->GetFaceAgeBracketResultsCache()[i];
        }
        return face_attribute_result;
    }

    std::vector<FaceEmotionResult> GetFaceEmotionResult() {
        auto num = m_face_session_->GetFaceEmotionResultsCache().size();
        std::vector<FaceEmotionResult> face_emotion_result;
        face_emotion_result.resize(num);
        for (size_t i = 0; i < num; ++i) {
            face_emotion_result[i].emotion = m_face_session_->GetFaceEmotionResultsCache()[i];
        }
        return face_emotion_result;
    }


    std::unique_ptr<FaceSession> m_face_session_;
};

Session::Session() : pImpl(std::make_unique<Impl>()) {}

Session::~Session() = default;

Session::Session(Session&&) noexcept = default;

Session& Session::operator=(Session&&) noexcept = default;

Session Session::Create(DetectModuleMode detect_mode, int32_t max_detect_face, const CustomPipelineParameter& param, int32_t detect_level_px,
                        int32_t track_by_detect_mode_fps) {
    Session session;
    session.pImpl->Configure(detect_mode, max_detect_face, param, detect_level_px, track_by_detect_mode_fps);
    return session;
}

void Session::ClearTrackingFace() {
    pImpl->ClearTrackingFace();
}

void Session::SetTrackLostRecoveryMode(bool value) {
    pImpl->SetTrackLostRecoveryMode(value);
}

void Session::SetLightTrackConfidenceThreshold(float value) {
    pImpl->SetLightTrackConfidenceThreshold(value);
}

void Session::SetTrackPreviewSize(int32_t preview_size) {
    pImpl->SetTrackPreviewSize(preview_size);
}

void Session::SetFilterMinimumFacePixelSize(int32_t min_face_pixel_size) {
    pImpl->SetFilterMinimumFacePixelSize(min_face_pixel_size);
}

void Session::SetFaceDetectThreshold(float threshold) {
    pImpl->SetFaceDetectThreshold(threshold);
}

void Session::SetTrackModeSmoothRatio(int32_t smooth_ratio) {
    pImpl->SetTrackModeSmoothRatio(smooth_ratio);
}

void Session::SetTrackModeNumSmoothCacheFrame(int32_t num_smooth_cache_frame) {
    pImpl->SetTrackModeNumSmoothCacheFrame(num_smooth_cache_frame);
}

void Session::SetTrackModeDetectInterval(int32_t detect_interval) {
    pImpl->SetTrackModeDetectInterval(detect_interval);
}

int32_t Session::FaceDetectAndTrack(inspirecv::FrameProcess& process, std::vector<FaceTrackWrap>& results) {
    return pImpl->FaceDetectAndTrack(process, results);
}

inspirecv::Rect2i Session::GetFaceBoundingBox(const FaceTrackWrap& face_data) {
    return pImpl->GetFaceBoundingBox(face_data);
}

std::vector<inspirecv::Point2f> Session::GetFaceDenseLandmark(const FaceTrackWrap& face_data) {
    return pImpl->GetFaceDenseLandmark(face_data);
}

std::vector<inspirecv::Point2f> Session::GetFaceFiveKeyPoints(const FaceTrackWrap& face_data) {
    return pImpl->GetFaceFiveKeyPoints(face_data);
}

int32_t Session::FaceFeatureExtract(inspirecv::FrameProcess& process, FaceTrackWrap& data, FaceEmbedding& embedding, bool normalize) {
    return pImpl->FaceFeatureExtract(process, data, embedding, normalize);
}

int32_t Session::FaceFeatureExtractWithAlignmentImage(inspirecv::FrameProcess& process, FaceEmbedding& embedding, bool normalize) {
    return pImpl->FaceFeatureExtractWithAlignmentImage(process, embedding, normalize);
}

int32_t Session::FaceFeatureExtractWithAlignmentImage(const inspirecv::Image& wrapped, FaceEmbedding& embedding, bool normalize) {
    return pImpl->FaceFeatureExtractWithAlignmentImage(wrapped, embedding, normalize);
}

void Session::GetFaceAlignmentImage(inspirecv::FrameProcess& process, FaceTrackWrap& data, inspirecv::Image& wrapped) {
    pImpl->GetFaceAlignmentImage(process, data, wrapped);
}

int32_t Session::MultipleFacePipelineProcess(inspirecv::FrameProcess& process, const CustomPipelineParameter& param,
                                             const std::vector<FaceTrackWrap>& face_data_list) {
    return pImpl->MultipleFacePipelineProcess(process, param, face_data_list);
}

std::vector<float> Session::GetRGBLivenessConfidence() {
    return pImpl->GetRGBLivenessConfidence();
}

std::vector<float> Session::GetFaceMaskConfidence() {
    return pImpl->GetFaceMaskConfidence();
}

std::vector<float> Session::GetFaceQualityConfidence() {
    return pImpl->GetFaceQualityConfidence();
}

std::vector<FaceInteractionState> Session::GetFaceInteractionState() {
    return pImpl->GetFaceInteractionState();
}

std::vector<FaceInteractionAction> Session::GetFaceInteractionAction() {
    return pImpl->GetFaceInteractionAction();
}

std::vector<FaceAttributeResult> Session::GetFaceAttributeResult() {
    return pImpl->GetFaceAttributeResult();
}

std::vector<FaceEmotionResult> Session::GetFaceEmotionResult() {
    return pImpl->GetFaceEmotionResult();
}

}  // namespace inspire
