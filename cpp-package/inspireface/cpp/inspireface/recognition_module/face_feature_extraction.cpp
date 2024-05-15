//
// Created by Tunm-Air13 on 2024/4/12.
//

#include "face_feature_extraction.h"
#include "feature_hub/simd.h"
#include "recognition_module/extract/alignment.h"
#include "track_module/landmark/face_landmark.h"
#include "herror.h"

namespace inspire {

FeatureExtraction::FeatureExtraction(InspireArchive &archive, bool enable_recognition):m_status_code_(SARC_SUCCESS) {
    if (enable_recognition) {
        InspireModel model;
        m_status_code_ = archive.LoadModel("feature", model);
        if (m_status_code_ != SARC_SUCCESS) {
            INSPIRE_LOGE("Load rec model error.");
        }
        m_status_code_ = InitExtractInteraction(model);
        if (m_status_code_ != 0) {
            INSPIRE_LOGE("FaceRecognition error.");
        }
    }

}

int32_t FeatureExtraction::InitExtractInteraction(InspireModel &model) {
    try {
        auto input_size = model.Config().get<std::vector<int>>("input_size");
        m_extract_ = std::make_shared<Extract>();
        auto ret = m_extract_->loadData(model, model.modelType);
        if (ret != InferenceHelper::kRetOk) {
            return HERR_ARCHIVE_LOAD_FAILURE;
        }
        return HSUCCEED;

    } catch (const std::runtime_error& e) {
        INSPIRE_LOGE("%s", e.what());
        return HERR_SESS_FACE_REC_OPTION_ERROR;
    }
}

int32_t FeatureExtraction::QueryStatus() const {
    return m_status_code_;
}

int32_t FeatureExtraction::FaceExtract(CameraStream &image, const HyperFaceData &face, Embedded &embedded) {
    if (m_extract_ == nullptr) {
        return HERR_SESS_REC_EXTRACT_FAILURE;
    }

    std::vector<cv::Point2f> pointsFive;
    for (const auto &p: face.keyPoints) {
        pointsFive.push_back(HPointToPoint2f(p));
    }
    auto trans = getTransformMatrix112(pointsFive);
    trans.convertTo(trans, CV_64F);
    auto crop = image.GetAffineRGBImage(trans, 112, 112);
//    cv::imshow("w", crop);
//    cv::waitKey(0);
    embedded = (*m_extract_)(crop);

    return 0;
}

int32_t FeatureExtraction::FaceExtract(CameraStream &image, const FaceObject &face, Embedded &embedded) {
    if (m_extract_ == nullptr) {
        return HERR_SESS_REC_EXTRACT_FAILURE;
    }

    auto lmk = face.landmark_;
    std::vector<cv::Point2f> lmk_5 = {lmk[FaceLandmark::LEFT_EYE_CENTER],
                                      lmk[FaceLandmark::RIGHT_EYE_CENTER],
                                      lmk[FaceLandmark::NOSE_CORNER],
                                      lmk[FaceLandmark::MOUTH_LEFT_CORNER],
                                      lmk[FaceLandmark::MOUTH_RIGHT_CORNER]};

    auto trans = getTransformMatrix112(lmk_5);
    trans.convertTo(trans, CV_64F);
    auto crop = image.GetAffineRGBImage(trans, 112, 112);
    embedded = (*m_extract_)(crop);

    return 0;
}

const std::shared_ptr<Extract> &FeatureExtraction::getMExtract() const {
    return m_extract_;
}

}   // namespace inspire