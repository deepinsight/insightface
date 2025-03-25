/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "face_feature_extraction_module.h"
#include "feature_hub/simd.h"
#include "track_module/landmark/face_landmark_adapt.h"
#include "herror.h"
#include "dest_const.h"

namespace inspire {

FeatureExtractionModule::FeatureExtractionModule(InspireArchive &archive, bool enable_recognition) : m_status_code_(SARC_SUCCESS) {
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

int32_t FeatureExtractionModule::InitExtractInteraction(InspireModel &model) {
    try {
        auto input_size = model.Config().get<std::vector<int>>("input_size");
        m_extract_ = std::make_shared<ExtractAdapt>();
        auto ret = m_extract_->loadData(model, model.modelType);
        if (ret != InferenceWrapper::WrapperOk) {
            return HERR_ARCHIVE_LOAD_FAILURE;
        }
        return HSUCCEED;

    } catch (const std::runtime_error &e) {
        INSPIRE_LOGE("%s", e.what());
        return HERR_SESS_FACE_REC_OPTION_ERROR;
    }
}

int32_t FeatureExtractionModule::QueryStatus() const {
    return m_status_code_;
}

int32_t FeatureExtractionModule::FaceExtract(inspirecv::FrameProcess &processor, const HyperFaceData &face, Embedded &embedded, float &norm,
                                             bool normalize) {
    if (m_extract_ == nullptr) {
        return HERR_SESS_REC_EXTRACT_FAILURE;
    }

    std::vector<inspirecv::Point2f> pointsFive;
    for (const auto &p : face.keyPoints) {
        pointsFive.push_back(inspirecv::Point2f(p.x, p.y));
    }
    auto trans = inspirecv::SimilarityTransformEstimateUmeyama(SIMILARITY_TRANSFORM_DEST, pointsFive);
    auto crop = processor.ExecuteImageAffineProcessing(trans, FACE_CROP_SIZE, FACE_CROP_SIZE);
    //    cv::imshow("w", crop);
    //    cv::waitKey(0);
    embedded = (*m_extract_)(crop, norm, normalize);

    return 0;
}

int32_t FeatureExtractionModule::FaceExtract(inspirecv::FrameProcess &processor, const FaceObjectInternal &face, Embedded &embedded, float &norm,
                                             bool normalize) {
    if (m_extract_ == nullptr) {
        return HERR_SESS_REC_EXTRACT_FAILURE;
    }

    auto lmk = face.landmark_;
    std::vector<inspirecv::Point2f> lmk_5 = {lmk[FaceLandmarkAdapt::LEFT_EYE_CENTER], lmk[FaceLandmarkAdapt::RIGHT_EYE_CENTER],
                                             lmk[FaceLandmarkAdapt::NOSE_CORNER], lmk[FaceLandmarkAdapt::MOUTH_LEFT_CORNER],
                                             lmk[FaceLandmarkAdapt::MOUTH_RIGHT_CORNER]};

    auto trans = inspirecv::SimilarityTransformEstimateUmeyama(SIMILARITY_TRANSFORM_DEST, lmk_5);
    auto crop = processor.ExecuteImageAffineProcessing(trans, FACE_CROP_SIZE, FACE_CROP_SIZE);

    embedded = (*m_extract_)(crop, norm, normalize);

    return 0;
}

const std::shared_ptr<ExtractAdapt> &FeatureExtractionModule::getMExtract() const {
    return m_extract_;
}

}  // namespace inspire