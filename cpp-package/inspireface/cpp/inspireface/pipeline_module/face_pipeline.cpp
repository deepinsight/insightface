//
// Created by tunm on 2023/9/7.
//

#include "face_pipeline.h"

#include "log.h"
#include "track_module/landmark/face_landmark.h"
#include "recognition_module/extract/alignment.h"
#include "herror.h"

namespace inspire {

FacePipeline::FacePipeline(InspireArchive &archive, bool enableLiveness, bool enableMaskDetect, bool enableAge,
                           bool enableGender, bool enableInteractionLiveness)
        : m_enable_liveness_(enableLiveness),
          m_enable_mask_detect_(enableMaskDetect),
          m_enable_age_(enableAge),
          m_enable_gender_(enableGender),
          m_enable_interaction_liveness_(enableInteractionLiveness) {

    if (m_enable_age_) {
        InspireModel ageModel;
        auto ret = InitAgePredict(ageModel);
        if (ret != 0) {
            INSPIRE_LOGE("InitAgePredict error.");
        }
    }

    // Initialize the gender prediction model (assuming Index is 0)
    if (m_enable_gender_) {
        InspireModel genderModel;
        auto ret = InitGenderPredict(genderModel);
        if (ret != 0) {
            INSPIRE_LOGE("InitGenderPredict error.");
        }
    }

    // Initialize the mask detection model
    if (m_enable_mask_detect_) {
        InspireModel maskModel;
        auto ret = archive.LoadModel("mask_detect", maskModel);
        if (ret != 0) {
            INSPIRE_LOGE("Load Mask model: %d", ret);
        }
        ret = InitMaskPredict(maskModel);
        if (ret != 0) {
            INSPIRE_LOGE("InitMaskPredict error.");
        }
    }

    // Initializing the RGB live detection model
    if (m_enable_liveness_) {
        InspireModel livenessModel;
        auto ret = archive.LoadModel("rgb_anti_spoofing", livenessModel);
        if (ret != 0) {
            INSPIRE_LOGE("Load anti-spoofing model.");
        }
        ret = InitRBGAntiSpoofing(livenessModel);
        if (ret != 0) {
            INSPIRE_LOGE("InitRBGAntiSpoofing error.");
        }
    }

    // Initializing the model for in-vivo detection (assuming Index is 0)
    if (m_enable_interaction_liveness_) {
        InspireModel actLivenessModel;
        auto ret = InitLivenessInteraction(actLivenessModel);
        if (ret != 0) {
            INSPIRE_LOGE("InitLivenessInteraction error.");
        }
    }

}


int32_t FacePipeline::Process(CameraStream &image, const HyperFaceData &face, FaceProcessFunction proc) {
    switch (proc) {
        case PROCESS_MASK: {
            if (m_mask_predict_ == nullptr) {
                return HERR_SESS_PIPELINE_FAILURE;       // uninitialized
            }
            std::vector<cv::Point2f> pointsFive;
            for (const auto &p: face.keyPoints) {
                pointsFive.push_back(HPointToPoint2f(p));
            }
            // debug
//            auto img = image.GetScaledImage(1.0f, true);
//            for (int i = 0; i < pointsFive.size(); ++i) {
//                cv::circle(img, pointsFive[i], 0, cv::Scalar(233, 2, 211), 4);
//            }
//            cv::imshow("wqwe", img);
//            cv::waitKey(0);
            auto trans = getTransformMatrix112(pointsFive);
            trans.convertTo(trans, CV_64F);
            auto crop = image.GetAffineRGBImage(trans, 112, 112);
//            cv::imshow("wq", crop);
//            cv::waitKey(0);
            auto mask_score = (*m_mask_predict_)(crop);
            faceMaskCache = mask_score;
            break;
        }
        case PROCESS_RGB_LIVENESS: {
            if (m_rgb_anti_spoofing_ == nullptr) {
                return HERR_SESS_PIPELINE_FAILURE;       // uninitialized
            }
//            auto trans27 = getTransformMatrixSafas(pointsFive);
//            trans27.convertTo(trans27, CV_64F);
//            auto align112x27 = image.GetAffineRGBImage(trans27, 112, 112);

            auto img = image.GetScaledImage(1.0, true);
            cv::Rect oriRect(face.rect.x, face.rect.y, face.rect.width, face.rect.height);
            auto rect = GetNewBox(img.cols, img.rows, oriRect, 2.7f);
            auto crop = img(rect);
//            cv::imwrite("crop.jpg", crop);
            auto score = (*m_rgb_anti_spoofing_)(crop);
//            auto i = cv::imread("zsb.jpg");
//            LOGE("SBA: %f", (*m_rgb_anti_spoofing_)(i));
            faceLivenessCache = score;
            break;
        }
        case PROCESS_AGE: {
            if (m_age_predict_ == nullptr) {
                return HERR_SESS_PIPELINE_FAILURE;       // uninitialized
            }
            break;
        }
        case PROCESS_GENDER: {
            if (m_gender_predict_ == nullptr) {
                return HERR_SESS_PIPELINE_FAILURE;       // uninitialized
            }
            break;
        }
    }
    return HSUCCEED;
}

int32_t FacePipeline::Process(CameraStream &image, FaceObject &face) {
    // In the tracking state, the count meets the requirements or the pipeline is executed in the detection state
    auto lmk = face.landmark_;
    std::vector<cv::Point2f> lmk_5 = {lmk[FaceLandmark::LEFT_EYE_CENTER],
                                 lmk[FaceLandmark::RIGHT_EYE_CENTER],
                                 lmk[FaceLandmark::NOSE_CORNER],
                                 lmk[FaceLandmark::MOUTH_LEFT_CORNER],
                                 lmk[FaceLandmark::MOUTH_RIGHT_CORNER]};
    auto trans = getTransformMatrix112(lmk_5);
    trans.convertTo(trans, CV_64F);
    auto align112x = image.GetAffineRGBImage(trans, 112, 112);
    if (m_mask_predict_ != nullptr) {
        auto mask_score = (*m_mask_predict_)(align112x);
        if (mask_score > 0.95) {
            face.faceProcess.maskInfo = MaskInfo::MASKED;
        } else {
            face.faceProcess.maskInfo = MaskInfo::UNMASKED;
        }
    }

    if (m_rgb_anti_spoofing_ != nullptr) {
//        auto trans27 = getTransformMatrixSafas(lmk_5);
//        trans27.convertTo(trans27, CV_64F);
//        auto align112x27 = image.GetAffineRGBImage(trans27, 112, 112);
        auto img = image.GetScaledImage(1.0, true);
        auto rect = GetNewBox(img.cols, img.rows, face.getBbox(), 2.7);
        auto crop = img(rect);
        auto score = (*m_rgb_anti_spoofing_)(crop);
        if (score > 0.88) {
            face.faceProcess.rgbLivenessInfo = RGBLivenessInfo::LIVENESS_REAL;
        } else {
            face.faceProcess.rgbLivenessInfo = RGBLivenessInfo::LIVENESS_FAKE;
        }
    }

    return HSUCCEED;
}


int32_t FacePipeline::InitAgePredict(InspireModel &) {

    return 0;
}


int32_t FacePipeline::InitGenderPredict(InspireModel &model) {
    return 0;
}

int32_t FacePipeline::InitMaskPredict(InspireModel &model) {
    m_mask_predict_ = std::make_shared<MaskPredict>();
    auto ret = m_mask_predict_->loadData(model, model.modelType);
    if (ret != InferenceHelper::kRetOk) {
        return HERR_ARCHIVE_LOAD_FAILURE;
    }
    return HSUCCEED;
}

int32_t FacePipeline::InitRBGAntiSpoofing(InspireModel &model) {
    auto input_size = model.Config().get<std::vector<int>>("input_size");
    m_rgb_anti_spoofing_ = std::make_shared<RBGAntiSpoofing>(input_size[0]);
    auto ret = m_rgb_anti_spoofing_->loadData(model, model.modelType);
    if (ret != InferenceHelper::kRetOk) {
        return HERR_ARCHIVE_LOAD_FAILURE;
    }
    return HSUCCEED;
}

int32_t FacePipeline::InitLivenessInteraction(InspireModel &model) {
    return 0;
}

const std::shared_ptr<RBGAntiSpoofing> &FacePipeline::getMRgbAntiSpoofing() const {
    return m_rgb_anti_spoofing_;
}


}