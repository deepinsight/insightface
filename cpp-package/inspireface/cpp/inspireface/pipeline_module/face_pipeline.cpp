//
// Created by tunm on 2023/9/7.
//

#include "face_pipeline.h"

#include "log.h"
#include "track_module/landmark/face_landmark.h"
#include "recognition_module/extract/alignment.h"
#include "middleware/utils.h"
#include "herror.h"

namespace inspire {

FacePipeline::FacePipeline(InspireArchive &archive, bool enableLiveness, bool enableMaskDetect, bool enableAttribute, 
                    bool enableInteractionLiveness)
        : m_enable_liveness_(enableLiveness),
          m_enable_mask_detect_(enableMaskDetect),
          m_enable_attribute_(enableAttribute),
          m_enable_interaction_liveness_(enableInteractionLiveness) {

    if (m_enable_attribute_) {
        InspireModel attrModel;
        auto ret = archive.LoadModel("face_attribute", attrModel);
        if (ret != 0) {
            INSPIRE_LOGE("Load Face attribute model: %d", ret);
        }

        ret = InitFaceAttributePredict(attrModel);
        if (ret != 0) {
            INSPIRE_LOGE("InitAgePredict error.");
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

    // There may be a combination of algorithms for facial interaction
    if (m_enable_interaction_liveness_) {
        // Blink model
        InspireModel blinkModel;
        auto ret = archive.LoadModel("blink_predict", blinkModel);
        if (ret != 0) {
            INSPIRE_LOGE("Load Blink model error.");
        }
        ret = InitBlinkFromLivenessInteraction(blinkModel);
        if (ret != 0) {
            INSPIRE_LOGE("InitBlinkFromLivenessInteraction error.");
        }
    }

}


int32_t FacePipeline::Process(CameraStream &image, const HyperFaceData &face, FaceProcessFunction proc) {
    cv::Mat originImage;
    cv::Mat crop112;
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
            if (crop112.empty())
            {
                auto trans = getTransformMatrix112(pointsFive);
                trans.convertTo(trans, CV_64F);
                crop112 = image.GetAffineRGBImage(trans, 112, 112);
            }
            
//            cv::imshow("wq", crop);
//            cv::waitKey(0);
            auto mask_score = (*m_mask_predict_)(crop112);
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
            if (originImage.empty()) {
                originImage = image.GetScaledImage(1.0, true);
            }
            cv::Rect oriRect(face.rect.x, face.rect.y, face.rect.width, face.rect.height);
            auto rect = GetNewBox(originImage.cols, originImage.rows, oriRect, 2.7f);
            auto crop = originImage(rect);
//            cv::imwrite("crop.jpg", crop);
            auto score = (*m_rgb_anti_spoofing_)(crop);
//            auto i = cv::imread("zsb.jpg");
//            LOGE("SBA: %f", (*m_rgb_anti_spoofing_)(i));
            faceLivenessCache = score;
            break;
        }
        case PROCESS_INTERACTION: {
            if (m_blink_predict_ == nullptr) {
                return HERR_SESS_PIPELINE_FAILURE;       // uninitialized
            }
            if (originImage.empty()) {
                originImage = image.GetScaledImage(1.0, true);
            }
            std::vector<std::vector<int>> order_list = {HLMK_LEFT_EYE_POINTS_INDEX, HLMK_RIGHT_EYE_POINTS_INDEX};
            eyesStatusCache = {0, 0};
            for (size_t i = 0; i < order_list.size(); i++)
            {   
                const auto &index = order_list[i];
                std::vector<cv::Point2f> points;
                for (const auto &idx: index)
                {   
                    points.emplace_back(face.densityLandmark[idx].x, face.densityLandmark[idx].y);
                }
                cv::Rect2f rect = cv::boundingRect(points);
                auto affine_scale = ComputeCropMatrix(rect, BlinkPredict::BLINK_EYE_INPUT_SIZE, BlinkPredict::BLINK_EYE_INPUT_SIZE);
                affine_scale.convertTo(affine_scale, CV_64F);
                auto pre_crop = image.GetAffineRGBImage(affine_scale, BlinkPredict::BLINK_EYE_INPUT_SIZE, BlinkPredict::BLINK_EYE_INPUT_SIZE);
                auto eyeStatus = (*m_blink_predict_)(pre_crop);
                eyesStatusCache[i] = eyeStatus;
            }
            break;
        }
        case PROCESS_ATTRIBUTE: {
            if (m_attribute_predict_ == nullptr) {
                return HERR_SESS_PIPELINE_FAILURE;       // uninitialized
            }
            std::vector<cv::Point2f> pointsFive;
            for (const auto &p: face.keyPoints) {
                pointsFive.push_back(HPointToPoint2f(p));
            }
            auto trans = getTransformMatrix112(pointsFive);
            trans.convertTo(trans, CV_64F);
            auto crop = image.GetAffineRGBImage(trans, 112, 112);
            auto outputs = (*m_attribute_predict_)(crop);
            faceAttributeCache = cv::Vec3i(outputs[0], outputs[1], outputs[2]);
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

int32_t FacePipeline::InitFaceAttributePredict(InspireModel &model) {
    m_attribute_predict_ = std::make_shared<FaceAttributePredict>();
    auto ret = m_attribute_predict_->loadData(model, model.modelType);
    if (ret != InferenceHelper::kRetOk) {
        return HERR_ARCHIVE_LOAD_FAILURE;
    }
    return HSUCCEED;
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

int32_t FacePipeline::InitBlinkFromLivenessInteraction(InspireModel &model) {
    m_blink_predict_ = std::make_shared<BlinkPredict>();
    auto ret = m_blink_predict_->loadData(model, model.modelType);
    if (ret != InferenceHelper::kRetOk) {
        return HERR_ARCHIVE_LOAD_FAILURE;
    }
    return HSUCCEED;
}

const std::shared_ptr<RBGAntiSpoofing> &FacePipeline::getMRgbAntiSpoofing() const {
    return m_rgb_anti_spoofing_;
}


}