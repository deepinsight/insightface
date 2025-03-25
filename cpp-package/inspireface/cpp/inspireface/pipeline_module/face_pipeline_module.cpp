/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "face_pipeline_module.h"

#include "log.h"
#include "track_module/landmark/face_landmark_adapt.h"
#include "recognition_module/dest_const.h"
#include "herror.h"
#include "liveness/order_of_hyper_landmark.h"

namespace inspire {

FacePipelineModule::FacePipelineModule(InspireArchive &archive, bool enableLiveness, bool enableMaskDetect, bool enableAttribute,
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

int32_t FacePipelineModule::Process(inspirecv::InspireImageProcess &processor, const HyperFaceData &face, FaceProcessFunctionOption proc) {
    inspirecv::Image originImage;
    inspirecv::Image scaleImage;
    switch (proc) {
        case PROCESS_MASK: {
            if (m_mask_predict_ == nullptr) {
                return HERR_SESS_PIPELINE_FAILURE;  // uninitialized
            }
            std::vector<inspirecv::Point2f> pointsFive;
            for (const auto &p : face.keyPoints) {
                pointsFive.push_back(inspirecv::Point2f(p.x, p.y));
            }

            auto trans = inspirecv::SimilarityTransformEstimateUmeyama(SIMILARITY_TRANSFORM_DEST, pointsFive);
            auto crop = processor.ExecuteImageAffineProcessing(trans, FACE_CROP_SIZE, FACE_CROP_SIZE);
            auto mask_score = (*m_mask_predict_)(crop);
            // crop.Show();
            faceMaskCache = mask_score;
            break;
        }
        case PROCESS_RGB_LIVENESS: {
            if (m_rgb_anti_spoofing_ == nullptr) {
                return HERR_SESS_PIPELINE_FAILURE;  // uninitialized
            }

            if (originImage.Empty()) {
                originImage = processor.ExecuteImageScaleProcessing(1.0, true);
            }
            inspirecv::Rect2i oriRect(face.rect.x, face.rect.y, face.rect.width, face.rect.height);
            auto rect = GetNewBox(originImage.Width(), originImage.Height(), oriRect, 2.7f);
            auto crop = originImage.Crop(rect);
            auto score = (*m_rgb_anti_spoofing_)(crop);
            // crop.Show();
            faceLivenessCache = score;
            break;
        }
        case PROCESS_INTERACTION: {
            if (m_blink_predict_ == nullptr) {
                return HERR_SESS_PIPELINE_FAILURE;  // uninitialized
            }
            if (originImage.Empty()) {
                originImage = processor.ExecuteImageScaleProcessing(1.0, true);
            }
            std::vector<std::vector<int>> order_list = {HLMK_LEFT_EYE_POINTS_INDEX, HLMK_RIGHT_EYE_POINTS_INDEX};
            eyesStatusCache = {0, 0};
            inspirecv::Point2f left_eye = inspirecv::Point2f(face.keyPoints[0].x, face.keyPoints[0].y);
            inspirecv::Point2f right_eye = inspirecv::Point2f(face.keyPoints[1].x, face.keyPoints[1].y);
            std::vector<inspirecv::Point2f> eyes = {left_eye, right_eye};
            auto new_eyes_points = inspirecv::ApplyTransformToPoints(eyes, processor.GetAffineMatrix().GetInverse());
            for (size_t i = 0; i < order_list.size(); i++) {
                const auto &index = order_list[i];
                std::vector<inspirecv::Point2i> points;
                for (const auto &idx : index) {
                    points.emplace_back(face.densityLandmark[idx].x, face.densityLandmark[idx].y);
                }
                auto rect = inspirecv::MinBoundingRect(points);
                auto mat = processor.GetAffineMatrix();
                auto new_rect = inspirecv::ApplyTransformToRect(rect, mat.GetInverse()).Square(1.3f);
                // Use more accurate 5 key point calibration
                auto cx = new_eyes_points[i].GetX();
                auto cy = new_eyes_points[i].GetY();
                new_rect.SetX(cx - new_rect.GetWidth() / 2);
                new_rect.SetY(cy - new_rect.GetHeight() / 2);

                // Ensure rect stays within image bounds while maintaining aspect ratio
                float originalAspectRatio = new_rect.GetWidth() / new_rect.GetHeight();

                // Adjust position and size to fit within image bounds
                if (new_rect.GetX() < 0) {
                    new_rect.SetWidth(new_rect.GetWidth() + new_rect.GetX());  // Reduce width by overflow amount
                    new_rect.SetX(0);
                }
                if (new_rect.GetY() < 0) {
                    new_rect.SetHeight(new_rect.GetHeight() + new_rect.GetY());  // Reduce height by overflow amount
                    new_rect.SetY(0);
                }

                float rightOverflow = (new_rect.GetX() + new_rect.GetWidth()) - originImage.Width();
                if (rightOverflow > 0) {
                    new_rect.SetWidth(new_rect.GetWidth() - rightOverflow);
                }

                float bottomOverflow = (new_rect.GetY() + new_rect.GetHeight()) - originImage.Height();
                if (bottomOverflow > 0) {
                    new_rect.SetHeight(new_rect.GetHeight() - bottomOverflow);
                }

                // Maintain minimum size (e.g., 20x20 ixels)
                const float minSize = 20.0f;
                if (new_rect.GetWidth() < minSize || new_rect.GetHeight() < minSize) {
                    continue;  // Skip this eye if the crop region is too small
                }

                auto crop = originImage.Crop(new_rect);
                auto score = (*m_blink_predict_)(crop);
                eyesStatusCache[i] = score;
            }
            break;
        }
        case PROCESS_ATTRIBUTE: {
            if (m_attribute_predict_ == nullptr) {
                return HERR_SESS_PIPELINE_FAILURE;  // uninitialized
            }
            std::vector<inspirecv::Point2f> pointsFive;
            for (const auto &p : face.keyPoints) {
                pointsFive.push_back(inspirecv::Point2f(p.x, p.y));
            }
            auto trans = inspirecv::SimilarityTransformEstimateUmeyama(SIMILARITY_TRANSFORM_DEST, pointsFive);
            auto crop = processor.ExecuteImageAffineProcessing(trans, FACE_CROP_SIZE, FACE_CROP_SIZE);
            auto outputs = (*m_attribute_predict_)(crop);
            faceAttributeCache = inspirecv::Vec3i{outputs[0], outputs[1], outputs[2]};
            break;
        }
    }
    return HSUCCEED;
}

int32_t FacePipelineModule::Process(inspirecv::InspireImageProcess &processor, FaceObjectInternal &face) {
    // In the tracking state, the count meets the requirements or the pipeline is executed in the detection state
    auto lmk = face.keyPointFive;
    std::vector<inspirecv::Point2f> lmk_5 = {lmk[FaceLandmarkAdapt::LEFT_EYE_CENTER], lmk[FaceLandmarkAdapt::RIGHT_EYE_CENTER],
                                             lmk[FaceLandmarkAdapt::NOSE_CORNER], lmk[FaceLandmarkAdapt::MOUTH_LEFT_CORNER],
                                             lmk[FaceLandmarkAdapt::MOUTH_RIGHT_CORNER]};
    auto trans = inspirecv::SimilarityTransformEstimateUmeyama(SIMILARITY_TRANSFORM_DEST, lmk_5);
    auto crop = processor.ExecuteImageAffineProcessing(trans, FACE_CROP_SIZE, FACE_CROP_SIZE);
    if (m_mask_predict_ != nullptr) {
        auto mask_score = (*m_mask_predict_)(crop);
        if (mask_score > 0.95) {
            face.faceProcess.maskInfo = MaskInfo::MASKED;
        } else {
            face.faceProcess.maskInfo = MaskInfo::UNMASKED;
        }
    }

    if (m_rgb_anti_spoofing_ != nullptr) {
        auto img = processor.ExecuteImageScaleProcessing(1.0, true);
        inspirecv::Rect2i oriRect(face.detect_bbox_.GetX(), face.detect_bbox_.GetY(), face.detect_bbox_.GetWidth(), face.detect_bbox_.GetHeight());
        auto rect = oriRect.Square(2.7f);
        auto crop = img.Crop(rect);
        auto score = (*m_rgb_anti_spoofing_)(crop);
        if (score > 0.88) {
            face.faceProcess.rgbLivenessInfo = RGBLivenessInfo::LIVENESS_REAL;
        } else {
            face.faceProcess.rgbLivenessInfo = RGBLivenessInfo::LIVENESS_FAKE;
        }
    }

    return HSUCCEED;
}

int32_t FacePipelineModule::InitFaceAttributePredict(InspireModel &model) {
    m_attribute_predict_ = std::make_shared<FaceAttributePredictAdapt>();
    auto ret = m_attribute_predict_->loadData(model, model.modelType);
    if (ret != InferenceWrapper::WrapperOk) {
        return HERR_ARCHIVE_LOAD_FAILURE;
    }
    return HSUCCEED;
}

int32_t FacePipelineModule::InitMaskPredict(InspireModel &model) {
    m_mask_predict_ = std::make_shared<MaskPredictAdapt>();
    auto ret = m_mask_predict_->loadData(model, model.modelType);
    if (ret != InferenceWrapper::WrapperOk) {
        return HERR_ARCHIVE_LOAD_FAILURE;
    }
    return HSUCCEED;
}

int32_t FacePipelineModule::InitRBGAntiSpoofing(InspireModel &model) {
    auto input_size = model.Config().get<std::vector<int>>("input_size");
#ifdef INFERENCE_WRAPPER_ENABLE_RKNN2
    m_rgb_anti_spoofing_ = std::make_shared<RBGAntiSpoofingAdapt>(input_size[0], true);
#else
    m_rgb_anti_spoofing_ = std::make_shared<RBGAntiSpoofingAdapt>(input_size[0]);
#endif
    auto ret = m_rgb_anti_spoofing_->loadData(model, model.modelType);
    if (ret != InferenceWrapper::WrapperOk) {
        return HERR_ARCHIVE_LOAD_FAILURE;
    }
    return HSUCCEED;
}

int32_t FacePipelineModule::InitBlinkFromLivenessInteraction(InspireModel &model) {
    m_blink_predict_ = std::make_shared<BlinkPredictAdapt>();
    auto ret = m_blink_predict_->loadData(model, model.modelType);
    if (ret != InferenceWrapper::WrapperOk) {
        return HERR_ARCHIVE_LOAD_FAILURE;
    }
    return HSUCCEED;
}

int32_t FacePipelineModule::InitLivenessInteraction(InspireModel &model) {
    return 0;
}

const std::shared_ptr<RBGAntiSpoofingAdapt> &FacePipelineModule::getMRgbAntiSpoofing() const {
    return m_rgb_anti_spoofing_;
}

}  // namespace inspire