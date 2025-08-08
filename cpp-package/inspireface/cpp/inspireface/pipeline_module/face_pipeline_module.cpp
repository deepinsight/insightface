/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "face_pipeline_module.h"

#include "log.h"
#include "track_module/landmark/face_landmark_adapt.h"
#include "track_module/landmark/landmark_param.h"
#include "recognition_module/dest_const.h"
#include "herror.h"

namespace inspire {

FacePipelineModule::FacePipelineModule(InspireArchive &archive, bool enableLiveness, bool enableMaskDetect, bool enableAttribute,
                                       bool enableInteractionLiveness, bool enableFaceEmotion)
: m_enable_liveness_(enableLiveness),
  m_enable_mask_detect_(enableMaskDetect),
  m_enable_attribute_(enableAttribute),
  m_enable_interaction_liveness_(enableInteractionLiveness),
  m_enable_face_emotion_(enableFaceEmotion) {
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
    m_landmark_param_ = archive.GetLandmarkParam();
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

    // Initialize the face emotion model
    if (m_enable_face_emotion_) {
        InspireModel faceEmotionModel;
        auto ret = archive.LoadModel("face_emotion", faceEmotionModel);
        if (ret != 0) {
            INSPIRE_LOGE("Load Face emotion model error.");
        }
        ret = InitFaceEmotion(faceEmotionModel);
        if (ret != 0) {
            INSPIRE_LOGE("InitFaceEmotion error.");
        }
    }
}

int32_t FacePipelineModule::Process(inspirecv::FrameProcess &processor, const FaceTrackWrap &face, FaceProcessFunctionOption proc) {
    // Original image
    inspirecv::Image originImage;
    std::vector<inspirecv::Point2f> stand_lmk;
    switch (proc) {
        case PROCESS_MASK: {
            if (m_mask_predict_ == nullptr) {
                INSPIRE_LOGE("Mask detection disabled");
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
                INSPIRE_LOGE("RGB liveness detection disabled");
                return HERR_SESS_PIPELINE_FAILURE;  // uninitialized
            }
            // New scheme: padding differences cause errors in inference results
            // inspirecv::TransformMatrix rotation_mode_affine = processor.GetAffineMatrix();
            // if (stand_lmk.empty()) {
            //     std::vector<inspirecv::Point2f> lmk;
            //     for (const auto &p : face.densityLandmark) {
            //         lmk.emplace_back(p.x, p.y);
            //     }
            //     stand_lmk = inspirecv::ApplyTransformToPoints(lmk, rotation_mode_affine.GetInverse());
            // }

            // auto rect_face = inspirecv::MinBoundingRect(stand_lmk);
            // auto rect_pts = rect_face.Square(2.7f).As<float>().ToFourVertices();
            // std::vector<inspirecv::Point2f> dst_pts = {{0, 0}, {112, 0}, {112, 112}, {0, 112}};
            // std::vector<inspirecv::Point2f> camera_pts = inspirecv::ApplyTransformToPoints(rect_pts, rotation_mode_affine);

            // auto affine = inspirecv::SimilarityTransformEstimate(camera_pts, dst_pts);
            // auto image_affine = processor.ExecuteImageAffineProcessing(affine, 112, 112);
            // image_affine.Write("liveness_affine.jpg");

            if (originImage.Empty()) {
                // This is a poor approach that impacts performance, 
                // but in order to capture clearer images and improve liveness detection accuracy, 
                // we have to keep it.
                originImage = processor.ExecuteImageScaleProcessing(1.0, true);
            }
            inspirecv::Rect2i oriRect(face.rect.x, face.rect.y, face.rect.width, face.rect.height);
            auto rect = GetNewBox(originImage.Width(), originImage.Height(), oriRect, 2.7f);
            if (Launch::GetInstance()->GetImageProcessingBackend() == Launch::IMAGE_PROCESSING_RGA) {
                // RKRGA must be aligned to 16
                rect = AlignmentBoxToStrideSquareBox(rect, 16);
            }
            auto crop = originImage.Crop(rect);
            auto score = (*m_rgb_anti_spoofing_)(crop);
            // crop.Show();
            // crop.Resize(112, 112).Write("liveness.jpg");
            faceLivenessCache = score;
            break;
        }
        case PROCESS_INTERACTION: {
            if (m_blink_predict_ == nullptr) {
                INSPIRE_LOGE("Interaction action detection disabled");
                return HERR_SESS_PIPELINE_FAILURE;  // uninitialized
            }
            std::vector<std::vector<int>> order_list = {m_landmark_param_->semantic_index.left_eye_region, m_landmark_param_->semantic_index.right_eye_region};
            eyesStatusCache = {0, 0};
            inspirecv::Point2f left_eye = inspirecv::Point2f(face.keyPoints[0].x, face.keyPoints[0].y);
            inspirecv::Point2f right_eye = inspirecv::Point2f(face.keyPoints[1].x, face.keyPoints[1].y);
            std::vector<inspirecv::Point2f> eyes = {left_eye, right_eye};
            // Get affine matrix
            inspirecv::TransformMatrix rotation_mode_affine = processor.GetAffineMatrix();
            // Get stand landmark
            if (stand_lmk.empty()) {
                std::vector<inspirecv::Point2f> lmk;
                for (const auto &p : face.densityLandmark) {
                    lmk.emplace_back(p.x, p.y);
                }
                stand_lmk = inspirecv::ApplyTransformToPoints(lmk, rotation_mode_affine.GetInverse());
            }
            for (size_t i = 0; i < order_list.size(); i++) {
                const auto &index = order_list[i];
                std::vector<inspirecv::Point2i> points;
                for (const auto &idx : index) {
                    points.emplace_back(stand_lmk[idx].GetX(), stand_lmk[idx].GetY());
                }
                auto rect_eye = inspirecv::MinBoundingRect(points).Square(1.5f);
                auto rect_pts_eye = rect_eye.As<float>().ToFourVertices();
                std::vector<inspirecv::Point2f> dst_pts_eye = {{0, 0}, {64, 0}, {64, 64}, {0, 64}};
                std::vector<inspirecv::Point2f> camera_pts_eye = inspirecv::ApplyTransformToPoints(rect_pts_eye, rotation_mode_affine);

                auto affine_eye = inspirecv::SimilarityTransformEstimate(camera_pts_eye, dst_pts_eye);
                auto eye_affine = processor.ExecuteImageAffineProcessing(affine_eye, 64, 64);
                // eye_affine.Write("eye_"+std::to_string(i)+".jpg");
                // auto crop = originImage.Crop(new_rect);
                auto score = (*m_blink_predict_)(eye_affine);
                eyesStatusCache[i] = score;
            }
            break;
        }
        case PROCESS_ATTRIBUTE: {
            if (m_attribute_predict_ == nullptr) {
                INSPIRE_LOGE("Face attribute detection disabled");
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
        case PROCESS_FACE_EMOTION: {
            if (m_face_emotion_ == nullptr) {
                INSPIRE_LOGE("Face emotion detection disabled");
                return HERR_SESS_PIPELINE_FAILURE;  // uninitialized
            }
            std::vector<inspirecv::Point2f> pointsFive;
            for (const auto &p : face.keyPoints) {
                pointsFive.push_back(inspirecv::Point2f(p.x, p.y));
            }
            auto trans = inspirecv::SimilarityTransformEstimateUmeyama(SIMILARITY_TRANSFORM_DEST, pointsFive);
            auto crop = processor.ExecuteImageAffineProcessing(trans, FACE_CROP_SIZE, FACE_CROP_SIZE);
            // crop.Show();
            faceEmotionCache = (*m_face_emotion_)(crop);
            break;
        }
    }
    return HSUCCEED;
}

int32_t FacePipelineModule::Process(inspirecv::FrameProcess &processor, FaceObjectInternal &face) {
    // In the tracking state, the count meets the requirements or the pipeline is executed in the detection state
    auto lmk = face.keyPointFive;
    std::vector<inspirecv::Point2f> lmk_5 = {lmk[m_landmark_param_->semantic_index.left_eye_center], lmk[m_landmark_param_->semantic_index.right_eye_center],
                                             lmk[m_landmark_param_->semantic_index.nose_corner], lmk[m_landmark_param_->semantic_index.mouth_left_corner],
                                             lmk[m_landmark_param_->semantic_index.mouth_right_corner]};
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
    auto ret = m_attribute_predict_->LoadData(model, model.modelType);
    if (ret != InferenceWrapper::WrapperOk) {
        return HERR_ARCHIVE_LOAD_FAILURE;
    }
    return HSUCCEED;
}

int32_t FacePipelineModule::InitMaskPredict(InspireModel &model) {
    m_mask_predict_ = std::make_shared<MaskPredictAdapt>();
    auto ret = m_mask_predict_->LoadData(model, model.modelType);
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
    auto ret = m_rgb_anti_spoofing_->LoadData(model, model.modelType);
    if (ret != InferenceWrapper::WrapperOk) {
        return HERR_ARCHIVE_LOAD_FAILURE;
    }
    return HSUCCEED;
}

int32_t FacePipelineModule::InitBlinkFromLivenessInteraction(InspireModel &model) {
    m_blink_predict_ = std::make_shared<BlinkPredictAdapt>();
    auto ret = m_blink_predict_->LoadData(model, model.modelType);
    if (ret != InferenceWrapper::WrapperOk) {
        return HERR_ARCHIVE_LOAD_FAILURE;
    }
    return HSUCCEED;
}

int32_t FacePipelineModule::InitLivenessInteraction(InspireModel &model) {
    return 0;
}

int32_t FacePipelineModule::InitFaceEmotion(InspireModel &model) {
    m_face_emotion_ = std::make_shared<FaceEmotionAdapt>();
    auto ret = m_face_emotion_->LoadData(model, model.modelType);
    if (ret != InferenceWrapper::WrapperOk) {
        return HERR_ARCHIVE_LOAD_FAILURE;
    }
    return HSUCCEED;
}

const std::shared_ptr<RBGAntiSpoofingAdapt> &FacePipelineModule::getMRgbAntiSpoofing() const {
    return m_rgb_anti_spoofing_;
}

}  // namespace inspire