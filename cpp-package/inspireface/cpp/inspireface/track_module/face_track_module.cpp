/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "face_track_module.h"
#include "log.h"
#include <algorithm>
#include <cstddef>
#include "middleware/costman.h"
#include "middleware/model_archive/inspire_archive.h"
#include "middleware/utils.h"
#include "herror.h"
#include "middleware/costman.h"
#include "cost_time.h"
#include "spend_timer.h"
#include "launch.h"

namespace inspire {

FaceTrackModule::FaceTrackModule(DetectModuleMode mode, int max_detected_faces, int detection_interval, int track_preview_size,
                                 int dynamic_detection_input_level, int TbD_mode_fps, bool detect_mode_landmark)
: m_mode_(mode),
  max_detected_faces_(max_detected_faces),
  detection_interval_(detection_interval),
  track_preview_size_(track_preview_size),
  m_dynamic_detection_input_level_(dynamic_detection_input_level),
  m_detect_mode_landmark_(detect_mode_landmark) {
    detection_index_ = -1;
    tracking_idx_ = 0;
    if (TbD_mode_fps < 0) {
        TbD_mode_fps = 30;
    }
    if (mode == DETECT_MODE_LIGHT_TRACK) {
        // In lightweight tracking mode, landmark detection is always required
        m_detect_mode_landmark_ = true;
    } else {
        // This version uses lmk106 to replace five key points, so lmk must be forcibly enabled!
        m_detect_mode_landmark_ = true;
    }
    if (m_mode_ == DETECT_MODE_TRACK_BY_DETECT) {
        m_TbD_tracker_ = std::make_shared<BYTETracker>(TbD_mode_fps, 30);
    }
}

void FaceTrackModule::SparseLandmarkPredict(const inspirecv::Image &raw_face_crop, std::vector<inspirecv::Point2f> &landmarks_output, float &score,
                                            float size) {
    COST_TIME_SIMPLE(SparseLandmarkPredict);
    landmarks_output.resize(m_landmark_param_->num_of_landmark);
    std::vector<float> lmk_out = (*m_landmark_predictor_)(raw_face_crop);
    for (int i = 0; i < m_landmark_param_->num_of_landmark; ++i) {
        float x = lmk_out[i * 2 + 0] * size;
        float y = lmk_out[i * 2 + 1] * size;
        landmarks_output[i] = inspirecv::Point<float>(x, y);
    }
}

float FaceTrackModule::PredictTrackScore(const inspirecv::Image &raw_face_crop) {
    return (*m_refine_net_)(raw_face_crop);
}

bool FaceTrackModule::TrackFace(inspirecv::FrameProcess &image, FaceObjectInternal &face) {
    COST_TIME_SIMPLE(TrackFace);
    // If the track lost recovery mode is enabled,  the lag information of the previous frame will not be used in the current frame
    if (face.GetConfidence() < m_light_track_confidence_threshold_ && !m_track_lost_recovery_mode_) {
        // If the face confidence level is below the threshold, disable tracking
        face.DisableTracking();
        return false;
    }

    inspirecv::TransformMatrix affine;
    std::vector<inspirecv::Point2f> landmark_back;

    // Increase track count
    face.IncrementTrackingCount();

    float score;
    // If it is a detection state, calculate the affine transformation matrix
    if (face.TrackingState() == ISF_DETECT) {
        COST_TIME_SIMPLE(GetRectSquare);
        inspirecv::Rect2i rect_square = face.GetRectSquare(0);

        std::vector<inspirecv::Point2f> rect_pts = rect_square.As<float>().ToFourVertices();
        inspirecv::TransformMatrix rotation_mode_affine = image.GetAffineMatrix();
        std::vector<inspirecv::Point2f> camera_pts = ApplyTransformToPoints(rect_pts, rotation_mode_affine);
        // camera_pts.erase(camera_pts.end() - 1);
        std::vector<inspirecv::Point2f> dst_pts = {{0, 0},
                                                   {(float)m_landmark_param_->input_size, 0},
                                                   {(float)m_landmark_param_->input_size, (float)m_landmark_param_->input_size},
                                                   {0, (float)m_landmark_param_->input_size}};
        affine = inspirecv::SimilarityTransformEstimate(camera_pts, dst_pts);
        face.setTransMatrix(affine);

        if (!m_detect_mode_landmark_) {
            /*If landmark is not extracted, the detection frame of the preview image needs to be changed
            back to the coordinate system of the original image */
            std::vector<inspirecv::Point2f> restore_rect_pts = inspirecv::ApplyTransformToPoints(rect_pts, rotation_mode_affine);
            inspirecv::Rect2f restore_rect = inspirecv::MinBoundingRect(restore_rect_pts);
            face.bbox_ = restore_rect.As<int>();
        }
    }

    if (m_face_quality_ != nullptr) {
        COST_TIME_SIMPLE(FaceQuality);
        auto affine_extensive = face.getTransMatrix();
        auto trans_e = ScaleAffineMatrixPreserveCenter(affine_extensive, m_crop_extensive_ratio_, m_landmark_param_->input_size);
        auto pre_crop = image.ExecuteImageAffineProcessing(trans_e, m_landmark_param_->input_size, m_landmark_param_->input_size);
        auto res = (*m_face_quality_)(pre_crop);
        // pre_crop.Show("pre_crop");

        auto affine_extensive_inv = affine_extensive.GetInverse();
        std::vector<inspirecv::Point2f> lmk_extensive = ApplyTransformToPoints(res.lmk, affine_extensive_inv);
        res.lmk = lmk_extensive;
        face.high_result = res;
    } else {
        // If face pose and quality model is not initialized, set the default value
        FacePoseQualityAdaptResult empty_result;
        empty_result.lmk = std::vector<inspirecv::Point2f>(5, inspirecv::Point2f(0, 0));
        empty_result.lmk_quality = std::vector<float>(5, 2.0f);
        empty_result.pitch = 0.0f;
        empty_result.yaw = 0.0f;
        empty_result.roll = 0.0f;
        face.high_result = empty_result;
    }

    if (m_detect_mode_landmark_) {
        // If Landmark need to be extracted in detection mode,
        // Landmark must be detected when fast tracing is enabled
        affine = face.getTransMatrix();
        inspirecv::TransformMatrix affine_inv = affine.GetInverse();
        std::vector<inspirecv::Point2f> landmark_rawout;
        std::vector<std::vector<inspirecv::Point2f>> multiscale_landmark_back;

        auto track_crop = image.ExecuteImageAffineProcessing(affine, m_landmark_param_->input_size, m_landmark_param_->input_size);
        score = PredictTrackScore(track_crop);
        // track_crop.Show("track_crop");

        // If the track lost recovery mode is enabled, 
        // it will determine whether to discard the invalid face that has been tracked in the current frame
        if (score < m_light_track_confidence_threshold_ && m_track_lost_recovery_mode_) {
            face.DisableTracking();
            return false;
        }

        for (int i = 0; i < m_multiscale_landmark_scales_.size(); i++) {
            inspirecv::Image crop;
            // Get the RGB image after affine transformation
            auto affine_scale = ScaleAffineMatrixPreserveCenter(affine, m_multiscale_landmark_scales_[i], m_landmark_param_->input_size);
            crop = image.ExecuteImageAffineProcessing(affine_scale, m_landmark_param_->input_size, m_landmark_param_->input_size);

            std::vector<inspirecv::Point2f> lmk_predict;

            // Predicted sparse key point
            SparseLandmarkPredict(crop, lmk_predict, score, m_landmark_param_->input_size);

            // Save the first scale landmark
            if (i == 0) {
                landmark_rawout = lmk_predict;
            }

            std::vector<inspirecv::Point2f> lmk_back;
            // Convert key points back to the original coordinate system
            lmk_back.resize(lmk_predict.size());
            lmk_back = inspirecv::ApplyTransformToPoints(lmk_predict, affine_scale.GetInverse());

            multiscale_landmark_back.push_back(lmk_back);
        }

        landmark_back = MultiFrameLandmarkMean(multiscale_landmark_back);

        // Extract 5 key points
        std::vector<inspirecv::Point2f> lmk_5 = {
          landmark_rawout[m_landmark_param_->semantic_index.left_eye_center], landmark_rawout[m_landmark_param_->semantic_index.right_eye_center],
          landmark_rawout[m_landmark_param_->semantic_index.nose_corner], landmark_rawout[m_landmark_param_->semantic_index.mouth_left_corner],
          landmark_rawout[m_landmark_param_->semantic_index.mouth_right_corner]};
        face.setAlignMeanSquareError(lmk_5);

        int MODE = 1;

        if (MODE > 0) {
            if (face.TrackingState() == ISF_DETECT) {
                face.ReadyTracking();
            } else if (face.TrackingState() == ISF_READY || face.TrackingState() == ISF_TRACKING) {
                COST_TIME_SIMPLE(LandmarkBack);
                inspirecv::TransformMatrix trans_m;
                // inspirecv::TransformMatrix tmp = face.getTransMatrix();
                std::vector<inspirecv::Point2f> inside_points;
                if (m_landmark_param_->input_size == 112) {
                    inside_points = landmark_rawout;
                } else {
                    inside_points = LandmarkCropped(landmark_rawout);
                }

                auto &mean_shape_ = m_landmark_param_->mean_shape_points;

                auto _affine = inspirecv::SimilarityTransformEstimate(inside_points, mean_shape_);
                auto mid_inside_points = ApplyTransformToPoints(inside_points, _affine);
                inside_points = FixPointsMeanshape(mid_inside_points, mean_shape_);

                trans_m = inspirecv::SimilarityTransformEstimate(landmark_back, inside_points);
                face.setTransMatrix(trans_m);
                face.EnableTracking();
            }
        }
        // Update face key points
        face.SetLandmark(landmark_back, true, true, m_track_mode_smooth_ratio_, m_track_mode_num_smooth_cache_frame_,
                         m_landmark_param_->num_of_landmark * 2);
        // Get the smoothed landmark
        auto &landmark_smooth = face.landmark_smooth_aux_.back();
        // Update the face key points
        face.high_result.lmk[0] = landmark_smooth[m_landmark_param_->semantic_index.left_eye_center];
        face.high_result.lmk[1] = landmark_smooth[m_landmark_param_->semantic_index.right_eye_center];
        face.high_result.lmk[2] = landmark_smooth[m_landmark_param_->semantic_index.nose_corner];
        face.high_result.lmk[3] = landmark_smooth[m_landmark_param_->semantic_index.mouth_left_corner];
        face.high_result.lmk[4] = landmark_smooth[m_landmark_param_->semantic_index.mouth_right_corner];
    }

    // If tracking status, update the confidence level
    if (face.TrackingState() == ISF_TRACKING) {
        face.SetConfidence(score);
    }

    return true;
}

void FaceTrackModule::UpdateStream(inspirecv::FrameProcess &image) {
    inspire::SpendTimer total("UpdateStream");
    total.Start();
    COST_TIME_SIMPLE(FaceTrackUpdateStream);
    detection_index_ += 1;
    if (m_mode_ == DETECT_MODE_ALWAYS_DETECT || m_mode_ == DETECT_MODE_TRACK_BY_DETECT)
        trackingFace.clear();

    // Record whether the detection has been performed in this frame
    bool detection_executed = false;
    if (trackingFace.empty() || (detection_interval_ > 0 && detection_index_ % detection_interval_ == 0) || m_mode_ == DETECT_MODE_ALWAYS_DETECT ||
        m_mode_ == DETECT_MODE_TRACK_BY_DETECT) {
        image.SetPreviewSize(track_preview_size_);
        inspirecv::Image image_detect = image.ExecutePreviewImageProcessing(true);
        m_debug_preview_image_size_ = image_detect.Width();

        nms();
        for (auto const &face : trackingFace) {
            inspirecv::Rect2i m_mask_rect = face.GetRect();
            std::vector<inspirecv::Point2f> pts = m_mask_rect.As<float>().ToFourVertices();
            inspirecv::TransformMatrix rotation_mode_affine = image.GetAffineMatrix();
            auto rotation_mode_affine_inv = rotation_mode_affine.GetInverse();
            std::vector<inspirecv::Point2f> affine_pts = inspirecv::ApplyTransformToPoints(pts, rotation_mode_affine_inv);
            inspirecv::Rect2f mask_rect = inspirecv::MinBoundingRect(affine_pts);
            BlackingTrackingRegion(image_detect, mask_rect);
        }

        Timer det_cost_time;
        DetectFace(image_detect, image.GetPreviewScale());
        detection_executed = true;
    }

    if (!candidate_faces_.empty()) {
        for (int i = 0; i < candidate_faces_.size(); i++) {
            trackingFace.push_back(candidate_faces_[i]);
        }
        candidate_faces_.clear();
    }

    // Record the number of faces before tracking
    size_t faces_before_tracking = trackingFace.size();

    for (std::vector<FaceObjectInternal>::iterator iter = trackingFace.begin(); iter != trackingFace.end();) {
        if (!TrackFace(image, *iter)) {
            iter = trackingFace.erase(iter);
        } else {
            iter++;
        }
    }

    // In the track lost recovery mode, if all faces are triggered to be lost, detection will be executed immediately 
    if (m_track_lost_recovery_mode_ && !detection_executed && faces_before_tracking > 0 && trackingFace.empty()) {
        image.SetPreviewSize(track_preview_size_);
        inspirecv::Image image_detect = image.ExecutePreviewImageProcessing(true);
        m_debug_preview_image_size_ = image_detect.Width();
        DetectFace(image_detect, image.GetPreviewScale());

        // Reset the detection index to 0
        detection_index_ = 0;

        // Add the detected faces to the tracking face list
        if (!candidate_faces_.empty()) {
            for (int i = 0; i < candidate_faces_.size(); i++) {
                trackingFace.push_back(candidate_faces_[i]);
            }
            candidate_faces_.clear();
        }

        // Track the faces  
        for (std::vector<FaceObjectInternal>::iterator iter = trackingFace.begin(); iter != trackingFace.end();) {
            if (!TrackFace(image, *iter)) {
                iter = trackingFace.erase(iter);
            } else {
                iter++;
            }
        }
    }

    total.Stop();
    // std::cout << total << std::endl;
}

void FaceTrackModule::nms(float th) {
    std::sort(trackingFace.begin(), trackingFace.end(), [](FaceObjectInternal a, FaceObjectInternal b) { return a.confidence_ > b.confidence_; });
    std::vector<float> area(trackingFace.size());
    for (int i = 0; i < int(trackingFace.size()); ++i) {
        area[i] = trackingFace.at(i).getBbox().Area();
    }
    for (int i = 0; i < int(trackingFace.size()); ++i) {
        for (int j = i + 1; j < int(trackingFace.size());) {
            float xx1 = (std::max)(trackingFace[i].getBbox().GetX(), trackingFace[j].getBbox().GetX());
            float yy1 = (std::max)(trackingFace[i].getBbox().GetY(), trackingFace[j].getBbox().GetY());
            float xx2 = (std::min)(trackingFace[i].getBbox().GetX() + trackingFace[i].getBbox().GetWidth(),
                                   trackingFace[j].getBbox().GetX() + trackingFace[j].getBbox().GetWidth());
            float yy2 = (std::min)(trackingFace[i].getBbox().GetY() + trackingFace[i].getBbox().GetHeight(),
                                   trackingFace[j].getBbox().GetY() + trackingFace[j].getBbox().GetHeight());
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (area[i] + area[j] - inter);
            if (ovr >= th) {
                trackingFace.erase(trackingFace.begin() + j);
                area.erase(area.begin() + j);
            } else {
                j++;
            }
        }
    }
}

void FaceTrackModule::BlackingTrackingRegion(inspirecv::Image &image, inspirecv::Rect2f &rect_mask) {
    COST_TIME_SIMPLE(BlackingTrackingRegion);
    int height = image.Height();
    int width = image.Width();
    auto ext = rect_mask.Square(1.5f);
    inspirecv::Rect2i safe_rect = ext.SafeRect(width, height).As<int>();
    image.Fill(safe_rect, {0, 0, 0});
}

void FaceTrackModule::DetectFace(const inspirecv::Image &input, float scale) {
    std::vector<FaceLoc> boxes = (*m_face_detector_)(input);

    if (m_mode_ == DETECT_MODE_TRACK_BY_DETECT) {
        std::vector<Object> objects;
        auto num_of_effective = std::min(boxes.size(), (size_t)max_detected_faces_);
        for (size_t i = 0; i < num_of_effective; i++) {
            Object obj;
            const auto box = boxes[i];
            obj.rect = inspirecv::Rect<int>(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
            if (!isShortestSideGreaterThan<int>(obj.rect, filter_minimum_face_px_size, scale)) {
                // Filter too small face detection box
                continue;
            }
            obj.label = 0;  // assuming all detections are faces
            obj.prob = box.score;
            objects.push_back(obj);
        }
        std::vector<STrack> output_stracks = m_TbD_tracker_->update(objects);
        for (const auto &st_track : output_stracks) {
            inspirecv::Rect<int> rect = inspirecv::Rect<int>(st_track.tlwh[0], st_track.tlwh[1], st_track.tlwh[2], st_track.tlwh[3]);
            FaceObjectInternal faceinfo(st_track.track_id, rect, FaceLandmarkAdapt::NUM_OF_LANDMARK + 10);
            faceinfo.detect_bbox_ = rect.As<int>();
            candidate_faces_.push_back(faceinfo);
        }
    } else {
        std::vector<inspirecv::Rect2i> bbox;
        bbox.resize(boxes.size());
        for (int i = 0; i < boxes.size(); i++) {
            bbox[i] = inspirecv::Rect<int>::Create(boxes[i].x1, boxes[i].y1, boxes[i].x2 - boxes[i].x1, boxes[i].y2 - boxes[i].y1);

            if (!isShortestSideGreaterThan<int>(bbox[i], filter_minimum_face_px_size, scale)) {
                // Filter too small face detection box
                continue;
            }
            if (m_mode_ == DETECT_MODE_ALWAYS_DETECT) {
                // Always detect mode without assigning an id
                tracking_idx_ = -1;
            } else {
                tracking_idx_ = tracking_idx_ + 1;
            }

            FaceObjectInternal faceinfo(tracking_idx_, bbox[i], m_landmark_param_->num_of_landmark + 10);
            faceinfo.detect_bbox_ = bbox[i];
            faceinfo.SetConfidence(boxes[i].score);

            // Control that the number of faces detected does not exceed the maximum limit
            if (candidate_faces_.size() >= max_detected_faces_) {
                continue;
            }

            candidate_faces_.push_back(faceinfo);
        }
    }
}

int FaceTrackModule::Configuration(inspire::InspireArchive &archive, const std::string &expansion_path, bool enable_face_pose_and_quality) {
    // Initialize the detection model
    m_landmark_param_ = archive.GetLandmarkParam();
    m_expansion_path_ = std::move(expansion_path);
    InspireModel detModel;
    auto scheme = ChoiceMultiLevelDetectModel(m_dynamic_detection_input_level_, track_preview_size_);
    auto ret = archive.LoadModel(scheme, detModel);
    if (ret != SARC_SUCCESS) {
        INSPIRE_LOGE("Load %s error: %d", scheme.c_str(), ret);
        return HERR_ARCHIVE_LOAD_MODEL_FAILURE;
    }
    InitDetectModel(detModel);

    // Initialize the landmark model
    InspireModel lmkModel;
    ret = archive.LoadModel(m_landmark_param_->landmark_engine_name, lmkModel);
    if (ret != SARC_SUCCESS) {
        INSPIRE_LOGE("Load %s error: %d", m_landmark_param_->landmark_engine_name.c_str(), ret);
        return HERR_ARCHIVE_LOAD_MODEL_FAILURE;
    }
    InitLandmarkModel(lmkModel);

    // Initialize the r-net model
    InspireModel rnetModel;
    ret = archive.LoadModel("refine_net", rnetModel);
    if (ret != SARC_SUCCESS) {
        INSPIRE_LOGE("Load %s error: %d", "refine_net", ret);
        return HERR_ARCHIVE_LOAD_MODEL_FAILURE;
    }
    InitRNetModel(rnetModel);
    if (enable_face_pose_and_quality) {
        // Initialize the pose quality model
        InspireModel pquModel;
        ret = archive.LoadModel("pose_quality", pquModel);
        if (ret != SARC_SUCCESS) {
            INSPIRE_LOGE("Load %s error: %d", "pose_quality", ret);
            return HERR_ARCHIVE_LOAD_MODEL_FAILURE;
        }
        InitFacePoseAndQualityModel(pquModel);
    }
    m_landmark_crop_ratio_ = m_landmark_param_->expansion_scale;
    m_multiscale_landmark_scales_ = GenerateCropScales(m_landmark_crop_ratio_, m_multiscale_landmark_loop_num_);
    return 0;
}

int FaceTrackModule::InitLandmarkModel(InspireModel &model) {
    m_landmark_predictor_ =
      std::make_shared<FaceLandmarkAdapt>(m_landmark_param_->input_size, m_landmark_param_->normalization_mode == "CenterScaling");
    auto ret = m_landmark_predictor_->LoadData(model, model.modelType);
    if (ret != InferenceWrapper::WrapperOk) {
        return HERR_ARCHIVE_LOAD_FAILURE;
    }
    return HSUCCEED;
}

int FaceTrackModule::InitDetectModel(InspireModel &model) {
    std::vector<int> input_size;
    input_size = model.Config().get<std::vector<int>>("input_size");

    m_face_detector_ = std::make_shared<FaceDetectAdapt>(input_size[0]);
    auto ret = m_face_detector_->LoadData(model, model.modelType, false);
    if (ret != InferenceWrapper::WrapperOk) {
        return HERR_ARCHIVE_LOAD_FAILURE;
    }
    return HSUCCEED;
}

int FaceTrackModule::InitRNetModel(InspireModel &model) {
    m_refine_net_ = std::make_shared<RNetAdapt>();
    auto ret = m_refine_net_->LoadData(model, model.modelType);
    if (ret != InferenceWrapper::WrapperOk) {
        return HERR_ARCHIVE_LOAD_FAILURE;
    }
    return HSUCCEED;
}

int FaceTrackModule::InitFacePoseAndQualityModel(InspireModel &model) {
    m_face_quality_ = std::make_shared<FacePoseQualityAdapt>();
    auto ret = m_face_quality_->LoadData(model, model.modelType);
    if (ret != InferenceWrapper::WrapperOk) {
        return HERR_ARCHIVE_LOAD_FAILURE;
    }
    return HSUCCEED;
}

void FaceTrackModule::SetDetectThreshold(float value) {
    m_face_detector_->SetClsThreshold(value);
}

void FaceTrackModule::SetMinimumFacePxSize(float value) {
    filter_minimum_face_px_size = value;
}

void FaceTrackModule::SetTrackPreviewSize(int preview_size) {
    track_preview_size_ = preview_size;
    if (track_preview_size_ == -1) {
        track_preview_size_ = m_face_detector_->GetInputSize();
    } else if (track_preview_size_ < 160) {
        INSPIRE_LOGW("Track preview size %d is less than the minimum input size %d", track_preview_size_, 160);
        track_preview_size_ = 160;
    }
}

int32_t FaceTrackModule::GetTrackPreviewSize() const {
    return track_preview_size_;
}

std::string FaceTrackModule::ChoiceMultiLevelDetectModel(const int32_t pixel_size, int32_t &final_size) {
    const auto face_detect_pixel_list = Launch::GetInstance()->GetFaceDetectPixelList();
    const auto face_detect_model_list = Launch::GetInstance()->GetFaceDetectModelList();
    const int32_t num_sizes = face_detect_pixel_list.size();
    if (pixel_size == -1) {
        // Find index with value 320, use index 1 as fallback
        int index = 1;
        for (int i = 0; i < face_detect_pixel_list.size(); ++i) {
            if (face_detect_pixel_list[i] == 320) {
                index = i;
                break;
            }
        }
        final_size = face_detect_pixel_list[index];
        return face_detect_model_list[index];
    }

    // Check for exact match
    for (int i = 0; i < num_sizes; ++i) {
        if (pixel_size == face_detect_pixel_list[i]) {
            final_size = face_detect_pixel_list[i];
            return face_detect_model_list[i];
        }
    }

    // Find the closest match
    int32_t closest_size = face_detect_pixel_list[0];
    std::string closest_scheme = face_detect_model_list[0];
    int32_t min_diff = std::abs(pixel_size - face_detect_pixel_list[0]);

    for (int i = 1; i < num_sizes; ++i) {
        int32_t diff = std::abs(pixel_size - face_detect_pixel_list[i]);
        if (diff < min_diff) {
            min_diff = diff;
            closest_size = face_detect_pixel_list[i];
            closest_scheme = face_detect_model_list[i];
        }
    }

    INSPIRE_LOGW(
      "Input pixel size %d is not supported. Choosing the closest scheme: %s closest_scheme for "
      "size %d.",
      pixel_size, closest_scheme.c_str(), closest_size);
    final_size = closest_size;

    return closest_scheme;
}

bool FaceTrackModule::IsDetectModeLandmark() const {
    return m_detect_mode_landmark_;
}

void FaceTrackModule::SetTrackModeSmoothRatio(float value) {
    m_track_mode_smooth_ratio_ = value;
}

void FaceTrackModule::SetTrackModeNumSmoothCacheFrame(int value) {
    m_track_mode_num_smooth_cache_frame_ = value;
}

void FaceTrackModule::SetTrackModeDetectInterval(int value) {
    detection_interval_ = value;
}

void FaceTrackModule::SetMultiscaleLandmarkLoop(int value) {
    m_multiscale_landmark_loop_num_ = value;
    m_multiscale_landmark_scales_ = GenerateCropScales(m_landmark_crop_ratio_, m_multiscale_landmark_loop_num_);
}

void FaceTrackModule::SetTrackLostRecoveryMode(bool value) {
    m_track_lost_recovery_mode_ = value;
}

void FaceTrackModule::SetLightTrackConfidenceThreshold(float value) {
    m_light_track_confidence_threshold_ = value;
}

void FaceTrackModule::ClearTrackingFace() {
    trackingFace.clear();
    candidate_faces_.clear();
}

int32_t FaceTrackModule::GetDebugPreviewImageSize() const {
    return m_debug_preview_image_size_;
}

}  // namespace inspire
