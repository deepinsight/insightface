/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "face_track_module.h"
#include "log.h"
#include "landmark/mean_shape.h"
#include <algorithm>
#include <cstddef>
#include "middleware/costman.h"
#include "middleware/model_archive/inspire_archive.h"
#include "middleware/utils.h"
#include "herror.h"
#include "middleware/costman.h"
#include "cost_time.h"
#include <inspirecv/time_spend.h>

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
        m_detect_mode_landmark_ = detect_mode_landmark;
    }
    if (m_mode_ == DETECT_MODE_TRACK_BY_DETECT) {
        m_TbD_tracker_ = std::make_shared<BYTETracker>(TbD_mode_fps, 30);
    }
}

void FaceTrackModule::SparseLandmarkPredict(const inspirecv::Image &raw_face_crop, std::vector<inspirecv::Point2f> &landmarks_output, float &score,
                                            float size) {
    COST_TIME_SIMPLE(SparseLandmarkPredict);
    landmarks_output.resize(FaceLandmarkAdapt::NUM_OF_LANDMARK);
    std::vector<float> lmk_out = (*m_landmark_predictor_)(raw_face_crop);
    for (int i = 0; i < FaceLandmarkAdapt::NUM_OF_LANDMARK; ++i) {
        float x = lmk_out[i * 2 + 0] * size;
        float y = lmk_out[i * 2 + 1] * size;
        landmarks_output[i] = inspirecv::Point<float>(x, y);
    }
    score = (*m_refine_net_)(raw_face_crop);
}

bool FaceTrackModule::TrackFace(inspirecv::InspireImageProcess &image, FaceObjectInternal &face) {
    COST_TIME_SIMPLE(TrackFace);
    // If the face confidence level is below 0.1, disable tracking
    if (face.GetConfidence() < 0.1) {
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
        std::vector<inspirecv::Point2f> dst_pts = {{0, 0}, {112, 0}, {112, 112}, {0, 112}};
        affine = inspirecv::SimilarityTransformEstimate(camera_pts, dst_pts);
        face.setTransMatrix(affine);

        std::vector<inspirecv::Point2f> dst_pts_extensive = {{0, 0},
                                                             {(float)m_crop_extensive_size_, 0},
                                                             {(float)m_crop_extensive_size_, (float)m_crop_extensive_size_},
                                                             {0, (float)m_crop_extensive_size_}};
        // Add extensive rect
        inspirecv::Rect2i extensive_rect = rect_square.Square(m_crop_extensive_ratio_);
        auto extensive_rect_pts = extensive_rect.As<float>().ToFourVertices();
        std::vector<inspirecv::Point2f> camera_pts_extensive = ApplyTransformToPoints(extensive_rect_pts, rotation_mode_affine);
        inspirecv::TransformMatrix extensive_affine = inspirecv::SimilarityTransformEstimate(camera_pts_extensive, dst_pts);
        face.setTransMatrixExtensive(extensive_affine);

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
        auto affine_extensive = face.getTransMatrixExtensive();
        auto pre_crop = image.ExecuteImageAffineProcessing(affine_extensive, m_crop_extensive_size_, m_crop_extensive_size_);
        auto res = (*m_face_quality_)(pre_crop);

        auto affine_extensive_inv = affine_extensive.GetInverse();
        std::vector<inspirecv::Point2f> lmk_extensive = ApplyTransformToPoints(res.lmk, affine_extensive_inv);
        res.lmk = lmk_extensive;
        face.high_result = res;
    }

    if (m_detect_mode_landmark_) {
        // If Landmark need to be extracted in detection mode,
        // Landmark must be detected when fast tracing is enabled
        affine = face.getTransMatrix();
        inspirecv::Image crop;
        // Get the RGB image after affine transformation
        crop = image.ExecuteImageAffineProcessing(affine, 112, 112);
        inspirecv::TransformMatrix affine_inv = affine.GetInverse();

        std::vector<inspirecv::Point2f> landmark_rawout;
        std::vector<float> bbox;

        Timer lmk_cost_time;
        // Predicted sparse key point
        SparseLandmarkPredict(crop, landmark_rawout, score, 112);
        // Extract 5 key points
        std::vector<inspirecv::Point2f> lmk_5 = {
          landmark_rawout[FaceLandmarkAdapt::LEFT_EYE_CENTER], landmark_rawout[FaceLandmarkAdapt::RIGHT_EYE_CENTER],
          landmark_rawout[FaceLandmarkAdapt::NOSE_CORNER], landmark_rawout[FaceLandmarkAdapt::MOUTH_LEFT_CORNER],
          landmark_rawout[FaceLandmarkAdapt::MOUTH_RIGHT_CORNER]};
        face.setAlignMeanSquareError(lmk_5);

        // Convert key points back to the original coordinate system
        landmark_back.resize(landmark_rawout.size());
        landmark_back = inspirecv::ApplyTransformToPoints(landmark_rawout, affine_inv);
        int MODE = 1;

        if (MODE > 0) {
            if (face.TrackingState() == ISF_DETECT) {
                face.ReadyTracking();
            } else if (face.TrackingState() == ISF_READY || face.TrackingState() == ISF_TRACKING) {
                COST_TIME_SIMPLE(LandmarkBack);
                inspirecv::TransformMatrix trans_m;
                inspirecv::TransformMatrix tmp = face.getTransMatrix();
                std::vector<inspirecv::Point2f> inside_points = landmark_rawout;

                std::vector<inspirecv::Point2f> mean_shape_(FaceLandmarkAdapt::NUM_OF_LANDMARK);
                for (int k = 0; k < FaceLandmarkAdapt::NUM_OF_LANDMARK; k++) {
                    mean_shape_[k].SetX(mean_shape[k * 2]);
                    mean_shape_[k].SetY(mean_shape[k * 2 + 1]);
                }

                auto _affine = inspirecv::SimilarityTransformEstimate(inside_points, mean_shape_);
                auto mid_inside_points = ApplyTransformToPoints(inside_points, _affine);
                inside_points = FixPointsMeanshape(mid_inside_points, mean_shape_);

                trans_m = inspirecv::SimilarityTransformEstimate(landmark_back, inside_points);
                face.setTransMatrix(trans_m);
                face.EnableTracking();

                Timer extensive_cost_time;
                // Add extensive rect
                // Calculate center point of landmarks
                inspirecv::Point2f center(0.0f, 0.0f);
                for (const auto &pt : landmark_back) {
                    center.SetX(center.GetX() + pt.GetX());
                    center.SetY(center.GetY() + pt.GetY());
                }
                center.SetX(center.GetX() / landmark_back.size());
                center.SetY(center.GetY() / landmark_back.size());

                // Create expanded points by scaling from center by 1.3
                std::vector<inspirecv::Point2f> lmk_back_rect = landmark_back;
                for (auto &pt : lmk_back_rect) {
                    pt.SetX(center.GetX() + (pt.GetX() - center.GetX()) * m_crop_extensive_ratio_);
                    pt.SetY(center.GetY() + (pt.GetY() - center.GetY()) * m_crop_extensive_ratio_);
                }
                inspirecv::TransformMatrix extensive_affine = inspirecv::SimilarityTransformEstimate(lmk_back_rect, mid_inside_points);
                face.setTransMatrixExtensive(extensive_affine);
                // INSPIRE_LOGD("Extensive Affine Cost %f", extensive_cost_time.GetCostTimeUpdate());
            }
        }
        // Add five key points to landmark_back
        for (int i = 0; i < 5; i++) {
            landmark_back.push_back(face.high_result.lmk[i]);
        }
        // Update face key points
        face.SetLandmark(landmark_back, true, true, m_track_mode_smooth_ratio_, m_track_mode_num_smooth_cache_frame_,
                         (FaceLandmarkAdapt::NUM_OF_LANDMARK + 10) * 2);
        // Get the smoothed landmark
        auto &landmark_smooth = face.landmark_smooth_aux_.back();
        // Update the face key points
        face.high_result.lmk[0] = landmark_smooth[FaceLandmarkAdapt::NUM_OF_LANDMARK + 0];
        face.high_result.lmk[1] = landmark_smooth[FaceLandmarkAdapt::NUM_OF_LANDMARK + 1];
        face.high_result.lmk[2] = landmark_smooth[FaceLandmarkAdapt::NUM_OF_LANDMARK + 2];
        face.high_result.lmk[3] = landmark_smooth[FaceLandmarkAdapt::NUM_OF_LANDMARK + 3];
        face.high_result.lmk[4] = landmark_smooth[FaceLandmarkAdapt::NUM_OF_LANDMARK + 4];
    }

    // If tracking status, update the confidence level
    if (face.TrackingState() == ISF_TRACKING) {
        face.SetConfidence(score);
    }

    return true;
}

void FaceTrackModule::UpdateStream(inspirecv::InspireImageProcess &image) {
    inspirecv::TimeSpend total("UpdateStream");
    total.Start();
    COST_TIME_SIMPLE(FaceTrackUpdateStream);
    detection_index_ += 1;
    if (m_mode_ == DETECT_MODE_ALWAYS_DETECT || m_mode_ == DETECT_MODE_TRACK_BY_DETECT)
        trackingFace.clear();
    if (trackingFace.empty() || detection_index_ % detection_interval_ == 0 || m_mode_ == DETECT_MODE_ALWAYS_DETECT ||
        m_mode_ == DETECT_MODE_TRACK_BY_DETECT) {
        image.SetPreviewSize(track_preview_size_);
        inspirecv::Image image_detect = image.ExecutePreviewImageProcessing(true);

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
    }

    if (!candidate_faces_.empty()) {
        for (int i = 0; i < candidate_faces_.size(); i++) {
            trackingFace.push_back(candidate_faces_[i]);
        }
        candidate_faces_.clear();
    }

    for (std::vector<FaceObjectInternal>::iterator iter = trackingFace.begin(); iter != trackingFace.end();) {
        if (!TrackFace(image, *iter)) {
            iter = trackingFace.erase(iter);
        } else {
            iter++;
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

            FaceObjectInternal faceinfo(tracking_idx_, bbox[i], FaceLandmarkAdapt::NUM_OF_LANDMARK + 10);
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

int FaceTrackModule::Configuration(inspire::InspireArchive &archive, const std::string &expansion_path) {
    // Initialize the detection model
    m_expansion_path_ = std::move(expansion_path);
    InspireModel detModel;
    auto scheme = ChoiceMultiLevelDetectModel(m_dynamic_detection_input_level_);
    auto ret = archive.LoadModel(scheme, detModel);
    if (ret != SARC_SUCCESS) {
        INSPIRE_LOGE("Load %s error: %d", scheme.c_str(), ret);
        return HERR_ARCHIVE_LOAD_MODEL_FAILURE;
    }
    InitDetectModel(detModel);

    // Initialize the landmark model
    InspireModel lmkModel;
    ret = archive.LoadModel("landmark", lmkModel);
    if (ret != SARC_SUCCESS) {
        INSPIRE_LOGE("Load %s error: %d", "landmark", ret);
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

    // Initialize the pose quality model
    InspireModel pquModel;
    ret = archive.LoadModel("pose_quality", pquModel);
    if (ret != SARC_SUCCESS) {
        INSPIRE_LOGE("Load %s error: %d", "pose_quality", ret);
        return HERR_ARCHIVE_LOAD_MODEL_FAILURE;
    }
    InitFacePoseModel(pquModel);

    return 0;
}

int FaceTrackModule::InitLandmarkModel(InspireModel &model) {
    m_landmark_predictor_ = std::make_shared<FaceLandmarkAdapt>(112);
    auto ret = m_landmark_predictor_->loadData(model, model.modelType);
    if (ret != InferenceWrapper::WrapperOk) {
        return HERR_ARCHIVE_LOAD_FAILURE;
    }
    return HSUCCEED;
}

int FaceTrackModule::InitDetectModel(InspireModel &model) {
    std::vector<int> input_size;
    input_size = model.Config().get<std::vector<int>>("input_size");

    m_face_detector_ = std::make_shared<FaceDetectAdapt>(input_size[0]);
    auto ret = m_face_detector_->loadData(model, model.modelType, false);
    if (ret != InferenceWrapper::WrapperOk) {
        return HERR_ARCHIVE_LOAD_FAILURE;
    }
    return HSUCCEED;
}

int FaceTrackModule::InitRNetModel(InspireModel &model) {
    m_refine_net_ = std::make_shared<RNetAdapt>();
    auto ret = m_refine_net_->loadData(model, model.modelType);
    if (ret != InferenceWrapper::WrapperOk) {
        return HERR_ARCHIVE_LOAD_FAILURE;
    }
    return HSUCCEED;
}

int FaceTrackModule::InitFacePoseModel(InspireModel &model) {
    m_face_quality_ = std::make_shared<FacePoseQualityAdapt>();
    auto ret = m_face_quality_->loadData(model, model.modelType);
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
}

std::string FaceTrackModule::ChoiceMultiLevelDetectModel(const int32_t pixel_size) {
    const int32_t supported_sizes[] = {160, 320, 640};
    const std::string scheme_names[] = {"face_detect_160", "face_detect_320", "face_detect_640"};
    const int32_t num_sizes = sizeof(supported_sizes) / sizeof(supported_sizes[0]);

    if (pixel_size == -1) {
        return scheme_names[1];
    }

    // Check for exact match
    for (int i = 0; i < num_sizes; ++i) {
        if (pixel_size == supported_sizes[i]) {
            return scheme_names[i];
        }
    }

    // Find the closest match
    int32_t closest_size = supported_sizes[0];
    std::string closest_scheme = scheme_names[0];
    int32_t min_diff = std::abs(pixel_size - supported_sizes[0]);

    for (int i = 1; i < num_sizes; ++i) {
        int32_t diff = std::abs(pixel_size - supported_sizes[i]);
        if (diff < min_diff) {
            min_diff = diff;
            closest_size = supported_sizes[i];
            closest_scheme = scheme_names[i];
        }
    }

    INSPIRE_LOGW(
      "Input pixel size %d is not supported. Choosing the closest scheme: %s closest_scheme for "
      "size %d.",
      pixel_size, closest_scheme.c_str(), closest_size);

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

}  // namespace inspire
