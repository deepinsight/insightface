#ifndef INSPIRE_FACE_FACE_INFO_INTERNAL_H
#define INSPIRE_FACE_FACE_INFO_INTERNAL_H

#include <memory>
#include <utility>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <inspirecv/inspirecv.h>
#include "middleware/utils.h"
#include "data_type.h"
#include "face_process.h"
#include "face_action_data.h"
#include "track_module/quality/face_pose_quality_adapt.h"
#include "track_module/landmark/landmark_param.h"

namespace inspire {

enum ISF_TRACK_STATE { ISF_UNTRACKING = -1, ISF_DETECT = 0, ISF_READY = 1, ISF_TRACKING = 2 };

class INSPIRE_API FaceObjectInternal {
public:
    FaceObjectInternal(int instance_id, inspirecv::Rect2i bbox, int num_landmark = 106) {
        const int safe_landmark_count = std::max(0, num_landmark);
        face_id_ = instance_id;
        landmark_.resize(safe_landmark_count);
        bbox_ = std::move(bbox);
        tracking_state_ = ISF_DETECT;
        confidence_ = 1.0;
        tracking_count_ = 0;
        pose_euler_angle_.resize(3, 0.0f);
        keyPointFive.resize(5);
        face_action_ = std::make_shared<FaceActionPredictor>(10);
        num_of_dense_landmark_ = safe_landmark_count;
        is_standard_ = false;
    }

    bool SetLandmark(const std::vector<inspirecv::Point2f> &lmk, bool update_rect = true, bool update_matrix = true, float h = 0.06f, int n = 5,
                     int num_of_lmk = 106 * 2) {
        if (num_of_lmk <= 0 || (num_of_lmk & 1) != 0 || n <= 0 || !std::isfinite(h) || h < 0.0f) {
            return false;
        }
        const size_t point_count = static_cast<size_t>(num_of_lmk / 2);
        if (point_count > lmk.size() || point_count > landmark_.size()) {
            return false;
        }
        for (const auto &frame : landmark_smooth_aux_) {
            if (frame.size() != point_count) {
                return false;
            }
        }

        std::copy_n(lmk.begin(), point_count, landmark_.begin());
        if (!DynamicSmoothParamUpdate(landmark_, landmark_smooth_aux_, num_of_lmk, h, n)) {
            return false;
        }

        if (update_rect) {
            if (point_count == lmk.size()) {
                bbox_ = inspirecv::MinBoundingRect(lmk).As<int>();
            } else {
                const std::vector<inspirecv::Point2f> active_landmarks(lmk.begin(), lmk.begin() + point_count);
                bbox_ = inspirecv::MinBoundingRect(active_landmarks).As<int>();
            }
        }

        if (point_count > 105 && keyPointFive.size() >= 5) {
            keyPointFive[0] = landmark_[55];
            keyPointFive[1] = landmark_[105];
            keyPointFive[2] = landmark_[69];
            keyPointFive[3] = landmark_[45];
            keyPointFive[4] = landmark_[50];
        }

        if (update_matrix && tracking_state_ == ISF_TRACKING) {
            // pass
        }
        return true;
    }

    bool setAlignMeanSquareError(const std::vector<inspirecv::Point2f> &lmk_5) {
        if (lmk_5.size() < 5) {
            return false;
        }
        float src_pts[] = {30.2946, 51.6963, 65.5318, 51.5014, 48.0252, 71.7366, 33.5493, 92.3655, 62.7299, 92.2041};
        for (int i = 0; i < 5; i++) {
            *(src_pts + 2 * i) += 8.0;
        }
        float sum = 0;
        for (int i = 0; i < 5; i++) {
            float l2 = L2norm(src_pts[i * 2 + 0], src_pts[i * 2 + 1], lmk_5[i].GetX(), lmk_5[i].GetY());
            sum += l2;
        }

        align_mse_ = sum / 5.0f;
        return true;
    }

    // Increment tracking count
    void IncrementTrackingCount() {
        tracking_count_++;
    }

    // Get tracking count
    int GetTrackingCount() const {
        return tracking_count_;
    }

    float GetAlignMSE() const {
        return align_mse_;
    }

    std::vector<inspirecv::Point2f> GetLanmdark() const {
        return landmark_;
    }

    inspirecv::Rect2i GetRect() const {
        return bbox_;
    }

    inspirecv::Rect2i GetRectSquare(float padding_ratio = 0.0) const {
        const int width = bbox_.GetWidth();
        const int height = bbox_.GetHeight();
        if (width <= 0 || height <= 0 || !std::isfinite(padding_ratio) || padding_ratio <= -1.0f) {
            return {};
        }

        const int64_t radius = static_cast<int64_t>(std::max(width, height)) / 2;
        const double padded_radius_value = static_cast<double>(radius) * (1.0 + static_cast<double>(padding_ratio));
        if (radius <= 0 || !std::isfinite(padded_radius_value) || padded_radius_value < 1.0 ||
            padded_radius_value > static_cast<double>(std::numeric_limits<int>::max()) / 2.0) {
            return {};
        }

        const int64_t padded_radius = static_cast<int64_t>(padded_radius_value);
        const int64_t center_x = static_cast<int64_t>(bbox_.GetX()) + width / 2;
        const int64_t center_y = static_cast<int64_t>(bbox_.GetY()) + height / 2;
        const int64_t x1 = center_x - padded_radius;
        const int64_t y1 = center_y - padded_radius;
        const int64_t side = padded_radius * 2;
        if (x1 < std::numeric_limits<int>::min() || x1 > std::numeric_limits<int>::max() ||
            y1 < std::numeric_limits<int>::min() || y1 > std::numeric_limits<int>::max() ||
            side > std::numeric_limits<int>::max()) {
            return {};
        }
        return inspirecv::Rect2i(static_cast<int>(x1), static_cast<int>(y1), static_cast<int>(side), static_cast<int>(side));
    }

    FaceActionList UpdateFaceAction(const SemanticIndex& semantic_index) {
        inspirecv::Vec3f euler{high_result.pitch, high_result.yaw, high_result.roll};
        inspirecv::Vec2f eyes{left_eye_status_.empty() ? 0.0f : left_eye_status_.back(),
                              right_eye_status_.empty() ? 0.0f : right_eye_status_.back()};
        face_action_->RecordActionFrame(landmark_, euler, eyes);
        return face_action_->AnalysisFaceAction(semantic_index);
    }

    void DisableTracking() {
        tracking_state_ = ISF_UNTRACKING;
    }

    void EnableTracking() {
        tracking_state_ = ISF_TRACKING;
    }

    void ReadyTracking() {
        tracking_state_ = ISF_READY;
    }

    ISF_TRACK_STATE TrackingState() const {
        return tracking_state_;
    }

    float GetConfidence() const {
        return confidence_;
    }

    void SetConfidence(float confidence) {
        confidence_ = confidence;
    }

    int GetTrackingId() const {
        return face_id_;
    }

    const inspirecv::TransformMatrix &getTransMatrix() const {
        return trans_matrix_;
    }

    const inspirecv::TransformMatrix &getTransMatrixExtensive() const {
        return trans_matrix_extensive_;
    }

    void setTransMatrix(const inspirecv::TransformMatrix &transMatrix) {
        trans_matrix_ = transMatrix.Clone();
    }

    void setTransMatrixExtensive(const inspirecv::TransformMatrix &transMatrixExtensive) {
        trans_matrix_extensive_ = transMatrixExtensive.Clone();
    }

    static float L2norm(float x0, float y0, float x1, float y1) {
        return sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1));
    }

    bool DynamicSmoothParamUpdate(std::vector<inspirecv::Point2f> &landmarks,
                                  std::vector<std::vector<inspirecv::Point2f>> &landmarks_lastNframes, int lm_length, float h = 0.06f,
                                  int n = 5) {
        if (lm_length <= 0 || (lm_length & 1) != 0 || n <= 0 || !std::isfinite(h) || h < 0.0f) {
            return false;
        }
        const size_t point_count = static_cast<size_t>(lm_length / 2);
        if (point_count > landmarks.size()) {
            return false;
        }
        for (const auto &frame : landmarks_lastNframes) {
            if (frame.size() != point_count) {
                return false;
            }
        }
        if (landmarks_lastNframes.size() > static_cast<size_t>(n)) {
            landmarks_lastNframes.erase(landmarks_lastNframes.begin(),
                                        landmarks_lastNframes.end() - static_cast<std::ptrdiff_t>(n));
        }
        if (landmarks_lastNframes.size() == static_cast<size_t>(n)) {
            for (size_t i = 0; i < point_count; i++) {
                const float current_x = landmarks[i].GetX();
                const float current_y = landmarks[i].GetY();
                float sum_d = 1;
                float max_d = 0;
                for (int j = 0; j < n; j++) {
                    float d = L2norm(current_x, current_y, landmarks_lastNframes[j][i].GetX(), landmarks_lastNframes[j][i].GetY());
                    if (d > max_d)
                        max_d = d;
                }
                for (int j = 0; j < n; j++) {
                    float d = exp(-max_d * (n - j) * h);
                    sum_d += d;
                    landmarks[i].SetX(landmarks[i].GetX() + d * landmarks_lastNframes[j][i].GetX());
                    landmarks[i].SetY(landmarks[i].GetY() + d * landmarks_lastNframes[j][i].GetY());
                }
                landmarks[i].SetX(landmarks[i].GetX() / sum_d);
                landmarks[i].SetY(landmarks[i].GetY() / sum_d);
            }
        }
        std::vector<inspirecv::Point2f> landmarks_frame;
        landmarks_frame.reserve(point_count);
        for (size_t i = 0; i < point_count; i++) {
            landmarks_frame.emplace_back(landmarks[i].GetX(), landmarks[i].GetY());
        }
        landmarks_lastNframes.push_back(std::move(landmarks_frame));
        if (landmarks_lastNframes.size() > static_cast<size_t>(n))
            landmarks_lastNframes.erase(landmarks_lastNframes.begin());
        return true;
    }

public:
    std::vector<inspirecv::Point2f> landmark_;
    std::vector<std::vector<inspirecv::Point2f>> landmark_smooth_aux_;
    inspirecv::Rect2i bbox_;
    inspirecv::Vec3f euler_angle_;
    std::vector<float> pose_euler_angle_;

    int num_of_dense_landmark_;

    float align_mse_{};

    const inspirecv::Vec3f &getEulerAngle() const {
        return euler_angle_;
    }

    const std::vector<float> &getPoseEulerAngle() const {
        return pose_euler_angle_;
    }

    bool setPoseEulerAngle(const std::vector<float> &poseEulerAngle) {
        if (poseEulerAngle.size() < 3 || !std::isfinite(poseEulerAngle[0]) || !std::isfinite(poseEulerAngle[1]) ||
            !std::isfinite(poseEulerAngle[2])) {
            return false;
        }
        pose_euler_angle_[0] = poseEulerAngle[0];
        pose_euler_angle_[1] = poseEulerAngle[1];
        pose_euler_angle_[2] = poseEulerAngle[2];

        is_standard_ = std::fabs(pose_euler_angle_[0]) < 0.5f && std::fabs(pose_euler_angle_[1]) < 0.48f;
        return true;
    }

    bool isStandard() const {
        return is_standard_;
    }

    const inspirecv::Rect2i &getBbox() const {
        return bbox_;
    }

    void setBbox(const inspirecv::Rect2i &bbox) {
        bbox_ = bbox;
    }

    std::vector<std::vector<float>> face_emotion_history_;
    
    inspirecv::TransformMatrix trans_matrix_;
    inspirecv::TransformMatrix trans_matrix_extensive_;
    float confidence_;
    inspirecv::Rect2i detect_bbox_;
    int tracking_count_;  // Tracking count

    bool is_standard_;

    FacePoseQualityAdaptResult high_result;

    FaceProcess faceProcess;

    std::vector<inspirecv::Point2f> keyPointFive;

    void setId(int id) {
        face_id_ = id;
    }

    std::vector<float> left_eye_status_;

    std::vector<float> right_eye_status_;

private:
    ISF_TRACK_STATE tracking_state_;
    std::shared_ptr<FaceActionPredictor> face_action_;
    int face_id_;
};

typedef std::vector<FaceObjectInternal> FaceObjectInternalList;

}  // namespace inspire

#endif  // INSPIRE_FACE_FACE_INFO_INTERNAL_H
