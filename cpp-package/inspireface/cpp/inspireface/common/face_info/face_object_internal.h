#ifndef INSPIRE_FACE_FACE_INFO_INTERNAL_H
#define INSPIRE_FACE_FACE_INFO_INTERNAL_H

#include <memory>
#include <utility>
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
        face_id_ = instance_id;
        landmark_.resize(num_landmark);
        bbox_ = std::move(bbox);
        tracking_state_ = ISF_DETECT;
        confidence_ = 1.0;
        tracking_count_ = 0;
        pose_euler_angle_.resize(3);
        keyPointFive.resize(5);
        face_action_ = std::make_shared<FaceActionPredictor>(10);
        num_of_dense_landmark_ = num_landmark;
    }

    void SetLandmark(const std::vector<inspirecv::Point2f> &lmk, bool update_rect = true, bool update_matrix = true, float h = 0.06f, int n = 5,
                     int num_of_lmk = 106 * 2) {
        // if (lmk.size() != landmark_.size()) {
        //     INSPIRE_LOGW("The SetLandmark function displays an exception indicating that the lmk number does not match");
        //     return;
        // }
        std::copy(lmk.begin(), lmk.end(), landmark_.begin());
        DynamicSmoothParamUpdate(landmark_, landmark_smooth_aux_, num_of_lmk, h, n);
        // std::cout << "smooth ratio: " << h << " num smooth cache frame: " << n << std::endl;

        // cv::Vec3d euler_angle;
        // EstimateHeadPose(landmark_, euler_angle_);
        // DynamicSmoothParamUpdate(landmark_, landmark_smooth_aux_, 106 * 2, 0.06);
        if (update_rect)
            bbox_ = inspirecv::MinBoundingRect(lmk).As<int>();
        if (update_matrix && tracking_state_ == ISF_TRACKING) {
            // pass
        }

        keyPointFive[0] = landmark_[55];
        keyPointFive[1] = landmark_[105];
        keyPointFive[2] = landmark_[69];
        keyPointFive[3] = landmark_[45];
        keyPointFive[4] = landmark_[50];
    }

    void setAlignMeanSquareError(const std::vector<inspirecv::Point2f> &lmk_5) {
        float src_pts[] = {30.2946, 51.6963, 65.5318, 51.5014, 48.0252, 71.7366, 33.5493, 92.3655, 62.7299, 92.2041};
        for (int i = 0; i < 5; i++) {
            *(src_pts + 2 * i) += 8.0;
        }
        float sum = 0;
        for (int i = 0; i < lmk_5.size(); i++) {
            float l2 = L2norm(src_pts[i * 2 + 0], src_pts[i * 2 + 1], lmk_5[i].GetX(), lmk_5[i].GetY());
            sum += l2;
        }

        align_mse_ = sum / 5.0f;
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
        int cx = bbox_.GetX() + bbox_.GetWidth() / 2;
        int cy = bbox_.GetY() + bbox_.GetHeight() / 2;
        int R = std::max(bbox_.GetWidth(), bbox_.GetHeight()) / 2;
        int R_padding = static_cast<int>(R * (1 + padding_ratio));
        int x1 = cx - R_padding;
        int y1 = cy - R_padding;
        int x2 = cx + R_padding;
        int y2 = cy + R_padding;
        int width = x2 - x1;
        int height = y2 - y1;
        assert(width > 0);
        assert(height > 0);
        assert(height == width);
        inspirecv::Rect2i box_square(x1, y1, width, height);
        return box_square;
    }

    FaceActionList UpdateFaceAction(const SemanticIndex& semantic_index) {
        inspirecv::Vec3f euler{high_result.pitch, high_result.yaw, high_result.roll};
        inspirecv::Vec2f eyes{left_eye_status_.back(), right_eye_status_.back()};
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

    void DynamicSmoothParamUpdate(std::vector<inspirecv::Point2f> &landmarks, std::vector<std::vector<inspirecv::Point2f>> &landmarks_lastNframes,
                                  int lm_length, float h = 0.06f, int n = 5) {
        std::vector<inspirecv::Point2f> landmarks_temp;
        landmarks_temp.assign(landmarks.begin(), landmarks.end());
        if (landmarks_lastNframes.size() == n) {
            for (int i = 0; i < lm_length / 2; i++) {
                float sum_d = 1;
                float max_d = 0;
                for (int j = 0; j < n; j++) {
                    float d = L2norm(landmarks_temp[i].GetX(), landmarks_temp[i].GetY(), landmarks_lastNframes[j][i].GetX(),
                                     landmarks_lastNframes[j][i].GetY());
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
        for (int i = 0; i < lm_length / 2; i++) {
            landmarks_frame.push_back(inspirecv::Point2f(landmarks[i].GetX(), landmarks[i].GetY()));
        }
        landmarks_lastNframes.push_back(landmarks_frame);
        if (landmarks_lastNframes.size() > n)
            landmarks_lastNframes.erase(landmarks_lastNframes.begin());
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

    void setPoseEulerAngle(const std::vector<float> &poseEulerAngle) {
        pose_euler_angle_[0] = poseEulerAngle[0];
        pose_euler_angle_[1] = poseEulerAngle[1];
        pose_euler_angle_[2] = poseEulerAngle[2];

        if (abs(pose_euler_angle_[0]) < 0.5 && abs(pose_euler_angle_[1]) < 0.48) {
            is_standard_ = true;
        }
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
