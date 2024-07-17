#ifndef FACE_INFO_H
#define FACE_INFO_H

#include <memory>
#include <utility>

//#include "face_action.h"
#include "opencv2/opencv.hpp"
#include "middleware/utils.h"
#include "data_type.h"
#include "face_process.h"
#include "track_module/quality/face_pose_quality.h"
#include "face_action.h"

namespace inspire {

enum TRACK_STATE {
    UNTRACKING = -1, DETECT = 0, READY = 1, TRACKING = 2
};

class INSPIRE_API FaceObject {
public:
    FaceObject(int instance_id, cv::Rect bbox, int num_landmark = 106) {
        face_id_ = instance_id;
        landmark_.resize(num_landmark);
        bbox_ = std::move(bbox);
        tracking_state_ = DETECT;
        confidence_ = 1.0;
        tracking_count_ = 0;
        pose_euler_angle_.resize(3);
        keyPointFive.resize(5);
        face_action_ = std::make_shared<FaceActionAnalyse>(10);
    }

    void UpdateMatrix(const cv::Mat &matrix) {
        assert(trans_matrix_.rows == 2 && trans_matrix_.cols == 3);
        double a00 = matrix.at<double>(0, 0);
        double a01 = matrix.at<double>(0, 1);
        double a10 = matrix.at<double>(1, 0);
        double a11 = matrix.at<double>(1, 1);
        double t1x = matrix.at<double>(0, 2);
        double t1y = matrix.at<double>(1, 2);

        double m00 = trans_matrix_.at<double>(0, 0);
        double m01 = trans_matrix_.at<double>(0, 1);
        double m10 = trans_matrix_.at<double>(1, 0);
        double m11 = trans_matrix_.at<double>(1, 1);
        double t0x = trans_matrix_.at<double>(0, 2);
        double t0y = trans_matrix_.at<double>(1, 2);

        double n_m00 = a00 * m00 + a01 * m10;
        double n_m01 = a00 * m01 + a01 * m11;
        double n_m02 = a00 * t0x + a01 * t0y + t1x;
        double n_m10 = a10 * m00 + a11 * m10;
        double n_m11 = a10 * m01 + a11 * m11;
        double n_m12 = a10 * t0x + a11 * t0y + t1y;

        trans_matrix_.at<double>(0, 0) = n_m00;
        trans_matrix_.at<double>(0, 1) = n_m01;
        trans_matrix_.at<double>(0, 2) = n_m02;
        trans_matrix_.at<double>(1, 0) = n_m10;
        trans_matrix_.at<double>(1, 1) = n_m11;
        trans_matrix_.at<double>(1, 2) = n_m12;
    }

    void SetLandmark(const std::vector<cv::Point2f> &lmk, bool update_rect = true,
                     bool update_matrix = true) {
        if (lmk.size() != landmark_.size()) {
            INSPIRE_LOGW("The SetLandmark function displays an exception indicating that the lmk number does not match");
            return;
        }
        std::copy(lmk.begin(), lmk.end(), landmark_.begin());
        DynamicSmoothParamUpdate(landmark_, landmark_smooth_aux_, 106 * 2, 0.06);

        // cv::Vec3d euler_angle;
        EstimateHeadPose(landmark_, euler_angle_);
        // DynamicSmoothParamUpdate(landmark_, landmark_smooth_aux_, 106 * 2, 0.06);
        if (update_rect)
            bbox_ = cv::boundingRect(lmk);
        if (update_matrix && tracking_state_ == TRACKING) {
            // pass
        }

        keyPointFive[0] = landmark_[55];
        keyPointFive[1] = landmark_[105];
        keyPointFive[2] = landmark_[69];
        keyPointFive[3] = landmark_[45];
        keyPointFive[4] = landmark_[50];

    }

    void setAlignMeanSquareError(const std::vector<cv::Point2f> &lmk_5) {
        float src_pts[] = {30.2946, 51.6963, 65.5318, 51.5014, 48.0252,
                           71.7366, 33.5493, 92.3655, 62.7299, 92.2041};
        for (int i = 0; i < 5; i++) {
            *(src_pts + 2 * i) += 8.0;
        }
        float sum = 0;
        for (int i = 0; i < lmk_5.size(); i++) {
            float l2 = L2norm(src_pts[i * 2 + 0], src_pts[i * 2 + 1], lmk_5[i].x, lmk_5[i].y);
            sum += l2;
        }

        align_mse_ = sum / 5.0f;
    }

    // 增加跟踪次数
    void IncrementTrackingCount() {
        tracking_count_++;
    }

    // 获取跟踪次数
    int GetTrackingCount() const {
        return tracking_count_;
    }

    float GetAlignMSE() const { return align_mse_; }

    std::vector<cv::Point2f> GetLanmdark() const { return landmark_; }

    cv::Rect GetRect() const { return bbox_; }

    cv::Rect GetRectSquare(float padding_ratio = 0.0) const {
        int cx = bbox_.x + bbox_.width / 2;
        int cy = bbox_.y + bbox_.height / 2;
        int R = std::max(bbox_.width, bbox_.height) / 2;
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
        cv::Rect box_square(x1, y1, width, height);
        return box_square;
    }

    FaceActions UpdateFaceAction() {
        cv::Vec3f euler(high_result.pitch, high_result.yaw, high_result.roll);
        cv::Vec2f eyes(left_eye_status_.back(), right_eye_status_.back());
        face_action_->RecordActionFrame(landmark_, euler, eyes);
        return face_action_->AnalysisFaceAction();
    }

    void DisableTracking() { tracking_state_ = UNTRACKING; }

    void EnableTracking() { tracking_state_ = TRACKING; }

    void ReadyTracking() { tracking_state_ = READY; }

    TRACK_STATE TrackingState() const { return tracking_state_; }

    float GetConfidence() const { return confidence_; }

    void SetConfidence(float confidence) { confidence_ = confidence; }

    int GetTrackingId() const { return face_id_; }

    const cv::Mat &getTransMatrix() const { return trans_matrix_; }

    void setTransMatrix(const cv::Mat &transMatrix) {
        transMatrix.copyTo(trans_matrix_);
    }

    static float L2norm(float x0, float y0, float x1, float y1) {
        return sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1));
    }

    void RequestFaceAction(
            std::vector<cv::Point2f> &landmarks,
            std::vector<std::vector<cv::Point2f>> &landmarks_lastNframes,
            int lm_length, float h) {
        int n = 5;
        std::vector<cv::Point2f> landmarks_temp;
        landmarks_temp.assign(landmarks.begin(), landmarks.end());
        if (landmarks_lastNframes.size() == n) {
            for (int i = 0; i < lm_length / 2; i++) {
                float sum_d = 1;
                float max_d = 0;
                for (int j = 0; j < n; j++) {
                    float d = L2norm(landmarks_temp[i].x, landmarks_temp[i].y,
                                     landmarks_lastNframes[j][i].x,
                                     landmarks_lastNframes[j][i].y);
                    if (d > max_d)
                        max_d = d;
                }
                for (int j = 0; j < n; j++) {
                    float d = exp(-max_d * (n - j) * h);
                    sum_d += d;
                    landmarks[i].x = landmarks[i].x + d * landmarks_lastNframes[j][i].x;
                    landmarks[i].y = landmarks[i].y + d * landmarks_lastNframes[j][i].y;
                }
                landmarks[i].x = landmarks[i].x / sum_d;
                landmarks[i].y = landmarks[i].y / sum_d;
            }
        }
        std::vector<cv::Point2f> landmarks_frame;
        for (int i = 0; i < lm_length / 2; i++) {
            landmarks_frame.push_back(cv::Point2f(landmarks[i].x, landmarks[i].y));
        }
        landmarks_lastNframes.push_back(landmarks_frame);
        if (landmarks_lastNframes.size() > 5)
            landmarks_lastNframes.erase(landmarks_lastNframes.begin());
    }

    void DynamicSmoothParamUpdate(
            std::vector<cv::Point2f> &landmarks,
            std::vector<std::vector<cv::Point2f>> &landmarks_lastNframes,
            int lm_length, float h) {
        int n = 5;
        std::vector<cv::Point2f> landmarks_temp;
        landmarks_temp.assign(landmarks.begin(), landmarks.end());
        if (landmarks_lastNframes.size() == n) {
            for (int i = 0; i < lm_length / 2; i++) {
                float sum_d = 1;
                float max_d = 0;
                for (int j = 0; j < n; j++) {
                    float d = L2norm(landmarks_temp[i].x, landmarks_temp[i].y,
                                     landmarks_lastNframes[j][i].x,
                                     landmarks_lastNframes[j][i].y);
                    if (d > max_d)
                        max_d = d;
                }
                for (int j = 0; j < n; j++) {
                    float d = exp(-max_d * (n - j) * h);
                    sum_d += d;
                    landmarks[i].x = landmarks[i].x + d * landmarks_lastNframes[j][i].x;
                    landmarks[i].y = landmarks[i].y + d * landmarks_lastNframes[j][i].y;
                }
                landmarks[i].x = landmarks[i].x / sum_d;
                landmarks[i].y = landmarks[i].y / sum_d;
            }
        }
        std::vector<cv::Point2f> landmarks_frame;
        for (int i = 0; i < lm_length / 2; i++) {
            landmarks_frame.push_back(cv::Point2f(landmarks[i].x, landmarks[i].y));
        }
        landmarks_lastNframes.push_back(landmarks_frame);
        if (landmarks_lastNframes.size() > 5)
            landmarks_lastNframes.erase(landmarks_lastNframes.begin());
    }

public:
    std::vector<cv::Point2f> landmark_;
    std::vector<std::vector<cv::Point2f>> landmark_smooth_aux_;
    cv::Rect bbox_;
    cv::Vec3f euler_angle_;
    std::vector<float> pose_euler_angle_;

    float align_mse_{};

    const cv::Vec3f &getEulerAngle() const { return euler_angle_; }

    const std::vector<float> &getPoseEulerAngle() const { return pose_euler_angle_; }

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

    const cv::Rect &getBbox() const { return bbox_; }

    std::vector<cv::Point2f> getRotateLandmark(int height, int width, int rotate = 0) {
        if (rotate != 0) {
            std::vector<cv::Point2f> result = RotatePoints(landmark_, rotate, cv::Size(height, width));
            return result;
        } else {
            return GetLanmdark();
        }
    }

    cv::Rect getRotateBbox(int height, int width, int rotate = 0, bool use_flip = false) {
        if (rotate != 0) {
            cv::Rect src_bbox = bbox_;
            std::vector<cv::Point2f> points;
            cv::Rect trans_rect;
            RotateRect(src_bbox, points, trans_rect, rotate, cv::Size(height, width));
            if (use_flip)
                trans_rect = flipRectWidth(trans_rect, cv::Size(width, height));
            return trans_rect;
        } else {
            return getBbox();
        }
    }

    void setBbox(const cv::Rect &bbox) { bbox_ = bbox; }

    cv::Mat trans_matrix_;
    float confidence_;
    cv::Rect detect_bbox_;
    int tracking_count_; // 跟踪次数

    bool is_standard_;

    FacePoseQualityResult high_result;

    FaceProcess faceProcess;

    std::vector<Point2f> keyPointFive;

    void setId(int id) {
        face_id_ = id;
    }

    std::vector<float> left_eye_status_;
    
    std::vector<float> right_eye_status_;

private:
    TRACK_STATE tracking_state_;
    std::shared_ptr<FaceActionAnalyse> face_action_;
    int face_id_;
};

typedef std::vector<FaceObject> FaceObjectList;

}   // namespace hyper

#endif // FACE_INFO_H
