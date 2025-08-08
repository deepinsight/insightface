/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#pragma once
#ifndef INSPIRE_FACE_TRACK_MODULE_FACE_TRACK_MODULE_H
#define INSPIRE_FACE_TRACK_MODULE_FACE_TRACK_MODULE_H
#include <iostream>
#include "face_detect/face_detect_adapt.h"
#include "face_detect/rnet_adapt.h"
#include "landmark/all.h"
#include "common/face_info/face_object_internal.h"
#include "frame_process.h"
#include "quality/face_pose_quality_adapt.h"
#include "middleware/model_archive/inspire_archive.h"
#include "tracker_optional/bytetrack/BYTETracker.h"
#include <data_type.h>
#include "landmark/landmark_param.h"

namespace inspire {

/**
 * @class FaceTrack
 * @brief Class for tracking faces in video streams.
 *
 * This class provides functionalities to track faces in real-time video streams. It integrates
 * face detection, landmark prediction, and face pose quality assessment to track and analyze faces.
 */
class INSPIRE_API FaceTrackModule {
public:
    /**
     * @brief Constructor for FaceTrack.
     * @param max_detected_faces Maximum number of faces to be detected.
     * @param detection_interval Interval between detections to track faces.
     * @param track_preview_size Size of the preview for tracking.
     * @param dynamic_detection_input_level Change the detector input size.
     */
    explicit FaceTrackModule(DetectModuleMode mode, int max_detected_faces = 1, int detection_interval = 20, int track_preview_size = -1,
                             int dynamic_detection_input_level = -1, int TbD_mode_fps = 30, bool detect_mode_landmark = true);

    /**
     * @brief Configures the face tracking with models.
     * @param archive Model archive for loading the required modes.
     * @param expansion_path Expand the path if you need it.
     * @return int Status of the configuration.
     */
    int Configuration(InspireArchive &archive, const std::string &expansion_path = "", bool enable_face_pose_and_quality = false);

    /**
     * @brief Updates the video stream for face tracking.
     * @param image Camera stream to process.
     * @param is_detect Flag to enable/disable face detection.
     */
    void UpdateStream(inspirecv::FrameProcess &image);

    /**
     * @brief Sets the preview size for tracking.
     * @param preview_size Size of the preview for tracking.
     */
    void SetTrackPreviewSize(int preview_size = -1);

    /**
     * @brief Gets the preview size for tracking.
     * @return Size of the preview for tracking.
     */
    int32_t GetTrackPreviewSize() const;

private:
    /**
     * @brief Predicts sparse landmarks for a cropped face image.
     * @param raw_face_crop Cropped face image.
     * @param landmarks_output Output vector for predicted landmarks.
     * @param score Confidence score for the landmarks prediction.
     * @param size Size for normalizing the face crop.
     */
    void SparseLandmarkPredict(const inspirecv::Image &raw_face_crop, std::vector<inspirecv::Point2f> &landmarks_output, float &score,
                               float size = 112.0);

    /**
     * @brief Predicts the tracking score for a cropped face image.
     * @param raw_face_crop Cropped face image.
     * @return float Tracking score.
     */
    float PredictTrackScore(const inspirecv::Image &raw_face_crop);

    /**
     * @brief Tracks a face in the given image stream.
     * @param image Camera stream containing the face.
     * @param face FaceObject to be tracked.
     * @return bool Status of face tracking.
     */
    bool TrackFace(inspirecv::FrameProcess &image, FaceObjectInternal &face);

    /**
     * @brief Blacks out the region specified in the image for tracking.
     * @param image Image in which the region needs to be blacked out.
     * @param rect_mask Rectangle specifying the region to black out.
     */
    static void BlackingTrackingRegion(inspirecv::Image &image, inspirecv::Rect2f &rect_mask);

    /**
     * @brief Performs non-maximum suppression for face detection.
     * @param th Threshold for suppression.
     */
    void nms(float th = 0.5);

    /**
     * @brief Detects faces in the given image.
     * @param input Image in which faces are to be detected.
     * @param scale Scale factor for image processing.
     */
    void DetectFace(const inspirecv::Image &input, float scale);

    /**
     * @brief Initializes the landmark model.
     * @param model Pointer to the landmark model.
     * @return int Status of initialization.
     */
    int InitLandmarkModel(InspireModel &model);

    /**
     * @brief Initializes the detection model.
     * @param model Pointer to the detection model to be initialized.
     * @return int Status of the initialization process. Returns 0 for success.
     */
    int InitDetectModel(InspireModel &model);

    /**
     * @brief Initializes the RNet (Refinement Network) model.
     * @param model Pointer to the RNet model to be initialized.
     * @return int Status of the initialization process. Returns 0 for success.
     */
    int InitRNetModel(InspireModel &model);

    /**
     * @brief Initializes the face pose and quality estimation model.
     * @param model Pointer to the face pose and quality model to be initialized.
     * @return int Status of the initialization process. Returns 0 for success.
     */
    int InitFacePoseAndQualityModel(InspireModel &model);

    /**
     * @brief Select the detection model scheme to be used according to the input pixel level.
     * @param pixel_size Currently, only 160, 320, and 640 pixel sizes are supported.
     * @return Return the corresponding scheme name, only ”face_detect_160”, ”face_detect_320”, ”face_detect_640” are supported.
     */
    std::string ChoiceMultiLevelDetectModel(const int32_t pixel_size, int32_t &final_size);

public:
    /**
     * @brief Set the detect threshold
     * */
    void SetDetectThreshold(float value);

    /**
     * @brief Set the minimum face size
     * */
    void SetMinimumFacePxSize(float value);

    /**
     * @brief Check if landmark detection is enabled in detection mode.
     * @return True if landmark detection is enabled, false otherwise.
     */
    bool IsDetectModeLandmark() const;

    /**
     * @brief Sets the smoothing ratio for landmark tracking
     * @param value Smoothing ratio between 0 and 1, smaller values mean stronger smoothing
     */
    void SetTrackModeSmoothRatio(float value);

    /**
     * @brief Set the number of smooth cache frame
     * @param value Number of frames to cache for smoothing
     */
    void SetTrackModeNumSmoothCacheFrame(int value);

    /**
     * @brief Set the detect interval
     * @param value Interval between detections
     */
    void SetTrackModeDetectInterval(int value);

    /**
     * @brief Set the multiscale landmark loop num
     * @param value Multiscale landmark loop num
     */
    void SetMultiscaleLandmarkLoop(int value);

    /**
     * @brief Set the track lost recovery mode
     * @param value Track lost recovery mode
     */
    void SetTrackLostRecoveryMode(bool value);

    /**
     * @brief Set the light track confidence threshold
     * @param value Light track confidence threshold
     */
    void SetLightTrackConfidenceThreshold(float value);

    /**
     * @brief Clear the tracking face
     */
    void ClearTrackingFace();

public:
    std::vector<FaceObjectInternal> trackingFace;  ///< Vector of FaceObjects currently being tracked.

public:
    int32_t GetDebugPreviewImageSize() const;

private:
    const int max_detected_faces_;                     ///< Maximum number of faces to detect.
    std::vector<FaceObjectInternal> candidate_faces_;  ///< Vector of candidate FaceObjects for tracking.
    int detection_index_;                              ///< Current detection index.
    int detection_interval_;                           ///< Interval between detections.
    int tracking_idx_;                                 ///< Current tracking index.
    int track_preview_size_;                           ///< Size of the tracking preview.
    int filter_minimum_face_px_size = 0;               ///< Minimum face pixel allowed to be retained (take the edge with the smallest Rect).

private:
    // Debug cache
    int32_t m_debug_preview_image_size_{0};  ///< Debug preview image size

private:
    std::shared_ptr<FaceDetectAdapt> m_face_detector_;         ///< Shared pointer to the face detector.
    std::shared_ptr<FaceLandmarkAdapt> m_landmark_predictor_;  ///< Shared pointer to the landmark predictor.
    std::shared_ptr<RNetAdapt> m_refine_net_;                  ///< Shared pointer to the RNet model.

    std::shared_ptr<FacePoseQualityAdapt> m_face_quality_;  ///< Shared pointer to the face pose quality assessor.

    std::shared_ptr<BYTETracker> m_TbD_tracker_;  ///< Shared pointer to the Bytetrack.
    int m_dynamic_detection_input_level_ = -1;    ///< Detector size class for dynamic input.

    float m_crop_extensive_ratio_ = 1.8f;  ///< Crop extensive ratio
    // float m_crop_extensive_ratio_ = 1.5f;  ///< Crop extensive ratio
    int m_crop_extensive_size_ = 96;  ///< Crop extensive size

    DetectModuleMode m_mode_;  ///< Detect mode

    std::string m_expansion_path_{""};  ///< Expand the path if you need it.

    bool m_detect_mode_landmark_{true};  ///< Detect mode landmark

    int m_track_mode_num_smooth_cache_frame_ = 5;  ///< Track mode number of smooth cache frame

    float m_track_mode_smooth_ratio_ = 0.05;  ///< Track mode smooth ratio

    int m_multiscale_landmark_loop_num_ = 1;  ///< Multiscale landmark loop num

    float m_landmark_crop_ratio_ = 1.1f;

    float m_light_track_confidence_threshold_ = 0.1;  ///< Light track confidence threshold

    std::vector<float> m_multiscale_landmark_scales_;

    bool m_track_lost_recovery_mode_{false};  ///< Track lost recovery mode(only for LightTrack mode)

    std::shared_ptr<LandmarkParam> m_landmark_param_;
};

}  // namespace inspire

#endif  // INSPIRE_FACE_TRACK_MODULE_FACE_TRACK_MODULE_H
