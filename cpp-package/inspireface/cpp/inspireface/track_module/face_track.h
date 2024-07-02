//
// Created by tunm on 2023/8/29.
//
#pragma once
#ifndef HYPERFACEREPO_FACETRACK_H
#define HYPERFACEREPO_FACETRACK_H
#include <iostream>
#include "face_detect/all.h"
#include "landmark/face_landmark.h"
#include "common/face_info/all.h"
#include "middleware/camera_stream/camera_stream.h"
#include "quality/face_pose_quality.h"
#include "middleware/model_archive/inspire_archive.h"
#ifdef ISF_ENABLE_TRACKING_BY_DETECTION
#include "tracker_optional/bytetrack/BYTETracker.h"
#endif

namespace inspire {


/**
 * @enum DetectMode
 * @brief Enumeration for different detection modes.
 */
enum DetectMode {
    DETECT_MODE_ALWAYS_DETECT = 0,      ///< Detection mode: Always detect
    DETECT_MODE_LIGHT_TRACK,            ///< Detection mode: Light face track
    DETECT_MODE_TRACK_BY_DETECT,        ///< Detection mode: Tracking by detection

};

/**
 * @class FaceTrack
 * @brief Class for tracking faces in video streams.
 *
 * This class provides functionalities to track faces in real-time video streams. It integrates
 * face detection, landmark prediction, and face pose quality assessment to track and analyze faces.
 */
class INSPIRE_API FaceTrack {
public:

    /**
     * @brief Constructor for FaceTrack.
     * @param max_detected_faces Maximum number of faces to be detected.
     * @param detection_interval Interval between detections to track faces.
     * @param track_preview_size Size of the preview for tracking.
     * @param dynamic_detection_input_level Change the detector input size.
     */
    explicit FaceTrack(DetectMode mode, int max_detected_faces = 1, int detection_interval = 20, int track_preview_size = 192, int dynamic_detection_input_level = -1, int TbD_mode_fps=30);

    /**
     * @brief Configures the face tracking with models.
     * @param archive Model archive for loading the required models.
     * @return int Status of the configuration.
     */
    int Configuration(InspireArchive &archive);

    /**
     * @brief Updates the video stream for face tracking.
     * @param image Camera stream to process.
     * @param is_detect Flag to enable/disable face detection.
     */
    void UpdateStream(CameraStream &image);

    /**
     * @brief Sets the preview size for tracking.
     * @param preview_size Size of the preview for tracking.
     */
    void SetTrackPreviewSize(int preview_size = 192);

private:


    /**
     * @brief Predicts sparse landmarks for a cropped face image.
     * @param raw_face_crop Cropped face image.
     * @param landmarks_output Output vector for predicted landmarks.
     * @param score Confidence score for the landmarks prediction.
     * @param size Size for normalizing the face crop.
     */
    void SparseLandmarkPredict(const cv::Mat &raw_face_crop,
                               std::vector<cv::Point2f> &landmarks_output,
                               float &score, float size = 112.0);

    /**
     * @brief Tracks a face in the given image stream.
     * @param image Camera stream containing the face.
     * @param face FaceObject to be tracked.
     * @return bool Status of face tracking.
     */
    bool TrackFace(CameraStream &image, FaceObject &face);

    /**
     * @brief Blacks out the region specified in the image for tracking.
     * @param image Image in which the region needs to be blacked out.
     * @param rect_mask Rectangle specifying the region to black out.
     */
    static void BlackingTrackingRegion(cv::Mat &image, cv::Rect &rect_mask);

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
    void DetectFace(const cv::Mat &input, float scale);

    /**
     * @brief Initializes the landmark model.
     * @param model Pointer to the landmark model.
     * @return int Status of initialization.
     */
    int InitLandmarkModel(InspireModel& model);

    /**
     * @brief Initializes the detection model.
     * @param model Pointer to the detection model to be initialized.
     * @return int Status of the initialization process. Returns 0 for success.
     */
    int InitDetectModel(InspireModel& model);

    /**
     * @brief Initializes the RNet (Refinement Network) model.
     * @param model Pointer to the RNet model to be initialized.
     * @return int Status of the initialization process. Returns 0 for success.
     */
    int InitRNetModel(InspireModel& model);

    /**
     * @brief Initializes the face pose estimation model.
     * @param model Pointer to the face pose model to be initialized.
     * @return int Status of the initialization process. Returns 0 for success.
     */
    int InitFacePoseModel(InspireModel& model);

    /**
     * @brief Select the detection model scheme to be used according to the input pixel level.
     * @param pixel_size Currently, only 160, 320, and 640 pixel sizes are supported.
     * @return Return the corresponding scheme name, only ”face_detect_160”, ”face_detect_320”, ”face_detect_640” are supported.
     */
    std::string ChoiceMultiLevelDetectModel(const int32_t pixel_size);
    
public:

    /**
     * @brief Gets the total time used for tracking.
     * @return double Total time used in tracking.
     */
    double GetTrackTotalUseTime() const;

    /**
     * @brief Fix detect threshold
     * */
    void SetDetectThreshold(float value);

    /**
     * @brief Fix detect threshold
     * */
    void SetMinimumFacePxSize(float value);

public:

    std::vector<FaceObject> trackingFace;                   ///< Vector of FaceObjects currently being tracked.

private:
    const int max_detected_faces_;                          ///< Maximum number of faces to detect.
    std::vector<FaceObject> candidate_faces_;               ///< Vector of candidate FaceObjects for tracking.
    int detection_index_;                                   ///< Current detection index.
    int detection_interval_;                                ///< Interval between detections.
    int tracking_idx_;                                      ///< Current tracking index.
    double det_use_time_;                                   ///< Time used for detection.
    double track_total_use_time_;                           ///< Total time used for tracking.
    int track_preview_size_;                                ///< Size of the tracking preview.
    int filter_minimum_face_px_size = 24;                   ///< Minimum face pixel allowed to be retained (take the edge with the smallest Rect).

private:

    std::shared_ptr<FaceDetect> m_face_detector_;          ///< Shared pointer to the face detector.
    std::shared_ptr<FaceLandmark> m_landmark_predictor_;   ///< Shared pointer to the landmark predictor.
    std::shared_ptr<RNet> m_refine_net_;                   ///< Shared pointer to the RNet model.
    std::shared_ptr<FacePoseQuality> m_face_quality_;      ///< Shared pointer to the face pose quality assessor.
#ifdef ISF_ENABLE_TRACKING_BY_DETECTION
    std::shared_ptr<BYTETracker> m_TbD_tracker_;           ///< Shared pointer to the Bytetrack.
#endif
    int m_dynamic_detection_input_level_ = -1;             ///< Detector size class for dynamic input.

    DetectMode m_mode_;                                    ///< Detect mode

    
};

}   // namespace hyper

#endif //HYPERFACEREPO_FACETRACK_H
