//
// Created by Tunm-Air13 on 2023/9/7.
//
#pragma once
#include <vector>
#ifndef HYPERFACEREPO_FACE_CONTEXT_H
#define HYPERFACEREPO_FACE_CONTEXT_H

/**
 * @file face_context.h
 * @brief Face context handling for HyperFaceRepo project.
 *        Includes definitions for face detection, tracking, and feature extraction.
 */

#include <memory>
#include "track_module/face_track.h"
#include "data_type.h"
#include "pipeline_module/face_pipeline.h"
#include "recognition_module/face_feature_extraction.h"
#include "middleware/model_archive/inspire_archive.h"

/**
 * @def DB_FILE_NAME
 * @brief Default database file name used in the FaceContext.
 */
#define DB_FILE_NAME ".E63520A95DD5B3892C56DA38C3B28E551D8173FD"

namespace inspire {

/**
 * @struct CustomPipelineParameter
 * @brief Structure to hold custom parameters for the face detection and processing pipeline.
 *
 * Includes options for enabling various features such as recognition, liveness detection, and quality assessment.
 */
typedef struct CustomPipelineParameter {
    bool enable_recognition = false;              ///< Enable face recognition feature
    bool enable_liveness = false;                 ///< Enable RGB liveness detection feature
    bool enable_ir_liveness = false;              ///< Enable IR (Infrared) liveness detection feature
    bool enable_mask_detect = false;              ///< Enable mask detection feature
    bool enable_face_attribute = false;           ///< Enable face attribute prediction feature
    bool enable_face_quality = false;             ///< Enable face quality assessment feature
    bool enable_interaction_liveness = false;     ///< Enable interactive liveness detection feature

} ContextCustomParameter;

/**
 * @class FaceContext
 * @brief Manages the context for face detection, tracking, and feature extraction in the HyperFaceRepo project.
 *
 * Provides interfaces to configure face detection modes, manage face tracking, perform recognition,
 * and handle other face-related features. Integrates with various modules such as FaceTrack, FaceRecognition, and FacePipeline.
 */
class INSPIRE_API FaceContext {
public:

    /**
     * @brief Constructor for the FaceContext class.
     */
    explicit FaceContext();

    /**
     * @brief Configures the face context with given parameters.
     * @param model_file_path Path to the model file for face detection.
     * @param detect_mode The detection mode to be used (image or video).
     * @param max_detect_face The maximum number of faces to detect.
     * @param param Custom parameters for the face pipeline.
     * @return int32_t Returns 0 on success, non-zero for any error.
     */
    int32_t Configuration(DetectMode detect_mode, 
                        int32_t max_detect_face, 
                        CustomPipelineParameter param, 
                        int32_t detect_level_px = -1,
                        int32_t track_by_detect_mode_fps = -1);

    /**
     * @brief Performs face detection and tracking on a given image stream.
     * @param image The camera stream to process for face detection and tracking.
     * @return int32_t Returns the number of faces detected and tracked.
     */// Method for face detection and tracking
    int32_t FaceDetectAndTrack(CameraStream &image);

    /**
     * @brief Set the threshold of face detection function, which only acts on the detection model
     * @param value threshold value
     * @return int32_t Returns the number of faces detected and tracked.
     * */
    int32_t SetFaceDetectThreshold(float value);

    /**
     * @brief Retrieves the list of currently tracked faces.
     * @return FaceObjectList A list of face objects currently being tracked.
     */
    FaceObjectList& GetTrackingFaceList();

    /**
     * @brief Processes faces using the provided pipeline parameters.
     * @param image Camera stream containing faces.
     * @param faces Vector of HyperFaceData for detected faces.
     * @param param Custom pipeline parameters.
     * @return int32_t Status code of the processing.
     */
    int32_t FacesProcess(CameraStream &image, const std::vector<HyperFaceData> &faces, const CustomPipelineParameter& param);

    /**
     * @brief Retrieves the face recognition module.
     * @return std::shared_ptr<FaceRecognition> Shared pointer to the FaceRecognition module.
     */
    const std::shared_ptr<FeatureExtraction>& FaceRecognitionModule();

    /**
     * @brief Retrieves the face pipeline module.
     * @return std::shared_ptr<FacePipeline> Shared pointer to the FacePipeline module.
     */
    const std::shared_ptr<FacePipeline>& FacePipelineModule();

    /**
     * @brief Gets the number of faces currently detected.
     * @return int32_t Number of faces currently detected.
     */
    const int32_t GetNumberOfFacesCurrentlyDetected() const;

    /**
     * @brief Extracts features of a face from an image.
     * @param image Camera stream containing the face.
     * @param data FaceBasicData to store extracted features.
     * @return int32_t Status code of the feature extraction.
     */
    int32_t FaceFeatureExtract(CameraStream &image, FaceBasicData& data);

    /**
     * @brief Retrieves the custom pipeline parameters.
     * @return CustomPipelineParameter Current custom pipeline parameters.
     */
    const CustomPipelineParameter &getMParameter() const;

    /**
     * @brief Static method for detecting face quality.
     * @param data FaceBasicData containing the face information.
     * @param result Float to store the face quality result.
     * @return int32_t Status code of the quality detection.
     */
    static int32_t FaceQualityDetect(FaceBasicData& data, float &result);

    /**
     * @brief Sets the preview size for face tracking.
     * @param preview_size Integer specifying the new preview size.
     * @return int32_t Status code of the operation.
     */
    int32_t SetTrackPreviewSize(int32_t preview_size);

    /**
     * @brief Filter the minimum face pixel size.
     * @param minSize The minimum pixel value.
     * @return int32_t Status code of the operation.
     */
    int32_t SetTrackFaceMinimumSize(int32_t minSize);

    /**
     * @brief Sets the mode for face detection.
     * @param mode You can select mode for track or detect.
     * @return int32_t Status code of the operation.
     * */
    int32_t SetDetectMode(DetectMode mode);

public:
    // Accessor methods for various cached data
    /**
     * @brief Retrieves the cache of detected face data.
     * @return std::vector<ByteArray> Cache of detected face data.
     */
    const std::vector<ByteArray>& GetDetectCache() const;

    /**
     * @brief Retrieves the cache of basic face data.
     * @return std::vector<FaceBasicData> Cache of basic face data.
     */
    const std::vector<FaceBasicData>& GetFaceBasicDataCache() const;

    /**
     * @brief Retrieves the cache of face rectangles.
     * @return std::vector<FaceRect> Cache of face rectangles.
     */
    const std::vector<FaceRect>& GetFaceRectsCache() const;

    /**
     * @brief Retrieves the cache of tracking IDs.
     * @return std::vector<int32_t> Cache of tracking IDs.
     */
    const std::vector<int32_t>& GetTrackIDCache() const;

    /**
     * @brief Retrieves the cache of roll results from face pose estimation.
     * @return std::vector<float> Cache of roll results.
     */
    const std::vector<float>& GetRollResultsCache() const;

    /**
     * @brief Retrieves the cache of yaw results from face pose estimation.
     * @return std::vector<float> Cache of yaw results.
     */
    const std::vector<float>& GetYawResultsCache() const;


    /**
     * @brief Gets the cache of pitch results from face pose estimation.
     * @return A const reference to a vector containing pitch results.
     */
    const std::vector<float>& GetPitchResultsCache() const;

    /**
     * @brief Gets the cache of face pose quality results.
     * @return A const reference to a vector of FacePoseQualityResult objects.
     */
    const std::vector<FacePoseQualityResult>& GetQualityResultsCache() const;

    /**
     * @brief Gets the cache of mask detection results.
     * @return A const reference to a vector containing mask detection results.
     */
    const std::vector<float>& GetMaskResultsCache() const;

    /**
     * @brief Gets the cache of RGB liveness detection results.
     * @return A const reference to a vector containing RGB liveness results.
     */
    const std::vector<float>& GetRgbLivenessResultsCache() const;

    /**
     * @brief Gets the cache of face quality predict results.
     * @return A const reference to a vector containing face quality predict results.
     */
    const std::vector<float>& GetFaceQualityScoresResultsCache() const;

    /**
     * @brief Gets the cache of left eye status predict results.
     * @return A const reference to a vector containing eye status predict results.
     */
    const std::vector<float>& GetFaceInteractionLeftEyeStatusCache() const;

    /**
     * @brief Gets the cache of right eye status predict results.
     * @return A const reference to a vector containing eye status predict results.
     */
    const std::vector<float>& GetFaceInteractionRightEyeStatusCache() const;

    /**
     * @brief Gets the cache of face attribute rece results.
     * @return A const reference to a vector containing face attribute rece results.
     */
    const std::vector<int>& GetFaceRaceResultsCache() const;

    /**
     * @brief Gets the cache of face attribute gender results.
     * @return A const reference to a vector containing face attribute gender results.
     */
    const std::vector<int>& GetFaceGenderResultsCache() const;

    /**
     * @brief Gets the cache of face attribute age bracket results.
     * @return A const reference to a vector containing face attribute age bracket results.
     */
    const std::vector<int>& GetFaceAgeBracketResultsCache() const;

    /**
     * @brief Gets the cache of face action normal results.
     * @return A const reference to a vector containing face action normal results.
     */
    const std::vector<int>& GetFaceNormalAactionsResultCache() const;
    
    /**
     * @brief Gets the cache of face action jaw open results.
     * @return A const reference to a vector containing face action jaw open results.
     */
    const std::vector<int>& GetFaceJawOpenAactionsResultCache() const;

    /**
     * @brief Gets the cache of face action blink results.
     * @return A const reference to a vector containing face action blink results.
     */
    const std::vector<int>& GetFaceBlinkAactionsResultCache() const;

    /**
     * @brief Gets the cache of face action shake results.
     * @return A const reference to a vector containing face action shake results.
     */
    const std::vector<int>& GetFaceShakeAactionsResultCache() const;

    /**
     * @brief Gets the cache of face action raise head results.
     * @return A const reference to a vector containing face action raise head results.
     */
    const std::vector<int>& GetFaceRaiseHeadAactionsResultCache() const;

    /**
     * @brief Gets the cache of the current face features.
     * @return A const reference to the Embedded object containing current face feature data.
     */
    const Embedded& GetFaceFeatureCache() const;


private:
    // Private member variables
    CustomPipelineParameter m_parameter_;                 ///< Stores custom parameters for the pipeline
    int32_t m_max_detect_face_{};                         ///< Maximum number of faces that can be detected
    DetectMode m_detect_mode_;                             ///< Current detection mode (image or video)
    bool m_always_detect_{};                               ///< Flag to determine if detection should always occur

    std::shared_ptr<FaceTrack> m_face_track_;              ///< Shared pointer to the FaceTrack object
    std::shared_ptr<FeatureExtraction> m_face_recognition_;  ///< Shared pointer to the FaceRecognition object
    std::shared_ptr<FacePipeline> m_face_pipeline_;        ///< Shared pointer to the FacePipeline object

private:
    // Cache data
    std::vector<ByteArray> m_detect_cache_;                        ///< Cache for storing serialized detected face data
    std::vector<FaceBasicData> m_face_basic_data_cache_;           ///< Cache for basic face data extracted from detection
    std::vector<FaceRect> m_face_rects_cache_;                     ///< Cache for face rectangle data from detection
    std::vector<int32_t> m_track_id_cache_;                        ///< Cache for tracking IDs of detected faces
    std::vector<float> m_roll_results_cache_;                       ///< Cache for storing roll results from face pose estimation
    std::vector<float> m_yaw_results_cache_;                        ///< Cache for storing yaw results from face pose estimation
    std::vector<float> m_pitch_results_cache_;                      ///< Cache for storing pitch results from face pose estimation
    std::vector<FacePoseQualityResult> m_quality_results_cache_;   ///< Cache for face pose quality results
    std::vector<float> m_mask_results_cache_;                       ///< Cache for mask detection results
    std::vector<float> m_rgb_liveness_results_cache_;               ///< Cache for RGB liveness detection results
    std::vector<float> m_quality_score_results_cache_;               ///< Cache for RGB face quality score results
    std::vector<float> m_react_left_eye_results_cache_;               ///< Cache for Left eye state in face interaction
    std::vector<float> m_react_right_eye_results_cache_;               ///< Cache for Right eye state in face interaction

    std::vector<int> m_action_normal_results_cache_;                ///< Cache for normal action in face interaction
    std::vector<int> m_action_shake_results_cache_;                 ///< Cache for shake action in face interaction
    std::vector<int> m_action_blink_results_cache_;                 ///< Cache for blink action in face interaction
    std::vector<int> m_action_jaw_open_results_cache_;              ///< Cache for jaw open action in face interaction
    std::vector<int> m_action_raise_head_results_cache_;            ///< Cache for raise head action in face interaction

    std::vector<int> m_attribute_race_results_cache_;
    std::vector<int> m_attribute_gender_results_cache_;
    std::vector<int> m_attribute_age_results_cache_;
    Embedded m_face_feature_cache_;                                ///< Cache for current face feature data

    std::mutex m_mtx_;                                             ///< Mutex for thread safety.

};

}   // namespace hyper

#endif //HYPERFACEREPO_FACE_CONTEXT_H
