//
// Created by tunm on 2023/9/7.
//

#ifndef HYPERFACEREPO_FACEPIPELINE_H
#define HYPERFACEREPO_FACEPIPELINE_H

#include "middleware/camera_stream/camera_stream.h"
#include "common/face_info/face_object.h"
#include "attribute/all.h"
#include "liveness/all.h"
#include "common/face_data/data_tools.h"
#include "middleware/model_archive/inspire_archive.h"

namespace inspire {

/**
 * @enum FaceProcessFunction
 * @brief Enumeration for different face processing functions in the FacePipeline.
 */
typedef enum FaceProcessFunction {
    PROCESS_MASK = 0,               ///< Mask detection.
    PROCESS_RGB_LIVENESS,           ///< RGB liveness detection.
    PROCESS_AGE,                    ///< Age estimation.
    PROCESS_GENDER,                 ///< Gender prediction.
} FaceProcessFunction;

/**
 * @class FacePipeline
 * @brief Class for performing face processing tasks in a pipeline.
 *
 * This class provides methods for processing faces in a pipeline, including mask detection, liveness detection,
 * age estimation, and gender prediction.
 */
class FacePipeline {
public:
    /**
     * @brief Constructor for FacePipeline class.
     *
     * @param archive Model archive instance for model loading.
     * @param enableLiveness Whether RGB liveness detection is enabled.
     * @param enableMaskDetect Whether mask detection is enabled.
     * @param enableAge Whether age estimation is enabled.
     * @param enableGender Whether gender prediction is enabled.
     * @param enableInteractionLiveness Whether interaction liveness detection is enabled.
     */
    explicit FacePipeline(InspireArchive &archive, bool enableLiveness, bool enableMaskDetect, bool enableAge,
                          bool enableGender, bool enableInteractionLiveness);

    /**
     * @brief Processes a face using the specified FaceProcessFunction.
     *
     * @param image CameraStream instance containing the image.
     * @param face FaceObject representing the detected face.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t Process(CameraStream &image, FaceObject &face);

    /**
     * @brief Processes a face using the specified FaceProcessFunction.
     *
     * @param image CameraStream instance containing the image.
     * @param face HyperFaceData representing the detected face.
     * @param proc The FaceProcessFunction to apply to the face.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t Process(CameraStream &image, const HyperFaceData &face, FaceProcessFunction proc);

    const std::shared_ptr<RBGAntiSpoofing> &getMRgbAntiSpoofing() const;

private:
    /**
     * @brief Initializes the AgePredict model.
     *
     * @param model Pointer to the AgePredict model.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t InitAgePredict(InspireModel &model);

    /**
     * @brief Initializes the GenderPredict model.
     *
     * @param model Pointer to the GenderPredict model.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t InitGenderPredict(InspireModel &model);

    /**
     * @brief Initializes the MaskPredict model.
     *
     * @param model Pointer to the MaskPredict model.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t InitMaskPredict(InspireModel &model);

    /**
     * @brief Initializes the RBGAntiSpoofing model.
     *
     * @param model Pointer to the RBGAntiSpoofing model.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t InitRBGAntiSpoofing(InspireModel &model);

    /**
     * @brief Initializes the LivenessInteraction model.
     *
     * @param model Pointer to the LivenessInteraction model.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t InitLivenessInteraction(InspireModel &model);

private:
    const bool m_enable_liveness_ = false;                 ///< Whether RGB liveness detection is enabled.
    const bool m_enable_mask_detect_ = false;              ///< Whether mask detection is enabled.
    const bool m_enable_age_ = false;                      ///< Whether age estimation is enabled.
    const bool m_enable_gender_ = false;                   ///< Whether gender prediction is enabled.
    const bool m_enable_interaction_liveness_ = false;     ///< Whether interaction liveness detection is enabled.

    std::shared_ptr<AgePredict> m_age_predict_;            ///< Pointer to AgePredict instance.
    std::shared_ptr<GenderPredict> m_gender_predict_;      ///< Pointer to GenderPredict instance.
    std::shared_ptr<MaskPredict> m_mask_predict_;          ///< Pointer to MaskPredict instance.
    std::shared_ptr<RBGAntiSpoofing> m_rgb_anti_spoofing_;   ///< Pointer to RBGAntiSpoofing instance.
    std::shared_ptr<LivenessInteraction> m_liveness_interaction_spoofing_; ///< Pointer to LivenessInteraction instance.

public:
    float faceMaskCache;    ///< Cache for face mask detection result.
    float faceLivenessCache; ///< Cache for face liveness detection result.
};

}

#endif //HYPERFACEREPO_FACEPIPELINE_H
