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
    PROCESS_ATTRIBUTE,              ///< Face attribute estimation.
    PROCESS_INTERACTION,            ///< Face interaction.
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
     * @param enableAttributee Whether face attribute estimation is enabled.
     * @param enableInteractionLiveness Whether interaction liveness detection is enabled.
     */
    explicit FacePipeline(InspireArchive &archive, bool enableLiveness, bool enableMaskDetect, bool enableAttribute, 
            bool enableInteractionLiveness);

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
     * @brief Initializes the FaceAttributePredict model.
     *
     * @param model Pointer to the FaceAttributePredict model.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t InitFaceAttributePredict(InspireModel &model);

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
     * @brief Initializes the Blink predict model.
     *
     * @param model Pointer to the Blink predict model.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t InitBlinkFromLivenessInteraction(InspireModel &model);

private:
    const bool m_enable_liveness_ = false;                 ///< Whether RGB liveness detection is enabled.
    const bool m_enable_mask_detect_ = false;              ///< Whether mask detection is enabled.
    const bool m_enable_attribute_ = false;                ///< Whether face attribute is enabled.
    const bool m_enable_interaction_liveness_ = false;     ///< Whether interaction liveness detection is enabled.

    std::shared_ptr<FaceAttributePredict> m_attribute_predict_;   ///< Pointer to Face attribute prediction instance.
    std::shared_ptr<MaskPredict> m_mask_predict_;          ///< Pointer to MaskPredict instance.
    std::shared_ptr<RBGAntiSpoofing> m_rgb_anti_spoofing_;   ///< Pointer to RBGAntiSpoofing instance.
    std::shared_ptr<BlinkPredict> m_blink_predict_;         ///< Pointer to Blink predict instance.

public:
    float faceMaskCache;    ///< Cache for face mask detection result.
    float faceLivenessCache; ///< Cache for face liveness detection result.
    cv::Vec2f eyesStatusCache;  ///< Cache for blink predict result.
    cv::Vec3i faceAttributeCache; ///< Cache for face attribute predict result.
};

}

#endif //HYPERFACEREPO_FACEPIPELINE_H
