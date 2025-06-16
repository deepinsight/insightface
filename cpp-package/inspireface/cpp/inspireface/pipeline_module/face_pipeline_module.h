/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#ifndef INSPIRE_FACE_PIPELINE_MODULE_H
#define INSPIRE_FACE_PIPELINE_MODULE_H

#include "frame_process.h"
#include "common/face_info/face_object_internal.h"
#include "attribute/face_attribute_adapt.h"
#include "attribute/mask_predict_adapt.h"
#include "liveness/rgb_anti_spoofing_adapt.h"
#include "liveness/blink_predict_adapt.h"
#include "middleware/model_archive/inspire_archive.h"
#include "face_wrapper.h"
#include "track_module/landmark/landmark_param.h"
#include "attribute/face_emotion_adapt.h"

namespace inspire {

/**
 * @enum FaceProcessFunction
 * @brief Enumeration for different face processing functions in the FacePipeline.
 */
typedef enum FaceProcessFunctionOption {
    PROCESS_MASK = 0,      ///< Mask detection.
    PROCESS_RGB_LIVENESS,  ///< RGB liveness detection.
    PROCESS_ATTRIBUTE,     ///< Face attribute estimation.
    PROCESS_INTERACTION,   ///< Face interaction.
    PROCESS_FACE_EMOTION,  ///< Face emotion recognition.
} FaceProcessFunctionOption;

/**
 * @class FacePipeline
 * @brief Class for performing face processing tasks in a pipeline.
 *
 * This class provides methods for processing faces in a pipeline, including mask detection, liveness detection,
 * age estimation, and gender prediction.
 */
class FacePipelineModule {
public:
    /**
     * @brief Constructor for FacePipeline class.
     *
     * @param archive Model archive instance for model loading.
     * @param enableLiveness Whether RGB liveness detection is enabled.
     * @param enableMaskDetect Whether mask detection is enabled.
     * @param enableAttributee Whether face attribute estimation is enabled.
     * @param enableInteractionLiveness Whether interaction liveness detection is enabled.
     * @param enableFaceEmotion Whether interaction emotion recognition is enabled.
     */
    explicit FacePipelineModule(InspireArchive &archive, bool enableLiveness, bool enableMaskDetect, bool enableAttribute,
                                bool enableInteractionLiveness, bool enableFaceEmotion);

    /**
     * @brief Processes a face using the specified FaceProcessFunction.
     *
     * @param image CameraStream instance containing the image.
     * @param face FaceObject representing the detected face.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t Process(inspirecv::FrameProcess &processor, FaceObjectInternal &face);

    /**
     * @brief Processes a face using the specified FaceProcessFunction.
     *
     * @param image CameraStream instance containing the image.
     * @param face FaceTrackWrap representing the detected face.
     * @param proc The FaceProcessFunction to apply to the face.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t Process(inspirecv::FrameProcess &processor, const FaceTrackWrap &face, FaceProcessFunctionOption proc);

    /**
     * @brief Get Rgb AntiSpoofing module
     * @return AntiSpoofing module
     */
    const std::shared_ptr<RBGAntiSpoofingAdapt> &getMRgbAntiSpoofing() const;

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
     * @brief Initializes the LivenessInteraction model.
     *
     * @param model Pointer to the LivenessInteraction model.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t InitLivenessInteraction(InspireModel &model);

    /**
     * @brief Initializes the Blink predict model.
     *
     * @param model Pointer to the Blink predict model.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t InitBlinkFromLivenessInteraction(InspireModel &model);

    /**
     * @brief Initializes the FaceEmotion model.
     *
     * @param model Pointer to the FaceEmotion model.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t InitFaceEmotion(InspireModel &model);

private:
    const bool m_enable_liveness_ = false;              ///< Whether RGB liveness detection is enabled.
    const bool m_enable_mask_detect_ = false;           ///< Whether mask detection is enabled.
    const bool m_enable_attribute_ = false;             ///< Whether face attribute is enabled.
    const bool m_enable_interaction_liveness_ = false;  ///< Whether interaction liveness detection is enabled.
    const bool m_enable_face_emotion_ = false;          ///< Whether face emotion is enabled.
    
    std::shared_ptr<FaceAttributePredictAdapt> m_attribute_predict_;  ///< Pointer to AgePredict instance.
    std::shared_ptr<MaskPredictAdapt> m_mask_predict_;                ///< Pointer to MaskPredict instance.
    std::shared_ptr<RBGAntiSpoofingAdapt> m_rgb_anti_spoofing_;       ///< Pointer to RBGAntiSpoofing instance.
    std::shared_ptr<BlinkPredictAdapt> m_blink_predict_;              ///< Pointer to Blink predict instance.
    std::shared_ptr<FaceEmotionAdapt> m_face_emotion_;                ///< Pointer to FaceEmotion instance.
    std::shared_ptr<LandmarkParam> m_landmark_param_;                ///< Pointer to LandmarkParam instance.

public:
    float faceMaskCache;                  ///< Cache for face mask detection result.
    float faceLivenessCache;              ///< Cache for face liveness detection result.
    inspirecv::Vec2f eyesStatusCache;     ///< Cache for blink predict result.
    inspirecv::Vec3i faceAttributeCache;  ///< Cache for face attribute predict result.
    std::vector<float> faceEmotionCache;   ///< Cache for face emotion recognition result.
};

}  // namespace inspire

#endif  // INSPIRE_FACE_PIPELINE_MODULE_H
