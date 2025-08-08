#ifndef INSPIRE_FACE_SESSION_H
#define INSPIRE_FACE_SESSION_H
#include <memory>
#include "data_type.h"
#include "frame_process.h"
#include "face_wrapper.h"

namespace inspire {

/**
 * @brief The face algorithm session class.
 */
class INSPIRE_API_EXPORT Session {
public:
    Session();
    ~Session();

    Session(Session&&) noexcept;
    Session& operator=(Session&&) noexcept;

    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    /**
     * @brief Create a new session with the given parameters.
     * @param detect_mode The mode of face detection.
     * @param max_detect_face The maximum number of faces to detect.
     * @param param The custom pipeline parameter.
     * @param detect_level_px The detection level in pixels.
     * @param track_by_detect_mode_fps The tracking frame rate.
     * @return A new session.
     */
    static Session Create(DetectModuleMode detect_mode, int32_t max_detect_face, const CustomPipelineParameter& param, int32_t detect_level_px = -1,
                          int32_t track_by_detect_mode_fps = -1);

    /**
     * @brief Create a new session pointer with the given parameters.
     * @param detect_mode The mode of face detection.
     * @param max_detect_face The maximum number of faces to detect.
     * @param param The custom pipeline parameter.
     * @param detect_level_px The detection level in pixels.
     * @param track_by_detect_mode_fps The tracking frame rate.
     * @return A raw pointer to new session. The caller is responsible for memory management.
     */
    static Session* CreatePtr(DetectModuleMode detect_mode, int32_t max_detect_face, const CustomPipelineParameter& param,
                              int32_t detect_level_px = -1, int32_t track_by_detect_mode_fps = -1) {
        return new Session(Create(detect_mode, max_detect_face, param, detect_level_px, track_by_detect_mode_fps));
    }

    /**
     * @brief Clear the tracking face
     */
    void ClearTrackingFace();

    /**
     * @brief Set the track lost recovery mode(only for LightTrack mode, default is false)
     * @param value The track lost recovery mode value
     */
    void SetTrackLostRecoveryMode(bool value);

    /**
     * @brief Set the light track confidence threshold
     * @param value Light track confidence threshold
     */
    void SetLightTrackConfidenceThreshold(float value);

    /**
     * @brief Set the track preview size.
     * @param preview_size The preview size.
     */
    void SetTrackPreviewSize(int32_t preview_size);

    /**
     * @brief Set the minimum face pixel size.
     * @param min_face_pixel_size The minimum face pixel size.
     */
    void SetFilterMinimumFacePixelSize(int32_t min_face_pixel_size);

    /**
     * @brief Set the face detect threshold.
     * @param threshold The face detect threshold.
     */
    void SetFaceDetectThreshold(float threshold);

    /**
     * @brief Set the track mode smooth ratio.
     * @param smooth_ratio The track mode smooth ratio.
     */
    void SetTrackModeSmoothRatio(int32_t smooth_ratio);

    /**
     * @brief Set the track mode num smooth cache frame.
     * @param num_smooth_cache_frame The track mode num smooth cache frame.
     */
    void SetTrackModeNumSmoothCacheFrame(int32_t num_smooth_cache_frame);

    /**
     * @brief Set the track mode detect interval.
     * @param detect_interval The track mode detect interval.
     */
    void SetTrackModeDetectInterval(int32_t detect_interval);

    /**
     * @brief Detect and track the faces in the frame.
     * @param process The frame process.
     * @param results The detected faces.
     */
    int32_t FaceDetectAndTrack(inspirecv::FrameProcess& process, std::vector<FaceTrackWrap>& results);

    /**
     * @brief Get the face bounding box.
     * @param face_data The face data.
     * @return The face bounding box.
     */
    inspirecv::Rect2i GetFaceBoundingBox(const FaceTrackWrap& face_data);

    /**
     * @brief Get the face dense landmark.
     * @param face_data The face data.
     * @return The face dense landmark.
     */
    std::vector<inspirecv::Point2f> GetFaceDenseLandmark(const FaceTrackWrap& face_data);

    /**
     * @brief Get the face five key points.
     * @param face_data The face data.
     * @return The face five key points.
     */
    std::vector<inspirecv::Point2f> GetFaceFiveKeyPoints(const FaceTrackWrap& face_data);

    /**
     * @brief Extract the face feature.
     * @param process The frame process.
     * @param data The face data.
     * @param embedding The face embedding.
     * @param normalize The normalize flag.
     */
    int32_t FaceFeatureExtract(inspirecv::FrameProcess& process, FaceTrackWrap& data, FaceEmbedding& embedding, bool normalize = true);

    /**
     * @brief Get the face alignment image.
     * @param process The frame process.
     * @param data The face data.
     * @param wrapped The wrapped image.
     */
    void GetFaceAlignmentImage(inspirecv::FrameProcess& process, FaceTrackWrap& data, inspirecv::Image& wrapped);

    /**
     * @brief Extract the face feature with alignment image.
     * @param process The frame process.
     * @param embedding The face embedding.
     * @param normalize The normalize flag.
     */
    int32_t FaceFeatureExtractWithAlignmentImage(inspirecv::FrameProcess& process, FaceEmbedding& embedding, bool normalize = true);

    /**
     * @brief Extract the face feature with alignment image.
     * @param wrapped The wrapped image.
     * @param embedding The face embedding.
     * @param normalize The normalize flag.
     */
    int32_t FaceFeatureExtractWithAlignmentImage(const inspirecv::Image& wrapped, FaceEmbedding& embedding, bool normalize = true);

    /**
     * @brief Multiple face pipeline process.
     * @param process The frame process.
     * @param param The custom pipeline parameter.
     * @param face_data_list The face data list.
     */
    int32_t MultipleFacePipelineProcess(inspirecv::FrameProcess& process, const CustomPipelineParameter& param,
                                        const std::vector<FaceTrackWrap>& face_data_list);

    /**
     * @brief Get the RGB liveness confidence.
     * @return The RGB liveness confidence.
     */
    std::vector<float> GetRGBLivenessConfidence();

    /**
     * @brief Get the face mask confidence.
     * @return The face mask confidence.
     */
    std::vector<float> GetFaceMaskConfidence();

    /**
     * @brief Get the face quality confidence.
     * @return The face quality confidence.
     */
    std::vector<float> GetFaceQualityConfidence();

    /**
     * @brief Get the face interaction state.
     * @return The face interaction state.
     */
    std::vector<FaceInteractionState> GetFaceInteractionState();

    /**
     * @brief Get the face interaction action.
     * @return The face interaction action.
     */
    std::vector<FaceInteractionAction> GetFaceInteractionAction();

    /**
     * @brief Get the face attribute result.
     * @return The face attribute result.
     */
    std::vector<FaceAttributeResult> GetFaceAttributeResult();

    /**
     * @brief Get the face emotion result.
     * @return The face emotion result.
     */
    std::vector<FaceEmotionResult> GetFaceEmotionResult();

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

}  // namespace inspire

#endif  // INSPIRE_FACE_SESSION_H