/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#pragma once
#ifndef INSPIRE_FACE_TRACK_MODULE_FACE_DETECT_FACE_DETECT_ADAPT_H
#define INSPIRE_FACE_TRACK_MODULE_FACE_DETECT_FACE_DETECT_ADAPT_H
#include "data_type.h"
#include "middleware/any_net_adapter.h"
#include "image_process/nexus_processor/image_processor.h"

namespace inspire {

/**
 * @class FaceDetect
 * @brief Class for face detection, inheriting from AnyNet.
 *
 * This class provides functionalities to detect faces in images using neural network models.
 */
class INSPIRE_API FaceDetectAdapt : public AnyNetAdapter {
public:
    /**
     * @brief Constructor for the FaceDetect class.
     * @param input_size The size of the input image for the neural network.
     * @param nms_threshold The threshold for non-maximum suppression.
     * @param cls_threshold The threshold for classification score.
     */
    explicit FaceDetectAdapt(int input_size = 160, float nms_threshold = 0.4f, float cls_threshold = 0.5f);

    /**
     * @brief Detects faces in a given image.
     * @param bgr The input image in BGR format.
     * @return FaceLocList List of detected faces with location and landmarks.
     */
    FaceLocList operator()(const inspirecv::Image &bgr);

    /**
     * @brief Detects faces while distinguishing failures from a valid empty result.
     * @param bgr The input image in BGR format.
     * @param results Receives detected faces and is cleared on failure.
     * @return HSUCCEED, HERR_DEVICE_IMAGE_PROCESS_FAILURE, or HERR_SESS_TRACKER_FAILURE.
     */
    int32_t Detect(const inspirecv::Image &bgr, FaceLocList &results);

    /** @brief Set non-maximum suppression threshold */
    void SetNmsThreshold(float mNmsThreshold);

    /** @brief Set face classification threshold */
    void SetClsThreshold(float mClsThreshold);

    /**
     * @brief Get the input size
     * @return int The input size
     */
    int GetInputSize() const;

private:
    /**
     * @brief Applies non-maximum suppression to reduce overlapping detected faces.
     * @param input_faces List of detected faces to be filtered.
     * @param nms_threshold The threshold for non-maximum suppression.
     */
    static void _nms(FaceLocList &input_faces, float nms_threshold);

    /**
     * @brief Decodes network outputs to face locations.
     * @param cls_pred Classification predictions.
     * @param box_pred Bounding box predictions.
     * @param lmk_pred Landmark predictions.
     * @param stride The stride of the detection.
     * @param results Decoded face locations.
     */
    void _decode(const float *cls_pred, const float *box_pred, const float *lmk_pred, int stride, std::vector<FaceLoc> &results);

private:
    float m_nms_threshold_;  ///< Threshold for non-maximum suppression.
    float m_cls_threshold_;  ///< Threshold for classification score.
    int m_input_size_;       ///< Input size for the neural network model.
};

}  // namespace inspire

#endif  // INSPIRE_FACE_TRACK_MODULE_FACE_DETECT_FACE_DETECT_ADAPT_H
