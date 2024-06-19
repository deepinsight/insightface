//
// Created by Tunm-Air13 on 2023/5/6.
//

#pragma once
#ifndef BIGGUYSMAIN_FACE_MODULE_FACE_DETECT_FACEDETECT_H
#define BIGGUYSMAIN_FACE_MODULE_FACE_DETECT_FACEDETECT_H
#include "../../data_type.h"
#include "middleware/any_net.h"

namespace inspire {

/**
 * @class FaceDetect
 * @brief Class for face detection, inheriting from AnyNet.
 *
 * This class provides functionalities to detect faces in images using neural network models.
 */
class INSPIRE_API FaceDetect: public AnyNet {
public:
    /**
     * @brief Constructor for the FaceDetect class.
     * @param input_size The size of the input image for the neural network.
     * @param nms_threshold The threshold for non-maximum suppression.
     * @param cls_threshold The threshold for classification score.
     */
    explicit FaceDetect(int input_size = 160, float nms_threshold = 0.4f, float cls_threshold = 0.5f);

    /**
     * @brief Detects faces in a given image.
     * @param bgr The input image in BGR format.
     * @return FaceLocList List of detected faces with location and landmarks.
     */
    FaceLocList operator()(const Matrix& bgr);

private:
    /**
     * @brief Applies non-maximum suppression to reduce overlapping detected faces.
     * @param input_faces List of detected faces to be filtered.
     * @param nms_threshold The threshold for non-maximum suppression.
     */
    static void _nms(FaceLocList &input_faces, float nms_threshold);

    /**
     * @brief Generates detection anchors based on stride.
     * @param stride The stride of the detection.
     * @param input_size The size of the input image.
     * @param num_anchors The number of anchors.
     * @param anchors The generated anchors.
     */
    void _generate_anchors(int stride, int input_size, int num_anchors, std::vector<float> &anchors);

    /**
     * @brief Decodes network outputs to face locations.
     * @param cls_pred Classification predictions.
     * @param box_pred Bounding box predictions.
     * @param lmk_pred Landmark predictions.
     * @param stride The stride of the detection.
     * @param results Decoded face locations.
     */
    void _decode(const std::vector<float> &cls_pred, const std::vector<float> &box_pred, const std::vector<float>& lmk_pred, int stride, std::vector<FaceLoc> &results);

public:
    /** @brief Set non-maximum suppression threshold */
    void SetNmsThreshold(float mNmsThreshold);
    /** @brief Set face classification threshold */
    void SetClsThreshold(float mClsThreshold);

private:
    float m_nms_threshold_; ///< Threshold for non-maximum suppression.
    float m_cls_threshold_; ///< Threshold for classification score.
    int m_input_size_;      ///< Input size for the neural network model.

};

/**
 * @brief Sorts FaceLoc objects in descending order of area.
 * @param a The first FaceLoc object.
 * @param b The second FaceLoc object.
 * @return bool True if 'a' is larger than 'b'.
 */
    bool SortBoxSize(const FaceLoc &a, const FaceLoc &b);

}

#endif //BIGGUYSMAIN_FACE_MODULE_FACE_DETECT_FACEDETECT_H
