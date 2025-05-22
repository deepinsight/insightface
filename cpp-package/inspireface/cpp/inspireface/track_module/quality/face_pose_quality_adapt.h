/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#pragma once
#ifndef INSPIRE_FACE_TRACK_MODULE_QUALITY_FACE_POSE_QUALITY_ADAPT_H
#define INSPIRE_FACE_TRACK_MODULE_QUALITY_FACE_POSE_QUALITY_ADAPT_H

#include "data_type.h"
#include "middleware/any_net_adapter.h"

namespace inspire {

/**
 * @struct FacePoseQualityResult
 * @brief Structure to store the results of face pose quality analysis.
 *
 * This includes pitch, yaw, roll angles, landmarks, and their corresponding quality scores.
 */
struct FacePoseQualityAdaptResult {
    float pitch;                          ///< Pitch angle of the face.
    float yaw;                            ///< Yaw angle of the face.
    float roll;                           ///< Roll angle of the face.
    std::vector<inspirecv::Point2f> lmk;  ///< Landmarks of the face.
    std::vector<float> lmk_quality;       ///< Quality scores for each landmark.
};

/**
 * @class FacePoseQuality
 * @brief Class for assessing the quality of face pose using neural networks.
 *
 * Inherits from AnyNet and provides functionalities to compute face pose quality metrics.
 */
class INSPIRE_API FacePoseQualityAdapt : public AnyNetAdapter {
public:
    /**
     * @brief Default constructor for FacePoseQuality.
     */
    FacePoseQualityAdapt();

    /**
     * @brief Computes face pose quality for a given affine-transformed face image.
     * @param bgr_affine Affine-transformed face image in BGR format.
     * @return FacePoseQualityResult The computed face pose quality metrics.
     */
    FacePoseQualityAdaptResult operator()(const inspirecv::Image& img);

    /**
     * @brief Computes the affine transformation matrix for face cropping.
     * @param rect Rectangle representing the face in the image.
     * @return cv::Mat The computed affine transformation matrix.
     */
    static inspirecv::TransformMatrix ComputeCropMatrix(const inspirecv::Rect2i& rect);

public:
    const static int INPUT_WIDTH = 96;   ///< Width of the input image for the network.
    const static int INPUT_HEIGHT = 96;  ///< Height of the input image for the network.
};

}  // namespace inspire

#endif  // INSPIRE_FACE_TRACK_MODULE_QUALITY_FACE_POSE_QUALITY_ADAPT_H
