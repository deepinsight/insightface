//
// Created by Tunm-Air13 on 2023/9/8.
//
#pragma once
#ifndef HYPERFACEREPO_FACEPOSE_H
#define HYPERFACEREPO_FACEPOSE_H

#include "../../data_type.h"
#include "middleware/any_net.h"

namespace inspire {

/**
 * @class FacePose
 * @brief Class for facial pose estimation, inheriting from AnyNet.
 *
 * This class provides functionalities for estimating the pose of a face in an image,
 * such as orientation in terms of pitch, yaw, and roll angles.
 */
class INSPIRE_API FacePose : public AnyNet {
public:
    /**
     * @brief Constructor for the FacePose class.
     */
    explicit FacePose();

    /**
     * @brief Operator to process an affine-transformed face image and return facial pose information.
     * @param bgr_affine Affine-transformed face image in BGR format.
     * @return std::vector<float> Vector containing the facial pose information (pitch, yaw, roll).
     */
    std::vector<float> operator()(const Matrix& bgr_affine);

};

}   // namespace hyper

#endif //HYPERFACEREPO_FACEPOSE_H
