//
// Created by tunm on 2023/9/7.
//
#pragma once
#ifndef HYPERFACEREPO_EXTRACT_H
#define HYPERFACEREPO_EXTRACT_H
#include "data_type.h"
#include "middleware/any_net.h"

namespace inspire {

/**
 * @class Extract
 * @brief Class for extracting features from faces, inheriting from AnyNet.
 *
 * This class specializes in processing face images to extract embedded facial features,
 * which can be used for further analysis like face recognition or verification.
 */
class INSPIRE_API Extract: public AnyNet {
public:
    /**
     * @brief Constructor for the Extract class.
     */
    Extract();

    /**
     * @brief Operator to process an affine-transformed face image and return the extracted features.
     * @param bgr_affine Affine-transformed face image in BGR format.
     * @return Embedded Vector of extracted features.
     */
    Embedded operator()(const Matrix& bgr_affine);

    /**
     * @brief Gets the facial features from an affine-transformed face image.
     * @param bgr_affine Affine-transformed face image in BGR format.
     * @return Embedded Vector of extracted facial features.
     */
    Embedded GetFaceFeature(const Matrix& bgr_affine);

};

}   // namespace hyper

#endif //HYPERFACEREPO_EXTRACT_H
