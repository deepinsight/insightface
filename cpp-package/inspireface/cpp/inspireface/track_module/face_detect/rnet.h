//
// Created by Tunm-Air13 on 2023/9/6.
//
#pragma once
#ifndef HYPERFACEREPO_RNET_H
#define HYPERFACEREPO_RNET_H
#include "../../data_type.h"
#include "middleware/any_net.h"

namespace inspire {

/**
 * @class RNet
 * @brief Class representing the RNet (Refinement Network), inheriting from AnyNet.
 *
 * This class is used for refining face detection results, typically as part of a cascaded
 * network system for facial recognition or detection tasks.
 */
class INSPIRE_API RNet: public AnyNet {
public:
    /**
     * @brief Constructor for the RNet class.
     */
    RNet();

    /**
     * @brief Operator to process an affine-transformed face image and return a score indicating the quality of the refinement.
     * @param bgr_affine Affine-transformed face image in BGR format.
     * @return float Score representing the quality or confidence of the refinement.
     */
    float operator()(const Matrix& bgr_affine);

};

}   //  namespace hyper

#endif //HYPERFACEREPO_RNET_H
