//
// Created by Tunm-Air13 on 2023/9/8.
//
#pragma once
#ifndef HYPERFACEREPO_BLINK_PREDICT_H
#define HYPERFACEREPO_BLINK_PREDICT_H
#include "data_type.h"
#include "middleware/any_net.h"

namespace inspire {

/**
 * @class BlinkPredict
 * @brief Prediction whether the eyes are open or closed.
 *
 * This class inherits from AnyNet and provides methods for performing blink prediction.
 */
class INSPIRE_API BlinkPredict : public AnyNet {
public:
    /**
     * @brief Constructor for MaskPredict class.
     */
    BlinkPredict();

    /**
     * @brief Operator for performing blink prediction on a BGR affine matrix.
     *
     * @param bgr_affine The BGR affine matrix to perform mask prediction on.
     * @return Blink prediction result.
     */
    float operator()(const Matrix& bgr_affine);

public:

    static const int BLINK_EYE_INPUT_SIZE = 64; ///< Input size

};

} // namespace inspire

#endif //HYPERFACEREPO_BLINK_PREDICT_H
