//
// Created by Tunm-Air13 on 2023/9/8.
//
#pragma once
#ifndef HYPERFACEREPO_MASKPREDICT_H
#define HYPERFACEREPO_MASKPREDICT_H
#include "data_type.h"
#include "middleware/any_net.h"

namespace inspire {

/**
 * @class MaskPredict
 * @brief Class for performing mask prediction.
 *
 * This class inherits from AnyNet and provides methods for performing mask prediction.
 */
class INSPIRE_API MaskPredict : public AnyNet {
public:
    /**
     * @brief Constructor for MaskPredict class.
     */
    MaskPredict();

    /**
     * @brief Operator for performing mask prediction on a BGR affine matrix.
     *
     * @param bgr_affine The BGR affine matrix to perform mask prediction on.
     * @return float The mask prediction result.
     */
    float operator()(const Matrix& bgr_affine);

private:
    const int m_input_size_ = 96; ///< The input size for the model.
};

}   // namespace hyper

#endif //HYPERFACEREPO_MASKPREDICT_H
