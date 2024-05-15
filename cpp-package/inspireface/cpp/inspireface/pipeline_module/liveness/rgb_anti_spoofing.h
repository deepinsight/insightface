//
// Created by Tunm-Air13 on 2023/9/8.
//
#pragma once
#ifndef HYPERFACEREPO_RBGANTISPOOFING_H
#define HYPERFACEREPO_RBGANTISPOOFING_H
#include "data_type.h"
#include "middleware/any_net.h"

namespace inspire {

/**
 * @class RBGAntiSpoofing
 * @brief Class for performing RGB anti-spoofing.
 *
 * This class inherits from AnyNet and provides methods for performing RGB anti-spoofing.
 */
class INSPIRE_API RBGAntiSpoofing : public AnyNet {
public:
    /**
     * @brief Constructor for RBGAntiSpoofing class.
     *
     * @param input_size The input size for the model (default is 112).
     * @param use_softmax Whether to use softmax activation (default is false).
     */
    RBGAntiSpoofing(int input_size = 112, bool use_softmax = false);

    /**
     * @brief Operator for performing RGB anti-spoofing on a BGR affine27 matrix.
     *
     * @param bgr_affine27 The BGR affine27 matrix to perform anti-spoofing on.
     * @return float The anti-spoofing result.
     */
    float operator()(const Matrix& bgr_affine27);

private:
    int m_input_size_; ///< The input size for the model.
    bool m_softmax_ = false; ///< Whether to use softmax activation.
};

}   // namespace hyper

#endif //HYPERFACEREPO_RBGANTISPOOFING_H
