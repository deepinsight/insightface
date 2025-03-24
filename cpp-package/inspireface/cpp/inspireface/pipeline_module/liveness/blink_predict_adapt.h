/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#pragma once
#ifndef INSPIREFACE_BLINK_PREDICT_H
#define INSPIREFACE_BLINK_PREDICT_H
#include "data_type.h"
#include "middleware/any_net_adapter.h"

namespace inspire {

/**
 * @class BlinkPredict
 * @brief Prediction whether the eyes are open or closed.
 *
 * This class inherits from AnyNet and provides methods for performing blink prediction.
 */
class INSPIRE_API BlinkPredictAdapt : public AnyNetAdapter {
public:
    /**
     * @brief Constructor for MaskPredict class.
     */
    BlinkPredictAdapt();

    /**
     * @brief Operator for performing blink prediction on a BGR affine matrix.
     *
     * @param bgr_affine The BGR affine matrix to perform mask prediction on.
     * @return Blink prediction result.
     */
    float operator()(const inspirecv::Image& bgr_affine);

public:
    static const int BLINK_EYE_INPUT_SIZE = 64;  ///< Input size
};

}  // namespace inspire

#endif  // INSPIREFACE_BLINK_PREDICT_H
