/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#pragma once
#ifndef INSPIRE_FACE_MASKPREDICT_ADAPT_H
#define INSPIRE_FACE_MASKPREDICT_ADAPT_H
#include "data_type.h"
#include "middleware/any_net_adapter.h"

namespace inspire {

/**
 * @class MaskPredict
 * @brief Class for performing mask prediction.
 *
 * This class inherits from AnyNet and provides methods for performing mask prediction.
 */
class INSPIRE_API MaskPredictAdapt : public AnyNetAdapter {
public:
    /**
     * @brief Constructor for MaskPredict class.
     */
    MaskPredictAdapt();

    /**
     * @brief Operator for performing mask prediction on a BGR affine matrix.
     *
     * @param bgr_affine The BGR affine matrix to perform mask prediction on.
     * @return float The mask prediction result.
     */
    float operator()(const inspirecv::Image& bgr_affine);

private:
    const int m_input_size_ = 96;  ///< The input size for the model.
};

}  // namespace inspire

#endif  // INSPIRE_FACE_MASKPREDICT_ADAPT_H
