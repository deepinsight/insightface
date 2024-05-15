//
// Created by Tunm-Air13 on 2023/9/8.
//

#include "mask_predict.h"

namespace inspire {


MaskPredict::MaskPredict(): AnyNet("MaskPredict") {}

float MaskPredict::operator()(const Matrix &bgr_affine) {
    cv::Mat input;
    cv::resize(bgr_affine, input, cv::Size(m_input_size_, m_input_size_));
    AnyTensorOutputs outputs;
    Forward(input, outputs);

    return outputs[0].second[0];

}


}   // namespace hyper