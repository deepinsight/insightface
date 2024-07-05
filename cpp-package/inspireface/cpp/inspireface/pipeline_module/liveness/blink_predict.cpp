//
// Created by Tunm-Air13 on 2023/9/8.
//

#include "blink_predict.h"
#include "middleware/utils.h"

namespace inspire {

BlinkPredict::BlinkPredict(): AnyNet("BlinkPredict") {}

float BlinkPredict::operator()(const Matrix &bgr_affine) {
    cv::Mat input;
    if (bgr_affine.cols == BLINK_EYE_INPUT_SIZE && bgr_affine.rows == BLINK_EYE_INPUT_SIZE)
    {
        input = bgr_affine;
    } else {
        cv::resize(bgr_affine, input, cv::Size(BLINK_EYE_INPUT_SIZE, BLINK_EYE_INPUT_SIZE));
    }
    cv::cvtColor(input, input, cv::COLOR_BGR2GRAY);
    AnyTensorOutputs outputs;
    Forward(input, outputs);
    auto &map = outputs[0].second;;
    
    return map[1];
}

} // namespace inspire