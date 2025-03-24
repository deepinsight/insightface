/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "blink_predict_adapt.h"
#include "middleware/utils.h"

namespace inspire {

BlinkPredictAdapt::BlinkPredictAdapt() : AnyNetAdapter("BlinkPredictAdapt") {}

float BlinkPredictAdapt::operator()(const inspirecv::Image &bgr_affine) {
    AnyTensorOutputs outputs;
    if (bgr_affine.Width() == BLINK_EYE_INPUT_SIZE && bgr_affine.Height() == BLINK_EYE_INPUT_SIZE) {
        auto input = bgr_affine.ToGray();
        Forward(input, outputs);
    } else {
        auto input = bgr_affine.ToGray();
        input = input.Resize(BLINK_EYE_INPUT_SIZE, BLINK_EYE_INPUT_SIZE);
        Forward(input, outputs);
    }
    auto &map = outputs[0].second;

    return map[1];
}

}  // namespace inspire