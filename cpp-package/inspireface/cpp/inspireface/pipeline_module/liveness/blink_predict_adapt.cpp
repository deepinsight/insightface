/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "blink_predict_adapt.h"
#include "middleware/utils.h"
#include <limits>

namespace inspire {

BlinkPredictAdapt::BlinkPredictAdapt() : AnyNetAdapter("BlinkPredictAdapt") {}

float BlinkPredictAdapt::operator()(const inspirecv::Image &bgr_affine) {
    AnyTensorOutputs outputs;
    if (bgr_affine.Width() == BLINK_EYE_INPUT_SIZE && bgr_affine.Height() == BLINK_EYE_INPUT_SIZE) {
        auto input = bgr_affine.ToGray();
        if (Forward(input, outputs) != InferenceWrapper::WrapperOk) {
            return std::numeric_limits<float>::quiet_NaN();
        }
    } else {
        auto input = bgr_affine.ToGray();
        input = input.Resize(BLINK_EYE_INPUT_SIZE, BLINK_EYE_INPUT_SIZE);
        if (Forward(input, outputs) != InferenceWrapper::WrapperOk) {
            return std::numeric_limits<float>::quiet_NaN();
        }
    }
    if (outputs.empty() || outputs[0].second.size() < 2) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    auto &map = outputs[0].second;

    return map[1];
}

}  // namespace inspire
