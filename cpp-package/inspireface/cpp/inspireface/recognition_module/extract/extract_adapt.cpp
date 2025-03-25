/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "extract_adapt.h"

namespace inspire {

Embedded ExtractAdapt::GetFaceFeature(const inspirecv::Image &bgr_affine) {
    AnyTensorOutputs outputs;
    Forward(bgr_affine, outputs);

    return outputs[0].second;
}

Embedded ExtractAdapt::operator()(const inspirecv::Image &bgr_affine, float &norm, bool normalize) {
    AnyTensorOutputs outputs;
    Forward(bgr_affine, outputs);

    auto &embedded = outputs[0].second;
    float mse = 0.0f;
    for (const auto &one : embedded) {
        mse += one * one;
    }
    mse = sqrt(mse);
    norm = mse;

    if (normalize) {
        for (float &one : embedded) {
            one /= mse;
        }
    }

    return embedded;
}

ExtractAdapt::ExtractAdapt() : AnyNetAdapter("ExtractAdapt") {}

}  // namespace inspire