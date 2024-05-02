//
// Created by tunm on 2023/9/7.
//

#include "extract.h"

namespace inspire {

Embedded Extract::GetFaceFeature(const Matrix& bgr_affine) {
    AnyTensorOutputs outputs;
    Forward(bgr_affine, outputs);

    return outputs[0].second;
}

Embedded Extract::operator()(const Matrix &bgr_affine) {

    AnyTensorOutputs outputs;
    Forward(bgr_affine, outputs);

    auto &embedded = outputs[0].second;
    float mse = 0.0f;
    for (const auto &one: embedded) {
        mse += one * one;
    }
    mse = sqrt(mse);
    for (float &one : embedded) {
        one /= mse;
    }

    return embedded;
}


Extract::Extract() : AnyNet("Extract") {}

}   // namespace hyper