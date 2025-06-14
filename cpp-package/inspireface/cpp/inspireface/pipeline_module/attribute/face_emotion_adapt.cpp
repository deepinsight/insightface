#include "face_emotion_adapt.h"

namespace inspire {

FaceEmotionAdapt::FaceEmotionAdapt() : AnyNetAdapter("FaceEmotionAdapt") {}

FaceEmotionAdapt::~FaceEmotionAdapt() {}

std::vector<float> FaceEmotionAdapt::operator()(const inspirecv::Image& bgr_affine) {
    AnyTensorOutputs outputs;
    if (bgr_affine.Width() != INPUT_WIDTH || bgr_affine.Height() != INPUT_HEIGHT) {
        auto resized = bgr_affine.Resize(INPUT_WIDTH, INPUT_HEIGHT);
        Forward(resized, outputs);
    } else {
        Forward(bgr_affine, outputs);
    }

    std::vector<float> &emotionOut = outputs[0].second;
    auto sm = Softmax(emotionOut);

    return sm;
}

}   // namespace inspire
