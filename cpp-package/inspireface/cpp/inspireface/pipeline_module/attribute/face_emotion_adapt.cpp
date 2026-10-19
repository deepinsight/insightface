#include "face_emotion_adapt.h"

namespace inspire {

FaceEmotionAdapt::FaceEmotionAdapt() : AnyNetAdapter("FaceEmotionAdapt") {}

FaceEmotionAdapt::~FaceEmotionAdapt() {}

std::vector<float> FaceEmotionAdapt::operator()(const inspirecv::Image& bgr_affine) {
    AnyTensorOutputs outputs;
    if (bgr_affine.Width() != INPUT_WIDTH || bgr_affine.Height() != INPUT_HEIGHT) {
        auto resized = bgr_affine.Resize(INPUT_WIDTH, INPUT_HEIGHT);
        if (Forward(resized, outputs) != InferenceWrapper::WrapperOk) {
            return {};
        }
    } else {
        if (Forward(bgr_affine, outputs) != InferenceWrapper::WrapperOk) {
            return {};
        }
    }

    if (outputs.empty() || outputs[0].second.size() != static_cast<size_t>(OUTPUT_SIZE)) {
        return {};
    }

    std::vector<float> &emotionOut = outputs[0].second;
    auto sm = Softmax(emotionOut);
    return sm.size() == static_cast<size_t>(OUTPUT_SIZE) ? sm : std::vector<float>{};
}

}   // namespace inspire
