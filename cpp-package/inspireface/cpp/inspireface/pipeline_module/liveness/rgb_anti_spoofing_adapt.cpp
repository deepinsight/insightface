/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "rgb_anti_spoofing_adapt.h"
#include "herror.h"
#include <cmath>
#include <limits>

namespace inspire {

RBGAntiSpoofingAdapt::RBGAntiSpoofingAdapt(int input_size, bool use_softmax) : AnyNetAdapter("RBGAntiSpoofingAdapt") {
    m_input_size_ = input_size;
    m_softmax_ = use_softmax;
}

float RBGAntiSpoofingAdapt::operator()(const inspirecv::Image& bgr_affine27) {
    float score = std::numeric_limits<float>::quiet_NaN();
    Predict(bgr_affine27, score);
    return score;
}

int32_t RBGAntiSpoofingAdapt::Predict(const inspirecv::Image& bgr_affine27, float& score) {
    score = std::numeric_limits<float>::quiet_NaN();
    AnyTensorOutputs outputs;
    inspirecv::Image resized;
    if (ResizeImageForInference(bgr_affine27, m_input_size_, m_input_size_, resized) != InferenceWrapper::WrapperOk) {
        m_processor_->MarkDone();
        return HERR_DEVICE_IMAGE_PROCESS_FAILURE;
    }
    if (Forward(resized, outputs) != InferenceWrapper::WrapperOk) {
        m_processor_->MarkDone();
        return HERR_SESS_PIPELINE_FAILURE;
    }
    if (m_processor_->MarkDone() != 0) {
        return HERR_DEVICE_IMAGE_PROCESS_FAILURE;
    }
    if (outputs.empty() || outputs[0].second.size() < 2) {
        return HERR_SESS_PIPELINE_FAILURE;
    }
    if (m_softmax_) {
        auto sm = Softmax(outputs[0].second);
        if (sm.size() < 2) {
            return HERR_SESS_PIPELINE_FAILURE;
        }
        score = sm[1];
    } else {
        score = outputs[0].second[1];
    }
    return std::isfinite(score) ? HSUCCEED : HERR_SESS_PIPELINE_FAILURE;
}

}  // namespace inspire
