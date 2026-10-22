/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "mask_predict_adapt.h"
#include "herror.h"
#include <cmath>
#include <limits>

namespace inspire {

MaskPredictAdapt::MaskPredictAdapt() : AnyNetAdapter("MaskPredictAdapt") {}

float MaskPredictAdapt::operator()(const inspirecv::Image& bgr_affine) {
    float score = std::numeric_limits<float>::quiet_NaN();
    Predict(bgr_affine, score);
    return score;
}

int32_t MaskPredictAdapt::Predict(const inspirecv::Image& bgr_affine, float& score) {
    score = std::numeric_limits<float>::quiet_NaN();
    AnyTensorOutputs outputs;
    inspirecv::Image resized;
    if (ResizeImageForInference(bgr_affine, m_input_size_, m_input_size_, resized) != InferenceWrapper::WrapperOk) {
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
    if (outputs.empty() || outputs[0].second.empty()) {
        return HERR_SESS_PIPELINE_FAILURE;
    }
#ifdef INFERENCE_WRAPPER_ENABLE_RKNN2
    auto sm = Softmax(outputs[0].second);
    if (sm.empty()) {
        return HERR_SESS_PIPELINE_FAILURE;
    }
    score = sm[0];
#else
    score = outputs[0].second[0];
#endif
    return std::isfinite(score) ? HSUCCEED : HERR_SESS_PIPELINE_FAILURE;
}

}  // namespace inspire
