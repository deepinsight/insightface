/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "rnet_adapt.h"

namespace inspire {

float RNetAdapt::operator()(const inspirecv::Image &bgr_affine) {
    inspirecv::Image resized;
    if (ResizeImageForInference(bgr_affine, 24, 24, resized) != InferenceWrapper::WrapperOk) {
        // Some RK devices seem unable to resize to 24x24, fallback to CPU processing
        m_processor_->MarkDone();
        resized = bgr_affine.Resize(24, 24);
        if (resized.Empty()) {
            return -1.0f;
        }
    }

    AnyTensorOutputs outputs;
    if (Forward(resized, outputs) != InferenceWrapper::WrapperOk || outputs.empty() || outputs[0].second.size() < 2) {
        m_processor_->MarkDone();
        return -1.0f;
    }
    if (m_processor_->MarkDone() != 0) {
        return -1.0f;
    }
#ifdef INFERENCE_WRAPPER_ENABLE_RKNN2
    auto sm = Softmax(outputs[0].second);
    if (sm.size() < 2) {
        return -1.0f;
    }
    return sm[1];
#else
    return outputs[0].second[1];
    // std::cout << outputs[0].second[0] << ", " << outputs[0].second[1] << std ::endl;
#endif
}

RNetAdapt::RNetAdapt() : AnyNetAdapter("RNetAdapt") {}

}  //  namespace inspire
