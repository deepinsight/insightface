/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "rnet_adapt.h"

namespace inspire {

float RNetAdapt::operator()(const inspirecv::Image &bgr_affine) {
    // auto resized = bgr_affine.Resize(24, 24);
    uint8_t *resized_data = nullptr;
    float scale;
    auto ret = m_processor_->Resize(bgr_affine.Data(), bgr_affine.Width(), bgr_affine.Height(), bgr_affine.Channels(), &resized_data, 24, 24);
    inspirecv::Image resized;
    if (ret == -1) {
        // Some RK devices seem unable to resize to 24x24, fallback to CPU processing
        resized = bgr_affine.Resize(24, 24);
    } else {
        // RGA resize success
        resized = inspirecv::Image::Create(24, 24, bgr_affine.Channels(), resized_data, false);
    }

    AnyTensorOutputs outputs;
    Forward(resized, outputs);
    m_processor_->MarkDone();
#ifdef INFERENCE_WRAPPER_ENABLE_RKNN2
    auto sm = Softmax(outputs[0].second);
    return sm[1];
#else
    return outputs[0].second[1];
    // std::cout << outputs[0].second[0] << ", " << outputs[0].second[1] << std ::endl;
#endif
}

RNetAdapt::RNetAdapt() : AnyNetAdapter("RNetAdapt") {}

}  //  namespace inspire