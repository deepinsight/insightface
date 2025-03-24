/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "mask_predict_adapt.h"

namespace inspire {

MaskPredictAdapt::MaskPredictAdapt() : AnyNetAdapter("MaskPredictAdapt") {}

float MaskPredictAdapt::operator()(const inspirecv::Image& bgr_affine) {
    AnyTensorOutputs outputs;
    if (bgr_affine.Height() == m_input_size_ && bgr_affine.Width() == m_input_size_) {
        Forward(bgr_affine, outputs);

    } else {
        // auto resized = bgr_affine.Resize(m_input_size_, m_input_size_);
        uint8_t* resized_data = nullptr;
        m_processor_->Resize(bgr_affine.Data(), bgr_affine.Width(), bgr_affine.Height(), bgr_affine.Channels(), &resized_data, m_input_size_,
                             m_input_size_);
        auto resized = inspirecv::Image::Create(m_input_size_, m_input_size_, bgr_affine.Channels(), resized_data, false);
        Forward(resized, outputs);
    }
    m_processor_->MarkDone();
#ifdef INFERENCE_WRAPPER_ENABLE_RKNN2
    auto sm = Softmax(outputs[0].second);
    return sm[0];
#else
    return outputs[0].second[0];
#endif
}

}  // namespace inspire