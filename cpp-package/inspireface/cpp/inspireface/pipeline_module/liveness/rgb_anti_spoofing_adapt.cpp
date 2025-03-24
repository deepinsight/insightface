/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "rgb_anti_spoofing_adapt.h"

namespace inspire {

RBGAntiSpoofingAdapt::RBGAntiSpoofingAdapt(int input_size, bool use_softmax) : AnyNetAdapter("RBGAntiSpoofingAdapt") {
    m_input_size_ = input_size;
    m_softmax_ = use_softmax;
}

float RBGAntiSpoofingAdapt::operator()(const inspirecv::Image& bgr_affine27) {
    AnyTensorOutputs outputs;
    if (bgr_affine27.Width() != m_input_size_ || bgr_affine27.Height() != m_input_size_) {
        // auto resized = bgr_affine27.Resize(m_input_size_, m_input_size_);
        uint8_t* resized_data = nullptr;
        float scale;
        m_processor_->Resize(bgr_affine27.Data(), bgr_affine27.Width(), bgr_affine27.Height(), bgr_affine27.Channels(), &resized_data, m_input_size_,
                             m_input_size_);
        auto resized = inspirecv::Image::Create(m_input_size_, m_input_size_, bgr_affine27.Channels(), resized_data, false);
        Forward(resized, outputs);
    } else {
        Forward(bgr_affine27, outputs);
    }
    if (m_softmax_) {
        auto sm = Softmax(outputs[0].second);
        return sm[1];
    } else {
        return outputs[0].second[1];
    }
}

}  // namespace inspire