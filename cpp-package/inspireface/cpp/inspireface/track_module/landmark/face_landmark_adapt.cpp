/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "face_landmark_adapt.h"
#include "cost_time.h"

namespace inspire {

std::vector<float> FaceLandmarkAdapt::operator()(const inspirecv::Image& bgr_affine) {
    COST_TIME_SIMPLE(FaceLandmarkAdapt);
    AnyTensorOutputs outputs;
    Forward(bgr_affine, outputs);
    auto& out = outputs[0].second;
    if (m_is_center_scaling_) {
        for (int i = 0; i < out.size(); ++i) {
            out[i] = (out[i] + 1) / 2;
        }
    }

    return out;
}

FaceLandmarkAdapt::FaceLandmarkAdapt(int input_size, bool is_center_scaling) 
    : AnyNetAdapter("FaceLandmarkAdapt"), m_input_size_(input_size), m_is_center_scaling_(is_center_scaling) {}

int FaceLandmarkAdapt::getInputSize() const {
    return m_input_size_;
}

}  //  namespace inspire