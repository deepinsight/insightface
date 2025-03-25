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
    const auto& out = outputs[0].second;

    return out;
}

FaceLandmarkAdapt::FaceLandmarkAdapt(int input_size) : AnyNetAdapter("FaceLandmarkAdapt"), m_input_size_(input_size) {}

int FaceLandmarkAdapt::getInputSize() const {
    return m_input_size_;
}

}  //  namespace inspire