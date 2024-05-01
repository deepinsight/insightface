//
// Created by Tunm-Air13 on 2023/9/8.
//

#include "face_pose.h"

namespace inspire {

FacePose::FacePose() : AnyNet("FacePose") {}

std::vector<float> FacePose::operator()(const Matrix& bgr_affine) {
    cv::Mat gray;
    cv::cvtColor(bgr_affine, gray, cv::COLOR_BGR2GRAY);
    AnyTensorOutputs outputs;
    Forward(gray, outputs);

    return outputs[0].second;
}

}   //  namespace hyper