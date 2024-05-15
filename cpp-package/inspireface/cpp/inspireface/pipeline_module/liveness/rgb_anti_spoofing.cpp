//
// Created by Tunm-Air13 on 2023/9/8.
//

#include "rgb_anti_spoofing.h"

namespace inspire {


std::vector<float> softmax(const std::vector<float>& input) {
    std::vector<float> result;
    float sum = 0.0;

    // Calculate the exponentials and the sum of exponentials
    for (float x : input) {
        float exp_x = std::exp(x);
        result.push_back(exp_x);
        sum += exp_x;
    }

    // Normalize by dividing each element by the sum
    for (float& value : result) {
        value /= sum;
    }

    return result;
}

RBGAntiSpoofing::RBGAntiSpoofing(int input_size, bool use_softmax): AnyNet("RBGAntiSpoofing") {
    m_input_size_ = input_size;
    m_softmax_ = use_softmax;
}


float RBGAntiSpoofing::operator()(const Matrix &bgr_affine27) {
    cv::Mat input;
    cv::resize(bgr_affine27, input, cv::Size(m_input_size_, m_input_size_));
//    cv::imwrite("bb.png", input);
    AnyTensorOutputs outputs;
    Forward(input, outputs);
    if (m_softmax_) {
        auto sm = softmax(outputs[0].second);
        return sm[1];
    } else {
        return outputs[0].second[1];
    }
}


}