//
// Created by tunm on 2023/9/8.
//
#include <iostream>
#include "track_module/face_detect/face_pose.h"

#include "middleware/model_archive/inspire_archive.h"

using namespace inspire;

int main(int argc, char** argv) {
    InspireArchive loader;
    loader.ReLoad("resource/pack/Pikachu");

    Configurable param;
    param.set<std::string>("input_layer", "data");
    param.set<std::vector<std::string>>("outputs_layers", {"ip3_pose", });
    param.set<std::vector<int>>("input_size", {112, 112});
    param.set<std::vector<float>>("mean", {0.0f, 0.0f, 0.0f});
    param.set<std::vector<float>>("norm", {1.0f, 1.0f, 1.0f});
    param.set<int>("input_channel", 1);        // Input Gray
    param.set<int>("input_image_channel", 1);        // BGR 2 Gray

    auto m_pose_net_ = std::make_shared<FacePose>();
    InspireModel model;
    loader.LoadModel("", model);
    m_pose_net_->loadData(model);

    auto image = cv::imread("resource/images/crop.png");

    cv::Mat gray;
    cv::resize(image, gray, cv::Size(112, 112));

    auto res = (*m_pose_net_)(gray);
    INSPIRE_LOGD("%f", res[0]);
    INSPIRE_LOGD("%f", res[1]);
    INSPIRE_LOGD("%f", res[2]);

    return 0;
}