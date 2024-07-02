//
// Created by tunm on 2024/5/26.
//
#include <cstddef>
#include <iostream>
#include <opencv2/core/types.hpp>
#ifndef DISABLE_GUI
#include <opencv2/highgui.hpp>
#endif
#include <opencv2/imgproc.hpp>
#include <vector>
#include "data_type.h"
#include "opencv2/opencv.hpp"
#include "inspireface/track_module/face_detect/all.h"
#include "inspireface/Initialization_module/launch.h"

using namespace inspire;

int main() {
    INSPIRE_LAUNCH->Load("test_res/pack/Megatron");
    auto archive = INSPIRE_LAUNCH->getMArchive();
    InspireModel detModel;
    auto ret = archive.LoadModel("face_detect", detModel);

    std::vector<int> input_size = {640, 640};
    detModel.Config().set<std::vector<int>>("input_size", input_size);

    FaceDetect detect(input_size[0]);
    detect.loadData(detModel, detModel.modelType, true);

    auto img = cv::imread("/Users/tunm/Downloads/xtl.png");

    double time;
    time = (double) cv::getTickCount();
    std::vector<FaceLoc> results = detect(img);
    time = ((double) cv::getTickCount() - time) / cv::getTickFrequency();
    std::cout << "use time：" << time << "秒\n";

    for (size_t i = 0; i < results.size(); i++) {
        auto &item = results[i];
        cv::rectangle(img, cv::Point2f(item.x1, item.y1), cv::Point2f(item.x2, item.y2), cv::Scalar(0, 0, 255), 4);
    }
#ifndef DISABLE_GUI
    cv::imshow("w", img);
    cv::waitKey(0);
#endif

    return 0;
}