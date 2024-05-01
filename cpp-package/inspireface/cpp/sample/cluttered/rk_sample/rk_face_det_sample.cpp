//
// Created by Tunm-Air13 on 2023/9/20.
//
#include "opencv2/opencv.hpp"
//#include "inspireface/middleware/model_loader/model_loader.h"
#include "inspireface/track_module/face_detect/all.h"

#include "inspireface/middleware/costman.h"
#include "middleware/model_archive/inspire_archive.h"
#include "log.h"

using namespace inspire;

int main() {
    auto detModel = "test_res/pack/Gundam_RV1109";
    InspireArchive inspireArchive;
    auto ret = inspireArchive.ReLoad(detModel);
    if (ret != SARC_SUCCESS) {
        LOGE("Error load");
        return ret;
    }
    InspireModel model;
    ret = inspireArchive.LoadModel("face_detect", model);
    if (ret != SARC_SUCCESS) {
        LOGE("Error model");
        return ret;
    }

    std::cout << model.Config().toString() << std::endl;

    std::shared_ptr<FaceDetect> m_face_detector_;
    m_face_detector_ = std::make_shared<FaceDetect>(320);
    m_face_detector_->loadData(model, InferenceHelper::kRknn);


    // Load a image
    cv::Mat image = cv::imread("test_res/images/face_sample.png");

    Timer timer;
    FaceLocList locs = (*m_face_detector_)(image);
    LOGD("cost: %f", timer.GetCostTimeUpdate());

    LOGD("Faces: %ld", locs.size());

    for (auto &loc: locs) {
        cv::rectangle(image, cv::Point2f(loc.x1, loc.y1), cv::Point2f(loc.x2, loc.y2), cv::Scalar(0, 0, 255), 3);
    }
    cv::imwrite("det.jpg", image);


    return 0;
}