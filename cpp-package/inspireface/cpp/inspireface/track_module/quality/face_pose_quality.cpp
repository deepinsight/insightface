//
// Created by Tunm-Air13 on 2023/9/15.
//

#include "face_pose_quality.h"
#include "middleware/utils.h"

namespace inspire {

FacePoseQuality::FacePoseQuality(): AnyNet("FacePoseQuality") {}

FacePoseQualityResult FacePoseQuality::operator()(const Matrix &bgr_affine) {
    FacePoseQualityResult res;
    AnyTensorOutputs outputs;
    Forward(bgr_affine, outputs);
    const auto &output = outputs[0].second;
    res.pitch = output[0] * 90;
    res.yaw = output[1] * 90;
    res.roll = output[2] * 90;
    std::vector<float> quality(output.begin() + 13, output.end());
    res.lmk_quality = quality;
    std::vector<float> face_pts5(output.begin() + 3, output.begin() + 13);
    res.lmk.resize(5);
    for (int i = 0; i < 5; i++) {
        res.lmk[i].x = (face_pts5[i * 2] + 1) * (INPUT_WIDTH / 2);
        res.lmk[i].y = (face_pts5[i * 2 + 1] + 1) * (INPUT_HEIGHT / 2);
    }

//    for (auto &p: res.lmk) {
//        cv::circle(bgr_affine, p, 0, cv::Scalar(0, 0, 255), 5);
//    }
//    cv::imshow("ww", bgr_affine);
//    cv::waitKey(0);

    return res;
}



}   // namespace hyper