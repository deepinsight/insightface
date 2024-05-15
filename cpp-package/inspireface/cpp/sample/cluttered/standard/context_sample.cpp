//
// Created by tunm on 2023/9/15.
//


#include <iostream>
#include "face_context.h"
#include "opencv2/opencv.hpp"
#include "sample/utils/test_helper.h"

using namespace inspire;

int main() {
#ifndef ISF_USE_MOBILE_OPENCV_IN_LOCAL
    FaceContext ctx;
    CustomPipelineParameter param;
    param.enable_liveness = true;
    param.enable_face_quality = true;
    int32_t ret = ctx.Configuration("test_res/pack/Pikachu-t1", DetectMode::DETECT_MODE_VIDEO, 1, param);
    if (ret != 0) {
        INSPIRE_LOGE("Initialization error");
        return -1;
    }
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Unable to open the camera." << std::endl;
        return -1;
    }

    cv::namedWindow("Webcam", cv::WINDOW_NORMAL);

    while (true) {
        cv::Mat frame;

        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Unable to obtain images from the camera." << std::endl;
            break;
        }

        CameraStream stream;
        stream.SetDataBuffer(frame.data, frame.rows, frame.cols);
        stream.SetDataFormat(BGR);
        stream.SetRotationMode(ROTATION_0);

        ctx.FaceDetectAndTrack(stream);

//        LOGD("Track Cost: %f", ctx.GetTrackTotalUseTime());

        auto &faces = ctx.GetTrackingFaceList();
        for (auto &face: faces) {
            auto rect = face.GetRect();
            int track_id = face.GetTrackingId();
            int track_count = face.GetTrackingCount();

            cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 2, 1);

            std::string text = "ID: " + std::to_string(track_id) + " Count: " + std::to_string(track_count);

            cv::Point text_position(rect.x, rect.y - 10);

            const auto& pose_and_quality = face.high_result;
            float mean_quality = 0.0f;
            for (int i = 0; i < pose_and_quality.lmk_quality.size(); ++i) {
                mean_quality += pose_and_quality.lmk_quality[i];
            }
            mean_quality /= pose_and_quality.lmk_quality.size();
            mean_quality = 1 - mean_quality;
            std::string pose_text = "pitch: " + std::to_string(pose_and_quality.pitch) + ",Yaw: " + std::to_string(pose_and_quality.yaw) + ",roll:" +std::to_string(pose_and_quality.roll) + ", q: " +
                    std::to_string(mean_quality);

            cv::Point pose_position(rect.x, rect.y + rect.height + 20);


            int font_face = cv::FONT_HERSHEY_SIMPLEX;
            double font_scale = 0.5;
            int font_thickness = 1;
            cv::Scalar font_color(255, 255, 255);

            cv::putText(frame, text, text_position, font_face, font_scale, font_color, font_thickness);
            cv::putText(frame, pose_text, pose_position, font_face, font_scale, font_color, font_thickness);
        }


        cv::imshow("Webcam", frame);

        if (cv::waitKey(1) == 27) {
            break;
        }
    }
    cap.release();

    cv::destroyAllWindows();
#endif
    return 0;
}