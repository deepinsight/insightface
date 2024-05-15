//
// Created by Tunm-Air13 on 2023/9/22.
//

#include "opencv2/opencv.hpp"

#include "inspireface/middleware/costman.h"
#include "inspireface/face_context.h"

using namespace inspire;

int main() {
    FaceContext ctx;
    CustomPipelineParameter param;
    int32_t ret = ctx.Configuration(
            "test_res/pack/Gundam_RV1109",
            DetectMode::DETECT_MODE_VIDEO,
            3,
            param);
    if (ret != HSUCCEED) {
        LOGE("Initiate error");
    }
    cv::Mat frame;
    std::string imageFolder = "test_res/video_frames/";

//    auto video_frame_num = 10;
    auto video_frame_num = 288;
    for (int i = 0; i < video_frame_num; ++i) {
        auto index = i + 1;
        std::stringstream frameFileName;
        frameFileName << imageFolder << "frame-" << std::setw(4) << std::setfill('0') << index << ".jpg";

        frame = cv::imread(frameFileName.str());

        CameraStream stream;
        stream.SetRotationMode(ROTATION_0);
        stream.SetDataFormat(BGR);
        stream.SetDataBuffer(frame.data, frame.rows, frame.cols);

        Timer timer;
        ctx.FaceDetectAndTrack(stream);
        LOGD("Cost: %f", timer.GetCostTimeUpdate());
        LOGD("faces: %d", ctx.GetNumberOfFacesCurrentlyDetected());

        LOGD("track id: %d", ctx.GetTrackingFaceList()[0].GetTrackingId());

        auto &face = ctx.GetTrackingFaceList()[0];
        for (auto &p: face.landmark_) {
            cv::circle(frame, p, 0, cv::Scalar(0, 0, 255), 3);
        }

        auto rect = face.GetRect();
        int track_id = face.GetTrackingId();
        int track_count = face.GetTrackingCount();

        cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 2, 1);

        std::string text = "ID: " + std::to_string(track_id) + " Count: " + std::to_string(track_count) + " Cf: " + std::to_string(face.GetConfidence());

        cv::Point text_position(rect.x, rect.y - 10);
        int font_face = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 0.5;
        int font_thickness = 1;
        cv::Scalar font_color(255, 255, 255);

        cv::putText(frame, text, text_position, font_face, font_scale, font_color, font_thickness);

        std::stringstream saveFile;
        saveFile << "track_frames/" << "result-" << std::setw(4) << std::setfill('0') << index << ".jpg";
        cv::imwrite(saveFile.str(), frame);
    }



    return 0;
}