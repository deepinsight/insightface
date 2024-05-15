//
// Created by tunm on 2023/8/29.
//
#include <iostream>
#include "inspireface/track_module/face_track.h"
#include "opencv2/opencv.hpp"

using namespace inspire;

int video_test(FaceTrack &ctx, int cam_id) {
#ifndef ISF_USE_MOBILE_OPENCV_IN_LOCAL
    cv::VideoCapture cap(cam_id);

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

        ctx.UpdateStream(stream, false);

        INSPIRE_LOGD("Track Cost: %f", ctx.GetTrackTotalUseTime());

        auto const &faces = ctx.trackingFace;
        for (auto const &face: faces) {
            auto rect = face.GetRect();
            int track_id = face.GetTrackingId();
            int track_count = face.GetTrackingCount();

            cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 2, 1);

            std::string text = "ID: " + std::to_string(track_id) + " Count: " + std::to_string(track_count);

            cv::Point text_position(rect.x, rect.y - 10);

            const auto& pose_and_quality = face.high_result;
            std::vector<float> euler = {pose_and_quality.yaw, pose_and_quality.roll, pose_and_quality.pitch};
            std::string pose_text = "P: " + std::to_string(euler[0]) + ",Yaw: " + std::to_string(euler[1]) + ",roll:" +std::to_string(euler[2]);

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

void video_file_test(FaceTrack& ctx, const std::string& video_filename) {
#ifndef ISF_USE_MOBILE_OPENCV_IN_LOCAL
    cv::VideoCapture cap(video_filename);

    if (!cap.isOpened()) {
        std::cerr << "Unable to open the video file: " << video_filename << std::endl;
        return;
    }

    cv::namedWindow("Video", cv::WINDOW_NORMAL);

    while (true) {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Unable to get frames from the video file." << std::endl;
            break;
        }

        CameraStream stream;
        stream.SetDataBuffer(frame.data, frame.rows, frame.cols);
        stream.SetDataFormat(BGR);
        stream.SetRotationMode(ROTATION_0);

        ctx.UpdateStream(stream, false);
        INSPIRE_LOGD("Track Cost: %f", ctx.GetTrackTotalUseTime());

        auto const &faces = ctx.trackingFace;
        for (auto const &face: faces) {
            auto rect = face.GetRect();
            int track_id = face.GetTrackingId();
            int track_count = face.GetTrackingCount();

            cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 2, 1);

            auto lmk = face.GetLanmdark();
            for (auto & p : lmk) {
                cv::circle(frame, p, 0, cv::Scalar(0, 0, 242), 2);
            }

            std::string text = "ID: " + std::to_string(track_id) + " Count: " + std::to_string(track_count);

            cv::Point text_position(rect.x, rect.y - 10);

            const auto& euler = face.high_result;
            std::string pose_text = "pitch: " + std::to_string(euler.pitch) + ",Yaw: " + std::to_string(euler.yaw) + ",roll:" +std::to_string(euler.roll);

            cv::Point pose_position(rect.x, rect.y + rect.height + 20);

            int font_face = cv::FONT_HERSHEY_SIMPLEX;
            double font_scale = 0.5;
            int font_thickness = 1;
            cv::Scalar font_color(255, 255, 255);

            cv::putText(frame, text, text_position, font_face, font_scale, font_color, font_thickness);
            cv::putText(frame, pose_text, pose_position, font_face, font_scale, font_color, font_thickness);
        }

        cv::imshow("Video", frame);

        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
#endif
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <source> <input>" << std::endl;
        return 1;
    }

    INSPIRE_SET_LOG_LEVEL(LogLevel::LOG_NONE);

    const std::string source = argv[1];
    const std::string input = argv[2];

    const std::string folder = "test_res/pack/Pikachu";
    INSPIRE_LOGD("%s", folder.c_str());
//    ModelLoader loader;
//    loader.Reset(folder);

    InspireArchive archive;
    archive.ReLoad(folder);
    std::cout << archive.QueryStatus() << std::endl;
    if (archive.QueryStatus() != SARC_SUCCESS) {
        INSPIRE_LOGE("error archive");
        return -1;
    }

    FaceTrack ctx;
    ctx.Configuration(archive);

    if (source == "webcam") {
        int cam_id = std::stoi(input);
        video_test(ctx, cam_id);
    } else if (source == "image") {
        cv::Mat image = cv::imread(input);
        if (!image.empty()) {
//            image_test(ctx, image);
        } else {
            std::cerr << "Unable to open the image file." << std::endl;
        }
    } else if (source == "video") {
        video_file_test(ctx, input);
    } else {
        std::cerr << "Invalid input source: " << source << std::endl;
        return 1;
    }

    return 0;
}