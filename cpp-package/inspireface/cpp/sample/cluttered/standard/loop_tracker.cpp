//
// Created by tunm on 2024/4/6.
//
#include <iostream>
#include "track_module/face_track.h"
#include "inspireface/recognition_module/face_feature_extraction.h"
#include "log.h"

using namespace inspire;

int main() {
    InspireArchive archive("test_res/pack/Pikachu");

    FaceTrack track;
//    FaceRecognition recognition(archive, true);

    auto ret = track.Configuration(archive);
    INSPIRE_LOGD("ret=%d", ret);

    auto image = cv::imread("test_res/data/bulk/kun.jpg");
    for (int i = 0; i < 10000000; ++i) {
        CameraStream stream;
        stream.SetDataBuffer(image.data, image.rows, image.cols);
        stream.SetDataFormat(BGR);
        stream.SetRotationMode(ROTATION_0);

        track.UpdateStream(stream, true);
    }


//    InspireModel model;
//    ret = archive.LoadModel("mask_detect", model);
//    std::cout << ret << std::endl;
//
//    archive.PublicPrintSubFiles();

    return 0;
}