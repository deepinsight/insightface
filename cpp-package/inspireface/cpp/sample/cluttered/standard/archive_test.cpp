//
// Created by tunm on 2024/4/6.
//
#include <iostream>
#include "track_module/face_track.h"
#include "inspireface/feature_hub/face_recognition.h"
#include "log.h"
#include "track_module/face_track.h"
#include "pipeline_module/face_pipeline.h"
#include "inspireface/feature_hub/face_recognition.h"
#include "middleware/inference_helper/customized/rknn_adapter.h"

using namespace inspire;

int main() {
    InspireArchive archive;
    auto ret = archive.ReLoad("test_res/pack/Gundam_RV1109");
    LOGD("ReLoad %d", ret);
//    InspireModel model;
//    ret = archive.LoadModel("mask_detect", model);
//    LOGD("LoadModel %d", ret);

    FaceTrack track;
    ret = track.Configuration(archive);
    LOGD("Configuration %d", ret);

    FacePipeline pipeline(archive, true, true, true, true, true);

    FaceRecognition recognition(archive, true);

//    std::shared_ptr<RKNNAdapter> rknet = std::make_shared<RKNNAdapter>();
//    ret = rknet->Initialize((unsigned char* )model.buffer, model.bufferSize);
//
//    LOGD("LoadModel %d", ret);


    return 0;
}