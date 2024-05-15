//
// Created by tunm on 2023/9/10.
//

#include <iostream>
#include "face_context.h"
#include "sample/utils/test_helper.h"
#include "inspireface/recognition_module/extract/alignment.h"
#include "recognition_module/face_feature_extraction.h"
#include "feature_hub/feature_hub.h"

using namespace inspire;

std::string GetFileNameWithoutExtension(const std::string& filePath) {
    size_t slashPos = filePath.find_last_of("/\\");
    if (slashPos != std::string::npos) {
        std::string fileName = filePath.substr(slashPos + 1);

        size_t dotPos = fileName.find_last_of('.');
        if (dotPos != std::string::npos) {
            return fileName.substr(0, dotPos);
        } else {
            return fileName;
        }
    }

    size_t dotPos = filePath.find_last_of('.');
    if (dotPos != std::string::npos) {
        return filePath.substr(0, dotPos);
    }

    return filePath;
}

int comparison1v1(FaceContext &ctx) {
    Embedded feature_1;
    Embedded feature_2;

    {
        auto image = cv::imread("");
        cv::Mat rot90;
        TestUtils::rotate(image, rot90, ROTATION_90);

        CameraStream stream;
        stream.SetDataFormat(BGR);
        stream.SetRotationMode(ROTATION_90);
        stream.SetDataBuffer(rot90.data, rot90.rows, rot90.cols);
        ctx.FaceDetectAndTrack(stream);
        const auto &faces = ctx.GetTrackingFaceList();
        if (faces.empty()) {
            INSPIRE_LOGD("image1 not face");
            return -1;
        }
        ctx.FaceRecognitionModule()->FaceExtract(stream, faces[0], feature_1);

    }

    {
        auto image = cv::imread("");
        CameraStream stream;
        stream.SetDataFormat(BGR);
        stream.SetRotationMode(ROTATION_0);
        stream.SetDataBuffer(image.data, image.rows, image.cols);
        ctx.FaceDetectAndTrack(stream);
        const auto &faces = ctx.GetTrackingFaceList();
        if (faces.empty()) {
            INSPIRE_LOGD("image1 not face");
            return -1;
        }
        ctx.FaceRecognitionModule()->FaceExtract(stream, faces[0], feature_2);

    }

    float rec;
    auto ret = FEATURE_HUB->CosineSimilarity(feature_1, feature_2, rec);
    INSPIRE_LOGD("rec: %f", rec);

    return 0;
}


int search(FaceContext &ctx) {

//    std::shared_ptr<FeatureBlock> block;
//    block.reset(FeatureBlock::Create(hyper::MC_OPENCV));

    std::vector<String> files_list = {
    };
    for (int i = 0; i < files_list.size(); ++i) {
        auto image = cv::imread(files_list[i]);
        CameraStream stream;
        stream.SetDataFormat(BGR);
        stream.SetRotationMode(ROTATION_0);
        stream.SetDataBuffer(image.data, image.rows, image.cols);
        ctx.FaceDetectAndTrack(stream);
        const auto &faces = ctx.GetTrackingFaceList();
        if (faces.empty()) {
            INSPIRE_LOGD("image1 not face");
            return -1;
        }
        Embedded feature;
        ctx.FaceRecognitionModule()->FaceExtract(stream, faces[0], feature);
        FEATURE_HUB->RegisterFaceFeature(feature, i, GetFileNameWithoutExtension(files_list[i]), 1000 + i);
    }

//    ctx.FaceRecognitionModule()->PrintMatrix();

//    auto ret = block->DeleteFeature(3);
//    LOGD("DEL: %d", ret);
//    block->PrintMatrix();

    FEATURE_HUB->DeleteFaceFeature(2);

    INSPIRE_LOGD("Number of faces in the library: %d", FEATURE_HUB->GetFaceFeatureCount());

    // Update or insert a face
    {
        Embedded feature;
        auto image = cv::imread("");
        CameraStream stream;
        stream.SetDataFormat(BGR);
        stream.SetRotationMode(ROTATION_0);
        stream.SetDataBuffer(image.data, image.rows, image.cols);
        ctx.FaceDetectAndTrack(stream);
        const auto &faces = ctx.GetTrackingFaceList();
        if (faces.empty()) {
            INSPIRE_LOGD("image1 not face");
            return -1;
        }
        ctx.FaceRecognitionModule()->FaceExtract(stream, faces[0], feature);

//        block->UpdateFeature(4, feature);
//        block->AddFeature(feature);
    }

    // Prepare an image to search
    {
        Embedded feature;
        auto image = cv::imread("");
        CameraStream stream;
        stream.SetDataFormat(BGR);
        stream.SetRotationMode(ROTATION_0);
        stream.SetDataBuffer(image.data, image.rows, image.cols);
        ctx.FaceDetectAndTrack(stream);
        const auto &faces = ctx.GetTrackingFaceList();
        if (faces.empty()) {
            INSPIRE_LOGD("image1 not face");
            return -1;
        }
        ctx.FaceRecognitionModule()->FaceExtract(stream, faces[0], feature);

        SearchResult result;
        auto timeStart = (double) cv::getTickCount();
        FEATURE_HUB->SearchFaceFeature(feature, result);
        double cost = ((double) cv::getTickCount() - timeStart) / cv::getTickFrequency() * 1000;
        INSPIRE_LOGD("Search time: %f", cost);
        INSPIRE_LOGD("Top1: %d, %f, %s %d", result.index, result.score, result.tag.c_str(), result.customId);
    }


    return 0;
}

int main(int argc, char** argv) {
    FaceContext ctx;
    CustomPipelineParameter param;
    param.enable_recognition = true;
    int32_t ret = ctx.Configuration("test_res/pack/Pikachu", DetectMode::DETECT_MODE_IMAGE, 1, param);
    if (ret != 0) {
        INSPIRE_LOGE("Initialization error");
        return -1;
    }
    
    comparison1v1(ctx);

//    search(ctx);

    return 0;

}