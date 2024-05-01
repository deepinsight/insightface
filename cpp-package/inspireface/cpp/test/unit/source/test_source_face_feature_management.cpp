//
// Created by Tunm-Air13 on 2023/9/12.
//

#include "settings/test_settings.h"
#include "inspireface/face_context.h"
#include "herror.h"
#include "../test_helper/test_help.h"
#include "feature_hub/feature_hub.h"

using namespace inspire;

TEST_CASE("test_FaceFeatureManagement", "[face_feature]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    SECTION("FeatureCURD") {
        DRAW_SPLIT_LINE
        // Initialize
        FaceContext ctx;
        CustomPipelineParameter param;
        param.enable_recognition = true;
        auto ret = ctx.Configuration(DetectMode::DETECT_MODE_IMAGE, 1, param);
        REQUIRE(ret == HSUCCEED);

        FEATURE_HUB->PrintFeatureMatrixInfo();

        // Know the location of 'kunkun' in advance
        int32_t KunkunIndex = 795;
        // Prepare a face photo in advance and extract the features
        auto image = cv::imread(GET_DATA("images/kun.jpg"));
        CameraStream stream;
        stream.SetDataFormat(BGR);
        stream.SetRotationMode(ROTATION_0);
        stream.SetDataBuffer(image.data, image.rows, image.cols);
        ret = ctx.FaceDetectAndTrack(stream);
        REQUIRE(ret == HSUCCEED);
        // Face detection
        ctx.FaceDetectAndTrack(stream);
        const auto &faces = ctx.GetTrackingFaceList();
        REQUIRE(faces.size() > 0);
        // Feature extraction of "Kunkun" was carried out
        Embedded feature;
        ret = ctx.FaceRecognitionModule()->FaceExtract(stream, faces[0], feature);
        CHECK(ret == HSUCCEED);

        // Import face feature vectors in batches
        String mat_path = GET_DATA("test_faceset/test_faces_A1.npy");
        String tags_path = GET_DATA("test_faceset/test_faces_A1.txt");
        auto result = LoadMatrixAndTags(mat_path, tags_path);
        // Gets the feature matrix and label names
        EmbeddedList featureMatrix = result.first;
        std::vector<std::string> tagNames = result.second;
        REQUIRE(featureMatrix.size() == 3000);
        REQUIRE(tagNames.size() == 3000);
        REQUIRE(featureMatrix[0].size() == 512);

        for (int i = 0; i < featureMatrix.size(); ++i) {
            auto &feat = featureMatrix[i];
            auto ret = FEATURE_HUB->RegisterFaceFeature(feat, i, tagNames[i], i);
            CHECK(ret == HSUCCEED);
        }

        std::cout << std::endl;
        REQUIRE(FEATURE_HUB->GetFaceFeatureCount() == 3000);
        spdlog::trace("All 3000 Faces embedded vector are loaded");

        // Prepare a face photo to search through the library
        SearchResult searchResult;
        ret = FEATURE_HUB->SearchFaceFeature(feature, searchResult, 0.5f);
        REQUIRE(ret == HSUCCEED);
        CHECK(searchResult.index != -1);
        CHECK(searchResult.index == KunkunIndex);
        CHECK(searchResult.tag == "Kunkun");
        CHECK(searchResult.score == Approx(0.76096).epsilon(1e-3));
        spdlog::info("Find Kunkun -> Location ID: {}, Confidence: {}, Tag: {}", searchResult.index, searchResult.score, searchResult.tag.c_str());
        // Save "Kunkun"'s library features and so on
        Embedded KunkunFeature;
        ret = FEATURE_HUB->GetFaceFeature(KunkunIndex, KunkunFeature);
        REQUIRE(ret == HSUCCEED);

        // The features of "Kunkun" library corresponding to those found above are deleted from the face library
        ret = FEATURE_HUB->DeleteFaceFeature(searchResult.index);
        CHECK(ret == HSUCCEED);
        // In search once
        SearchResult secondSearchResult;
        ret = FEATURE_HUB->SearchFaceFeature(feature, secondSearchResult, 0.5f);
        REQUIRE(ret == HSUCCEED);
        CHECK(secondSearchResult.index == -1);
        spdlog::info("Kunkun被删除了无法找到: {}, {}", secondSearchResult.index, secondSearchResult.tag);

        // Just take a random place and change the eigenvector for that place and put "Kunkun" back in there
        auto newIndex = 2888;
        // Try inserting an unused location first
        ret = FEATURE_HUB->UpdateFaceFeature(KunkunFeature, 3001, "Chicken", 3001);
        REQUIRE(ret == HERR_SESS_REC_BLOCK_UPDATE_FAILURE);
        ret = FEATURE_HUB->UpdateFaceFeature(KunkunFeature, newIndex, "Chicken", 3001);
        REQUIRE(ret == HSUCCEED);
        SearchResult thirdlySearchResult;
        ret = FEATURE_HUB->SearchFaceFeature(feature, thirdlySearchResult, 0.5f);
        REQUIRE(ret == HSUCCEED);
        CHECK(thirdlySearchResult.index != -1);
        CHECK(thirdlySearchResult.index == newIndex);
        CHECK(thirdlySearchResult.tag == "Chicken");
        spdlog::info("Find Kunkun again -> New Location ID: {}, Confidence: {}, Tag: {}", thirdlySearchResult.index, thirdlySearchResult.score, thirdlySearchResult.tag.c_str());

    }

#if ENABLE_BENCHMARK
    SECTION("FeatureSearchBenchmark") {
        DRAW_SPLIT_LINE

        // Initialize
        FaceContext ctx;
        CustomPipelineParameter param;
        param.enable_recognition = true;
        auto ret = ctx.Configuration(DetectMode::DETECT_MODE_IMAGE, 1, param);
        REQUIRE(ret == HSUCCEED);

        FEATURE_HUB->PrintFeatureMatrixInfo();

        // Import face feature vectors in batches
        String mat_path = GET_DATA("test_faceset/test_faces_A1.npy");
        String tags_path = GET_DATA("test_faceset/test_faces_A1.txt");
        auto result = LoadMatrixAndTags(mat_path, tags_path);
        // Gets the feature matrix and label names
        EmbeddedList featureMatrix = result.first;
        std::vector<std::string> tagNames = result.second;
        REQUIRE(featureMatrix.size() == 3000);
        REQUIRE(tagNames.size() == 3000);
        REQUIRE(featureMatrix[0].size() == 512);

        for (int i = 0; i < featureMatrix.size(); ++i) {
            auto &feat = featureMatrix[i];
            auto ret = FEATURE_HUB->RegisterFaceFeature(feat, i, tagNames[i], i);
            CHECK(ret == HSUCCEED);
        }

        std::cout << std::endl;
        REQUIRE(FEATURE_HUB->GetFaceFeatureCount() == 3000);
        spdlog::trace("3000个特征向量全部载入");

        // Prepare a picture of a face
        auto image = cv::imread(GET_DATA("images/face_sample.png"));
        CameraStream stream;
        stream.SetDataFormat(BGR);
        stream.SetRotationMode(ROTATION_0);
        stream.SetDataBuffer(image.data, image.rows, image.cols);
        ret = ctx.FaceDetectAndTrack(stream);
        REQUIRE(ret == HSUCCEED);
        // Face detection
        ctx.FaceDetectAndTrack(stream);
        const auto &faces = ctx.GetTrackingFaceList();
        REQUIRE(faces.size() > 0);
        // Feature extraction of "kunkun" was carried out
        Embedded feature;
        ret = ctx.FaceRecognitionModule()->FaceExtract(stream, faces[0], feature);
        CHECK(ret == HSUCCEED);

        // Insert the face further back
        auto regIndex = 4000;
        ret = FEATURE_HUB->RegisterFaceFeature(feature, regIndex, "test", 4000);
        REQUIRE(ret == HSUCCEED);

        const auto loop = 1000;
        double total = 0.0f;
        spdlog::info("Start performing {} searches: ", loop);
        auto out = (double) cv::getTickCount();
        for (int i = 0; i < loop; ++i) {

            // Prepare a face photo to look it up from the library
            SearchResult searchResult;
            auto timeStart = (double) cv::getTickCount();
            ret = FEATURE_HUB->SearchFaceFeature(feature, searchResult, 0.5f);
            double cost = ((double) cv::getTickCount() - timeStart) / cv::getTickFrequency() * 1000;
            REQUIRE(ret == HSUCCEED);
            CHECK(searchResult.index == regIndex);
            total += cost;
        }
        auto end = ((double) cv::getTickCount() - out) / cv::getTickFrequency() * 1000;

        spdlog::info("Execute {} times Total Cost: {}ms, Average Cost: {}ms", loop, end, total / loop);
    }
#endif
}