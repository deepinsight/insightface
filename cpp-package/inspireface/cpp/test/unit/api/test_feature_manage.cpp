//
// Created by tunm on 2023/10/11.
//

#include <iostream>
#include "settings/test_settings.h"
#include "inspireface/c_api/inspireface.h"
#include "opencv2/opencv.hpp"
#include "unit/test_helper/simple_csv_writer.h"
#include "unit/test_helper/test_help.h"

TEST_CASE("test_FeatureManage", "[feature_manage]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    SECTION("Face feature management basic functions") {
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        parameter.enable_recognition = 1;
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, -1, -1, &session);
        REQUIRE(ret == HSUCCEED);
        HFFeatureHubConfiguration configuration = {0};
        auto dbPath = GET_SAVE_DATA(".test");
        HString dbPathStr = new char[dbPath.size() + 1];
        std::strcpy(dbPathStr, dbPath.c_str());
        configuration.enablePersistence = 1;
        configuration.dbPath = dbPathStr;
        configuration.featureBlockNum = 20;
        configuration.searchMode = HF_SEARCH_MODE_EXHAUSTIVE;
        configuration.searchThreshold = 0.48f;
        // Delete the previous data before testing
        if (std::remove(configuration.dbPath) != 0) {
            spdlog::trace("Error deleting file");
        }
        ret = HFFeatureHubDataEnable(configuration);
        REQUIRE(ret == HSUCCEED);

        // Get a face picture
        cv::Mat kunImage = cv::imread(GET_DATA("data/bulk/kun.jpg"));
        HFImageData imageData = {0};
        imageData.data = kunImage.data;
        imageData.height = kunImage.rows;
        imageData.width = kunImage.cols;
        imageData.format = HF_STREAM_BGR;
        imageData.rotation = HF_CAMERA_ROTATION_0;
        HFImageStream imgHandle;
        ret = HFCreateImageStream(&imageData, &imgHandle);
        REQUIRE(ret == HSUCCEED);

        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum > 0);

        // Extract face feature
        HFFaceFeature feature = {0};
        ret = HFFaceFeatureExtract(session, imgHandle, multipleFaceData.tokens[0], &feature);
        REQUIRE(ret == HSUCCEED);

        // Insert data into feature management
        HFFaceFeatureIdentity identity = {0};
        identity.feature = &feature;
        identity.tag = "chicken";
        identity.customId = 1234;
        ret = HFFeatureHubInsertFeature(identity);
        REQUIRE(ret == HSUCCEED);

        // Check number
        HInt32 num;
        ret = HFFeatureHubGetFaceCount(&num);
        REQUIRE(ret == HSUCCEED);
        CHECK(num == 1);

        // Update Face info
        HFFaceFeatureIdentity updatedIdentity = {0};
        updatedIdentity.feature = identity.feature;
        updatedIdentity.customId = identity.customId;
        updatedIdentity.tag = "iKun";
        ret = HFFeatureHubFaceUpdate(updatedIdentity);
        REQUIRE(ret == HSUCCEED);

        // Trying to update an identity that doesn't exist
        HFFaceFeatureIdentity nonIdentity = {0};
        nonIdentity.customId = 234;
        nonIdentity.tag = "no";
        nonIdentity.feature = &feature;
        ret = HFFeatureHubFaceUpdate(nonIdentity);
        REQUIRE(ret != HSUCCEED);

        // Trying to delete an identity that doesn't exist
        ret = HFFeatureHubFaceRemove(nonIdentity.customId);
        REQUIRE(ret != HSUCCEED);

        // Delete kunkun
        ret = HFFeatureHubFaceRemove(identity.customId);
        REQUIRE(ret == HSUCCEED);

        ret = HFFeatureHubGetFaceCount(&num);
        REQUIRE(ret == HSUCCEED);
        CHECK(num == 0);


        // Finish
        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);


        ret = HFReleaseImageStream(imgHandle);
        REQUIRE(ret == HSUCCEED);

        ret = HFFeatureHubDataDisable();
        REQUIRE(ret == HSUCCEED);
        delete[]dbPathStr;
    }

    SECTION("Import a large faces data") {
#ifdef ISF_ENABLE_USE_LFW_DATA
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        parameter.enable_recognition = 1;
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, -1, -1, &session);
        REQUIRE(ret == HSUCCEED);
        HFFeatureHubConfiguration configuration = {0};
        auto dbPath = GET_SAVE_DATA(".test");
        HString dbPathStr = new char[dbPath.size() + 1];
        std::strcpy(dbPathStr, dbPath.c_str());
        configuration.enablePersistence = 1;
        configuration.dbPath = dbPathStr;
        configuration.featureBlockNum = 20;
        configuration.searchMode = HF_SEARCH_MODE_EXHAUSTIVE;
        configuration.searchThreshold = 0.48f;
        // Delete the previous data before testing
        if (std::remove(configuration.dbPath) != 0) {
            spdlog::trace("Error deleting file");
        }
        ret = HFFeatureHubDataEnable(configuration);
        REQUIRE(ret == HSUCCEED);

        auto lfwDir = getLFWFunneledDir();
        auto dataList = LoadLFWFunneledValidData(lfwDir, getTestLFWFunneledTxt());
        size_t numOfNeedImport = 1000;
        auto importStatus = ImportLFWFunneledValidData(session, dataList, numOfNeedImport);
        REQUIRE(importStatus);
        HInt32 count;
        ret = HFFeatureHubGetFaceCount(&count);
        REQUIRE(ret == HSUCCEED);
        CHECK(count == numOfNeedImport);


        // Finish
        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);

        delete[]dbPathStr;

#else
        TEST_PRINT("The test case that uses LFW is not enabled, so it will be skipped.");
#endif
    }

    SECTION("Faces feature CURD") {
#ifdef ISF_ENABLE_USE_LFW_DATA
        // This section needs to be connected to the "Import a large faces data" section before it can be executed
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        parameter.enable_recognition = 1;
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, -1, -1, &session);
        REQUIRE(ret == HSUCCEED);

        // Face track
        cv::Mat dstImage = cv::imread(GET_DATA("data/bulk/Nathalie_Baye_0002.jpg"));
        HFImageData imageData = {0};
        imageData.data = dstImage.data;
        imageData.height = dstImage.rows;
        imageData.width = dstImage.cols;
        imageData.format = HF_STREAM_BGR;
        imageData.rotation = HF_CAMERA_ROTATION_0;
        HFImageStream imgHandle;
        ret = HFCreateImageStream(&imageData, &imgHandle);
        REQUIRE(ret == HSUCCEED);

        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum > 0);

        // Extract face feature
        HFFaceFeature feature = {0};
        ret = HFFaceFeatureExtract(session, imgHandle, multipleFaceData.tokens[0], &feature);
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseImageStream(imgHandle);
        REQUIRE(ret == HSUCCEED);

        // Search for a face
        HFloat confidence;
        HFFaceFeatureIdentity searchedIdentity = {0};
        ret = HFFeatureHubFaceSearch(feature, &confidence, &searchedIdentity);
        REQUIRE(ret == HSUCCEED);
        CHECK(searchedIdentity.customId == 898);
        CHECK(std::string(searchedIdentity.tag) == "Nathalie_Baye");

        // Delete kunkun and search
        ret = HFFeatureHubFaceRemove(searchedIdentity.customId);
        REQUIRE(ret == HSUCCEED);
        // Search again
        ret = HFFeatureHubFaceSearch(feature, &confidence, &searchedIdentity);
//        spdlog::info("{}", confidence);
        REQUIRE(ret == HSUCCEED);
        CHECK(searchedIdentity.customId == -1);

        // Insert again
        HFFaceFeatureIdentity againIdentity = {0};
        againIdentity.customId = 898;
        againIdentity.tag = "Cover";
        againIdentity.feature = &feature;
        ret = HFFeatureHubInsertFeature(againIdentity);
        REQUIRE(ret == HSUCCEED);


        // Search again
        HFFaceFeatureIdentity searchedAgainIdentity = {0};
        ret = HFFeatureHubFaceSearch(feature, &confidence, &searchedAgainIdentity);
        REQUIRE(ret == HSUCCEED);
        CHECK(searchedAgainIdentity.customId == 898);

        // Update any feature
        HInt32 updateId = 909;
        cv::Mat zyImage = cv::imread(GET_DATA("data/bulk/woman.png"));
        HFImageData imageDataZy = {0};
        imageDataZy.data = zyImage.data;
        imageDataZy.height = zyImage.rows;
        imageDataZy.width = zyImage.cols;
        imageDataZy.format = HF_STREAM_BGR;
        imageDataZy.rotation = HF_CAMERA_ROTATION_0;
        HFImageStream imgHandleZy;
        ret = HFCreateImageStream(&imageDataZy, &imgHandleZy);
        REQUIRE(ret == HSUCCEED);

        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceDataZy = {0};
        ret = HFExecuteFaceTrack(session, imgHandleZy, &multipleFaceDataZy);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceDataZy.detectedNum > 0);

        // Extract face feature
        HFFaceFeature featureZy = {0};
        ret = HFFaceFeatureExtract(session, imgHandleZy, multipleFaceDataZy.tokens[0], &featureZy);
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseImageStream(imgHandleZy);
        REQUIRE(ret == HSUCCEED);

        // Update id: 11297
        HFFaceFeatureIdentity updateIdentity = {0};
        updateIdentity.customId = updateId;
        updateIdentity.tag = "ZY";
        updateIdentity.feature = &featureZy;
        ret = HFFeatureHubFaceUpdate(updateIdentity);
        REQUIRE(ret == HSUCCEED);

//
        // Prepare a zy query image
        cv::Mat zyImageQuery = cv::imread(GET_DATA("data/bulk/woman_search.jpeg"));
        HFImageData imageDataZyQuery = {0};
        imageDataZyQuery.data = zyImageQuery.data;
        imageDataZyQuery.height = zyImageQuery.rows;
        imageDataZyQuery.width = zyImageQuery.cols;
        imageDataZyQuery.format = HF_STREAM_BGR;
        imageDataZyQuery.rotation = HF_CAMERA_ROTATION_0;
        HFImageStream imgHandleZyQuery;
        ret = HFCreateImageStream(&imageDataZyQuery, &imgHandleZyQuery);
        REQUIRE(ret == HSUCCEED);
//
        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceDataZyQuery = {0};
        ret = HFExecuteFaceTrack(session, imgHandleZyQuery, &multipleFaceDataZyQuery);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceDataZyQuery.detectedNum > 0);
//
        // Extract face feature
        HFFaceFeature featureZyQuery = {0};
        ret = HFFaceFeatureExtract(session, imgHandleZyQuery, multipleFaceDataZyQuery.tokens[0], &featureZyQuery);
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseImageStream(imgHandleZyQuery);
        REQUIRE(ret == HSUCCEED);

        // Search
        HFloat confidenceQuery;
        HFFaceFeatureIdentity searchedIdentityQuery = {0};
        ret = HFFeatureHubFaceSearch(featureZyQuery, &confidenceQuery, &searchedIdentityQuery);
        REQUIRE(ret == HSUCCEED);
        CHECK(searchedIdentityQuery.customId == updateId);

        ret = HFFeatureHubDataDisable();
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);

#else
        TEST_PRINT("The test case that uses LFW is not enabled, so it will be skipped.");
#endif
    }

}

TEST_CASE("test_SearchTopK", "[feature_search_top_k]") {
#ifdef ISF_ENABLE_USE_LFW_DATA
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    SECTION("Face feature management basic functions") {
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        parameter.enable_recognition = 1;
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, -1, -1, &session);
        REQUIRE(ret == HSUCCEED);
        HFFeatureHubConfiguration configuration = {0};
        auto dbPath = GET_SAVE_DATA(".test");
        HString dbPathStr = new char[dbPath.size() + 1];
        std::strcpy(dbPathStr, dbPath.c_str());
        configuration.enablePersistence = 1;
        configuration.dbPath = dbPathStr;
        configuration.featureBlockNum = 20;
        configuration.searchMode = HF_SEARCH_MODE_EXHAUSTIVE;
        configuration.searchThreshold = 0.46f;
        // Delete the previous data before testing
        if (std::remove(configuration.dbPath) != 0) {
            spdlog::trace("Error deleting file");
        }
        ret = HFFeatureHubDataEnable(configuration);
        REQUIRE(ret == HSUCCEED);

        // Import 1k faces
        auto lfwDir = getLFWFunneledDir();
        auto dataList = LoadLFWFunneledValidData(lfwDir, getTestLFWFunneledTxt());
        size_t numOfNeedImport = 1000;
        auto importStatus = ImportLFWFunneledValidData(session, dataList, numOfNeedImport);
        REQUIRE(importStatus);
        HInt32 count;
        ret = HFFeatureHubGetFaceCount(&count);
        REQUIRE(ret == HSUCCEED);
        CHECK(count == numOfNeedImport);

        // Prepare multiple photos of a person
        std::vector<std::string> photos = {
                GET_DATA("data/RD/d1.jpeg"),
                GET_DATA("data/RD/d2.jpeg"),
                GET_DATA("data/RD/d3.jpeg"),
                GET_DATA("data/RD/d4.jpeg"),
        };
        std::vector<std::string> tags = {
                "d1", "d2", "d3", "d4",
        };
        std::vector<HInt32> updateIds = {
                5, 163, 670, 971,
        };
        REQUIRE(photos.size() == tags.size());
        REQUIRE(updateIds.size() == tags.size());

        // Replace the face features in the photo with each target in FeatureHub
        for (int i = 0; i < photos.size(); ++i) {
            // Face track
            cv::Mat dstImage = cv::imread(photos[i]);
            HFImageData imageData = {0};
            imageData.data = dstImage.data;
            imageData.height = dstImage.rows;
            imageData.width = dstImage.cols;
            imageData.format = HF_STREAM_BGR;
            imageData.rotation = HF_CAMERA_ROTATION_0;
            HFImageStream imgHandle;
            ret = HFCreateImageStream(&imageData, &imgHandle);
            REQUIRE(ret == HSUCCEED);

            // Extract basic face information from photos
            HFMultipleFaceData multipleFaceData = {0};
            ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
            REQUIRE(ret == HSUCCEED);
            REQUIRE(multipleFaceData.detectedNum > 0);

            // Extract face feature
            HFFaceFeature feature = {0};
            ret = HFFaceFeatureExtract(session, imgHandle, multipleFaceData.tokens[0], &feature);
            REQUIRE(ret == HSUCCEED);

            char* cstr = new char[tags[i].size() + 1]; // Dynamically allocate memory for the name
            strcpy(cstr, tags[i].c_str()); // Copy the name into the allocated memory

            // Create identity
            HFFaceFeatureIdentity identity = {0};
            identity.customId = updateIds[i];
            identity.feature = &feature;
            identity.tag = cstr;

            // Update
            ret = HFFeatureHubFaceUpdate(identity);
            REQUIRE(ret == HSUCCEED);

            ret = HFReleaseImageStream(imgHandle);
            REQUIRE(ret == HSUCCEED);
            delete[] cstr; // Clean up the dynamically allocated memory
        }

        // Prepare a target photo for a face top-k search
        cv::Mat image = cv::imread(GET_DATA("data/RD/d5.jpeg"));
        HFImageData imageData = {0};
        imageData.data = image.data;
        imageData.height = image.rows;
        imageData.width = image.cols;
        imageData.format = HF_STREAM_BGR;
        imageData.rotation = HF_CAMERA_ROTATION_0;
        HFImageStream imgHandle;
        ret = HFCreateImageStream(&imageData, &imgHandle);
        REQUIRE(ret == HSUCCEED);

        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum > 0);

        HFFaceFeature feature = {0};
        ret = HFFaceFeatureExtract(session, imgHandle, multipleFaceData.tokens[0], &feature);
        REQUIRE(ret == HSUCCEED);

        // Run the top-k search
        HFSearchTopKResults topk;
        ret = HFFeatureHubFaceSearchTopK(feature, 10, &topk);
        REQUIRE(ret == HSUCCEED);

        // Check whether the top-k result is consistent with the expectation
        CHECK(topk.size == photos.size());
        for (int i = 0; i < topk.size; ++i) {
            TEST_PRINT("Top-{} -> id: {}, {}", i + 1, topk.customIds[i], topk.confidence[i]);
            CHECK(std::find(updateIds.begin(), updateIds.end(), topk.customIds[i]) != updateIds.end());
        }

        ret = HFReleaseImageStream(imgHandle);
        REQUIRE(ret == HSUCCEED);

        // Finish
        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);

        ret = HFFeatureHubDataDisable();
        REQUIRE(ret == HSUCCEED);
        delete[]dbPathStr;
    }
#endif
}

TEST_CASE("test_FeatureBenchmark", "[feature_benchmark]") {

    // Test the search time at 1k, 5k and 10k of the face library (the target face is at the back).
    SECTION("Search face benchmark from 1k") {
#if defined(ISF_ENABLE_BENCHMARK) && defined(ISF_ENABLE_USE_LFW_DATA)
        size_t loop = 1000;
        size_t numOfNeedImport = 1000;
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        parameter.enable_recognition = 1;
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, -1, -1, &session);
        REQUIRE(ret == HSUCCEED);
        HFFeatureHubConfiguration configuration = {0};
        auto dbPath = GET_SAVE_DATA(".test");
        HString dbPathStr = new char[dbPath.size() + 1];
        std::strcpy(dbPathStr, dbPath.c_str());
        configuration.enablePersistence = 1;
        configuration.dbPath = dbPathStr;
        configuration.featureBlockNum = 20;
        configuration.searchMode = HF_SEARCH_MODE_EXHAUSTIVE;
        configuration.searchThreshold = 0.48f;
        // Delete the previous data before testing
        if (std::remove(configuration.dbPath) != 0) {
            spdlog::trace("Error deleting file");
        }
        ret = HFFeatureHubDataEnable(configuration);
        REQUIRE(ret == HSUCCEED);

        auto lfwDir = getLFWFunneledDir();
        auto dataList = LoadLFWFunneledValidData(lfwDir, getTestLFWFunneledTxt());
//        TEST_PRINT("{}", dataList.size());
        auto importStatus = ImportLFWFunneledValidData(session, dataList, numOfNeedImport);
        REQUIRE(importStatus);
        HInt32 count;
        ret = HFFeatureHubGetFaceCount(&count);
        REQUIRE(ret == HSUCCEED);
        CHECK(count == numOfNeedImport);

        // Face track
        cv::Mat dstImage = cv::imread(GET_DATA("data/search/Teresa_Williams_0001_1k.jpg"));
        HFImageData imageData = {0};
        imageData.data = dstImage.data;
        imageData.height = dstImage.rows;
        imageData.width = dstImage.cols;
        imageData.format = HF_STREAM_BGR;
        imageData.rotation = HF_CAMERA_ROTATION_0;
        HFImageStream imgHandle;
        ret = HFCreateImageStream(&imageData, &imgHandle);
        REQUIRE(ret == HSUCCEED);

        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum > 0);

        // Extract face feature
        HFFaceFeature feature = {0};
        ret = HFFaceFeatureExtract(session, imgHandle, multipleFaceData.tokens[0], &feature);
        REQUIRE(ret == HSUCCEED);

        // Search for a face
        HFloat confidence;
        HFFaceFeatureIdentity searchedIdentity = {0};
        auto start = (double) cv::getTickCount();
        for (int i = 0; i < loop; ++i) {
            ret = HFFeatureHubFaceSearch(feature, &confidence, &searchedIdentity);
        }
        auto cost = ((double) cv::getTickCount() - start) / cv::getTickFrequency() * 1000;

        REQUIRE(ret == HSUCCEED);
        REQUIRE(searchedIdentity.customId == 999);
        REQUIRE(std::string(searchedIdentity.tag) == "Teresa_Williams");

        TEST_PRINT("<Benchmark> Search Face from 1k -> Loop: {}, Total Time: {:.5f}ms, Average Time: {:.5f}ms", loop, cost, cost / loop);

        BenchmarkRecord record(getBenchmarkRecordFile());
        record.insertBenchmarkData("Search Face from 1k", loop, cost, cost / loop);
        // Finish
        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);


        ret = HFReleaseImageStream(imgHandle);
        REQUIRE(ret == HSUCCEED);

        ret = HFFeatureHubDataDisable();
        REQUIRE(ret == HSUCCEED);
        delete []dbPathStr;
#else
        TEST_PRINT("Skip face search benchmark test, you need to enable both lfw and benchmark test.");
#endif
    }

    SECTION("Search face benchmark from 5k") {
#if defined(ISF_ENABLE_BENCHMARK) && defined(ISF_ENABLE_USE_LFW_DATA)
        size_t loop = 1000;
        size_t numOfNeedImport = 5000;
        HResult ret;
        std::string modelPath = GET_MODEL_FILE();
        HPath path = modelPath.c_str();
        HFSessionCustomParameter parameter = {0};
        parameter.enable_recognition = 1;
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, -1, -1, &session);
        REQUIRE(ret == HSUCCEED);
        HFFeatureHubConfiguration configuration = {0};
        auto dbPath = GET_SAVE_DATA(".test");
        HString dbPathStr = new char[dbPath.size() + 1];
        std::strcpy(dbPathStr, dbPath.c_str());
        configuration.enablePersistence = 1;
        configuration.dbPath = dbPathStr;
        configuration.featureBlockNum = 20;
        configuration.searchMode = HF_SEARCH_MODE_EXHAUSTIVE;
        configuration.searchThreshold = 0.48f;
        // Delete the previous data before testing
        if (std::remove(configuration.dbPath) != 0) {
            spdlog::trace("Error deleting file");
        }
        ret = HFFeatureHubDataEnable(configuration);
        REQUIRE(ret == HSUCCEED);

        auto lfwDir = getLFWFunneledDir();
        auto dataList = LoadLFWFunneledValidData(lfwDir, getTestLFWFunneledTxt());
        auto importStatus = ImportLFWFunneledValidData(session, dataList, numOfNeedImport);
        REQUIRE(importStatus);
        HInt32 count;
        ret = HFFeatureHubGetFaceCount(&count);
        REQUIRE(ret == HSUCCEED);
        CHECK(count == numOfNeedImport);

        // Face track
        cv::Mat dstImage = cv::imread(GET_DATA("data/search/Mary_Katherine_Smart_0001_5k.jpg"));
        HFImageData imageData = {0};
        imageData.data = dstImage.data;
        imageData.height = dstImage.rows;
        imageData.width = dstImage.cols;
        imageData.format = HF_STREAM_BGR;
        imageData.rotation = HF_CAMERA_ROTATION_0;
        HFImageStream imgHandle;
        ret = HFCreateImageStream(&imageData, &imgHandle);
        REQUIRE(ret == HSUCCEED);

        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum > 0);

        // Extract face feature
        HFFaceFeature feature = {0};
        ret = HFFaceFeatureExtract(session, imgHandle, multipleFaceData.tokens[0], &feature);
        REQUIRE(ret == HSUCCEED);

        // Search for a face
        HFloat confidence;
        HFFaceFeatureIdentity searchedIdentity = {0};
        auto start = (double) cv::getTickCount();
        for (int i = 0; i < loop; ++i) {
            ret = HFFeatureHubFaceSearch(feature, &confidence, &searchedIdentity);
        }
        auto cost = ((double) cv::getTickCount() - start) / cv::getTickFrequency() * 1000;

        REQUIRE(ret == HSUCCEED);
        REQUIRE(searchedIdentity.customId == 4998);
        REQUIRE(std::string(searchedIdentity.tag) == "Mary_Katherine_Smart");

        TEST_PRINT("<Benchmark> Search Face from 5k -> Loop: {}, Total Time: {:.5f}ms, Average Time: {:.5f}ms", loop, cost, cost / loop);

        BenchmarkRecord record(getBenchmarkRecordFile());
        record.insertBenchmarkData("Search Face from 5k", loop, cost, cost / loop);
        // Finish
        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseImageStream(imgHandle);
        REQUIRE(ret == HSUCCEED);

        ret = HFFeatureHubDataDisable();
        REQUIRE(ret == HSUCCEED);
        delete []dbPathStr;
#else
        TEST_PRINT("Skip face search benchmark test, you need to enable both lfw and benchmark test.");
#endif
    }

    SECTION("Search face benchmark from 10k") {
#if defined(ISF_ENABLE_BENCHMARK) && defined(ISF_ENABLE_USE_LFW_DATA)
        size_t loop = 1000;
        size_t numOfNeedImport = 10000;
        HResult ret;
        std::string modelPath = GET_MODEL_FILE();
        HPath path = modelPath.c_str();
        HFSessionCustomParameter parameter = {0};
        parameter.enable_recognition = 1;
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, -1, -1, &session);
        REQUIRE(ret == HSUCCEED);
        HFFeatureHubConfiguration configuration = {0};
        auto dbPath = GET_SAVE_DATA(".test");
        HString dbPathStr = new char[dbPath.size() + 1];
        std::strcpy(dbPathStr, dbPath.c_str());
        configuration.enablePersistence = 1;
        configuration.dbPath = dbPathStr;
        configuration.featureBlockNum = 20;
        configuration.searchMode = HF_SEARCH_MODE_EXHAUSTIVE;
        configuration.searchThreshold = 0.48f;
        // Delete the previous data before testing
        if (std::remove(configuration.dbPath) != 0) {
            spdlog::trace("Error deleting file");
        }
        ret = HFFeatureHubDataEnable(configuration);
        REQUIRE(ret == HSUCCEED);

        auto lfwDir = getLFWFunneledDir();
        auto dataList = LoadLFWFunneledValidData(lfwDir, getTestLFWFunneledTxt());
//        TEST_PRINT("{}", dataList.size());
        auto importStatus = ImportLFWFunneledValidData(session, dataList, numOfNeedImport);
        REQUIRE(importStatus);
        HInt32 count;
        ret = HFFeatureHubGetFaceCount(&count);
        REQUIRE(ret == HSUCCEED);
        CHECK(count == numOfNeedImport);

        // Update any feature
        HInt32 updateId = numOfNeedImport - 1;
        cv::Mat zyImage = cv::imread(GET_DATA("data/bulk/woman.png"));
        HFImageData imageDataZy = {0};
        imageDataZy.data = zyImage.data;
        imageDataZy.height = zyImage.rows;
        imageDataZy.width = zyImage.cols;
        imageDataZy.format = HF_STREAM_BGR;
        imageDataZy.rotation = HF_CAMERA_ROTATION_0;
        HFImageStream imgHandleZy;
        ret = HFCreateImageStream(&imageDataZy, &imgHandleZy);
        REQUIRE(ret == HSUCCEED);

        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceDataZy = {0};
        ret = HFExecuteFaceTrack(session, imgHandleZy, &multipleFaceDataZy);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceDataZy.detectedNum > 0);

        // Extract face feature
        HFFaceFeature featureZy = {0};
        ret = HFFaceFeatureExtract(session, imgHandleZy, multipleFaceDataZy.tokens[0], &featureZy);
        REQUIRE(ret == HSUCCEED);

        // Update id: 11297
        HFFaceFeatureIdentity updateIdentity = {0};
        updateIdentity.customId = updateId;
        updateIdentity.tag = "ZY";
        updateIdentity.feature = &featureZy;
        ret = HFFeatureHubFaceUpdate(updateIdentity);
        REQUIRE(ret == HSUCCEED);

        HFReleaseImageStream(imgHandleZy);

        // Face track
        cv::Mat dstImage = cv::imread(GET_DATA("data/bulk/woman_search.jpeg"));
        HFImageData imageData = {0};
        imageData.data = dstImage.data;
        imageData.height = dstImage.rows;
        imageData.width = dstImage.cols;
        imageData.format = HF_STREAM_BGR;
        imageData.rotation = HF_CAMERA_ROTATION_0;
        HFImageStream imgHandle;
        ret = HFCreateImageStream(&imageData, &imgHandle);
        REQUIRE(ret == HSUCCEED);

        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum > 0);

        // Extract face feature
        HFFaceFeature feature = {0};
        ret = HFFaceFeatureExtract(session, imgHandle, multipleFaceData.tokens[0], &feature);
        REQUIRE(ret == HSUCCEED);

        // Search for a face
        HFloat confidence;
        HFFaceFeatureIdentity searchedIdentity = {0};
        auto start = (double) cv::getTickCount();
        for (int i = 0; i < loop; ++i) {
            ret = HFFeatureHubFaceSearch(feature, &confidence, &searchedIdentity);
        }
        auto cost = ((double) cv::getTickCount() - start) / cv::getTickFrequency() * 1000;

        REQUIRE(ret == HSUCCEED);
        REQUIRE(searchedIdentity.customId == updateId);
        REQUIRE(std::string(searchedIdentity.tag) == "ZY");

        TEST_PRINT("<Benchmark> Search Face from 10k -> Loop: {}, Total Time: {:.5f}ms, Average Time: {:.5f}ms", loop, cost, cost / loop);

        BenchmarkRecord record(getBenchmarkRecordFile());
        record.insertBenchmarkData("Search Face from 10k", loop, cost, cost / loop);

        // Finish
        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);


        ret = HFReleaseImageStream(imgHandle);
        REQUIRE(ret == HSUCCEED);

        ret = HFFeatureHubDataDisable();
        REQUIRE(ret == HSUCCEED);
        delete []dbPathStr;
#else
        TEST_PRINT("Skip face search benchmark test, you need to enable both lfw and benchmark test.");
#endif
    }

    SECTION("Face comparison benchmark") {
#ifdef ISF_ENABLE_BENCHMARK
        int loop = 1000;
        HResult ret;
        std::string modelPath = GET_MODEL_FILE();
        HPath path = modelPath.c_str();
        HFSessionCustomParameter parameter = {0};
        parameter.enable_recognition = 1;
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3,  -1, -1, &session);
        REQUIRE(ret == HSUCCEED);
        HFFeatureHubConfiguration configuration = {0};
        auto dbPath = GET_SAVE_DATA(".test");
        HString dbPathStr = new char[dbPath.size() + 1];
        std::strcpy(dbPathStr, dbPath.c_str());
        configuration.enablePersistence = 1;
        configuration.dbPath = dbPathStr;
        configuration.featureBlockNum = 20;
        configuration.searchMode = HF_SEARCH_MODE_EXHAUSTIVE;
        configuration.searchThreshold = 0.48f;
        // Delete the previous data before testing
        if (std::remove(configuration.dbPath) != 0) {
            spdlog::trace("Error deleting file");
        }
        ret = HFFeatureHubDataEnable(configuration);
        REQUIRE(ret == HSUCCEED);

        cv::Mat zyImage = cv::imread(GET_DATA("data/bulk/woman.png"));
        HFImageData imageDataZy = {0};
        imageDataZy.data = zyImage.data;
        imageDataZy.height = zyImage.rows;
        imageDataZy.width = zyImage.cols;
        imageDataZy.format = HF_STREAM_BGR;
        imageDataZy.rotation = HF_CAMERA_ROTATION_0;
        HFImageStream imgHandleZy;
        ret = HFCreateImageStream(&imageDataZy, &imgHandleZy);
        REQUIRE(ret == HSUCCEED);

        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceDataZy = {0};
        ret = HFExecuteFaceTrack(session, imgHandleZy, &multipleFaceDataZy);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceDataZy.detectedNum > 0);

        HInt32 featureNum;
        HFGetFeatureLength(&featureNum);

        // Extract face feature
        HFloat featureCacheZy[featureNum];
        ret = HFFaceFeatureExtractCpy(session, imgHandleZy, multipleFaceDataZy.tokens[0], featureCacheZy);
        HFFaceFeature featureZy = {0};
        featureZy.size = featureNum;
        featureZy.data = featureCacheZy;
        REQUIRE(ret == HSUCCEED);

        cv::Mat zyImageQuery = cv::imread(GET_DATA("data/bulk/woman_search.jpeg"));
        HFImageData imageDataZyQuery = {0};
        imageDataZyQuery.data = zyImageQuery.data;
        imageDataZyQuery.height = zyImageQuery.rows;
        imageDataZyQuery.width = zyImageQuery.cols;
        imageDataZyQuery.format = HF_STREAM_BGR;
        imageDataZyQuery.rotation = HF_CAMERA_ROTATION_0;
        HFImageStream imgHandleZyQuery;
        ret = HFCreateImageStream(&imageDataZyQuery, &imgHandleZyQuery);
        REQUIRE(ret == HSUCCEED);
//
        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceDataZyQuery = {0};
        ret = HFExecuteFaceTrack(session, imgHandleZyQuery, &multipleFaceDataZyQuery);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceDataZyQuery.detectedNum > 0);
//
        // Extract face feature
        HFloat featureCacheZyQuery[featureNum];
        ret = HFFaceFeatureExtractCpy(session, imgHandleZyQuery, multipleFaceDataZyQuery.tokens[0], featureCacheZyQuery);
        HFFaceFeature featureZyQuery = {0};
        featureZyQuery.data = featureCacheZyQuery;
        featureZyQuery.size = featureNum;
        REQUIRE(ret == HSUCCEED);

        auto start = (double) cv::getTickCount();
        for (int i = 0; i < loop; ++i) {
            HFloat compRes;
            ret = HFFaceComparison(featureZy, featureZyQuery, &compRes);
        }
        auto cost = ((double) cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        REQUIRE(ret == HSUCCEED);
        TEST_PRINT("<Benchmark> Face Comparison -> Loop: {}, Total Time: {:.5f}ms, Average Time: {:.5f}ms", loop, cost, cost / loop);

        BenchmarkRecord record(getBenchmarkRecordFile());
        record.insertBenchmarkData("Face Comparison", loop, cost, cost / loop);

        HFReleaseImageStream(imgHandleZy);
        HFReleaseImageStream(imgHandleZyQuery);

        // Finish
        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);
        ret = HFFeatureHubDataDisable();
        REQUIRE(ret == HSUCCEED);
        delete []dbPathStr;
#else
        TEST_PRINT("The benchmark is not enabled, so all relevant test cases are skipped.");
#endif
    }

    SECTION("Face feature extract benchmark") {
#ifdef ISF_ENABLE_BENCHMARK
        int loop = 1000;
        HResult ret;
        std::string modelPath = GET_MODEL_FILE();
        HPath path = modelPath.c_str();
        HFSessionCustomParameter parameter = {0};
        parameter.enable_recognition = 1;
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3,-1, -1, &session);
        REQUIRE(ret == HSUCCEED);
        HFFeatureHubConfiguration configuration = {0};
        auto dbPath = GET_SAVE_DATA(".test");
        HString dbPathStr = new char[dbPath.size() + 1];
        std::strcpy(dbPathStr, dbPath.c_str());
        configuration.enablePersistence = 1;
        configuration.dbPath = dbPathStr;
        configuration.featureBlockNum = 20;
        configuration.searchMode = HF_SEARCH_MODE_EXHAUSTIVE;
        configuration.searchThreshold = 0.48f;
        // Delete the previous data before testing
        if (std::remove(configuration.dbPath) != 0) {
            spdlog::trace("Error deleting file");
        }
        ret = HFFeatureHubDataEnable(configuration);
        REQUIRE(ret == HSUCCEED);

        // Face track
        cv::Mat dstImage = cv::imread(GET_DATA("data/search/Teresa_Williams_0001_1k.jpg"));
        HFImageData imageData = {0};
        imageData.data = dstImage.data;
        imageData.height = dstImage.rows;
        imageData.width = dstImage.cols;
        imageData.format = HF_STREAM_BGR;
        imageData.rotation = HF_CAMERA_ROTATION_0;
        HFImageStream imgHandle;
        ret = HFCreateImageStream(&imageData, &imgHandle);
        REQUIRE(ret == HSUCCEED);

        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum > 0);

        // Extract face feature
        HFFaceFeature feature = {0};
        auto start = (double) cv::getTickCount();
        for (int i = 0; i < loop; ++i) {
            ret = HFFaceFeatureExtract(session, imgHandle, multipleFaceData.tokens[0], &feature);
        }
        auto cost = ((double) cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        REQUIRE(ret == HSUCCEED);
        TEST_PRINT("<Benchmark> Face Extract -> Loop: {}, Total Time: {:.5f}ms, Average Time: {:.5f}ms", loop, cost, cost / loop);

        BenchmarkRecord record(getBenchmarkRecordFile());
        record.insertBenchmarkData("Face Extract", loop, cost, cost / loop);

        HFReleaseImageStream(imgHandle);
        // Finish
        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);
        ret = HFFeatureHubDataDisable();
        REQUIRE(ret == HSUCCEED);
        delete []dbPathStr;
#else
        TEST_PRINT("Skip the face feature extraction benchmark test. To run it, you need to turn on the benchmark test.");
#endif
    }
}