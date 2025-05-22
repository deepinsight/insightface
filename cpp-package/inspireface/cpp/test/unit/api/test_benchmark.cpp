/**
 * Created by Jingyu Yan
 * @date 2025-01-12
 */

#include <iostream>
#include "settings/test_settings.h"
#include "inspireface/c_api/inspireface.h"
#include "unit/test_helper/simple_csv_writer.h"
#include "unit/test_helper/test_help.h"
#include "unit/test_helper/test_tools.h"
#include "middleware/costman.h"

#ifdef ISF_ENABLE_BENCHMARK

TEST_CASE("test_BenchmarkFaceDetect", "[benchmark]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    const int loop = 1000;

    SECTION("Benchmark face detection@160") {
        auto pixLevel = 160;
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, pixLevel, -1, &session);
        REQUIRE(ret == HSUCCEED);

        // Get a face picture
        HFImageStream imgHandle;
        auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
        ret = CVImageToImageStream(image, imgHandle);
        REQUIRE(ret == HSUCCEED);

        inspire::SpendTimer timeSpend("Face Detect@160");
        for (size_t i = 0; i < loop; i++) {
            timeSpend.Start();
            // Extract basic face information from photos
            HFMultipleFaceData multipleFaceData = {0};
            ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
            REQUIRE(ret == HSUCCEED);
            REQUIRE(multipleFaceData.detectedNum == 1);
            timeSpend.Stop();
        }
        std::cout << timeSpend << std::endl;

        ret = HFReleaseImageStream(imgHandle);
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);
    }

    SECTION("Benchmark face detection@320") {
        auto pixLevel = 320;
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, pixLevel, -1, &session);
        REQUIRE(ret == HSUCCEED);

        // Get a face picture
        HFImageStream imgHandle;
        auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
        ret = CVImageToImageStream(image, imgHandle);
        REQUIRE(ret == HSUCCEED);

        inspire::SpendTimer timeSpend("Face Detect@320");
        for (size_t i = 0; i < loop; i++) {
            timeSpend.Start();
            // Extract basic face information from photos
            HFMultipleFaceData multipleFaceData = {0};
            ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
            REQUIRE(ret == HSUCCEED);
            REQUIRE(multipleFaceData.detectedNum == 1);
            timeSpend.Stop();
        }
        std::cout << timeSpend << std::endl;

        ret = HFReleaseImageStream(imgHandle);
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);
    }

    SECTION("Benchmark face detection@640") {
        auto pixLevel = 640;
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, pixLevel, -1, &session);
        REQUIRE(ret == HSUCCEED);

        // Get a face picture
        HFImageStream imgHandle;
        auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
        ret = CVImageToImageStream(image, imgHandle);
        REQUIRE(ret == HSUCCEED);

        inspire::SpendTimer timeSpend("Face Detect@640");
        for (size_t i = 0; i < loop; i++) {
            timeSpend.Start();
            // Extract basic face information from photos
            HFMultipleFaceData multipleFaceData = {0};
            ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
            REQUIRE(ret == HSUCCEED);
            REQUIRE(multipleFaceData.detectedNum == 1);
            timeSpend.Stop();
        }
        std::cout << timeSpend << std::endl;

        ret = HFReleaseImageStream(imgHandle);
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);
    }
}

TEST_CASE("test_BenchmarkFaceTrack", "[benchmark]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    const int loop = 1000;
    auto pixLevel = 160;
    HResult ret;
    HFSessionCustomParameter parameter = {0};
    HFDetectMode detMode = HF_DETECT_MODE_LIGHT_TRACK;
    HFSession session;
    ret = HFCreateInspireFaceSession(parameter, detMode, 3, pixLevel, -1, &session);
    REQUIRE(ret == HSUCCEED);

    // Get a face picture
    HFImageStream imgHandle;
    auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
    ret = CVImageToImageStream(image, imgHandle);
    REQUIRE(ret == HSUCCEED);

    inspire::SpendTimer timeSpend("Face Track");
    for (size_t i = 0; i < loop; i++) {
        timeSpend.Start();
        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum == 1);
        timeSpend.Stop();
    }
    std::cout << timeSpend << std::endl;

    ret = HFReleaseImageStream(imgHandle);
    REQUIRE(ret == HSUCCEED);

    ret = HFReleaseInspireFaceSession(session);
    REQUIRE(ret == HSUCCEED);
}

TEST_CASE("test_BenchmarkFaceExtractWithAlign", "[benchmark]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    const int loop = 1000;
    auto pixLevel = 160;
    HResult ret;
    HFSessionCustomParameter parameter = {0};
    HFDetectMode detMode = HF_DETECT_MODE_LIGHT_TRACK;
    HFSession session;

    ret = HFCreateInspireFaceSessionOptional(HF_ENABLE_FACE_RECOGNITION, detMode, 3, -1, -1, &session);
    REQUIRE(ret == HSUCCEED);

    // Face track
    auto dstImage = inspirecv::Image::Create(GET_DATA("data/search/Teresa_Williams_0001_1k.jpg"));

    // Get a face picture
    HFImageStream imgHandle;
    ret = CVImageToImageStream(dstImage, imgHandle);
    REQUIRE(ret == HSUCCEED);

    HFMultipleFaceData multipleFaceData = {0};
    ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
    REQUIRE(ret == HSUCCEED);

    inspire::SpendTimer timeSpend("Face Extract With Align");
    for (size_t i = 0; i < loop; i++) {
        timeSpend.Start();
        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        timeSpend.Stop();
    }
    std::cout << timeSpend << std::endl;

    ret = HFReleaseImageStream(imgHandle);
    REQUIRE(ret == HSUCCEED);

    ret = HFReleaseInspireFaceSession(session);
    REQUIRE(ret == HSUCCEED);
}

TEST_CASE("test_BenchmarkFaceComparison", "[benchmark]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    int loop = 1000;
    HResult ret;
    HFSessionCustomParameter parameter = {0};
    parameter.enable_recognition = 1;
    HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
    HFSession session;
    ret = HFCreateInspireFaceSession(parameter, detMode, 3, -1, -1, &session);
    REQUIRE(ret == HSUCCEED);

    auto image = inspirecv::Image::Create(GET_DATA("data/bulk/woman.png"));
    HFImageStream imgHandle;
    ret = CVImageToImageStream(image, imgHandle);
    REQUIRE(ret == HSUCCEED);

    // Extract basic face information from photos
    HFMultipleFaceData multipleFaceDataZy = {0};
    ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceDataZy);
    REQUIRE(ret == HSUCCEED);
    REQUIRE(multipleFaceDataZy.detectedNum > 0);

    HInt32 featureNum;
    HFGetFeatureLength(&featureNum);

    // Extract face feature
    HFloat featureCacheZy[featureNum];
    ret = HFFaceFeatureExtractCpy(session, imgHandle, multipleFaceDataZy.tokens[0], featureCacheZy);
    HFFaceFeature featureZy = {0};
    featureZy.size = featureNum;
    featureZy.data = featureCacheZy;
    REQUIRE(ret == HSUCCEED);

    auto imageQuery = inspirecv::Image::Create(GET_DATA("data/bulk/woman_search.jpeg"));
    HFImageStream imgHandleQuery;
    ret = CVImageToImageStream(imageQuery, imgHandleQuery);
    REQUIRE(ret == HSUCCEED);

    HFMultipleFaceData multipleFaceDataQuery = {0};
    ret = HFExecuteFaceTrack(session, imgHandleQuery, &multipleFaceDataQuery);
    REQUIRE(ret == HSUCCEED);
    REQUIRE(multipleFaceDataQuery.detectedNum > 0);

    // Extract face feature
    HFloat featureCacheZyQuery[featureNum];
    ret = HFFaceFeatureExtractCpy(session, imgHandleQuery, multipleFaceDataQuery.tokens[0], featureCacheZyQuery);
    HFFaceFeature featureZyQuery = {0};
    featureZyQuery.data = featureCacheZyQuery;
    featureZyQuery.size = featureNum;
    REQUIRE(ret == HSUCCEED);

    inspire::SpendTimer timeSpend("Face Comparison");
    for (int i = 0; i < loop; ++i) {
        timeSpend.Start();
        HFloat compRes;
        ret = HFFaceComparison(featureZy, featureZyQuery, &compRes);
        REQUIRE(ret == HSUCCEED);
        timeSpend.Stop();
    }
    std::cout << timeSpend << std::endl;

    ret = HFReleaseImageStream(imgHandle);
    REQUIRE(ret == HSUCCEED);

    ret = HFReleaseImageStream(imgHandleQuery);
    REQUIRE(ret == HSUCCEED);

    // Finish
    ret = HFReleaseInspireFaceSession(session);
    REQUIRE(ret == HSUCCEED);
}

TEST_CASE("test_BenchmarkFaceHubSearchPersistence", "[benchmark]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    SECTION("Benchmark search 1k@Persistence") {
        const int loop = 1000;
        HResult ret;
        HFFeatureHubConfiguration configuration;
        auto dbPath = GET_SAVE_DATA(".test");
        HString dbPathStr = new char[dbPath.size() + 1];
        std::strcpy(dbPathStr, dbPath.c_str());
        configuration.primaryKeyMode = HF_PK_AUTO_INCREMENT;
        configuration.enablePersistence = 1;
        configuration.persistenceDbPath = dbPathStr;
        configuration.searchMode = HF_SEARCH_MODE_EXHAUSTIVE;
        configuration.searchThreshold = 0.48f;
        // Delete the previous data before testing
        if (std::remove(configuration.persistenceDbPath) != 0) {
            spdlog::trace("Error deleting file");
        }
        ret = HFFeatureHubDataEnable(configuration);
        REQUIRE(ret == HSUCCEED);

        std::vector<std::vector<HFloat>> baseFeatures;
        size_t genSizeOfBase = 1000;
        HInt32 featureLength;
        HFGetFeatureLength(&featureLength);
        REQUIRE(featureLength > 0);
        for (int i = 0; i < genSizeOfBase; ++i) {
            auto feat = GenerateRandomFeature(featureLength);
            baseFeatures.push_back(feat);
            // Construct face feature
            HFFaceFeature feature = {0};
            feature.size = feat.size();
            feature.data = feat.data();
            HFFaceFeatureIdentity identity = {0};
            identity.feature = &feature;
            HFaceId allocId;
            ret = HFFeatureHubInsertFeature(identity, &allocId);
            REQUIRE(ret == HSUCCEED);
        }
        HInt32 totalFace;
        ret = HFFeatureHubGetFaceCount(&totalFace);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(totalFace == genSizeOfBase);

        HInt32 targetId = 800;
        auto targetFeature = baseFeatures[targetId - 1];

        auto searchFeat = SimulateSimilarVector(targetFeature);
        HFFaceFeature searchFeature = {0};
        searchFeature.size = searchFeat.size();
        searchFeature.data = searchFeat.data();
        HFloat confidence = 0.0f;
        HFFaceFeatureIdentity mostSimilar = {0};
        inspire::SpendTimer timeSpend("Face Search 1k@Persistence");
        for (size_t i = 0; i < loop; i++) {
            timeSpend.Start();
            ret = HFFeatureHubFaceSearch(searchFeature, &confidence, &mostSimilar);
            REQUIRE(ret == HSUCCEED);
            REQUIRE(mostSimilar.id == targetId);
            REQUIRE(confidence > 0.88f);
            timeSpend.Stop();
        }
        std::cout << timeSpend << std::endl;

        ret = HFFeatureHubDataDisable();
        REQUIRE(ret == HSUCCEED);

        delete[] dbPathStr;
    }

    SECTION("Benchmark search 5k@Persistence") {
        const int loop = 1000;
        HResult ret;
        HFFeatureHubConfiguration configuration;
        auto dbPath = GET_SAVE_DATA(".test");
        HString dbPathStr = new char[dbPath.size() + 1];
        std::strcpy(dbPathStr, dbPath.c_str());
        configuration.primaryKeyMode = HF_PK_AUTO_INCREMENT;
        configuration.enablePersistence = 1;
        configuration.persistenceDbPath = dbPathStr;
        configuration.searchMode = HF_SEARCH_MODE_EXHAUSTIVE;
        configuration.searchThreshold = 0.48f;
        // Delete the previous data before testing
        if (std::remove(configuration.persistenceDbPath) != 0) {
            spdlog::trace("Error deleting file");
        }
        ret = HFFeatureHubDataEnable(configuration);
        REQUIRE(ret == HSUCCEED);

        std::vector<std::vector<HFloat>> baseFeatures;
        size_t genSizeOfBase = 5000;
        HInt32 featureLength;
        HFGetFeatureLength(&featureLength);
        REQUIRE(featureLength > 0);
        for (int i = 0; i < genSizeOfBase; ++i) {
            auto feat = GenerateRandomFeature(featureLength);
            baseFeatures.push_back(feat);
            // Construct face feature
            HFFaceFeature feature = {0};
            feature.size = feat.size();
            feature.data = feat.data();
            HFFaceFeatureIdentity identity = {0};
            identity.feature = &feature;
            HFaceId allocId;
            ret = HFFeatureHubInsertFeature(identity, &allocId);
            REQUIRE(ret == HSUCCEED);
        }
        HInt32 totalFace;
        ret = HFFeatureHubGetFaceCount(&totalFace);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(totalFace == genSizeOfBase);

        HInt32 targetId = 4800;
        auto targetFeature = baseFeatures[targetId - 1];

        auto searchFeat = SimulateSimilarVector(targetFeature);
        HFFaceFeature searchFeature = {0};
        searchFeature.size = searchFeat.size();
        searchFeature.data = searchFeat.data();
        HFloat confidence = 0.0f;
        HFFaceFeatureIdentity mostSimilar = {0};
        inspire::SpendTimer timeSpend("Face Search 5k@Persistence");
        for (size_t i = 0; i < loop; i++) {
            timeSpend.Start();
            ret = HFFeatureHubFaceSearch(searchFeature, &confidence, &mostSimilar);
            REQUIRE(ret == HSUCCEED);
            REQUIRE(mostSimilar.id == targetId);
            REQUIRE(confidence > 0.88f);
            timeSpend.Stop();
        }
        std::cout << timeSpend << std::endl;

        ret = HFFeatureHubDataDisable();
        REQUIRE(ret == HSUCCEED);

        delete[] dbPathStr;
    }

    SECTION("Benchmark search 10k@Persistence") {
        const int loop = 1000;
        HResult ret;
        HFFeatureHubConfiguration configuration;
        auto dbPath = GET_SAVE_DATA(".test");
        HString dbPathStr = new char[dbPath.size() + 1];
        std::strcpy(dbPathStr, dbPath.c_str());
        configuration.primaryKeyMode = HF_PK_AUTO_INCREMENT;
        configuration.enablePersistence = 1;
        configuration.persistenceDbPath = dbPathStr;
        configuration.searchMode = HF_SEARCH_MODE_EXHAUSTIVE;
        configuration.searchThreshold = 0.48f;
        // Delete the previous data before testing
        if (std::remove(configuration.persistenceDbPath) != 0) {
            spdlog::trace("Error deleting file");
        }
        ret = HFFeatureHubDataEnable(configuration);
        REQUIRE(ret == HSUCCEED);

        std::vector<std::vector<HFloat>> baseFeatures;
        size_t genSizeOfBase = 10000;
        HInt32 featureLength;
        HFGetFeatureLength(&featureLength);
        REQUIRE(featureLength > 0);
        for (int i = 0; i < genSizeOfBase; ++i) {
            auto feat = GenerateRandomFeature(featureLength);
            baseFeatures.push_back(feat);
            // Construct face feature
            HFFaceFeature feature = {0};
            feature.size = feat.size();
            feature.data = feat.data();
            HFFaceFeatureIdentity identity = {0};
            identity.feature = &feature;
            HFaceId allocId;
            ret = HFFeatureHubInsertFeature(identity, &allocId);
            REQUIRE(ret == HSUCCEED);
        }
        HInt32 totalFace;
        ret = HFFeatureHubGetFaceCount(&totalFace);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(totalFace == genSizeOfBase);

        HInt32 targetId = 9800;
        auto targetFeature = baseFeatures[targetId - 1];

        auto searchFeat = SimulateSimilarVector(targetFeature);
        HFFaceFeature searchFeature = {0};
        searchFeature.size = searchFeat.size();
        searchFeature.data = searchFeat.data();
        HFloat confidence = 0.0f;
        HFFaceFeatureIdentity mostSimilar = {0};
        inspire::SpendTimer timeSpend("Face Search 10k@Persistence");
        for (size_t i = 0; i < loop; i++) {
            timeSpend.Start();
            ret = HFFeatureHubFaceSearch(searchFeature, &confidence, &mostSimilar);
            REQUIRE(ret == HSUCCEED);
            REQUIRE(mostSimilar.id == targetId);
            REQUIRE(confidence > 0.88f);
            timeSpend.Stop();
        }
        std::cout << timeSpend << std::endl;

        ret = HFFeatureHubDataDisable();
        REQUIRE(ret == HSUCCEED);

        delete[] dbPathStr;
    }
}

TEST_CASE("test_BenchmarkFaceHubSearchMemory", "[benchmark]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    SECTION("Benchmark search 1k@Memory") {
        const int loop = 1000;
        HResult ret;
        HFFeatureHubConfiguration configuration;
        configuration.primaryKeyMode = HF_PK_AUTO_INCREMENT;
        configuration.enablePersistence = 0;
        configuration.searchMode = HF_SEARCH_MODE_EXHAUSTIVE;
        configuration.searchThreshold = 0.48f;
        ret = HFFeatureHubDataEnable(configuration);
        REQUIRE(ret == HSUCCEED);

        std::vector<std::vector<HFloat>> baseFeatures;
        size_t genSizeOfBase = 1000;
        HInt32 featureLength;
        HFGetFeatureLength(&featureLength);
        REQUIRE(featureLength > 0);
        for (int i = 0; i < genSizeOfBase; ++i) {
            auto feat = GenerateRandomFeature(featureLength);
            baseFeatures.push_back(feat);
            // Construct face feature
            HFFaceFeature feature = {0};
            feature.size = feat.size();
            feature.data = feat.data();
            HFFaceFeatureIdentity identity = {0};
            identity.feature = &feature;
            HFaceId allocId;
            ret = HFFeatureHubInsertFeature(identity, &allocId);
            REQUIRE(ret == HSUCCEED);
        }
        HInt32 totalFace;
        ret = HFFeatureHubGetFaceCount(&totalFace);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(totalFace == genSizeOfBase);

        HInt32 targetId = 800;
        auto targetFeature = baseFeatures[targetId - 1];

        auto searchFeat = SimulateSimilarVector(targetFeature);
        HFFaceFeature searchFeature = {0};
        searchFeature.size = searchFeat.size();
        searchFeature.data = searchFeat.data();
        HFloat confidence = 0.0f;
        HFFaceFeatureIdentity mostSimilar = {0};
        inspire::SpendTimer timeSpend("Face Search 1k@Memory");
        for (size_t i = 0; i < loop; i++) {
            timeSpend.Start();
            ret = HFFeatureHubFaceSearch(searchFeature, &confidence, &mostSimilar);
            REQUIRE(ret == HSUCCEED);
            REQUIRE(mostSimilar.id == targetId);
            REQUIRE(confidence > 0.88f);
            timeSpend.Stop();
        }
        std::cout << timeSpend << std::endl;

        ret = HFFeatureHubDataDisable();
        REQUIRE(ret == HSUCCEED);
    }

    SECTION("Benchmark search 5k@Persistence") {
        const int loop = 1000;
        HResult ret;
        HFFeatureHubConfiguration configuration;
        configuration.primaryKeyMode = HF_PK_AUTO_INCREMENT;
        configuration.enablePersistence = 0;
        configuration.searchMode = HF_SEARCH_MODE_EXHAUSTIVE;
        configuration.searchThreshold = 0.48f;
        ret = HFFeatureHubDataEnable(configuration);
        REQUIRE(ret == HSUCCEED);

        std::vector<std::vector<HFloat>> baseFeatures;
        size_t genSizeOfBase = 5000;
        HInt32 featureLength;
        HFGetFeatureLength(&featureLength);
        REQUIRE(featureLength > 0);
        for (int i = 0; i < genSizeOfBase; ++i) {
            auto feat = GenerateRandomFeature(featureLength);
            baseFeatures.push_back(feat);
            // Construct face feature
            HFFaceFeature feature = {0};
            feature.size = feat.size();
            feature.data = feat.data();
            HFFaceFeatureIdentity identity = {0};
            identity.feature = &feature;
            HFaceId allocId;
            ret = HFFeatureHubInsertFeature(identity, &allocId);
            REQUIRE(ret == HSUCCEED);
        }
        HInt32 totalFace;
        ret = HFFeatureHubGetFaceCount(&totalFace);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(totalFace == genSizeOfBase);

        HInt32 targetId = 4800;
        auto targetFeature = baseFeatures[targetId - 1];

        auto searchFeat = SimulateSimilarVector(targetFeature);
        HFFaceFeature searchFeature = {0};
        searchFeature.size = searchFeat.size();
        searchFeature.data = searchFeat.data();
        HFloat confidence = 0.0f;
        HFFaceFeatureIdentity mostSimilar = {0};
        inspire::SpendTimer timeSpend("Face Search 5k@Memory");
        for (size_t i = 0; i < loop; i++) {
            timeSpend.Start();
            ret = HFFeatureHubFaceSearch(searchFeature, &confidence, &mostSimilar);
            REQUIRE(ret == HSUCCEED);
            REQUIRE(mostSimilar.id == targetId);
            REQUIRE(confidence > 0.88f);
            timeSpend.Stop();
        }
        std::cout << timeSpend << std::endl;

        ret = HFFeatureHubDataDisable();
        REQUIRE(ret == HSUCCEED);
    }

    SECTION("Benchmark search 10k@Persistence") {
        const int loop = 1000;
        HResult ret;
        HFFeatureHubConfiguration configuration;
        configuration.primaryKeyMode = HF_PK_AUTO_INCREMENT;
        configuration.enablePersistence = 0;
        configuration.searchMode = HF_SEARCH_MODE_EXHAUSTIVE;
        configuration.searchThreshold = 0.48f;

        ret = HFFeatureHubDataEnable(configuration);
        REQUIRE(ret == HSUCCEED);

        std::vector<std::vector<HFloat>> baseFeatures;
        size_t genSizeOfBase = 10000;
        HInt32 featureLength;
        HFGetFeatureLength(&featureLength);
        REQUIRE(featureLength > 0);
        for (int i = 0; i < genSizeOfBase; ++i) {
            auto feat = GenerateRandomFeature(featureLength);
            baseFeatures.push_back(feat);
            // Construct face feature
            HFFaceFeature feature = {0};
            feature.size = feat.size();
            feature.data = feat.data();
            HFFaceFeatureIdentity identity = {0};
            identity.feature = &feature;
            HFaceId allocId;
            ret = HFFeatureHubInsertFeature(identity, &allocId);
            REQUIRE(ret == HSUCCEED);
        }
        HInt32 totalFace;
        ret = HFFeatureHubGetFaceCount(&totalFace);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(totalFace == genSizeOfBase);

        HInt32 targetId = 9800;
        auto targetFeature = baseFeatures[targetId - 1];

        auto searchFeat = SimulateSimilarVector(targetFeature);
        HFFaceFeature searchFeature = {0};
        searchFeature.size = searchFeat.size();
        searchFeature.data = searchFeat.data();
        HFloat confidence = 0.0f;
        HFFaceFeatureIdentity mostSimilar = {0};
        inspire::SpendTimer timeSpend("Face Search 10k@Memory");
        for (size_t i = 0; i < loop; i++) {
            timeSpend.Start();
            ret = HFFeatureHubFaceSearch(searchFeature, &confidence, &mostSimilar);
            REQUIRE(ret == HSUCCEED);
            REQUIRE(mostSimilar.id == targetId);
            REQUIRE(confidence > 0.88f);
            timeSpend.Stop();
        }
        std::cout << timeSpend << std::endl;

        ret = HFFeatureHubDataDisable();
        REQUIRE(ret == HSUCCEED);
    }
}

#endif