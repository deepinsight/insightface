/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#include <iostream>
#include <vector>
#include <inspireface.h>

int main(int argc, char* argv[]) {
    // Check whether the number of parameters is correct
    if (argc != 4) {
        HFLogPrint(HF_LOG_ERROR, "Usage: %s <pack_path> <img1_path> <img2_path>", argv[0]);
        return 1;
    }

    auto packPath = argv[1];
    auto imgPath1 = argv[2];
    auto imgPath2 = argv[3];

    HFLogPrint(HF_LOG_INFO, "Pack file Path: %s", packPath);
    HFLogPrint(HF_LOG_INFO, "Source file Path 1: %s", imgPath1);
    HFLogPrint(HF_LOG_INFO, "Source file Path 2: %s", imgPath2);

    HResult ret;
    // The resource file must be loaded before it can be used
    ret = HFLaunchInspireFace(packPath);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Load Resource error: %d", ret);
        return ret;
    }

    // Create a session for face recognition
    HOption option = HF_ENABLE_FACE_RECOGNITION;
    HFSession session;
    ret = HFCreateInspireFaceSessionOptional(option, HF_DETECT_MODE_ALWAYS_DETECT, 1, -1, -1, &session);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create session error: %d", ret);
        return ret;
    }

    std::vector<char*> twoImg = {imgPath1, imgPath2};
    std::vector<std::vector<float>> vec(2, std::vector<float>(512));
    for (int i = 0; i < twoImg.size(); ++i) {
        HFImageBitmap imageBitmap = {0};
        ret = HFCreateImageBitmapFromFilePath(twoImg[i], 3, &imageBitmap);
        if (ret != HSUCCEED) {
            HFLogPrint(HF_LOG_ERROR, "Create image bitmap error: %d", ret);
            return ret;
        }
        // Prepare image data for processing

        HFImageStream stream;
        ret = HFCreateImageStreamFromImageBitmap(imageBitmap, HF_CAMERA_ROTATION_0, &stream);  // Create an image stream for processing
        if (ret != HSUCCEED) {
            HFLogPrint(HF_LOG_ERROR, "Create stream error: %d", ret);
            return ret;
        }

        // Execute face tracking on the image
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, stream, &multipleFaceData);  // Track faces in the image
        if (ret != HSUCCEED) {
            HFLogPrint(HF_LOG_ERROR, "Run face track error: %d", ret);
            return ret;
        }
        if (multipleFaceData.detectedNum == 0) {  // Check if any faces were detected
            HFLogPrint(HF_LOG_ERROR, "No face was detected: %s", twoImg[i]);
            return ret;
        }

        // Extract facial features from the first detected face, an interface that uses copy features in a comparison scenario
        ret = HFFaceFeatureExtractCpy(session, stream, multipleFaceData.tokens[0], vec[i].data());  // Extract features
        if (ret != HSUCCEED) {
            HFLogPrint(HF_LOG_ERROR, "Extract feature error: %d", ret);
            return ret;
        }

        ret = HFReleaseImageStream(stream);
        if (ret != HSUCCEED) {
            HFLogPrint(HF_LOG_ERROR, "Release image stream error: %d", ret);
        }
        ret = HFReleaseImageBitmap(imageBitmap);
        if (ret != HSUCCEED) {
            HFLogPrint(HF_LOG_ERROR, "Release image bitmap error: %d", ret);
            return ret;
        }
    }

    // Make feature1
    HFFaceFeature feature1 = {0};
    feature1.data = vec[0].data();
    feature1.size = vec[0].size();

    // Make feature2
    HFFaceFeature feature2 = {0};
    feature2.data = vec[1].data();
    feature2.size = vec[1].size();

    // Run comparison
    HFloat similarity;
    ret = HFFaceComparison(feature1, feature2, &similarity);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Feature comparison error: %d", ret);
        return ret;
    }

    HFloat recommended_cosine_threshold;
    ret = HFGetRecommendedCosineThreshold(&recommended_cosine_threshold);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Get recommended cosine threshold error: %d", ret);
        return ret;
    }

    if (similarity > recommended_cosine_threshold) {
        HFLogPrint(HF_LOG_INFO, "%.3f > %.3f ✓ Same face", similarity, recommended_cosine_threshold);
    } else {
        HFLogPrint(HF_LOG_WARN, "%.3f < %.3f ✗ Different face", similarity, recommended_cosine_threshold);
    }
    HFLogPrint(HF_LOG_INFO, "Similarity score: %.3f", similarity);

    // Convert cosine similarity to percentage similarity.
    // Note: conversion parameters are not optimal and should be adjusted based on your specific use case.
    HFloat percentage;
    ret = HFCosineSimilarityConvertToPercentage(similarity, &percentage);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Convert similarity to percentage error: %d", ret);
        return ret;
    }
    HFLogPrint(HF_LOG_INFO, "Percentage similarity: %f", percentage);

    // The memory must be freed at the end of the program
    ret = HFReleaseInspireFaceSession(session);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Release session error: %d", ret);
        return ret;
    }
}