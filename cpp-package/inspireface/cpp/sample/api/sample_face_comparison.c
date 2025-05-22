/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inspireface.h>

#define NUM_IMAGES 2

int main(int argc, char* argv[]) {
    HResult ret;
    const char* packPath;
    const char* imgPath1;
    const char* imgPath2;
    HOption option;
    HFSession session;
    HFFaceFeature features[NUM_IMAGES];
    const char* imgPaths[NUM_IMAGES];
    int i;
    HFloat similarity;
    HFloat recommended_cosine_threshold;
    HFloat percentage;

    /* Check whether the number of parameters is correct */
    if (argc != 4) {
        HFLogPrint(HF_LOG_ERROR, "Usage: %s <pack_path> <img1_path> <img2_path>", argv[0]);
        return 1;
    }

    packPath = argv[1];
    imgPath1 = argv[2];
    imgPath2 = argv[3];

    /* Initialize features array to NULL */
    memset(features, 0, sizeof(features));

    /* Allocate memory for feature vectors */
    for (i = 0; i < NUM_IMAGES; i++) {
        ret = HFCreateFaceFeature(&features[i]);
        if (ret != HSUCCEED) {
            HFLogPrint(HF_LOG_ERROR, "Create face feature error: %d", ret);
            goto cleanup;
        }
    }

    /* Set the image path array */
    imgPaths[0] = imgPath1;
    imgPaths[1] = imgPath2;

    HFLogPrint(HF_LOG_INFO, "Pack file Path: %s", packPath);
    HFLogPrint(HF_LOG_INFO, "Source file Path 1: %s", imgPath1);
    HFLogPrint(HF_LOG_INFO, "Source file Path 2: %s", imgPath2);

    /* The resource file must be loaded before it can be used */
    ret = HFLaunchInspireFace(packPath);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Load Resource error: %d", ret);
        goto cleanup;
    }

    /* Create a session for face recognition */
    option = HF_ENABLE_FACE_RECOGNITION;
    ret = HFCreateInspireFaceSessionOptional(option, HF_DETECT_MODE_ALWAYS_DETECT, 1, -1, -1, &session);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create session error: %d", ret);
        goto cleanup;
    }

    /* Process two images */
    for (i = 0; i < NUM_IMAGES; i++) {
        HFImageBitmap imageBitmap = {0};
        HFImageStream stream;
        HFMultipleFaceData multipleFaceData = {0};

        ret = HFCreateImageBitmapFromFilePath(imgPaths[i], 3, &imageBitmap);
        if (ret != HSUCCEED) {
            HFReleaseImageBitmap(imageBitmap);
            HFLogPrint(HF_LOG_ERROR, "Create image bitmap error: %d", ret);
            goto cleanup;
        }

        ret = HFCreateImageStreamFromImageBitmap(imageBitmap, HF_CAMERA_ROTATION_0, &stream);
        if (ret != HSUCCEED) {
            HFReleaseImageStream(stream);
            HFReleaseImageBitmap(imageBitmap);
            HFLogPrint(HF_LOG_ERROR, "Create stream error: %d", ret);
            goto cleanup;
        }

        ret = HFExecuteFaceTrack(session, stream, &multipleFaceData);
        if (ret != HSUCCEED) {
            HFReleaseImageStream(stream);
            HFReleaseImageBitmap(imageBitmap);
            HFLogPrint(HF_LOG_ERROR, "Run face track error: %d", ret);
            goto cleanup;
        }

        if (multipleFaceData.detectedNum == 0) {
            HFReleaseImageStream(stream);
            HFReleaseImageBitmap(imageBitmap);
            HFLogPrint(HF_LOG_ERROR, "No face was detected: %s", imgPaths[i]);
            goto cleanup;
        }

        ret = HFFaceFeatureExtractTo(session, stream, multipleFaceData.tokens[0], features[i]);
        if (ret != HSUCCEED) {
            HFReleaseImageStream(stream);
            HFReleaseImageBitmap(imageBitmap);
            HFLogPrint(HF_LOG_ERROR, "Extract feature error: %d", ret);
            goto cleanup;
        }

        HFReleaseImageStream(stream);
        HFReleaseImageBitmap(imageBitmap);
    }

    HFFaceFeature feature1 = features[0];
    HFFaceFeature feature2 = features[1];

    /* Run comparison */
    ret = HFFaceComparison(feature1, feature2, &similarity);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Feature comparison error: %d", ret);
        goto cleanup;
    }

    ret = HFGetRecommendedCosineThreshold(&recommended_cosine_threshold);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Get recommended cosine threshold error: %d", ret);
        goto cleanup;
    }

    if (similarity > recommended_cosine_threshold) {
        HFLogPrint(HF_LOG_INFO, "%.3f > %.3f ✓ Same face", similarity, recommended_cosine_threshold);
    } else {
        HFLogPrint(HF_LOG_WARN, "%.3f < %.3f ✗ Different face", similarity, recommended_cosine_threshold);
    }
    HFLogPrint(HF_LOG_INFO, "Similarity score: %.3f", similarity);

    ret = HFCosineSimilarityConvertToPercentage(similarity, &percentage);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Convert similarity to percentage error: %d", ret);
        goto cleanup;
    }
    HFLogPrint(HF_LOG_INFO, "Percentage similarity: %f", percentage);

    /* Clean up resources */
    ret = HFReleaseInspireFaceSession(session);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Release session error: %d", ret);
    }
    
cleanup:
    /* Release the feature vector memory */
    for (i = 0; i < NUM_IMAGES; i++) {
        if (features[i].data != NULL) {  // Only release features that were successfully created
            HFReleaseFaceFeature(&features[i]);
        }
    }
    
    HFDeBugShowResourceStatistics();
    
    return ret;
}