#include <inspireface.h>
#include <unistd.h>
#include <stdio.h>


int main(int argc, char* argv[]) {
    if (argc != 4) {
        HFLogPrint(HF_LOG_ERROR, "Usage: %s <pack_path> <feature_path> <image_path>", argv[0]);
        return -1;
    }
    
    char* packPath = argv[1];
    char* featurePath = argv[2];
    char* imagePath = argv[3];
    
    HResult ret;
    ret = HFLaunchInspireFace(packPath);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Load Resource error: %d", ret);
        return ret;
    }
    
    // 1. Enable feature hub
    HFFeatureHubConfiguration featureHubConfiguration;
    featureHubConfiguration.primaryKeyMode = HF_PK_AUTO_INCREMENT; 
    featureHubConfiguration.enablePersistence = 1;
    featureHubConfiguration.persistenceDbPath = featurePath;
    featureHubConfiguration.searchThreshold = 0.32f;
    featureHubConfiguration.searchMode = HF_SEARCH_MODE_EXHAUSTIVE;
    ret = HFFeatureHubDataEnable(featureHubConfiguration);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Enable feature hub error: %d", ret);
        return ret;
    }

    // 2. Create session
    HFSession session;
    ret = HFCreateInspireFaceSessionOptional(HF_ENABLE_FACE_RECOGNITION, HF_DETECT_MODE_ALWAYS_DETECT, 1, 320, -1, &session);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create session error: %d", ret);
        return ret;
    }
    
    // 3. Load image
    HFImageBitmap image;
    ret = HFCreateImageBitmapFromFilePath(imagePath, 3, &image);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create image bitmap error: %d", ret);
        HFReleaseInspireFaceSession(session);
        return ret;
    }
    
    // 4. Create image stream
    HFImageStream imageHandle;
    ret = HFCreateImageStreamFromImageBitmap(image, HF_CAMERA_ROTATION_0, &imageHandle);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create image stream error: %d", ret);
        HFReleaseImageBitmap(image);
        HFReleaseInspireFaceSession(session);
        return ret;
    }

    // 5. Detect face
    HFMultipleFaceData multipleFaceData;
    ret = HFExecuteFaceTrack(session, imageHandle, &multipleFaceData);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Execute face track error: %d", ret);
        HFReleaseImageStream(imageHandle);
        HFReleaseImageBitmap(image);
        HFReleaseInspireFaceSession(session);
        return ret;
    }

    if (multipleFaceData.detectedNum == 0) {
        HFLogPrint(HF_LOG_WARN, "No face detected in image: %s", imagePath);
        HFReleaseImageStream(imageHandle);
        HFReleaseImageBitmap(image);
        HFReleaseInspireFaceSession(session);
        return -1;
    }

    HFLogPrint(HF_LOG_INFO, "Face detected: %d", multipleFaceData.detectedNum);

    // 6. Extract feature
    HFFaceFeature feature;
    ret = HFCreateFaceFeature(&feature);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create face feature error: %d", ret);
        HFReleaseImageStream(imageHandle);
        HFReleaseImageBitmap(image);
        HFReleaseInspireFaceSession(session);
        return ret;
    }

    ret = HFFaceFeatureExtractTo(session, imageHandle, multipleFaceData.tokens[0], feature);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Extract feature error: %d", ret);
        HFReleaseFaceFeature(&feature);
        HFReleaseImageStream(imageHandle);
        HFReleaseImageBitmap(image);
        HFReleaseInspireFaceSession(session);
        return ret;
    }

    // 7. Search face
    HFloat confidence;
    HFFaceFeatureIdentity searchResult = {0};
    ret = HFFeatureHubFaceSearch(feature, &confidence, &searchResult);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Search face feature error: %d", ret);
    } else {
        if (searchResult.id != -1) {
            HFLogPrint(HF_LOG_INFO, "Search face feature success!");
            HFLogPrint(HF_LOG_INFO, "Found matching face ID: %d", searchResult.id);
            HFLogPrint(HF_LOG_INFO, "Confidence score: %.4f", confidence);
        } else {
            HFLogPrint(HF_LOG_INFO, "No matching face found in database");
        }
    }

    // 8. Clean up
    HFReleaseFaceFeature(&feature);
    HFReleaseImageStream(imageHandle);
    HFReleaseImageBitmap(image);
    HFReleaseInspireFaceSession(session);

    HFDeBugShowResourceStatistics();

    return ret;
}