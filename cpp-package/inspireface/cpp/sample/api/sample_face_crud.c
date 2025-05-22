#include <inspireface.h>
#include <stdio.h>

int main() {
    HResult ret;
    // The resource file must be loaded before it can be used
    ret = HFLaunchInspireFace("test_res/pack/Pikachu");
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Load Resource error: %d", ret);
        return ret;
    }

    char *db_path = "case_crud.db";
    if (remove(db_path) != 0) {
        HFLogPrint(HF_LOG_ERROR, "Remove database file error: %d", ret);
        return ret;
    }
    
    HFFeatureHubConfiguration configuration;
    configuration.primaryKeyMode = HF_PK_AUTO_INCREMENT;
    configuration.enablePersistence = 1;
    configuration.persistenceDbPath = db_path;
    configuration.searchMode = HF_SEARCH_MODE_EXHAUSTIVE;
    configuration.searchThreshold = 0.48f;
    ret = HFFeatureHubDataEnable(configuration);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Enable feature hub error: %d", ret);
        return ret;
    }

    // Create a session
    HFSession session;
    ret = HFCreateInspireFaceSessionOptional(HF_ENABLE_FACE_RECOGNITION, HF_DETECT_MODE_ALWAYS_DETECT, 1, 320, -1, &session);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create session error: %d", ret);
        return ret;
    }
    
    // Prepare an image for insertion into the hub
    HFImageBitmap image;
    ret = HFCreateImageBitmapFromFilePath("test_res/data/bulk/kun.jpg", 3, &image);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create image bitmap error: %d", ret);
        return ret;
    }
    
    // Create an image stream
    HFImageStream imageHandle;
    ret = HFCreateImageStreamFromImageBitmap(image, HF_CAMERA_ROTATION_0, &imageHandle);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create image stream error: %d", ret);
        return ret;
    }

    // Detect and track
    HFMultipleFaceData multipleFaceData;
    ret = HFExecuteFaceTrack(session, imageHandle, &multipleFaceData);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Execute face track error: %d", ret);
        return ret;
    }

    if (multipleFaceData.detectedNum > 0) {
        HFLogPrint(HF_LOG_INFO, "Face detected: %d", multipleFaceData.detectedNum);
    }

    HFFaceFeature feature;
    ret = HFCreateFaceFeature(&feature);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create face feature error: %d", ret);
        return ret;
    }

    ret = HFFaceFeatureExtractCpy(session, imageHandle, multipleFaceData.tokens[0], feature.data);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Extract feature error: %d", ret);
        return ret;
    }

    // Insert face feature into the hub
    HFFaceFeatureIdentity featureIdentity;
    featureIdentity.feature = &feature;
    featureIdentity.id = -1;
    HFaceId result_id;
    ret = HFFeatureHubInsertFeature(featureIdentity, &result_id);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Insert feature error: %d", ret);
        return ret;
    }

    // Prepare a photo of the same person for the query
    HFImageBitmap query_image;
    ret = HFCreateImageBitmapFromFilePath("test_res/data/bulk/jntm.jpg", 3, &query_image);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create image bitmap error: %d", ret);
        return ret;
    }

    // Create an image stream
    HFImageStream query_imageHandle;
    ret = HFCreateImageStreamFromImageBitmap(query_image, HF_CAMERA_ROTATION_0, &query_imageHandle);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create image stream error: %d", ret);
        return ret;
    }

    // Detect and track
    ret = HFExecuteFaceTrack(session, query_imageHandle, &multipleFaceData);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Execute face track error: %d", ret);
        return ret;
    }

    if (multipleFaceData.detectedNum > 0) {
        HFLogPrint(HF_LOG_INFO, "Face detected: %d", multipleFaceData.detectedNum);
    }

    HFFaceFeature query_feature;
    ret = HFCreateFaceFeature(&query_feature);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create face feature error: %d", ret);
        return ret;
    }
    
    // Extract face feature
    ret = HFFaceFeatureExtractTo(session, query_imageHandle, multipleFaceData.tokens[0], query_feature);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Extract feature error: %d", ret);
        return ret;
    }

    // Search face feature
    HFFaceFeatureIdentity query_featureIdentity;
    query_featureIdentity.feature = &query_feature;
    query_featureIdentity.id = -1;
    HFloat confidence;
    ret = HFFeatureHubFaceSearch(query_feature, &confidence, &query_featureIdentity);

    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Search feature error: %d", ret);
        return ret;
    }

    HFLogPrint(HF_LOG_INFO, "Search feature result: %d", query_featureIdentity.id);
    HFLogPrint(HF_LOG_INFO, "Search feature confidence: %f", confidence);

    // Remove face feature
    ret = HFFeatureHubFaceRemove(result_id);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Remove feature error: %d", ret);
        return ret;
    }
    HFLogPrint(HF_LOG_INFO, "Remove feature result: %d", result_id);    

    // Query again
    ret = HFFeatureHubFaceSearch(query_feature, &confidence, &query_featureIdentity);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Search feature error: %d", ret);
        return ret;
    }
    HFLogPrint(HF_LOG_INFO, "Query again, search feature result: %d", query_featureIdentity.id);
    if (query_featureIdentity.id != -1) {
        HFLogPrint(HF_LOG_INFO, "Remove feature failed");
    }

    // Release resources
    HFReleaseFaceFeature(&feature);
    HFReleaseFaceFeature(&query_feature);
    HFReleaseImageStream(imageHandle);
    HFReleaseImageStream(query_imageHandle);
    HFReleaseImageBitmap(image);
    HFReleaseImageBitmap(query_image);
    HFReleaseInspireFaceSession(session);

    HFDeBugShowResourceStatistics();

    return 0;
}