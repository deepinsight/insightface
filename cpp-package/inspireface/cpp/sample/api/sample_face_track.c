/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#include <stdio.h>
#include <stdlib.h>
#include <inspireface.h>

int main(int argc, char* argv[]) {
    HResult ret;
    const char* packPath;
    const char* sourcePath;
    int rotation;
    HFRotation rotation_enum;
    HOption option;
    HFDetectMode detMode;
    HInt32 maxDetectNum;
    HInt32 detectPixelLevel;
    HFSession session;
    HFImageBitmap image;
    HFImageStream imageHandle;
    HFMultipleFaceData multipleFaceData;
    int faceNum;
    HFImageBitmap drawImage;
    HFImageBitmapData data;
    int index;
    HFFaceMaskConfidence maskConfidence;
    HFFaceQualityConfidence qualityConfidence;
    HOption pipelineOption;
    HFFaceDetectPixelList pixelLevels;
    HFFaceEmotionResult faceEmotionResult;

    /* Check whether the number of parameters is correct */
    if (argc < 3 || argc > 4) {
        HFLogPrint(HF_LOG_ERROR, "Usage: %s <pack_path> <source_path> [rotation]", argv[0]);
        return 1;
    }

    packPath = argv[1];
    sourcePath = argv[2];
    rotation = 0;

    /* If rotation is provided, check and set the value */
    if (argc == 4) {
        rotation = atoi(argv[3]);
        if (rotation != 0 && rotation != 90 && rotation != 180 && rotation != 270) {
            HFLogPrint(HF_LOG_ERROR, "Invalid rotation value. Allowed values are 0, 90, 180, 270.");
            return 1;
        }
    }

    /* Set rotation based on input parameter */
    switch (rotation) {
        case 90:
            rotation_enum = HF_CAMERA_ROTATION_90;
            break;
        case 180:
            rotation_enum = HF_CAMERA_ROTATION_180;
            break;
        case 270:
            rotation_enum = HF_CAMERA_ROTATION_270;
            break;
        case 0:
        default:
            rotation_enum = HF_CAMERA_ROTATION_0;
            break;
    }

    HFLogPrint(HF_LOG_INFO, "Pack file Path: %s", packPath);
    HFLogPrint(HF_LOG_INFO, "Source file Path: %s", sourcePath);
    HFLogPrint(HF_LOG_INFO, "Rotation: %d", rotation);

    HFSetLogLevel(HF_LOG_DEBUG);

    /* The resource file must be loaded before it can be used */
    ret = HFLaunchInspireFace(packPath);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Load Resource error: %d", ret);
        return ret;
    }

    ret = HFQuerySupportedPixelLevelsForFaceDetection(&pixelLevels);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "HFQuerySupportedPixelLevelsForFaceDetection error: %d", ret);
        return ret;
    }
    HFLogPrint(HF_LOG_INFO, "Supported pixel levels for face detection: %d", pixelLevels.size);
    for (int i = 0; i < pixelLevels.size; i++) {
        HFLogPrint(HF_LOG_INFO, "Supported pixel level %d: %d", i + 1, pixelLevels.pixel_level[i]);
    }

    /* Enable the functions in the pipeline: mask detection, live detection, and face quality
     * detection */
    option = HF_ENABLE_QUALITY | HF_ENABLE_MASK_DETECT | HF_ENABLE_LIVENESS | HF_ENABLE_FACE_EMOTION;
    /* Non-video or frame sequence mode uses IMAGE-MODE, which is always face detection without
     * tracking */
    detMode = HF_DETECT_MODE_LIGHT_TRACK;
    /* Maximum number of faces detected */
    maxDetectNum = 20;
    /* Face detection image input level */
    detectPixelLevel = 320;
    /* Handle of the current face SDK algorithm context */
    session = NULL;
    ret = HFCreateInspireFaceSessionOptional(option, detMode, maxDetectNum, detectPixelLevel, -1, &session);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create FaceContext error: %d", ret);
        return ret;
    }

    HFSessionSetTrackPreviewSize(session, detectPixelLevel);
    HFSessionSetFilterMinimumFacePixelSize(session, 4);

    /* Load a image */
    ret = HFCreateImageBitmapFromFilePath(sourcePath, 3, &image);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "The source entered is not a picture or read error.");
        return ret;
    }
    /* Prepare an image parameter structure for configuration */
    ret = HFCreateImageStreamFromImageBitmap(image, rotation_enum, &imageHandle);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create ImageStream error: %d", ret);
        return ret;
    }

    /* Execute HF_FaceContextRunFaceTrack captures face information in an image */
    ret = HFExecuteFaceTrack(session, imageHandle, &multipleFaceData);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Execute HFExecuteFaceTrack error: %d", ret);
        return ret;
    }

    /* Print the number of faces detected */
    faceNum = multipleFaceData.detectedNum;
    HFLogPrint(HF_LOG_INFO, "Num of face: %d", faceNum);

    /* Copy a new image to draw */
    ret = HFImageBitmapCopy(image, &drawImage);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Copy ImageBitmap error: %d", ret);
        return ret;
    }
    ret = HFImageBitmapGetData(drawImage, &data);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Get ImageBitmap data error: %d", ret);
        return ret;
    }
    for (index = 0; index < faceNum; ++index) {
        HInt32 numOfLmk;
        HPoint2f* denseLandmarkPoints;
        HPoint2f fiveKeyPoints[5];
        float area;
        size_t i;

        HFLogPrint(HF_LOG_INFO, "========================================");
        HFLogPrint(HF_LOG_INFO, "Token size: %d", multipleFaceData.tokens[index].size);
        HFLogPrint(HF_LOG_INFO, "Process face index: %d", index);
        HFLogPrint(HF_LOG_INFO, "DetConfidence: %f", multipleFaceData.detConfidence[index]);
        HFLogPrint(HF_LOG_INFO, "TrackCount: %d", multipleFaceData.trackCounts[index]);
        
        HFImageBitmapDrawRect(drawImage, multipleFaceData.rects[index], (HColor){0, 100, 255}, 4);

        /* Get the number of dense landmark points */
        HFGetNumOfFaceDenseLandmark(&numOfLmk);
        denseLandmarkPoints = (HPoint2f*)malloc(sizeof(HPoint2f) * numOfLmk);
        if (denseLandmarkPoints == NULL) {
            HFLogPrint(HF_LOG_ERROR, "Memory allocation failed!");
            return -1;
        }

        ret = HFGetFaceDenseLandmarkFromFaceToken(multipleFaceData.tokens[index], denseLandmarkPoints, numOfLmk);
        if (ret != HSUCCEED) {
            free(denseLandmarkPoints);
            HFLogPrint(HF_LOG_ERROR, "HFGetFaceDenseLandmarkFromFaceToken error!!");
            return -1;
        }

        /* Draw dense landmark points */
        for (i = 0; i < numOfLmk; i++) {
            HFImageBitmapDrawCircleF(drawImage, 
                                   (HPoint2f){denseLandmarkPoints[i].x, denseLandmarkPoints[i].y}, 
                                   0, 
                                   (HColor){100, 100, 0}, 
                                   2);
        }
        free(denseLandmarkPoints);

        HFaceRect rt = multipleFaceData.rects[index];
        area = ((float)(rt.height * rt.width)) / (data.width * data.height);
        HFLogPrint(HF_LOG_INFO, "area: %f", area);

        ret = HFGetFaceFiveKeyPointsFromFaceToken(multipleFaceData.tokens[index], fiveKeyPoints, 5);
        if (ret != HSUCCEED) {
            HFLogPrint(HF_LOG_ERROR, "HFGetFaceFiveKeyPointsFromFaceToken error!!");
            return -1;
        }
        for (i = 0; i < 5; i++) {
            HFImageBitmapDrawCircleF(drawImage, (HPoint2f){fiveKeyPoints[i].x, fiveKeyPoints[i].y}, 0, (HColor){0, 0, 232}, 2);
        }
    }
    HFImageBitmapWriteToFile(drawImage, "draw_detected.jpg");
    HFLogPrint(HF_LOG_WARN, "Write to file success: %s", "draw_detected.jpg");

    /* Run pipeline function */
    /* Select the pipeline function that you want to execute, provided that it is already enabled
     * when FaceContext is created! */
    pipelineOption = HF_ENABLE_QUALITY | HF_ENABLE_MASK_DETECT | HF_ENABLE_LIVENESS | HF_ENABLE_FACE_EMOTION;
    /* In this loop, all faces are processed */
    ret = HFMultipleFacePipelineProcessOptional(session, imageHandle, &multipleFaceData, pipelineOption);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Execute Pipeline error: %d", ret);
        return ret;
    }

    /* Get mask detection results from the pipeline cache */
    ret = HFGetFaceMaskConfidence(session, &maskConfidence);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Get mask detect result error: %d", ret);
        return -1;
    }

    /* Get face quality results from the pipeline cache */
    ret = HFGetFaceQualityConfidence(session, &qualityConfidence);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Get face quality result error: %d", ret);
        return -1;
    }

    ret = HFGetFaceEmotionResult(session, &faceEmotionResult);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Get face emotion result error: %d", ret);
        return -1;
    }

    for (index = 0; index < faceNum; ++index) {
        HFLogPrint(HF_LOG_INFO, "========================================");
        HFLogPrint(HF_LOG_INFO, "Process face index from pipeline: %d", index);
        HFLogPrint(HF_LOG_INFO, "Mask detect result: %f", maskConfidence.confidence[index]);
        HFLogPrint(HF_LOG_INFO, "Quality predict result: %f", qualityConfidence.confidence[index]);
        HFLogPrint(HF_LOG_INFO, "Emotion result: %d", faceEmotionResult.emotion[index]);
        /* We set the threshold of wearing a mask as 0.85. If it exceeds the threshold, it will be
         * judged as wearing a mask. The threshold can be adjusted according to the scene */
        if (maskConfidence.confidence[index] > 0.85) {
            HFLogPrint(HF_LOG_INFO, "Mask");
        } else {
            HFLogPrint(HF_LOG_INFO, "Non Mask");
        }
    }

    ret = HFReleaseImageStream(imageHandle);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Release image stream error: %d", ret);
    }
    /* The memory must be freed at the end of the program */
    ret = HFReleaseInspireFaceSession(session);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Release session error: %d", ret);
        return ret;
    }

    ret = HFReleaseImageBitmap(image);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Release image bitmap error: %d", ret);
        return ret;
    }

    ret = HFReleaseImageBitmap(drawImage);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Release draw image bitmap error: %d", ret);
        return ret;
    }

    HFLogPrint(HF_LOG_INFO, "");
    HFDeBugShowResourceStatistics();

    return 0;
}
