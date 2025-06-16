/*
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#include <stdio.h>
#include <stdlib.h>
#include <inspireface.h>

int main(int argc, char* argv[]) {
    /* Check whether the number of parameters is correct */
    if (argc < 3 || argc > 4) {
        HFLogPrint(HF_LOG_ERROR, "Usage: %s <pack_path> <source_path> [rotation]", argv[0]);
        return 1;
    }

    const char* packPath = argv[1];
    const char* sourcePath = argv[2];
    int rotation = 0;

    /* If rotation is provided, check and set the value */
    if (argc == 4) {
        rotation = atoi(argv[3]);
        if (rotation != 0 && rotation != 90 && rotation != 180 && rotation != 270) {
            HFLogPrint(HF_LOG_ERROR, "Invalid rotation value. Allowed values are 0, 90, 180, 270.");
            return 1;
        }
    }
    
    HFRotation rotation_enum;
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

    HFSetLogLevel(HF_LOG_INFO);

    HResult ret;
    /* The resource file must be loaded before it can be used */
    ret = HFLaunchInspireFace(packPath);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Load Resource error: %d", ret);
        return ret;
    }

    /* Enable the functions in the pipeline: mask detection, live detection, and face quality
     * detection */
    HOption option = HF_ENABLE_QUALITY | HF_ENABLE_MASK_DETECT | HF_ENABLE_LIVENESS;
    /* Non-video or frame sequence mode uses IMAGE-MODE, which is always face detection without
     * tracking */
    HFDetectMode detMode = HF_DETECT_MODE_LIGHT_TRACK;
    /* Maximum number of faces detected */
    HInt32 maxDetectNum = 20;
    /* Face detection image input level */
    HInt32 detectPixelLevel = 160;
    /* Handle of the current face SDK algorithm context */
    HFSession session = {0};
    ret = HFCreateInspireFaceSessionOptional(option, detMode, maxDetectNum, detectPixelLevel, -1, &session);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create FaceContext error: %d", ret);
        return ret;
    }

    HFSessionSetTrackPreviewSize(session, detectPixelLevel);
    HFSessionSetFilterMinimumFacePixelSize(session, 4);

    /* Load a image */
    HFImageBitmap image;
    ret = HFCreateImageBitmapFromFilePath(sourcePath, 3, &image);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "The source entered is not a picture or read error.");
        return ret;
    }
    /* Prepare an image parameter structure for configuration */
    HFImageStream imageHandle = {0};
    ret = HFCreateImageStreamFromImageBitmap(image, rotation_enum, &imageHandle);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create ImageStream error: %d", ret);
        return ret;
    }

    int loop = 100;

    /* Enable the cost spend */
    HFSessionSetEnableTrackCostSpend(session, 1);

    int i;
    /* Execute HF_FaceContextRunFaceTrack captures face information in an image */
    HFMultipleFaceData multipleFaceData = {0};
    for (i = 0; i < loop; i++) {
        ret = HFExecuteFaceTrack(session, imageHandle, &multipleFaceData);
        if (ret != HSUCCEED) {
            HFLogPrint(HF_LOG_ERROR, "Execute HFExecuteFaceTrack error: %d", ret);
            return ret;
        }
    }
    HFLogPrint(HF_LOG_INFO, "Number of Detection: %d", multipleFaceData.detectedNum);
    HFSessionPrintTrackCostSpend(session);

    if (multipleFaceData.detectedNum > 0) {
        HFLogPrint(HF_LOG_INFO, "========================================");
        for (i = 0; i < multipleFaceData.detectedNum; i++) {
            HFLogPrint(HF_LOG_INFO, "TrackId: %d", multipleFaceData.trackIds[i]);
            HFLogPrint(HF_LOG_INFO, "TrackCount: %d", multipleFaceData.trackCounts[i]);
        }
    } else {
        HFLogPrint(HF_LOG_WARN, "The face cannot be detected, and the tracking test results may be invalid!");
    }

    ret = HFReleaseImageStream(imageHandle);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Release image stream error: %d", ret);
    }


    ret = HFReleaseImageBitmap(image);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Release image bitmap error: %d", ret);
        return ret;
    }    
    
    /* The memory must be freed at the end of the program */
    ret = HFReleaseInspireFaceSession(session);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Release session error: %d", ret);
        return ret;
    }

    return 0;
}
