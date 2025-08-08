/**
 * Created by InspireFace
 * @date 2025-06-22
 * Sample code for creating InspireFace session with all features enabled
 */
#include <stdio.h>
#include <stdlib.h>
#include <inspireface.h>

int main(int argc, char* argv[]) {
    HResult ret;
    const char* packPath;
    HOption option;
    HFDetectMode detMode;
    HInt32 maxDetectNum;
    HInt32 detectPixelLevel;
    HFSession session;
    HFFaceDetectPixelList pixelLevels;

    packPath = argv[1];
    HFLogPrint(HF_LOG_INFO, "Pack file Path: %s", packPath);

    /* Set log level to debug for detailed output */
    HFSetLogLevel(HF_LOG_DEBUG);

    /* The resource file must be loaded before it can be used */
    ret = HFLaunchInspireFace(packPath);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Load Resource error: %d", ret);
        return ret;
    }
    HFLogPrint(HF_LOG_INFO, "Resource loaded successfully");

    /* Query supported pixel levels for face detection */
    ret = HFQuerySupportedPixelLevelsForFaceDetection(&pixelLevels);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "HFQuerySupportedPixelLevelsForFaceDetection error: %d", ret);
        return ret;
    }
    HFLogPrint(HF_LOG_INFO, "Supported pixel levels for face detection: %d", pixelLevels.size);
    for (int i = 0; i < pixelLevels.size; i++) {
        HFLogPrint(HF_LOG_INFO, "Supported pixel level %d: %d", i + 1, pixelLevels.pixel_level[i]);
    }

    /* Enable ALL available functions in the pipeline */
    option = HF_ENABLE_FACE_RECOGNITION |    // Face recognition
             HF_ENABLE_LIVENESS |            // RGB liveness detection
             HF_ENABLE_IR_LIVENESS |         // IR liveness detection
             HF_ENABLE_MASK_DETECT |         // Mask detection
             HF_ENABLE_FACE_ATTRIBUTE |      // Face attribute prediction
             HF_ENABLE_QUALITY |             // Face quality assessment
             HF_ENABLE_INTERACTION |         // Interaction feature
             HF_ENABLE_FACE_POSE |           // Face pose estimation
             HF_ENABLE_FACE_EMOTION;         // Face emotion recognition

    HFLogPrint(HF_LOG_INFO, "Enabled features:");
    HFLogPrint(HF_LOG_INFO, "- Face Recognition: YES");
    HFLogPrint(HF_LOG_INFO, "- RGB Liveness Detection: YES");
    HFLogPrint(HF_LOG_INFO, "- IR Liveness Detection: YES");
    HFLogPrint(HF_LOG_INFO, "- Mask Detection: YES");
    HFLogPrint(HF_LOG_INFO, "- Face Attributes: YES");
    HFLogPrint(HF_LOG_INFO, "- Face Quality: YES");
    HFLogPrint(HF_LOG_INFO, "- Interaction: YES");
    HFLogPrint(HF_LOG_INFO, "- Face Pose: YES");
    HFLogPrint(HF_LOG_INFO, "- Face Emotion: YES");

    /* Set detection mode - use light track for general purpose */
    detMode = HF_DETECT_MODE_LIGHT_TRACK;
    HFLogPrint(HF_LOG_INFO, "Detection mode: HF_DETECT_MODE_LIGHT_TRACK");

    /* Maximum number of faces to detect */
    maxDetectNum = 20;
    HFLogPrint(HF_LOG_INFO, "Maximum faces to detect: %d", maxDetectNum);

    /* Face detection image input level */
    detectPixelLevel = 320;
    HFLogPrint(HF_LOG_INFO, "Detection pixel level: %d", detectPixelLevel);

    /* Create InspireFace session with all features enabled */
    session = NULL;
    ret = HFCreateInspireFaceSessionOptional(option, detMode, maxDetectNum, detectPixelLevel, -1, &session);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create InspireFace session error: %d", ret);
        return ret;
    }
    HFLogPrint(HF_LOG_INFO, "InspireFace session created successfully");

    /* Configure session parameters */
    ret = HFSessionSetTrackPreviewSize(session, detectPixelLevel);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Set track preview size error: %d", ret);
    } else {
        HFLogPrint(HF_LOG_INFO, "Track preview size set to: %d", detectPixelLevel);
    }

    ret = HFSessionSetFilterMinimumFacePixelSize(session, 4);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Set minimum face pixel size error: %d", ret);
    } else {
        HFLogPrint(HF_LOG_INFO, "Minimum face pixel size set to: 4");
    }

    /* Session is now ready for use */
    HFLogPrint(HF_LOG_INFO, "========================================");
    HFLogPrint(HF_LOG_INFO, "Session created successfully with ALL features enabled!");
    HFLogPrint(HF_LOG_INFO, "You can now use this session for:");
    HFLogPrint(HF_LOG_INFO, "- Face detection and tracking");
    HFLogPrint(HF_LOG_INFO, "- Face recognition");
    HFLogPrint(HF_LOG_INFO, "- Liveness detection (RGB & IR)");
    HFLogPrint(HF_LOG_INFO, "- Mask detection");
    HFLogPrint(HF_LOG_INFO, "- Face attribute analysis");
    HFLogPrint(HF_LOG_INFO, "- Face quality assessment");
    HFLogPrint(HF_LOG_INFO, "- Interaction detection");
    HFLogPrint(HF_LOG_INFO, "- Face pose estimation");
    HFLogPrint(HF_LOG_INFO, "- Emotion recognition");
    HFLogPrint(HF_LOG_INFO, "========================================");

    /* Clean up resources */
    ret = HFReleaseInspireFaceSession(session);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Release session error: %d", ret);
        return ret;
    }
    HFLogPrint(HF_LOG_INFO, "Session released successfully");

    /* Show resource statistics */
    HFLogPrint(HF_LOG_INFO, "");
    HFDeBugShowResourceStatistics();

    return 0;
}
