/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#include <stdio.h>
#include <stdlib.h>
#include <inspireface.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
    const char* resourcePath = argv[1];
    HResult ret = HFReloadInspireFace(resourcePath);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Failed to launch InspireFace: %d", ret);
        return 1;
    }

    // Enable the functions in the pipeline: mask detection, live detection, and face quality
    // detection
    HOption option = HF_ENABLE_QUALITY | HF_ENABLE_MASK_DETECT | HF_ENABLE_LIVENESS | HF_ENABLE_IR_LIVENESS | HF_ENABLE_FACE_RECOGNITION;
    // Non-video or frame sequence mode uses IMAGE-MODE, which is always face detection without
    // tracking
    HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
    // Maximum number of faces detected
    HInt32 maxDetectNum = 20;
    // Face detection image input level
    HInt32 detectPixelLevel = 160;
    // Handle of the current face SDK algorithm context
    HFSession session = {0};
    ret = HFCreateInspireFaceSessionOptional(option, detMode, maxDetectNum, detectPixelLevel, -1, &session);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create FaceContext error: %d", ret);
        return ret;
    }

    // Wait for 2000 seconds, watching the cpu usage
    sleep(2000);

    ret = HFReleaseInspireFaceSession(session);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Release InspireFaceSession error: %d", ret);
        return ret;
    }

    return 0;
}