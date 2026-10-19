#ifndef INSPIREFACE_FACE_RESULT_SNAPSHOT_H
#define INSPIREFACE_FACE_RESULT_SNAPSHOT_H

#include <vector>

#include "inspireface.h"

/**
 * @brief Owns a deep copy of one face detection result.
 */
struct HF_FaceResultSnapshot {
    HResult CopyFrom(const HFMultipleFaceData& source);
    void GetView(PHFMultipleFaceData result);

private:
    std::vector<HFaceRect> rects_;
    std::vector<HInt32> track_ids_;
    std::vector<HInt32> track_counts_;
    std::vector<HFloat> detection_confidence_;
    std::vector<HFloat> roll_;
    std::vector<HFloat> yaw_;
    std::vector<HFloat> pitch_;
    std::vector<std::vector<HUInt8>> token_storage_;
    std::vector<HFFaceBasicToken> token_views_;
};

#endif  // INSPIREFACE_FACE_RESULT_SNAPSHOT_H
