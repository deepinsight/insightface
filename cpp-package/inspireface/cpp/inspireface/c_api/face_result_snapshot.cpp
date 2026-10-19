#include "face_result_snapshot.h"

#include <cstddef>

HResult HF_FaceResultSnapshot::CopyFrom(const HFMultipleFaceData& source) {
    if (source.detectedNum < 0) {
        return HERR_INVALID_FACE_LIST;
    }
    const std::size_t count = static_cast<std::size_t>(source.detectedNum);
    if (count == 0) {
        return HSUCCEED;
    }
    if (source.rects == nullptr || source.trackIds == nullptr || source.trackCounts == nullptr ||
        source.detConfidence == nullptr || source.angles.roll == nullptr || source.angles.yaw == nullptr ||
        source.angles.pitch == nullptr || source.tokens == nullptr) {
        return HERR_INVALID_FACE_LIST;
    }

    rects_.assign(source.rects, source.rects + count);
    track_ids_.assign(source.trackIds, source.trackIds + count);
    track_counts_.assign(source.trackCounts, source.trackCounts + count);
    detection_confidence_.assign(source.detConfidence, source.detConfidence + count);
    roll_.assign(source.angles.roll, source.angles.roll + count);
    yaw_.assign(source.angles.yaw, source.angles.yaw + count);
    pitch_.assign(source.angles.pitch, source.angles.pitch + count);
    token_storage_.resize(count);
    token_views_.resize(count);
    for (std::size_t index = 0; index < count; ++index) {
        const HFFaceBasicToken& token = source.tokens[index];
        if (token.size < 0 || (token.size > 0 && token.data == nullptr)) {
            return HERR_INVALID_FACE_TOKEN;
        }
        const auto* begin = static_cast<const HUInt8*>(token.data);
        if (token.size > 0) {
            token_storage_[index].assign(begin, begin + token.size);
        }
        token_views_[index].size = token.size;
        token_views_[index].data = token_storage_[index].empty() ? nullptr : token_storage_[index].data();
    }
    return HSUCCEED;
}

void HF_FaceResultSnapshot::GetView(PHFMultipleFaceData result) {
    *result = HFMultipleFaceData{};
    result->detectedNum = static_cast<HInt32>(rects_.size());
    if (rects_.empty()) {
        return;
    }
    result->rects = rects_.data();
    result->trackIds = track_ids_.data();
    result->trackCounts = track_counts_.data();
    result->detConfidence = detection_confidence_.data();
    result->angles.roll = roll_.data();
    result->angles.yaw = yaw_.data();
    result->angles.pitch = pitch_.data();
    result->tokens = token_views_.data();
}
