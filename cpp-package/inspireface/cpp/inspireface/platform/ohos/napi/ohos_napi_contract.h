#ifndef INSPIREFACE_OHOS_NAPI_CONTRACT_H
#define INSPIREFACE_OHOS_NAPI_CONTRACT_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "c_api/inspireface.h"

namespace inspire {
namespace ohos {

enum class ImageByteSizeStatus {
    kOk,
    kInvalidDimensions,
    kUnsupportedFormat,
    kOddYuvDimensions,
    kOverflow,
};

struct ImageByteSizeResult {
    ImageByteSizeStatus status = ImageByteSizeStatus::kInvalidDimensions;
    size_t bytes = 0;
};

inline ImageByteSizeResult CalculateImageByteSize(HFImageFormat format, int32_t width, int32_t height) {
    ImageByteSizeResult result;
    if (width <= 0 || height <= 0) {
        return result;
    }

    size_t numerator = 0;
    size_t denominator = 1;
    switch (format) {
        case HF_STREAM_RGB:
        case HF_STREAM_BGR:
            numerator = 3;
            break;
        case HF_STREAM_RGBA:
        case HF_STREAM_BGRA:
            numerator = 4;
            break;
        case HF_STREAM_YUV_NV12:
        case HF_STREAM_YUV_NV21:
        case HF_STREAM_I420:
            if ((width & 1) != 0 || (height & 1) != 0) {
                result.status = ImageByteSizeStatus::kOddYuvDimensions;
                return result;
            }
            numerator = 3;
            denominator = 2;
            break;
        case HF_STREAM_GRAY:
            numerator = 1;
            break;
        default:
            result.status = ImageByteSizeStatus::kUnsupportedFormat;
            return result;
    }

    const size_t safe_width = static_cast<size_t>(width);
    const size_t safe_height = static_cast<size_t>(height);
    if (safe_width > std::numeric_limits<size_t>::max() / safe_height) {
        result.status = ImageByteSizeStatus::kOverflow;
        return result;
    }
    const size_t pixels = safe_width * safe_height;
    if (pixels > std::numeric_limits<size_t>::max() / numerator) {
        result.status = ImageByteSizeStatus::kOverflow;
        return result;
    }

    result.bytes = pixels * numerator / denominator;
    result.status = ImageByteSizeStatus::kOk;
    return result;
}

inline bool CopyExactImageBytes(const uint8_t* source, size_t source_size, HFImageFormat format, int32_t width,
                                int32_t height, std::vector<uint8_t>* destination) {
    if (!source || !destination) {
        return false;
    }
    const ImageByteSizeResult expected = CalculateImageByteSize(format, width, height);
    if (expected.status != ImageByteSizeStatus::kOk || expected.bytes != source_size) {
        return false;
    }
    destination->assign(source, source + source_size);
    return true;
}

}  // namespace ohos
}  // namespace inspire

#endif  // INSPIREFACE_OHOS_NAPI_CONTRACT_H
