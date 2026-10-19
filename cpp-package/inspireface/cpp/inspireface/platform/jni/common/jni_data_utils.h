#ifndef INSPIREFACE_JNI_DATA_UTILS_H
#define INSPIREFACE_JNI_DATA_UTILS_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

#include "c_api/inspireface.h"

namespace inspire {
namespace jni {

inline bool CheckedImageByteSize(int32_t format, int32_t width, int32_t height, size_t *size) {
    if (!size || width <= 0 || height <= 0) {
        return false;
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
                return false;
            }
            numerator = 3;
            denominator = 2;
            break;
        case HF_STREAM_GRAY:
            numerator = 1;
            break;
        default:
            return false;
    }

    const size_t safeWidth = static_cast<size_t>(width);
    const size_t safeHeight = static_cast<size_t>(height);
    if (safeWidth > std::numeric_limits<size_t>::max() / safeHeight) {
        return false;
    }
    const size_t pixels = safeWidth * safeHeight;
    if (pixels > std::numeric_limits<size_t>::max() / numerator) {
        return false;
    }
    *size = pixels * numerator / denominator;
    return *size != 0;
}

inline bool CopyPackedRows(const uint8_t *source, size_t sourceStride, size_t rowBytes, int32_t height,
                           std::vector<uint8_t> *destination) {
    if (!source || !destination || height <= 0 || rowBytes == 0 || sourceStride < rowBytes ||
        rowBytes > std::numeric_limits<size_t>::max() / static_cast<size_t>(height)) {
        return false;
    }
    destination->resize(rowBytes * static_cast<size_t>(height));
    for (int32_t row = 0; row < height; ++row) {
        std::memcpy(destination->data() + static_cast<size_t>(row) * rowBytes, source + static_cast<size_t>(row) * sourceStride, rowBytes);
    }
    return true;
}

inline uint8_t Expand5To8(uint16_t value) {
    return static_cast<uint8_t>((value << 3) | (value >> 2));
}

inline uint8_t Expand6To8(uint16_t value) {
    return static_cast<uint8_t>((value << 2) | (value >> 4));
}

inline bool ConvertRgb565RowsToRgb(const uint8_t *source, size_t sourceStride, int32_t width, int32_t height,
                                   std::vector<uint8_t> *destination) {
    size_t outputSize = 0;
    if (!source || !destination || !CheckedImageByteSize(HF_STREAM_RGB, width, height, &outputSize) ||
        static_cast<size_t>(width) > std::numeric_limits<size_t>::max() / sizeof(uint16_t) ||
        sourceStride < static_cast<size_t>(width) * sizeof(uint16_t)) {
        return false;
    }

    destination->resize(outputSize);
    for (int32_t row = 0; row < height; ++row) {
        const uint8_t *sourceRow = source + static_cast<size_t>(row) * sourceStride;
        uint8_t *destinationRow = destination->data() + static_cast<size_t>(row) * static_cast<size_t>(width) * 3;
        for (int32_t column = 0; column < width; ++column) {
            uint16_t pixel = 0;
            std::memcpy(&pixel, sourceRow + static_cast<size_t>(column) * sizeof(pixel), sizeof(pixel));
            destinationRow[column * 3] = Expand5To8(static_cast<uint16_t>((pixel >> 11) & 0x1F));
            destinationRow[column * 3 + 1] = Expand6To8(static_cast<uint16_t>((pixel >> 5) & 0x3F));
            destinationRow[column * 3 + 2] = Expand5To8(static_cast<uint16_t>(pixel & 0x1F));
        }
    }
    return true;
}

inline bool ReadEulerAngle(const HFFaceEulerAngle &angles, int32_t index, float *roll, float *yaw, float *pitch) {
    if (index < 0 || !roll || !yaw || !pitch || !angles.roll || !angles.yaw || !angles.pitch) {
        return false;
    }
    *roll = angles.roll[index];
    *yaw = angles.yaw[index];
    *pitch = angles.pitch[index];
    return true;
}

class OwnedImageBufferRegistry {
public:
    bool Store(HFImageStream stream, std::shared_ptr<std::vector<uint8_t>> buffer) {
        if (!stream || !buffer || buffer->empty()) {
            return false;
        }
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            return buffers_.emplace(stream, std::move(buffer)).second;
        } catch (const std::bad_alloc &) {
            return false;
        }
    }

    bool Erase(HFImageStream stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffers_.erase(stream) == 1;
    }

    std::shared_ptr<const std::vector<uint8_t>> Find(HFImageStream stream) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto iterator = buffers_.find(stream);
        return iterator == buffers_.end() ? nullptr : iterator->second;
    }

    size_t Size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffers_.size();
    }

private:
    mutable std::mutex mutex_;
    std::map<HFImageStream, std::shared_ptr<std::vector<uint8_t>>> buffers_;
};

}  // namespace jni
}  // namespace inspire

#endif
