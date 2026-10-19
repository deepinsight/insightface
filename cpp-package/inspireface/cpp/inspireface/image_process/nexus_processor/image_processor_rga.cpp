#include "image_processor_rga.h"

#if defined(ISF_ENABLE_RGA)

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>

namespace inspire {

namespace nexus {

namespace {

bool IsPowerOfTwo(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

bool AlignDimension(int value, int alignment, int& aligned) {
    if (value <= 0 || !IsPowerOfTwo(alignment) || value > std::numeric_limits<int>::max() - (alignment - 1)) {
        return false;
    }
    aligned = (value + alignment - 1) & ~(alignment - 1);
    return true;
}

bool IsValidRequest(const uint8_t* src_data, int src_width, int src_height, int channels, uint8_t** dst_data) {
    if (dst_data == nullptr) {
        return false;
    }
    *dst_data = nullptr;
    // The RGA buffers below are explicitly wrapped as RK_FORMAT_RGB_888.
    return src_data != nullptr && src_width > 0 && src_height > 0 && channels == 3;
}

}  // namespace

RgaImageProcessor::RgaImageProcessor() {
    aligned_width_ = 4;
}

RgaImageProcessor::~RgaImageProcessor() {}

int32_t RgaImageProcessor::GetAlignedWidth(int width) const {
    int aligned = 0;
    return AlignDimension(width, aligned_width_, aligned) ? aligned : 0;
}

void RgaImageProcessor::SetAlignedWidth(int width) {
    if (!IsPowerOfTwo(width)) {
        INSPIRECV_LOG(ERROR) << "RGA alignment must be a positive power of two: " << width;
        return;
    }
    aligned_width_ = width;
}

bool RgaImageProcessor::BeginCpuAccess(RGABuffer& buffer) {
    if (buffer.cpu_access_active) {
        return true;
    }
    if (dma_sync_device_to_cpu(buffer.dma_fd) != 0) {
        INSPIRECV_LOG(ERROR) << "Failed to begin CPU access for RGA DMA buffer";
        return false;
    }
    buffer.cpu_access_active = true;
    return true;
}

bool RgaImageProcessor::EndCpuAccess(RGABuffer& buffer) {
    if (!buffer.cpu_access_active) {
        return true;
    }
    if (dma_sync_cpu_to_device(buffer.dma_fd) != 0) {
        INSPIRECV_LOG(ERROR) << "Failed to end CPU access for RGA DMA buffer";
        return false;
    }
    buffer.cpu_access_active = false;
    return true;
}

int32_t RgaImageProcessor::Resize(const uint8_t* src_data, int src_width, int src_height, int channels, uint8_t** dst_data, int dst_width,
                                  int dst_height) {
    if (!IsValidRequest(src_data, src_width, src_height, channels, dst_data) || dst_width <= 0 || dst_height <= 0) {
        return -1;
    }

    int aligned_src_width = 0;
    int aligned_dst_width = 0;
    if (!AlignDimension(src_width, aligned_width_, aligned_src_width) ||
        !AlignDimension(dst_width, aligned_width_, aligned_dst_width) || aligned_dst_width != dst_width) {
        // Image::Create does not carry a row stride. Returning a wider DMA row would corrupt the logical image.
        INSPIRECV_LOG(ERROR) << "RGA resize output width is not aligned: " << dst_width;
        return -1;
    }

    try {
        BufferKey src_key{aligned_src_width, src_height, channels, true};
        auto& src_buffer = GetOrCreateBuffer(src_key);
        BufferKey dst_key{aligned_dst_width, dst_height, channels, false};
        auto& dst_buffer = GetOrCreateBuffer(dst_key, false);

        if (!BeginCpuAccess(src_buffer)) {
            return -1;
        }
        for (int y = 0; y < src_height; ++y) {
            auto* row = static_cast<uint8_t*>(src_buffer.virtual_addr) + static_cast<size_t>(y) * aligned_src_width * channels;
            memcpy(row, src_data + static_cast<size_t>(y) * src_width * channels, static_cast<size_t>(src_width) * channels);
            if (aligned_src_width != src_width) {
                memset(row + static_cast<size_t>(src_width) * channels, 0,
                       static_cast<size_t>(aligned_src_width - src_width) * channels);
            }
        }
        if (!EndCpuAccess(src_buffer) || !EndCpuAccess(dst_buffer)) {
            return -1;
        }

        int ret = imcheck(src_buffer.buffer, dst_buffer.buffer, {}, {});
        if (IM_STATUS_NOERROR != ret) {
            INSPIRECV_LOG(ERROR) << "RGA parameter check failed: " << imStrError((IM_STATUS)ret);
            return -1;
        }
        ret = imresize(src_buffer.buffer, dst_buffer.buffer);
        if (ret != IM_STATUS_SUCCESS) {
            INSPIRECV_LOG(ERROR) << "RGA resize failed: " << imStrError((IM_STATUS)ret);
            return -1;
        }
        if (!BeginCpuAccess(dst_buffer)) {
            return -1;
        }
        *dst_data = static_cast<uint8_t*>(dst_buffer.virtual_addr);
        return 0;
    } catch (const std::exception& error) {
        INSPIRECV_LOG(ERROR) << "RGA resize failed: " << error.what();
        return -1;
    }
}

int32_t RgaImageProcessor::MarkDone() {
    int32_t status = 0;
    for (auto& pair : buffer_cache_) {
        if (!EndCpuAccess(pair.second)) {
            status = -1;
        }
    }
    return status;
}

int32_t RgaImageProcessor::SwapColor(const uint8_t* src_data, int src_width, int src_height, int channels, uint8_t** dst_data) {
    if (!IsValidRequest(src_data, src_width, src_height, channels, dst_data)) {
        return -1;
    }

    int aligned_src_width = 0;
    if (!AlignDimension(src_width, aligned_width_, aligned_src_width) || aligned_src_width != src_width) {
        INSPIRECV_LOG(ERROR) << "RGA color conversion output width is not aligned: " << src_width;
        return -1;
    }

    try {
        BufferKey src_key{aligned_src_width, src_height, channels, true};
        auto& src_buffer = GetOrCreateBuffer(src_key);
        BufferKey dst_key{aligned_src_width, src_height, channels, false};
        auto& dst_buffer = GetOrCreateBuffer(dst_key, false);

        if (!BeginCpuAccess(src_buffer)) {
            return -1;
        }
        memcpy(src_buffer.virtual_addr, src_data, static_cast<size_t>(src_width) * src_height * channels);
        if (!EndCpuAccess(src_buffer) || !EndCpuAccess(dst_buffer)) {
            return -1;
        }

        int ret = imcheck(src_buffer.buffer, dst_buffer.buffer, {}, {});
        if (IM_STATUS_NOERROR != ret) {
            INSPIRECV_LOG(ERROR) << "RGA parameter check failed: " << imStrError((IM_STATUS)ret);
            return -1;
        }
        ret = imcvtcolor(src_buffer.buffer, dst_buffer.buffer, RK_FORMAT_RGB_888, RK_FORMAT_BGR_888);
        if (ret != IM_STATUS_SUCCESS) {
            INSPIRECV_LOG(ERROR) << "RGA color conversion failed: " << imStrError((IM_STATUS)ret);
            return -1;
        }
        if (!BeginCpuAccess(dst_buffer)) {
            return -1;
        }
        *dst_data = static_cast<uint8_t*>(dst_buffer.virtual_addr);
        return 0;
    } catch (const std::exception& error) {
        INSPIRECV_LOG(ERROR) << "RGA color conversion failed: " << error.what();
        return -1;
    }
}

int32_t RgaImageProcessor::Padding(const uint8_t* src_data, int src_width, int src_height, int channels, int top, int bottom, int left, int right,
                                   uint8_t** dst_data, int& dst_width, int& dst_height) {
    dst_width = 0;
    dst_height = 0;
    if (!IsValidRequest(src_data, src_width, src_height, channels, dst_data) || top < 0 || bottom < 0 || left < 0 || right < 0 ||
        src_width > std::numeric_limits<int>::max() - left - right || src_height > std::numeric_limits<int>::max() - top - bottom) {
        return -1;
    }

    const int output_width = src_width + left + right;
    const int output_height = src_height + top + bottom;
    int aligned_src_width = 0;
    int aligned_dst_width = 0;
    if (!AlignDimension(src_width, aligned_width_, aligned_src_width) ||
        !AlignDimension(output_width, aligned_width_, aligned_dst_width) || aligned_dst_width != output_width) {
        INSPIRECV_LOG(ERROR) << "RGA padding output width is not aligned: " << output_width;
        return -1;
    }

    try {
        BufferKey src_key{aligned_src_width, src_height, channels, true};
        auto& src_buffer = GetOrCreateBuffer(src_key, true);
        BufferKey dst_key{aligned_dst_width, output_height, channels, false};
        auto& dst_buffer = GetOrCreateBuffer(dst_key, false);

        if (!BeginCpuAccess(src_buffer)) {
            return -1;
        }
        for (int y = 0; y < src_height; ++y) {
            auto* row = static_cast<uint8_t*>(src_buffer.virtual_addr) + static_cast<size_t>(y) * aligned_src_width * channels;
            memcpy(row, src_data + static_cast<size_t>(y) * src_width * channels, static_cast<size_t>(src_width) * channels);
            if (aligned_src_width != src_width) {
                memset(row + static_cast<size_t>(src_width) * channels, 0,
                       static_cast<size_t>(aligned_src_width - src_width) * channels);
            }
        }
        if (!EndCpuAccess(src_buffer) || !EndCpuAccess(dst_buffer)) {
            return -1;
        }

        im_rect src_rect = {0, 0, src_width, src_height};
        im_rect dst_rect = {left, top, src_width, src_height};
        int ret = imcheck(src_buffer.buffer, dst_buffer.buffer, src_rect, dst_rect);
        if (IM_STATUS_NOERROR != ret) {
            INSPIRECV_LOG(ERROR) << "RGA parameter check failed: " << imStrError((IM_STATUS)ret);
            return -1;
        }
        ret = imfill(dst_buffer.buffer, {0, 0, output_width, output_height}, 0x000000);
        if (ret != IM_STATUS_SUCCESS) {
            INSPIRECV_LOG(ERROR) << "RGA fill failed: " << imStrError((IM_STATUS)ret);
            return -1;
        }
        ret = improcess(src_buffer.buffer, dst_buffer.buffer, {}, src_rect, dst_rect, {}, IM_SYNC);
        if (ret != IM_STATUS_SUCCESS) {
            INSPIRECV_LOG(ERROR) << "RGA copy failed: " << imStrError((IM_STATUS)ret);
            return -1;
        }
        if (!BeginCpuAccess(dst_buffer)) {
            return -1;
        }
        *dst_data = static_cast<uint8_t*>(dst_buffer.virtual_addr);
        dst_width = output_width;
        dst_height = output_height;
        return 0;
    } catch (const std::exception& error) {
        INSPIRECV_LOG(ERROR) << "RGA image padding failed: " << error.what();
        return -1;
    }
}

int32_t RgaImageProcessor::ResizeAndPadding(const uint8_t* src_data, int src_width, int src_height, int channels, int dst_width, int dst_height,
                                            uint8_t** dst_data, float& scale) {
    scale = 0.0f;
    if (!IsValidRequest(src_data, src_width, src_height, channels, dst_data) || dst_width <= 0 || dst_height <= 0) {
        return -1;
    }

    int aligned_src_width = 0;
    int aligned_dst_width = 0;
    int aligned_dst_height = 0;
    if (!AlignDimension(src_width, aligned_width_, aligned_src_width) ||
        !AlignDimension(dst_width, aligned_width_, aligned_dst_width) ||
        !AlignDimension(dst_height, aligned_width_, aligned_dst_height) || aligned_dst_width != dst_width || aligned_dst_height != dst_height) {
        INSPIRECV_LOG(ERROR) << "RGA resize-and-pad output size is not aligned: " << dst_width << "x" << dst_height;
        return -1;
    }

    const float output_scale = std::min(static_cast<float>(dst_width) / src_width, static_cast<float>(dst_height) / src_height);
    int resized_w = static_cast<int>(src_width * output_scale);
    int resized_h = static_cast<int>(src_height * output_scale);
    if (!std::isfinite(output_scale) || output_scale <= 0.0f ||
        !AlignDimension(resized_w, aligned_width_, resized_w) || !AlignDimension(resized_h, aligned_width_, resized_h) ||
        resized_w > dst_width || resized_h > dst_height) {
        return -1;
    }

    try {
        BufferKey src_key{aligned_src_width, src_height, channels, true};
        auto& src_buffer = GetOrCreateBuffer(src_key);
        BufferKey dst_key{aligned_dst_width, aligned_dst_height, channels, false};
        auto& dst_buffer = GetOrCreateBuffer(dst_key, false);

        if (!BeginCpuAccess(src_buffer)) {
            return -1;
        }
        for (int y = 0; y < src_height; ++y) {
            auto* row = static_cast<uint8_t*>(src_buffer.virtual_addr) + static_cast<size_t>(y) * aligned_src_width * channels;
            memcpy(row, src_data + static_cast<size_t>(y) * src_width * channels, static_cast<size_t>(src_width) * channels);
            if (aligned_src_width != src_width) {
                memset(row + static_cast<size_t>(src_width) * channels, 0,
                       static_cast<size_t>(aligned_src_width - src_width) * channels);
            }
        }
        if (!EndCpuAccess(src_buffer) || !EndCpuAccess(dst_buffer)) {
            return -1;
        }

        im_rect src_rect = {0, 0, src_width, src_height};
        im_rect dst_rect = {0, 0, resized_w, resized_h};
        int ret = imcheck(src_buffer.buffer, dst_buffer.buffer, src_rect, dst_rect);
        if (IM_STATUS_NOERROR != ret) {
            INSPIRECV_LOG(ERROR) << "RGA parameter check failed: " << imStrError((IM_STATUS)ret);
            return -1;
        }
        ret = imfill(dst_buffer.buffer, {0, 0, dst_width, dst_height}, 0x000000);
        if (ret != IM_STATUS_SUCCESS) {
            INSPIRECV_LOG(ERROR) << "RGA fill failed: " << imStrError((IM_STATUS)ret);
            return -1;
        }
        ret = improcess(src_buffer.buffer, dst_buffer.buffer, {}, src_rect, dst_rect, {}, IM_SYNC);
        if (ret != IM_STATUS_SUCCESS) {
            INSPIRECV_LOG(ERROR) << "RGA resize failed: " << imStrError((IM_STATUS)ret);
            return -1;
        }
        if (!BeginCpuAccess(dst_buffer)) {
            return -1;
        }
        *dst_data = static_cast<uint8_t*>(dst_buffer.virtual_addr);
        scale = output_scale;
        return 0;
    } catch (const std::exception& error) {
        INSPIRECV_LOG(ERROR) << "RGA resize-and-pad failed: " << error.what();
        return -1;
    }
}

}  // namespace nexus

}  // namespace inspire

#endif  // ISF_ENABLE_RGA
