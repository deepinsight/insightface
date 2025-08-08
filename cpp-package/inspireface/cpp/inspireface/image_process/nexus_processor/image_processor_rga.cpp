#include "image_processor_rga.h"

#if defined(ISF_ENABLE_RGA)

namespace inspire {

namespace nexus {

RgaImageProcessor::RgaImageProcessor() {
    aligned_width_ = 4;
}

RgaImageProcessor::~RgaImageProcessor() {}

int32_t RgaImageProcessor::GetAlignedWidth(int width) const {
    return aligned_width_;
}

void RgaImageProcessor::SetAlignedWidth(int width) {
    aligned_width_ = width;
}

int32_t RgaImageProcessor::Resize(const uint8_t* src_data, int src_width, int src_height, int channels, uint8_t** dst_data, int dst_width,
                                  int dst_height) {
    // Calculate width aligned to 4 bytes
    int aligned_src_width = (src_width + aligned_width_ - 1) & ~(aligned_width_ - 1);  // Round up to nearest multiple of aligned_width_
    int aligned_dst_width = (dst_width + aligned_width_ - 1) & ~(aligned_width_ - 1);
    // std::cout << "aligned_src_width: " << aligned_src_width << ", aligned_dst_width: " << aligned_dst_width << std::endl;

    // 1. Get or create source buffer with aligned width
    BufferKey src_key{aligned_src_width, src_height, channels};
    auto& src_buffer = GetOrCreateBuffer(src_key);

    // 2. Get or create destination buffer with aligned width
    BufferKey dst_key{aligned_dst_width, dst_height, channels};
    auto& dst_buffer = GetOrCreateBuffer(dst_key, false);

    // 3. Copy source data to RGA buffer, handle padding row by row
    for (int y = 0; y < src_height; y++) {
        memcpy(static_cast<uint8_t*>(src_buffer.virtual_addr) + y * aligned_src_width * channels, src_data + y * src_width * channels,
               src_width * channels);
        // Padding area remains zero
    }

    dma_sync_cpu_to_device(src_buffer.dma_fd);
    dma_sync_cpu_to_device(dst_buffer.dma_fd);

    // 4. Execute RGA resize
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

    // 5. Return pointer to destination buffer
    *dst_data = static_cast<uint8_t*>(dst_buffer.virtual_addr);

    return 0;
}

int32_t RgaImageProcessor::MarkDone() {
    // Sync all buffers
    for (const auto& pair : buffer_cache_) {
        dma_sync_device_to_cpu(pair.second.dma_fd);
    }

    // // Print current cache status for debugging
    // INSPIRECV_LOG(INFO) << "MarkDone: Current cache status:";
    // INSPIRECV_LOG(INFO) << "Cache size: " << buffer_cache_.size();
    // for (const auto& pair : buffer_cache_) {
    //     INSPIRECV_LOG(INFO) << "Buffer: " << pair.second.width << "x" << pair.second.height << " dma_fd=" << pair.second.dma_fd;
    // }
    return 0;
}

int32_t RgaImageProcessor::SwapColor(const uint8_t* src_data, int src_width, int src_height, int channels, uint8_t** dst_data) {
    // Calculate width aligned to 4 bytes
    int aligned_src_width = (src_width + aligned_width_ - 1) & ~(aligned_width_ - 1);  // Round up to nearest multiple of aligned_width_
    // 1. Get or create source buffer
    BufferKey src_key{aligned_src_width, src_height, channels};
    auto& src_buffer = GetOrCreateBuffer(src_key);

    // 2. Get or create destination buffer
    BufferKey dst_key{aligned_src_width, src_height, channels};
    auto& dst_buffer = GetOrCreateBuffer(dst_key, false);

    // 3. Copy source data to RGA buffer, handle padding row by row
    for (int y = 0; y < src_height; y++) {
        memcpy(static_cast<uint8_t*>(src_buffer.virtual_addr) + y * aligned_src_width * channels, src_data + y * src_width * channels,
               src_width * channels);
        // Padding area remains zero
    }

    dma_sync_cpu_to_device(src_buffer.dma_fd);
    dma_sync_cpu_to_device(dst_buffer.dma_fd);

    // 3. Execute RGA swap color
    int ret = imcheck(src_buffer.buffer, dst_buffer.buffer, {}, {});
    if (IM_STATUS_NOERROR != ret) {
        INSPIRECV_LOG(ERROR) << "RGA parameter check failed: " << imStrError((IM_STATUS)ret);
        return -1;
    }

    // 4. Execute RGA swap color
    ret = imcvtcolor(src_buffer.buffer, dst_buffer.buffer, RK_FORMAT_RGB_888, RK_FORMAT_BGR_888);
    if (ret != IM_STATUS_SUCCESS) {
        INSPIRECV_LOG(ERROR) << "RGA color conversion failed: " << imStrError((IM_STATUS)ret);
        return -1;
    }

    dma_sync_device_to_cpu(dst_buffer.dma_fd);

    // 5. Return pointer to destination buffer
    *dst_data = static_cast<uint8_t*>(dst_buffer.virtual_addr);

    return 0;
}

int32_t RgaImageProcessor::Padding(const uint8_t* src_data, int src_width, int src_height, int channels, int top, int bottom, int left, int right,
                                   uint8_t** dst_data, int& dst_width, int& dst_height) {
    // Calculate final dimensions
    dst_width = src_width + left + right;
    dst_height = src_height + top + bottom;

    // Calculate width aligned to 4 bytes
    int aligned_src_width = (src_width + aligned_width_ - 1) & ~(aligned_width_ - 1);  // Round up to nearest multiple of aligned_width_
    int aligned_dst_width = (dst_width + aligned_width_ - 1) & ~(aligned_width_ - 1);  // Round up to nearest multiple of aligned_width_

    // 1. Get or create source buffer with aligned width
    BufferKey src_key{aligned_src_width, src_height, channels};
    auto& src_buffer = GetOrCreateBuffer(src_key, true);

    // 2. Create destination buffer with padded dimensions
    BufferKey dst_key{aligned_dst_width, dst_height, channels};
    auto& dst_buffer = GetOrCreateBuffer(dst_key, false);

    // 3. Copy source data to RGA buffer
    for (int y = 0; y < src_height; y++) {
        memcpy(static_cast<uint8_t*>(src_buffer.virtual_addr) + y * aligned_src_width * channels, src_data + y * src_width * channels,
               src_width * channels);
    }

    dma_sync_cpu_to_device(src_buffer.dma_fd);
    dma_sync_cpu_to_device(dst_buffer.dma_fd);

    // 4. Execute padding operation
    // Set source and destination regions
    im_rect src_rect = {0, 0, src_width, src_height};
    im_rect dst_rect = {left, top, src_width, src_height};  // Specify padding position

    int ret = imcheck(src_buffer.buffer, dst_buffer.buffer, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret) {
        INSPIRECV_LOG(ERROR) << "RGA parameter check failed: " << imStrError((IM_STATUS)ret);
        return -1;
    }

    // 5. Fill entire destination area with black first
    ret = imfill(dst_buffer.buffer, {0, 0, dst_width, dst_height}, 0x000000);
    if (ret != IM_STATUS_SUCCESS) {
        INSPIRECV_LOG(ERROR) << "RGA fill failed: " << imStrError((IM_STATUS)ret);
        return -1;
    }

    // 6. Copy source image to specified position in destination
    ret = improcess(src_buffer.buffer, dst_buffer.buffer, {}, src_rect, dst_rect, {}, IM_SYNC);
    if (ret != IM_STATUS_SUCCESS) {
        INSPIRECV_LOG(ERROR) << "RGA copy failed: " << imStrError((IM_STATUS)ret);
        return -1;
    }

    dma_sync_device_to_cpu(dst_buffer.dma_fd);

    *dst_data = static_cast<uint8_t*>(dst_buffer.virtual_addr);
    return 0;
}

int32_t RgaImageProcessor::ResizeAndPadding(const uint8_t* src_data, int src_width, int src_height, int channels, int dst_width, int dst_height,
                                            uint8_t** dst_data, float& scale) {
    // Ensure target dimensions are multiples of 4
    int aligned_dst_width = (dst_width + aligned_width_ - 1) & ~(aligned_width_ - 1);  // Round up to nearest multiple of aligned_width_
    int aligned_dst_height = (dst_height + aligned_width_ - 1) & ~(aligned_width_ - 1);  // Round up to nearest multiple of aligned_width_

    // Calculate scale (take minimum to fit target box)
    scale = std::min(static_cast<float>(aligned_dst_width) / src_width, static_cast<float>(aligned_dst_height) / src_height);

    // Calculate scaled dimensions
    int resized_w = static_cast<int>(src_width * scale);
    int resized_h = static_cast<int>(src_height * scale);

    // Ensure scaled dimensions are multiples of 4
    resized_w = (resized_w + aligned_width_ - 1) & ~(aligned_width_ - 1);  // Round up to nearest multiple of aligned_width_
    resized_h = (resized_h + aligned_width_ - 1) & ~(aligned_width_ - 1);  // Round up to nearest multiple of aligned_width_

    int aligned_src_width = (src_width + aligned_width_ - 1) & ~(aligned_width_ - 1);  // Round up to nearest multiple of aligned_width_

    // 1. Get source buffer
    BufferKey src_key{aligned_src_width, src_height, channels};
    auto& src_buffer = GetOrCreateBuffer(src_key);

    // 2. Get destination buffer
    BufferKey dst_key{aligned_dst_width, aligned_dst_height, channels};
    auto& dst_buffer = GetOrCreateBuffer(dst_key, false);

    // 3. Copy source data to RGA buffer
    for (int y = 0; y < src_height; y++) {
        memcpy(static_cast<uint8_t*>(src_buffer.virtual_addr) + y * aligned_src_width * channels, src_data + y * src_width * channels,
               src_width * channels);
    }

    dma_sync_cpu_to_device(src_buffer.dma_fd);
    dma_sync_cpu_to_device(dst_buffer.dma_fd);

    // 4. Set source and destination regions
    im_rect src_rect = {0, 0, src_width, src_height};
    im_rect dst_rect = {0, 0, resized_w, resized_h};  // Image placed in top-left corner

    int ret = imcheck(src_buffer.buffer, dst_buffer.buffer, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret) {
        INSPIRECV_LOG(ERROR) << "RGA parameter check failed: " << imStrError((IM_STATUS)ret);
        return -1;
    }

    // 5. Fill entire destination area with black first
    im_rect fill_rect = {0, 0, aligned_dst_width, aligned_dst_height};
    ret = imfill(dst_buffer.buffer, fill_rect, 0x000000);  // Fill with black
    if (ret != IM_STATUS_SUCCESS) {
        INSPIRECV_LOG(ERROR) << "RGA fill failed: " << imStrError((IM_STATUS)ret);
        return -1;
    }

    // 6. Execute resize operation, image will be placed in top-left corner
    ret = improcess(src_buffer.buffer, dst_buffer.buffer, {}, src_rect, dst_rect, {}, IM_SYNC);
    if (ret != IM_STATUS_SUCCESS) {
        INSPIRECV_LOG(ERROR) << "RGA resize failed: " << imStrError((IM_STATUS)ret);
        return -1;
    }

    dma_sync_device_to_cpu(dst_buffer.dma_fd);

    // 7. Return processed data
    *dst_data = static_cast<uint8_t*>(dst_buffer.virtual_addr);
    return 0;
}

}  // namespace nexus

}  // namespace inspire

#endif  // ISF_ENABLE_RGA