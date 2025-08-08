#ifndef INSPIRE_FACE_NEXUS_IMAGE_PROCESSOR_RGA_H
#define INSPIRE_FACE_NEXUS_IMAGE_PROCESSOR_RGA_H

#if defined(ISF_ENABLE_RGA)

#include "image_processor.h"
#include <linux/stddef.h>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#include <memory>
#include <unordered_map>
#include "im2d.hpp"
#include "im2d_single.h"
#include "RgaUtils.h"
#include "rga/utils.h"
#include "rga/dma_alloc.h"
#include <launch.h>

namespace inspire {

namespace nexus {

class INSPIRE_API_EXPORT RgaImageProcessor : public ImageProcessor {
public:
    RgaImageProcessor();
    ~RgaImageProcessor() override;

    int32_t Resize(const uint8_t* src_data, int src_width, int src_height, int channels, uint8_t** dst_data, int dst_width, int dst_height) override;

    int32_t SwapColor(const uint8_t* src_data, int src_width, int src_height, int channels, uint8_t** dst_data) override;

    int32_t Padding(const uint8_t* src_data, int src_width, int src_height, int channels, int top, int bottom, int left, int right,
                    uint8_t** dst_data, int& dst_width, int& dst_height) override;

    int32_t ResizeAndPadding(const uint8_t* src_data, int src_width, int src_height, int channels, int dst_width, int dst_height, uint8_t** dst_data,
                             float& scale) override;

    int32_t MarkDone() override;

    int32_t GetAlignedWidth(int width) const override;

    void SetAlignedWidth(int width) override;

public:
    struct BufferInfo {
        int dma_fd;
        int width;
        int height;
        int channels;
        size_t buffer_size;
    };

    BufferInfo GetCurrentSrcBufferInfo() const {
        auto it = buffer_cache_.find(last_src_key_);
        if (it != buffer_cache_.end()) {
            const auto& buffer = it->second;
            return {buffer.dma_fd, buffer.width, buffer.height, buffer.channels, buffer.buffer_size};
        }
        return {-1, 0, 0, 0, 0};  // Return invalid values if cache doesn't exist
    }

    BufferInfo GetCurrentDstBufferInfo() const {
        auto it = buffer_cache_.find(last_dst_key_);
        if (it != buffer_cache_.end()) {
            const auto& buffer = it->second;
            return {buffer.dma_fd, buffer.width, buffer.height, buffer.channels, buffer.buffer_size};
        }
        return {-1, 0, 0, 0, 0};  // Return invalid values if cache doesn't exist
    }

    size_t GetCacheSize() const {
        return buffer_cache_.size();
    }

    void DumpCacheStatus() const override {
        INSPIRECV_LOG(INFO) << "Current cache status:";
        INSPIRECV_LOG(INFO) << "Cache size: " << buffer_cache_.size();

        auto src_info = GetCurrentSrcBufferInfo();
        INSPIRECV_LOG(INFO) << "Source buffer: "
                            << "dma_fd=" << src_info.dma_fd << ", size=" << src_info.width << "x" << src_info.height << "x" << src_info.channels;

        auto dst_info = GetCurrentDstBufferInfo();
        INSPIRECV_LOG(INFO) << "Destination buffer: "
                            << "dma_fd=" << dst_info.dma_fd << ", size=" << dst_info.width << "x" << dst_info.height << "x" << dst_info.channels;
    }

private:
    struct RGABuffer {
        int width{0};
        int height{0};
        int channels{0};
        int dma_fd{-1};
        void* virtual_addr{nullptr};
        size_t buffer_size{0};
        rga_buffer_handle_t handle{0};
        rga_buffer_t buffer{};

        bool Allocate(int w, int h, int c) {
            width = w;
            height = h;
            channels = c;
            buffer_size = width * height * channels;

            int ret = dma_buf_alloc(INSPIREFACE_CONTEXT->GetRockchipDmaHeapPath().c_str(), buffer_size, &dma_fd, &virtual_addr);
            if (ret < 0) {
                INSPIRECV_LOG(ERROR) << "Failed to allocate DMA buffer: " << ret;
                return false;
            }

            handle = importbuffer_fd(dma_fd, buffer_size);
            if (handle == 0) {
                INSPIRECV_LOG(ERROR) << "Failed to import buffer";
                Release();
                return false;
            }

            buffer = wrapbuffer_handle(handle, w, h, RK_FORMAT_RGB_888);

            return true;
        }

        void Release() {
            if (handle) {
                releasebuffer_handle(handle);
                handle = 0;
            }
            if (dma_fd >= 0) {
                dma_buf_free(buffer_size, &dma_fd, virtual_addr);
                dma_fd = -1;
                virtual_addr = nullptr;
            }
        }

        ~RGABuffer() {
            Release();
        }
    };

    struct BufferKey {
        int width;
        int height;
        int channels;

        bool operator==(const BufferKey& other) const {
            return width == other.width && height == other.height && channels == other.channels;
        }
    };

    struct BufferKeyHash {
        std::size_t operator()(const BufferKey& key) const {
            return std::hash<int>()(key.width) ^ (std::hash<int>()(key.height) << 1) ^ (std::hash<int>()(key.channels) << 2);
        }
    };

    RGABuffer& GetOrCreateBuffer(const BufferKey& key, bool is_src = true) {
        auto it = buffer_cache_.find(key);
        if (it != buffer_cache_.end()) {
            if (is_src) {
                last_src_key_ = key;
            } else {
                last_dst_key_ = key;
            }
            return it->second;
        }

        if (buffer_cache_.size() >= 3) {  // Keep max 3 buffers in cache
            for (auto it = buffer_cache_.begin(); it != buffer_cache_.end();) {
                if (!(it->first == last_src_key_) && !(it->first == last_dst_key_)) {
                    it = buffer_cache_.erase(it);
                } else {
                    ++it;
                }
            }
        }

        auto& buffer = buffer_cache_[key];
        if (!buffer.Allocate(key.width, key.height, key.channels)) {
            INSPIRECV_LOG(ERROR) << "Failed to allocate RGA buffer";
            throw std::runtime_error("RGA buffer allocation failed");
        }

        if (is_src) {
            last_src_key_ = key;
        } else {
            last_dst_key_ = key;
        }

        return buffer;
    }

private:
    std::unordered_map<BufferKey, RGABuffer, BufferKeyHash> buffer_cache_;
    BufferKey last_src_key_{0, 0, 0};
    BufferKey last_dst_key_{0, 0, 0};
    int32_t aligned_width_{0};
};

}  // namespace nexus

}  // namespace inspire

#endif  // ISF_ENABLE_RGA

#endif  // INSPIRE_FACE_NEXUS_IMAGE_PROCESSOR_RGA_H
