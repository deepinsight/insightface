#include "image_processor_general.h"
#include "log.h"
#include <cmath>
#include <exception>
#include <limits>

namespace inspire {

namespace nexus {

namespace {

bool IsSafeImageShape(int width, int height, int channels) {
    return width > 0 && height > 0 && channels > 0 &&
           static_cast<size_t>(width) <= std::numeric_limits<size_t>::max() / static_cast<size_t>(height) &&
           static_cast<size_t>(width) * static_cast<size_t>(height) <=
             std::numeric_limits<size_t>::max() / static_cast<size_t>(channels);
}

bool IsValidImageRequest(const uint8_t* src_data, int src_width, int src_height, int channels, uint8_t** dst_data) {
    if (dst_data == nullptr) {
        return false;
    }
    *dst_data = nullptr;
    return src_data != nullptr && IsSafeImageShape(src_width, src_height, channels);
}

}  // namespace

int32_t GeneralImageProcessor::Resize(const uint8_t* src_data, int src_width, int src_height, int channels, uint8_t** dst_data, int dst_width,
                                      int dst_height) {
    if (!IsValidImageRequest(src_data, src_width, src_height, channels, dst_data) || !IsSafeImageShape(dst_width, dst_height, channels)) {
        return -1;
    }
    try {
        inspirecv::Image src_img(src_width, src_height, channels, src_data, false);
        last_buffer_.image = src_img.Resize(dst_width, dst_height);
        *dst_data = last_buffer_.GetData();
        return *dst_data == nullptr ? -1 : 0;
    } catch (const std::exception& error) {
        INSPIRE_LOGE("CPU image resize failed: %s", error.what());
        return -1;
    }
}

int32_t GeneralImageProcessor::SwapColor(const uint8_t* src_data, int src_width, int src_height, int channels, uint8_t** dst_data) {
    if (!IsValidImageRequest(src_data, src_width, src_height, channels, dst_data)) {
        return -1;
    }
    try {
        inspirecv::Image src_img(src_width, src_height, channels, src_data, false);
        last_buffer_.image = src_img.SwapRB();
        *dst_data = last_buffer_.GetData();
        return *dst_data == nullptr ? -1 : 0;
    } catch (const std::exception& error) {
        INSPIRE_LOGE("CPU color conversion failed: %s", error.what());
        return -1;
    }
}

int32_t GeneralImageProcessor::Padding(const uint8_t* src_data, int src_width, int src_height, int channels, int top, int bottom, int left, int right,
                                       uint8_t** dst_data, int& dst_width, int& dst_height) {
    dst_width = 0;
    dst_height = 0;
    if (!IsValidImageRequest(src_data, src_width, src_height, channels, dst_data) || top < 0 || bottom < 0 || left < 0 || right < 0 ||
        src_width > std::numeric_limits<int>::max() - left - right || src_height > std::numeric_limits<int>::max() - top - bottom) {
        return -1;
    }
    const int output_width = src_width + left + right;
    const int output_height = src_height + top + bottom;
    if (!IsSafeImageShape(output_width, output_height, channels)) {
        return -1;
    }
    try {
        inspirecv::Image src_img(src_width, src_height, channels, src_data, false);
        last_buffer_.image = src_img.Pad(top, bottom, left, right, inspirecv::Color::Black);
        *dst_data = last_buffer_.GetData();
        if (*dst_data == nullptr) {
            return -1;
        }
        dst_width = output_width;
        dst_height = output_height;
        return 0;
    } catch (const std::exception& error) {
        INSPIRE_LOGE("CPU image padding failed: %s", error.what());
        return -1;
    }
}
int32_t GeneralImageProcessor::ResizeAndPadding(const uint8_t* src_data, int src_width, int src_height, int channels, int dst_width, int dst_height,
                                                uint8_t** dst_data, float& scale) {
    scale = 0.0f;
    if (!IsValidImageRequest(src_data, src_width, src_height, channels, dst_data) || !IsSafeImageShape(dst_width, dst_height, channels)) {
        return -1;
    }
    const float output_scale = std::min(static_cast<float>(dst_width) / src_width, static_cast<float>(dst_height) / src_height);
    const int resized_w = static_cast<int>(src_width * output_scale);
    const int resized_h = static_cast<int>(src_height * output_scale);
    if (!std::isfinite(output_scale) || output_scale <= 0.0f || resized_w <= 0 || resized_h <= 0) {
        return -1;
    }

    try {
        inspirecv::Image src_img(src_width, src_height, channels, src_data, false);
        inspirecv::Image resized_img = src_img.Resize(resized_w, resized_h);
        last_buffer_.image = resized_img.Pad(0, dst_height - resized_h, 0, dst_width - resized_w, inspirecv::Color::Black);
        *dst_data = last_buffer_.GetData();
        if (*dst_data == nullptr) {
            return -1;
        }
        scale = output_scale;
        return 0;
    } catch (const std::exception& error) {
        INSPIRE_LOGE("CPU resize-and-pad failed: %s", error.what());
        return -1;
    }
}

int32_t GeneralImageProcessor::MarkDone() {
    return 0;
}

void GeneralImageProcessor::DumpCacheStatus() const {
    INSPIRECV_LOG(INFO) << "GeneralImageProcessor has no cache to dump";
}

int32_t GeneralImageProcessor::GetAlignedWidth(int width) const {
    // Not Supported
    INSPIRE_LOGE("GeneralImageProcessor::GetAlignedWidth is not supported");
    return 0;
}

void GeneralImageProcessor::SetAlignedWidth(int width) {
    // Not Supported
    INSPIRE_LOGE("GeneralImageProcessor::SetAlignedWidth is not supported");
}

}  // namespace nexus

}  // namespace inspire
