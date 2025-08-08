#include "image_processor_general.h"
#include "log.h"

namespace inspire {

namespace nexus {

int32_t GeneralImageProcessor::Resize(const uint8_t* src_data, int src_width, int src_height, int channels, uint8_t** dst_data, int dst_width,
                                      int dst_height) {
    inspirecv::Image src_img(src_width, src_height, channels, src_data, false);
    last_buffer_.image = src_img.Resize(dst_width, dst_height);
    *dst_data = last_buffer_.GetData();
    return 0;
}

int32_t GeneralImageProcessor::SwapColor(const uint8_t* src_data, int src_width, int src_height, int channels, uint8_t** dst_data) {
    inspirecv::Image src_img(src_width, src_height, channels, src_data, false);
    last_buffer_.image = src_img.SwapRB();
    *dst_data = last_buffer_.GetData();
    return 0;
}

int32_t GeneralImageProcessor::Padding(const uint8_t* src_data, int src_width, int src_height, int channels, int top, int bottom, int left, int right,
                                       uint8_t** dst_data, int& dst_width, int& dst_height) {
    inspirecv::Image src_img(src_width, src_height, channels, src_data, false);
    dst_width = src_width + left + right;
    dst_height = src_height + top + bottom;
    last_buffer_.image = src_img.Pad(top, bottom, left, right, inspirecv::Color::Black);
    *dst_data = last_buffer_.GetData();
    return 0;
}
int32_t GeneralImageProcessor::ResizeAndPadding(const uint8_t* src_data, int src_width, int src_height, int channels, int dst_width, int dst_height,
                                                uint8_t** dst_data, float& scale) {
    inspirecv::Image src_img(src_width, src_height, channels, src_data, false);

    scale = std::min(static_cast<float>(dst_width) / src_width, static_cast<float>(dst_height) / src_height);

    int resized_w = static_cast<int>(src_width * scale);
    int resized_h = static_cast<int>(src_height * scale);

    int wpad = dst_width - resized_w;
    int hpad = dst_height - resized_h;

    inspirecv::Image resized_img = src_img.Resize(resized_w, resized_h);
    last_buffer_.image = resized_img.Pad(0, hpad, 0, wpad, inspirecv::Color::Black);
    *dst_data = last_buffer_.GetData();
    return 0;
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