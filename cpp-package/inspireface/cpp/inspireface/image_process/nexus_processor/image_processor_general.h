#ifndef INSPIRE_FACE_NEXUS_IMAGE_PROCESSOR_GENERAL_H
#define INSPIRE_FACE_NEXUS_IMAGE_PROCESSOR_GENERAL_H

#include "image_processor.h"
#include <inspirecv/inspirecv.h>

namespace inspire {

namespace nexus {

class INSPIRE_API_EXPORT GeneralImageProcessor : public ImageProcessor {
public:
    GeneralImageProcessor() = default;
    ~GeneralImageProcessor() override = default;

    int32_t Resize(const uint8_t* src_data, int src_width, int src_height, int channels, uint8_t** dst_data, int dst_width, int dst_height) override;

    int32_t SwapColor(const uint8_t* src_data, int src_width, int src_height, int channels, uint8_t** dst_data) override;

    int32_t Padding(const uint8_t* src_data, int src_width, int src_height, int channels, int top, int bottom, int left, int right,
                    uint8_t** dst_data, int& dst_width, int& dst_height) override;

    int32_t ResizeAndPadding(const uint8_t* src_data, int src_width, int src_height, int channels, int dst_width, int dst_height, uint8_t** dst_data,
                             float& scale) override;

    int32_t MarkDone() override;

    void DumpCacheStatus() const override;

    int32_t GetAlignedWidth(int width) const override;

    void SetAlignedWidth(int width) override;

private:
    struct BufferWrapper {
        inspirecv::Image image;
        uint8_t* GetData() {
            return const_cast<uint8_t*>(image.Data());
        }
    };
    BufferWrapper last_buffer_;

};  // GeneralImageProcessor

}  // namespace nexus

}  // namespace inspire

#endif  // INSPIRE_FACE_NEXUS_IMAGE_PROCESSOR_GENERAL_H