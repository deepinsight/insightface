#ifndef INSPIRE_FACE_NEXUS_IMAGE_PROCESSOR_H
#define INSPIRE_FACE_NEXUS_IMAGE_PROCESSOR_H

#include <iostream>
#include <vector>
#include <inspirecv/inspirecv.h>
#include <memory>
#include "launch.h"
#include "data_type.h"

namespace inspire {

namespace nexus {

/**
 * @brief Extensible image processing interface that supports hardware acceleration backends
 *
 * This interface provides common image processing operations like resize, color conversion,
 * padding etc. It can be implemented by different backends based on compile options:
 * - Default CPU-based implementation using InspireCV (always available)
 * - Hardware accelerated implementation like Rockchip RGA (enabled with ISF_ENABLE_RGA) 
 *
 * The backend implementation is selected at runtime based on the backend parameter.
 */
class INSPIRE_API_EXPORT ImageProcessor {
public:
    static std::unique_ptr<ImageProcessor> Create(inspire::Launch::ImageProcessingBackend backend = inspire::Launch::IMAGE_PROCESSING_CPU);

public:
    // Virtual destructor
    virtual ~ImageProcessor() = 0;

    // Resize image to specified dimensions
    virtual int32_t Resize(const uint8_t* src_data, int src_width, int src_height, int channels, uint8_t** dst_data, int dst_width,
                           int dst_height) = 0;

    // Swap color channels of the image
    virtual int32_t SwapColor(const uint8_t* src_data, int src_width, int src_height, int channels, uint8_t** dst_data) = 0;

    // Add padding to image borders
    virtual int32_t Padding(const uint8_t* src_data, int src_width, int src_height, int channels, int top, int bottom, int left, int right,
                            uint8_t** dst_data, int& dst_width, int& dst_height) = 0;

    // Resize image and add padding as needed
    virtual int32_t ResizeAndPadding(const uint8_t* src_data, int src_width, int src_height, int channels, int dst_width, int dst_height,
                                     uint8_t** dst_data, float& scale) = 0;

    // Mark processing as complete
    virtual int32_t MarkDone() = 0;

    // Display cache status information
    virtual void DumpCacheStatus() const = 0;

    // Get aligned width of the image
    virtual int32_t GetAlignedWidth(int width) const = 0;

    // Set aligned width of the image
    virtual void SetAlignedWidth(int width) = 0;

};  // class ImageProcessor

}  // namespace nexus

}  // namespace inspire

#endif  // INSPIRE_FACE_NEXUS_IMAGE_PROCESSOR_H
