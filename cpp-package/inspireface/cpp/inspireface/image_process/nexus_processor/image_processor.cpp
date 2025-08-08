#include "image_processor.h"
#include "log.h"

#if defined(ISF_ENABLE_RGA)
#include "image_processor_rga.h"
#endif
#include "image_processor_general.h"

namespace inspire {

namespace nexus {

ImageProcessor::~ImageProcessor() = default;

std::unique_ptr<ImageProcessor> ImageProcessor::Create(inspire::Launch::ImageProcessingBackend backend) {
    switch (backend) {
        case inspire::Launch::IMAGE_PROCESSING_RGA:
            #if defined(ISF_ENABLE_RGA)
            return std::make_unique<RgaImageProcessor>();
            #else
            INSPIRE_LOGE("RGA backend is not enabled, using CPU backend instead");
            return std::make_unique<GeneralImageProcessor>();
            #endif
        case inspire::Launch::IMAGE_PROCESSING_CPU:
            return std::make_unique<GeneralImageProcessor>();
        default:
            // Default to CPU backend
            return std::make_unique<GeneralImageProcessor>();
    }
}

}  // namespace nexus

}  // namespace inspire
