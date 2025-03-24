#include "image_processor.h"

#if defined(ISF_ENABLE_RGA)
#include "image_processor_rga.h"
#else
#include "image_processor_general.h"
#endif

namespace inspire {

namespace nexus {

ImageProcessor::~ImageProcessor() = default;

std::unique_ptr<ImageProcessor> ImageProcessor::Create() {
#if defined(ISF_ENABLE_RGA)
    return std::make_unique<RgaImageProcessor>();
#else
    return std::make_unique<GeneralImageProcessor>();
#endif
}

}  // namespace nexus

}  // namespace inspire
