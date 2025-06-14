#ifndef INSPIREFACE_PIPELINE_MODULE_ATTRIBUTE_FACE_EMOTION_ADAPT_H
#define INSPIREFACE_PIPELINE_MODULE_ATTRIBUTE_FACE_EMOTION_ADAPT_H

#include "data_type.h"
#include "middleware/any_net_adapter.h"

namespace inspire {
    
class INSPIRE_API FaceEmotionAdapt : public AnyNetAdapter {
public:

    const std::vector<std::string> EMOTION_LABELS = {"Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"};
    const int32_t INPUT_WIDTH = 112;
    const int32_t INPUT_HEIGHT = 112;
    const int32_t OUTPUT_SIZE = 7;
public:
    FaceEmotionAdapt();
    ~FaceEmotionAdapt();

    std::vector<float> operator()(const inspirecv::Image& bgr_affine);
    
};  // class FaceEmotionAdapt

}   // namespace inspire

#endif // INSPIREFACE_PIPELINE_MODULE_ATTRIBUTE_FACE_EMOTION_ADAPT_H
