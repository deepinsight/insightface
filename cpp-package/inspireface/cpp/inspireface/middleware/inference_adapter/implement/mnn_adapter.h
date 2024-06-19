#ifndef MNN_ADAPTER_IMPL__
#define MNN_ADAPTER_IMPL__
#include "../inference_adapter.h"
#include "opencv2/opencv.hpp"
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include <MNN/MNNForwardType.h>

class MNNCVAdapter : public InferenceAdapter {
public:

    MNNCVAdapter() {};

    ~MNNCVAdapter() override {};

private:
    std::shared_ptr<MNN::Interpreter> detect_model_;
    MNN::Tensor *input_{};
    std::vector<MNN::Tensor*> output_tensors_;
    MNN::Session *sess{};
    std::vector<int> tensor_shape_;
    MNN::ScheduleConfig _config;
    MNNForwardType backend_;
    int width_{};
    int height_{};
    float mean[3]{};
    float normal[3]{};

};

#endif  // MNN_ADAPTER_IMPL__