//
// Created by tunm on 2023/9/21.
//


#include "opencv2/opencv.hpp"

#include "inspireface/middleware/costman.h"
#include "inspireface/feature_hub/face_recognition.h"
#include "inspireface/feature_hub/simd.h"
#include "inspireface/middleware/model_archive/inspire_archive.h"

using namespace inspire;

std::shared_ptr<InspireArchive> loader;

void rec_function() {

    std::shared_ptr<Extract> m_extract_;

    Configurable param;
//    param.set<int>("model_index", ModelIndex::_03_extract);
    param.set<std::string>("input_layer", "input");
    param.set<std::vector<std::string>>("outputs_layers", {"267", });
    param.set<std::vector<int>>("input_size", {112, 112});
    param.set<std::vector<float>>("mean", {0.0f, 0.0f, 0.0f});
    param.set<std::vector<float>>("norm", {1.0f, 1.0f, 1.0f});
    param.set<int>("data_type", InputTensorInfo::kDataTypeImage);
    param.set<int>("input_tensor_type", InputTensorInfo::kTensorTypeUint8);
    param.set<int>("output_tensor_type", InputTensorInfo::kTensorTypeFp32);
    param.set<bool>("nchw", false);
    param.set<bool>("swap_color", true);        // RK requires rgb input

    m_extract_ = std::make_shared<Extract>();
    InspireModel model;
    loader->LoadModel("feature", model);
    m_extract_->loadData(model, InferenceHelper::kRknn);

    loader.reset();

    std::vector<std::string> files = {
            "test_res/images/test_data/0.jpg",
            "test_res/images/test_data/1.jpg",
            "test_res/images/test_data/2.jpg",
    };
    EmbeddedList embedded_list;
    for (int i = 0; i < files.size(); ++i) {
        auto warped = cv::imread(files[i]);
        Timer timer;
        auto emb = (*m_extract_)(warped);
        LOGD("耗时: %f", timer.GetCostTimeUpdate());
        embedded_list.push_back(emb);
        LOGD("%lu", emb.size());
    }

    float _0v1;
    float _0v2;
    float _1v2;
    FaceRecognition::CosineSimilarity(embedded_list[0], embedded_list[1], _0v1);
    FaceRecognition::CosineSimilarity(embedded_list[0], embedded_list[2], _0v2);
    FaceRecognition::CosineSimilarity(embedded_list[1], embedded_list[2], _1v2);
    LOGD("0 vs 1 : %f", _0v1);
    LOGD("0 vs 2 : %f", _0v2);
    LOGD("1 vs 2 : %f", _1v2);

//    LOGD("size: %lu", embedded_list.size());
//    LOGD("num of vector: %lu", embedded_list[2].size());
//
//    float _0v1 = simd_dot(embedded_list[0].data(), embedded_list[1].data(), 512);
//    float _0v2 = simd_dot(embedded_list[0].data(), embedded_list[2].data(), 512);
//    float _1v2 = simd_dot(embedded_list[1].data(), embedded_list[2].data(), 512);
//    LOGD("0 vs 1 : %f", _0v1);
//    LOGD("0 vs 2 : %f", _0v2);
//    LOGD("1 vs 2 : %f", _1v2);

}



int main() {
    loader = std::make_shared<InspireArchive>();
    loader->ReLoad("test_res/pack/Gundam_RV1109");



    rec_function();

    return 0;
}