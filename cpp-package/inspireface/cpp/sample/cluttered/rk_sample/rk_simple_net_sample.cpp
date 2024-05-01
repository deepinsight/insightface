//
// Created by Tunm-Air13 on 2023/9/21.
//

#include "opencv2/opencv.hpp"
#include "inspireface/track_module/face_detect/all.h"
#include "inspireface/pipeline_module/attribute/mask_predict.h"

#include "inspireface/middleware/costman.h"
#include "inspireface/track_module/quality/face_pose_quality.h"
#include "inspireface/track_module/landmark/face_landmark.h"
#include "inspireface/pipeline_module/liveness/rgb_anti_spoofing.h"
#include "inspireface/face_context.h"

using namespace inspire;

InspireArchive loader;


void test_rnet() {
    std::shared_ptr<RNet> m_rnet_;
    Configurable param;
//    param.set<int>("model_index", ModelIndex::_04_refine_net);
    param.set<std::string>("input_layer", "input_1");
    param.set<std::vector<std::string>>("outputs_layers", {"conv5-1/Softmax", "conv5-2/BiasAdd"});
    param.set<std::vector<int>>("input_size", {24, 24});
    param.set<std::vector<float>>("mean", {0.0f, 0.0f, 0.0f});
    param.set<std::vector<float>>("norm", {1.0f, 1.0f, 1.0f});
    param.set<bool>("swap_color", true);        // RGB mode
    param.set<int>("data_type", InputTensorInfo::kDataTypeImage);
    param.set<int>("input_tensor_type", InputTensorInfo::kTensorTypeUint8);
    param.set<int>("output_tensor_type", InputTensorInfo::kTensorTypeFp32);
    param.set<bool>("nchw", false);

    InspireModel model;
    loader.LoadModel("refine_net", model);
    m_rnet_ = std::make_shared<RNet>();
    m_rnet_->loadData(model, InferenceHelper::kRknn);

    {
        // Load a image
        cv::Mat image = cv::imread("test_res/images/test_data/hasface.jpg");

        Timer timer;
        auto score = (*m_rnet_)(image);
        LOGD("RNETcost: %f", timer.GetCostTimeUpdate());
        LOGD("has face: %f", score);
    }

    {
        // Load a image
        cv::Mat image = cv::imread("test_res/images/test_data/noface.jpg");

        Timer timer;
        auto score = (*m_rnet_)(image);
        LOGD("cost: %f", timer.GetCostTimeUpdate());
        LOGD("non face: %f", score);
    }

}

void test_mask() {
    Configurable param;
//    param.set<int>("model_index", ModelIndex::_05_mask);
    param.set<std::string>("input_layer", "input_1");
    param.set<std::vector<std::string>>("outputs_layers", {"activation_1/Softmax",});
    param.set<std::vector<int>>("input_size", {96, 96});
    param.set<std::vector<float>>("mean", {0.0f, 0.0f, 0.0f});
    param.set<std::vector<float>>("norm", {1.0f, 1.0f, 1.0f});
    param.set<bool>("swap_color", true);        // RGB mode
    param.set<int>("data_type", InputTensorInfo::kDataTypeImage);
    param.set<int>("input_tensor_type", InputTensorInfo::kTensorTypeUint8);
    param.set<int>("output_tensor_type", InputTensorInfo::kTensorTypeFp32);
    param.set<bool>("nchw", false);

    std::shared_ptr<MaskPredict> m_mask_predict_;
    m_mask_predict_ = std::make_shared<MaskPredict>();
    InspireModel model;
    loader.LoadModel("mask_detect", model);
    m_mask_predict_->loadData(model, InferenceHelper::kRknn);

    {
        // Load a image
        cv::Mat image = cv::imread("test_res/images/test_data/mask.jpg");

        Timer timer;
        auto score = (*m_mask_predict_)(image);
        LOGD("cost: %f", timer.GetCostTimeUpdate());
        LOGD("mask: %f", score);
    }

    {
        // Load a image
        cv::Mat image = cv::imread("test_res/images/test_data/nomask.jpg");

        Timer timer;
        auto score = (*m_mask_predict_)(image);
        LOGD("cost: %f", timer.GetCostTimeUpdate());
        LOGD("maskless: %f", score);
    }

}

void test_quality() {
    Configurable param;
//    param.set<int>("model_index", ModelIndex::_07_pose_q_fp16);
    param.set<std::string>("input_layer", "data");
    param.set<std::vector<std::string>>("outputs_layers", {"fc1", });
    param.set<std::vector<int>>("input_size", {96, 96});
    param.set<std::vector<float>>("mean", {0.0f, 0.0f, 0.0f});
    param.set<std::vector<float>>("norm", {1.0f, 1.0f, 1.0f});
    param.set<bool>("swap_color", true);        // RGB mode
    param.set<int>("data_type", InputTensorInfo::kDataTypeImage);
    param.set<int>("input_tensor_type", InputTensorInfo::kTensorTypeUint8);
    param.set<int>("output_tensor_type", InputTensorInfo::kTensorTypeFp32);
    param.set<bool>("nchw", false);
    std::shared_ptr<FacePoseQuality> m_face_quality_;
    m_face_quality_ = std::make_shared<FacePoseQuality>();
    InspireModel model;
    loader.LoadModel("pose_quality", model);
    m_face_quality_->loadData(model, InferenceHelper::kRknn);

    {
        std::vector<std::string> names = {
                "test_res/images/test_data/p3.jpg",
//                "test_res/images/test_data/p1.jpg",
        };
        for (int i = 0; i < names.size(); ++i) {
            LOGD("Image: %s", names[i].c_str());
            cv::Mat image = cv::imread(names[i]);

            Timer timer;
            auto pose_res = (*m_face_quality_)(image);
            LOGD("质量cost: %f", timer.GetCostTimeUpdate());

            for (auto &p: pose_res.lmk) {
                cv::circle(image, p, 0, cv::Scalar(0, 0, 255), 2);
            }
            cv::imwrite("pose.jpg", image);
            LOGD("pitch: %f", pose_res.pitch);
            LOGD("yam: %f", pose_res.yaw);
            LOGD("roll: %f", pose_res.roll);

            for (auto q: pose_res.lmk_quality) {
                std::cout << q << ", ";
            }
            std::cout << std::endl;
        }

    }

}


void test_landmark_mnn() {
    Configurable param;
//    param.set<int>("model_index", ModelIndex::_01_lmk);
    param.set<std::string>("input_layer", "input_1");
    param.set<std::vector<std::string>>("outputs_layers", {"prelu1/add", });
    param.set<std::vector<int>>("input_size", {112, 112});
    param.set<std::vector<float>>("mean", {127.5f, 127.5f, 127.5f});
    param.set<std::vector<float>>("norm", {0.0078125f, 0.0078125f, 0.0078125f});

    std::shared_ptr<FaceLandmark> m_landmark_predictor_;
    m_landmark_predictor_ = std::make_shared<FaceLandmark>(112);
    InspireModel model;
    loader.LoadModel("landmark", model);
    m_landmark_predictor_->loadData(model);

    cv::Mat image = cv::imread("test_res/images/test_data/crop.png");
    cv::resize(image, image, cv::Size(112, 112));

    std::vector<float> lmk;
    Timer timer;
    for (int i = 0; i < 50; ++i) {
        lmk = (*m_landmark_predictor_)(image);
        LOGD("cost: %f", timer.GetCostTimeUpdate());
    }

    for (int i = 0; i < FaceLandmark::NUM_OF_LANDMARK; ++i) {
        float x = lmk[i * 2 + 0] * 112;
        float y = lmk[i * 2 + 1] * 112;
        cv::circle(image, cv::Point2f(x, y), 0, cv::Scalar(0, 0, 255), 1);
    }

    cv::imwrite("lmk.jpg", image);


}



void test_landmark() {
    Configurable param;
//    param.set<int>("model_index", ModelIndex::_01_lmk);
    param.set<std::string>("input_layer", "input_1");
    param.set<std::vector<std::string>>("outputs_layers", {"prelu1/add", });
    param.set<std::vector<int>>("input_size", {112, 112});
    param.set<std::vector<float>>("mean", {0.0f, 0.0f, 0.0f});
    param.set<std::vector<float>>("norm", {1.0f, 1.0f, 1.0f});
    param.set<int>("data_type", InputTensorInfo::kDataTypeImage);
    param.set<int>("input_tensor_type", InputTensorInfo::kTensorTypeUint8);
    param.set<int>("output_tensor_type", InputTensorInfo::kTensorTypeFp32);
    param.set<bool>("nchw", false);

    std::shared_ptr<FaceLandmark> m_landmark_predictor_;
    m_landmark_predictor_ = std::make_shared<FaceLandmark>(112);
    InspireModel model;
    loader.LoadModel("landmark", model);
    m_landmark_predictor_->loadData(model, InferenceHelper::kRknn);

    cv::Mat image = cv::imread("test_res/images/test_data/0.jpg");
    cv::resize(image, image, cv::Size(112, 112));

    std::vector<float> lmk;
    Timer timer;
    for (int i = 0; i < 50; ++i) {
        lmk = (*m_landmark_predictor_)(image);
        LOGD("LMKcost: %f", timer.GetCostTimeUpdate());
    }

    for (int i = 0; i < FaceLandmark::NUM_OF_LANDMARK; ++i) {
        float x = lmk[i * 2 + 0] * 112;
        float y = lmk[i * 2 + 1] * 112;
        cv::circle(image, cv::Point2f(x, y), 0, cv::Scalar(0, 0, 255), 1);
    }

    cv::imwrite("lmk.jpg", image);


}


void test_liveness() {

    Configurable param;
//    param.set<int>("model_index", ModelIndex::_06_msafa27);
    param.set<std::string>("input_layer", "data");
    param.set<std::vector<std::string>>("outputs_layers", {"556",});
    param.set<std::vector<int>>("input_size", {80, 80});
    param.set<std::vector<float>>("mean", {0.0f, 0.0f, 0.0f});
    param.set<std::vector<float>>("norm", {1.0f, 1.0f, 1.0f});
    param.set<bool>("swap_color", false);        // RGB mode
    param.set<int>("data_type", InputTensorInfo::kDataTypeImage);
    param.set<int>("input_tensor_type", InputTensorInfo::kTensorTypeUint8);
    param.set<int>("output_tensor_type", InputTensorInfo::kTensorTypeFp32);
    param.set<bool>("nchw", false);

    std::shared_ptr<RBGAntiSpoofing> m_rgb_anti_spoofing_;

    InspireModel model;
    loader.LoadModel("rgb_anti_spoofing", model);
    m_rgb_anti_spoofing_ = std::make_shared<RBGAntiSpoofing>(80, true);
    m_rgb_anti_spoofing_->loadData(model, InferenceHelper::kRknn);

    std::vector<std::string> names = {
            "test_res/images/test_data/real.jpg",
            "test_res/images/test_data/fake.jpg",
            "test_res/images/test_data/live.jpg",
            "test_res/images/test_data/ttt.jpg",
            "test_res/images/test_data/w.jpg",
            "test_res/images/test_data/w2.jpg",
    };

    for (int i = 0; i < names.size(); ++i) {
        auto image = cv::imread(names[i]);
        Timer timer;
        auto score = (*m_rgb_anti_spoofing_)(image);
        LOGD("cost: %f", timer.GetCostTimeUpdate());
        LOGD("%s : %f", names[i].c_str(), score);
    }

}

int test_liveness_ctx() {
    CustomPipelineParameter parameter;
    parameter.enable_liveness = true;
    FaceContext ctx;
    ctx.Configuration("test_res/pack/Gundam_RV1109", inspire::DETECT_MODE_IMAGE, 3, parameter);
    std::vector<std::string> names = {
            "test_res/images/test_data/real.jpg",
            "test_res/images/test_data/fake.jpg",
            "test_res/images/test_data/live.jpg",
            "test_res/images/test_data/ttt.jpg",
            "test_res/images/test_data/w.jpg",
            "test_res/images/test_data/w2.jpg",
            "test_res/images/test_data/bb.png",
    };

    for (int i = 0; i < names.size(); ++i) {
        auto image = cv::imread(names[i]);
        auto score = (*ctx.FacePipelineModule()->getMRgbAntiSpoofing())(image);
        LOGD("%s : %f", names[i].c_str(), score);
    }


    return 0;
}

int main() {
    loader.ReLoad("test_res/pack/Gundam_RV1109");

//    test_rnet();

//    test_mask();

//    test_quality();

//    test_landmark_mnn();

//    test_landmark();

    test_liveness();
    test_liveness_ctx();
    return 0;
}