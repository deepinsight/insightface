#include <inspirecv/inspirecv.h>
#include <inspireface/include/inspireface/launch.h>
#include "inspireface/middleware/model_archive/inspire_archive.h"
#include "inspireface/track_module/face_detect/face_detect_adapt.h"
#include "inspireface/track_module/landmark/face_landmark_adapt.h"
#include "inspireface/track_module/quality/face_pose_quality_adapt.h"
#include "inspireface/recognition_module/extract/extract_adapt.h"
#include "inspireface/include/inspireface/spend_timer.h"

void test_face_detect() {
    inspire::InspireModel model;
    INSPIREFACE_CONTEXT->getMArchive().LoadModel("face_detect_160", model);
    auto input_size = 160;
    inspire::FaceDetectAdapt faceDetectAdapt(input_size);
    faceDetectAdapt.LoadData(model, model.modelType);
    inspirecv::Image image = inspirecv::Image::Create("test_res/data/bulk/kun.jpg");
    inspire::FaceLocList faces;
    inspire::SpendTimer timeSpend("Face Detect@" + std::to_string(input_size));
    for (int i = 0; i < 1000; i++) {
        timeSpend.Start();
        faces = faceDetectAdapt(image);
        timeSpend.Stop();
    }
    std::cout << timeSpend << std::endl;
    ;
    std::cout << "faces size: " << faces.size() << std::endl;
    for (auto &face : faces) {
        inspirecv::Rect2i rect = inspirecv::Rect2i::Create(face.x1, face.y1, face.x2 - face.x1, face.y2 - face.y1);
        image.DrawRect(rect, {0, 0, 255});
    }
    image.Write("im.jpg");
}

void test_landmark() {
    inspire::InspireModel model;
    INSPIREFACE_CONTEXT->getMArchive().LoadModel("landmark", model);
    auto input_size = 112;
    inspire::FaceLandmarkAdapt landmarkAdapt(input_size);
    landmarkAdapt.LoadData(model, model.modelType);
    inspirecv::Image image = inspirecv::Image::Create("test_res/data/crop/crop.png");
    image = image.Resize(input_size, input_size);
    std::vector<float> lmk;
    inspire::SpendTimer timeSpend("Landmark@" + std::to_string(input_size));
    timeSpend.Start();
    for (int i = 0; i < 10; i++) {
        lmk = landmarkAdapt(image);
    }
    timeSpend.Stop();
    std::cout << timeSpend << std::endl;
    ;
    for (int i = 0; i < inspire::FaceLandmarkAdapt::NUM_OF_LANDMARK; i++) {
        auto p = inspirecv::Point2i::Create(lmk[i * 2] * input_size, lmk[i * 2 + 1] * input_size);
        image.DrawCircle(p, 5, {0, 0, 255});
    }
    image.Write("lm.jpg");
}

void test_quality() {
    inspire::InspireModel model;
    INSPIREFACE_CONTEXT->getMArchive().LoadModel("pose_quality", model);
    auto input_size = 96;
    inspire::FacePoseQualityAdapt poseQualityAdapt;
    poseQualityAdapt.LoadData(model, model.modelType);
    inspirecv::Image image = inspirecv::Image::Create("test_res/data/crop/crop.png");
    image = image.Resize(input_size, input_size);
    inspire::FacePoseQualityAdaptResult quality;
    inspire::SpendTimer timeSpend("Pose Quality@" + std::to_string(input_size));
    timeSpend.Start();
    for (int i = 0; i < 10; i++) {
        quality = poseQualityAdapt(image);
    }
    timeSpend.Stop();
    std::cout << timeSpend << std::endl;
    ;
    std::cout << "quality: " << quality.pitch << ", " << quality.yaw << ", " << quality.roll << std::endl;
    for (int i = 0; i < quality.lmk.size(); i++) {
        std::cout << "lmk: " << quality.lmk[i].GetX() << ", " << quality.lmk[i].GetY() << std::endl;
        auto p = inspirecv::Point2i::Create(quality.lmk[i].GetX(), quality.lmk[i].GetY());
        image.DrawCircle(p, 3, {0, 0, 255});
    }
    image.Write("qu.jpg");
}

void test_feature() {
    inspire::InspireModel model;
    INSPIREFACE_CONTEXT->getMArchive().LoadModel("feature", model);
    auto input_size = 112;
    inspire::ExtractAdapt extractAdapt;
    extractAdapt.LoadData(model, model.modelType);
    inspirecv::Image image = inspirecv::Image::Create("test_res/data/crop/crop.png");
    image = image.Resize(input_size, input_size);
    float norm;
    bool normalize = true;
    inspire::SpendTimer timeSpend("Extract@" + std::to_string(input_size));
    timeSpend.Start();
    inspire::Embedded feature;
    for (int i = 0; i < 10; i++) {
        feature = extractAdapt(image, norm, normalize);
    }
    timeSpend.Stop();
    std::cout << timeSpend << std::endl;
    ;
    std::cout << "feature: " << feature.size() << std::endl;
}

int main() {
    std::string archivePath = "test_res/pack/Pikachu_Apple";
    INSPIREFACE_CONTEXT->Load(archivePath);
    // Test face detect
    test_face_detect();

    // Test landmark
    // test_landmark();

    // Test quality
    // test_quality();

    // Test feature
    // test_feature();
    return 0;
}
