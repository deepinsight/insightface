#include <iostream>
#include <inspirecv/inspirecv.h>
#include <inspireface/include/inspireface/launch.h>
#include <inspireface/include/inspireface/frame_process.h>
#include "inspireface/track_module/landmark/face_landmark_adapt.h"
int main() {
    std::string expansion_path = "";
    INSPIREFACE_CONTEXT->Load("test_res/pack/Pikachu-t4");
    auto archive = INSPIREFACE_CONTEXT->getMArchive();

    inspire::InspireModel lmkModel;
    auto ret = archive.LoadModel("landmark", lmkModel);
    if (ret != 0) {
        INSPIRE_LOGE("Load %s error: %d", "landmark", ret);
        return -1;
    }

    inspire::FaceLandmarkAdapt lmk;
    lmk.LoadData(lmkModel, lmkModel.modelType);

    auto image = inspirecv::Image::Create("test_res/data/crop/crop.png");
    auto data = image.Resize(112, 112);
    auto lmk_out = lmk(data);
    std::vector<inspirecv::Point2i> landmarks_output(inspire::FaceLandmarkAdapt::NUM_OF_LANDMARK);
    for (int i = 0; i < inspire::FaceLandmarkAdapt::NUM_OF_LANDMARK; ++i) {
        float x = lmk_out[i * 2 + 0] * image.Width();
        float y = lmk_out[i * 2 + 1] * image.Height();
        landmarks_output[i] = inspirecv::Point<int>(x, y);
    }

    for (int i = 0; i < landmarks_output.size(); ++i) {
        image.DrawCircle(landmarks_output[i], 5, {0, 0, 255});
    }
    image.Write("crop_lmk.png");
}
