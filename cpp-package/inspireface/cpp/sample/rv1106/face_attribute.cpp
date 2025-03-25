#include <inspirecv/inspirecv.h>
#include <inspireface/pipeline_module/attribute/face_attribute_adapt.h>
#include "inspireface/initialization_module/launch.h"
#include <inspireface/middleware/inspirecv_image_process.h>
#include <inspirecv/time_spend.h>
#include <log.h>

using namespace inspire;

int main() {
    INSPIRE_SET_LOG_LEVEL(ISF_LOG_DEBUG);
    std::string expansion_path = "";
    INSPIRE_LAUNCH->Load("test_res/pack/Gundam_RV1106");
    auto archive = INSPIRE_LAUNCH->getMArchive();
    InspireModel detModel;
    auto ret = archive.LoadModel("face_attribute", detModel);
    if (ret != SARC_SUCCESS) {
        INSPIRE_LOGE("Load %s error: %d", "face_detect_160", ret);
        return HERR_ARCHIVE_LOAD_MODEL_FAILURE;
    }

    FaceAttributePredictAdapt face_attribute;
    face_attribute.loadData(detModel, detModel.modelType, false);

    auto img = inspirecv::Image::Create("test_res/data/crop/crop.png");
    auto result = face_attribute(img);
    std::cout << "result: " << result[0] << ", " << result[1] << ", " << result[2] << std::endl;

    return 0;
}