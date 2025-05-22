#include <inspirecv/inspirecv.h>
#include <inspireface/pipeline_module/attribute/face_attribute_adapt.h>
#include <inspireface/include/inspireface/launch.h>
#include <inspireface/include/inspireface/frame_process.h>
#include <inspireface/include/inspireface/spend_timer.h>
#include <inspireface/include/inspireface/herror.h>
#include <log.h>

using namespace inspire;

int main() {
    INSPIRE_SET_LOG_LEVEL(ISF_LOG_DEBUG);
    std::string expansion_path = "";
    INSPIREFACE_CONTEXT->Load("test_res/pack/Gundam_RV1106");
    auto archive = INSPIREFACE_CONTEXT->getMArchive();
    InspireModel detModel;
    auto ret = archive.LoadModel("face_attribute", detModel);
    if (ret != SARC_SUCCESS) {
        INSPIRE_LOGE("Load %s error: %d", "face_detect_160", ret);
        return HERR_ARCHIVE_LOAD_MODEL_FAILURE;
    }

    FaceAttributePredictAdapt face_attribute;
    face_attribute.LoadData(detModel, detModel.modelType, false);

    auto img = inspirecv::Image::Create("test_res/data/crop/crop.png");
    auto result = face_attribute(img);
    std::cout << "result: " << result[0] << ", " << result[1] << ", " << result[2] << std::endl;

    return 0;
}