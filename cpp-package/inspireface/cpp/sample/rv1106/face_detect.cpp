#include <inspirecv/inspirecv.h>
#include <inspireface/track_module/face_track_module.h>
#include <inspireface/include/inspireface/launch.h>
#include <inspireface/include/inspireface/frame_process.h>
#include <inspireface/include/inspireface/spend_timer.h>
#include <inspireface/include/inspireface/herror.h>

using namespace inspire;

int main() {
    INSPIRE_SET_LOG_LEVEL(ISF_LOG_DEBUG);
    std::string expansion_path = "";
    INSPIREFACE_CONTEXT->Load("test_res/pack/Gundam_RV1106");
    auto archive = INSPIREFACE_CONTEXT->getMArchive();
    InspireModel detModel;
    auto ret = archive.LoadModel("face_detect_160", detModel);
    if (ret != SARC_SUCCESS) {
        INSPIRE_LOGE("Load %s error: %d", "face_detect_160", ret);
        return HERR_ARCHIVE_LOAD_MODEL_FAILURE;
    }
    FaceDetectAdapt face_detect(160);
    std::vector<int> input_size;
    input_size = detModel.Config().get<std::vector<int>>("input_size");

    ret = face_detect.LoadData(detModel, detModel.modelType, false);
    if (ret != 0) {
        INSPIRE_LOGE("Load %s error: %d", "face_detect_160", ret);
        return HERR_ARCHIVE_LOAD_MODEL_FAILURE;
    }

    std::cout << "Load model success" << std::endl;

    auto img = inspirecv::Image::Create("data/bulk/kun.jpg");

    inspire::SpendTimer time_spend("Detect");
    FaceLocList results;
    for (int i = 0; i < 10; i++) {
        time_spend.Start();
        results = face_detect(img);
        time_spend.Stop();
        std::cout << "================" << std::endl;
    }
    std::cout << time_spend << std::endl;
    std::cout << "Face detect success:" << results.size() << std::endl;
    for (auto &face : results) {
        std::cout << "Face detect success:" << face.x1 << " " << face.y1 << " " << face.x2 << " " << face.y2 << std::endl;
    }

    return 0;
}