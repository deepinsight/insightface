#include <inspirecv/inspirecv.h>
#include <inspireface/track_module/face_track_module.h>
#include <inspireface/include/inspireface/launch.h>
#include <inspireface/include/inspireface/frame_process.h>
#include <inspireface/pipeline_module/face_pipeline_module.h>
#include <inspireface/common/face_data/face_serialize_tools.h>

using namespace inspire;

int main() {
    std::string expansion_path = "";
    INSPIREFACE_CONTEXT->Load("test_res/pack/Pikachu");
    auto archive = INSPIREFACE_CONTEXT->getMArchive();
    auto mode = inspire::DetectModuleMode::DETECT_MODE_LIGHT_TRACK;
    FaceTrackModule tracker(mode, 10, 20, 320, -1);
    tracker.Configuration(archive, expansion_path);

    FacePipelineModule pipe(archive, true, true, true, true, true);

    auto image = inspirecv::Image::Create("test_res/data/bulk/r90.jpg");
    inspirecv::FrameProcess processor;
    processor.SetDataBuffer(image.Data(), image.Height(), image.Width());
    processor.SetDataFormat(inspirecv::DATA_FORMAT::BGR);
    processor.SetRotationMode(inspirecv::ROTATION_MODE::ROTATION_90);
    std::vector<FaceProcessFunctionOption> methods = {PROCESS_MASK, PROCESS_RGB_LIVENESS, PROCESS_ATTRIBUTE, PROCESS_INTERACTION};
    for (int i = 0; i < 1; i++) {
        auto show = image.Clone();
        tracker.UpdateStream(processor);
        auto faces = tracker.trackingFace;
        int index = 0;
        if (faces.size() > 0) {
            auto &face = faces[index];
            auto hyper_face = FaceObjectInternalToHyperFaceData(face);
            PrintHyperFaceDataDetail(hyper_face);

            std::cout << face.getTransMatrix() << std::endl;

            for (auto method : methods) {
                pipe.Process(processor, hyper_face, method);
            }
            std::cout << "eyes status: " << pipe.eyesStatusCache[0] << " " << pipe.eyesStatusCache[1] << std::endl;
        }

        // show.Show("faces", );
    }

    return 0;
}