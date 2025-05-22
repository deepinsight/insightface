#include <inspirecv/inspirecv.h>
#include <inspireface/track_module/face_track_module.h>
#include <inspireface/include/inspireface/launch.h>
#include <inspireface/include/inspireface/frame_process.h>

using namespace inspire;

int main() {
    std::string expansion_path = "";
    INSPIREFACE_CONTEXT->Load("test_res/pack/Pikachu");
    auto archive = INSPIREFACE_CONTEXT->getMArchive();
    auto mode = inspire::DetectModuleMode::DETECT_MODE_ALWAYS_DETECT;
    FaceTrackModule tracker(mode, 10, 20, 320, -1);
    tracker.Configuration(archive, expansion_path);

    auto image = inspirecv::Image::Create("test_res/data/bulk/r0.jpg");
    inspirecv::FrameProcess processor;
    processor.SetDataBuffer(image.Data(), image.Height(), image.Width());
    processor.SetDataFormat(inspirecv::DATA_FORMAT::BGR);
    processor.SetRotationMode(inspirecv::ROTATION_MODE::ROTATION_0);
    for (int i = 0; i < 100; i++) {
        auto show = image.Clone();
        tracker.UpdateStream(processor);
        auto faces = tracker.trackingFace;
        int index = 0;
        if (faces.size() > 0) {
            auto &face = faces[index];
            for (auto &p : face.high_result.lmk) {
                show.DrawCircle(p.As<int>(), 5, {0, 255, 0});
            }
            for (auto &p : face.landmark_) {
                show.DrawCircle(p.As<int>(), 5, {0, 0, 255});
            }
        }
        show.Show("faces", 0);
    }

    return 0;
}