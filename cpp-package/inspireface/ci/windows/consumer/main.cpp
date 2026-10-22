#include <inspireface/inspireface.hpp>
#include <inspirecv/time_spend.h>

#include <cstdio>
#include <cstring>
#include <exception>
#include <vector>

#if defined(ISF_EXPORTS)
#error "A consumer must not inherit the SDK producer's ISF_EXPORTS"
#endif
#if defined(_WIN32)
#if ISF_CONSUMER_EXPECT_SHARED && !defined(ISF_BUILD_SHARED_LIBS)
#error "The shared package must propagate ISF_BUILD_SHARED_LIBS"
#elif !ISF_CONSUMER_EXPECT_SHARED && defined(ISF_BUILD_SHARED_LIBS)
#error "The static package must not propagate DLL import declarations"
#endif
#endif

static int check_image_and_timer() {
    const uint8_t pixels[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const auto image = inspirecv::Image::Create(2, 2, 3, pixels);
    if (image.Width() != 2 || image.Height() != 2 || image.Channels() != 3 ||
        !image.Data() || std::memcmp(image.Data(), pixels, sizeof(pixels)) != 0) return 1;
    auto frame = inspirecv::FrameProcess::Create(image);
    frame.SetPreviewScale(1.0f);
    const auto preview = frame.ExecutePreviewImageProcessing(true);
    if (preview.Width() != 2 || preview.Height() != 2 || preview.Channels() != 3 ||
        !preview.Data() || std::memcmp(preview.Data(), pixels, sizeof(pixels)) != 0) return 1;

    // These inline methods directly reference TimeSpend's exported static data.
    // They catch a missing INSPIRECV_API=dllimport in the installed target.
    inspirecv::TimeSpend timer("installed consumer");
    inspirecv::TimeSpend::Disable();
    if (timer.Report() != "Timer Disabled.") return 1;
    return 0;
}

static int check_model(const char* path) {
    auto launch = inspire::Launch::GetInstance();
    const int32_t status = launch->Load(path);
    if (status != 0 || !launch->isMLoad()) {
        std::fprintf(stderr, "Installed C++ model load failed: %d\n", (int)status);
        return 1;
    }
    int result = 0;
    {
        std::vector<uint8_t> pixels(64 * 64 * 3, 0);
        auto frame = inspirecv::FrameProcess::Create(pixels.data(), 64, 64);
        auto session = inspire::Session::Create(inspire::DETECT_MODE_ALWAYS_DETECT,
                                                1, inspire::CustomPipelineParameter{});
        std::vector<inspire::FaceTrackWrap> faces;
        const int32_t detected = session.FaceDetectAndTrack(frame, faces);
        if (detected != 0 || faces.size() > 1) {
            std::fprintf(stderr, "Installed C++ inference failed: status=%d, faces=%zu\n",
                         (int)detected, faces.size());
            result = 1;
        }
    }
    launch->Unload();
    if (launch->isMLoad()) result = 1;
    return result;
}

int main(int argc, char** argv) {
    if (argc > 2) {
        std::fprintf(stderr, "Usage: %s [resource-pack]\n", argv[0]);
        return 2;
    }
    try {
        const char* major = GetInspireFaceVersionMajorStr();
        if (!major || !*major || check_image_and_timer()) return 1;
        if (argc == 2 && check_model(argv[1])) return 1;
        std::printf("Installed C++ consumer passed: version major %s, model=%s\n",
                    major, argc == 2 ? "checked" : "not requested");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Installed C++ consumer exception: %s\n", error.what());
        return 1;
    }
}
