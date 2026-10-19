#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include "inspireface/c_api/inspireface.h"
#include "settings/test_settings.h"
#include "unit/test_helper/c_api_guard.h"

using inspireface_test::UniqueImageBitmap;
using inspireface_test::UniqueImageStream;
using inspireface_test::UniqueSession;

namespace {

UniqueSession CreateHandleTestSession() {
    HFSessionCustomParameter parameter = {};
    UniqueSession session;
    REQUIRE(HFCreateInspireFaceSession(parameter, HF_DETECT_MODE_ALWAYS_DETECT, 2, -1, -1, session.Put()) == HSUCCEED);
    return session;
}

UniqueImageStream CreateHandleTestStream(const inspirecv::Image& image) {
    REQUIRE_FALSE(image.Empty());
    HFImageData data = {};
    data.data = const_cast<HUInt8*>(image.Data());
    data.width = image.Width();
    data.height = image.Height();
    data.format = HF_STREAM_BGR;
    data.rotation = HF_CAMERA_ROTATION_0;
    UniqueImageStream stream;
    REQUIRE(HFCreateImageStream(&data, stream.Put()) == HSUCCEED);
    return stream;
}

}  // namespace

TEST_CASE("C API opaque handles reject every wrong resource type without invalidating the owner",
          "[api][contract][handle_lifecycle][boundary]") {
    auto session = CreateHandleTestSession();
    UniqueImageStream stream;
    REQUIRE(HFCreateImageStreamEmpty(stream.Put()) == HSUCCEED);

    std::vector<HUInt8> pixels(4 * 4 * 3, 7);
    HFImageBitmapData bitmap_data = {pixels.data(), 4, 4, 3};
    UniqueImageBitmap bitmap;
    REQUIRE(HFCreateImageBitmap(&bitmap_data, bitmap.Put()) == HSUCCEED);

    const HFSession stream_as_session = reinterpret_cast<HFSession>(stream.Get());
    const HFImageStream session_as_stream = reinterpret_cast<HFImageStream>(session.Get());
    const HFImageBitmap stream_as_bitmap = reinterpret_cast<HFImageBitmap>(stream.Get());
    const HFFaceResultSnapshot session_as_snapshot = reinterpret_cast<HFFaceResultSnapshot>(session.Get());

    CHECK(HFSessionClearTrackingFace(stream_as_session) == HERR_INVALID_CONTEXT_HANDLE);
    CHECK(HFReleaseInspireFaceSession(stream_as_session) == HERR_INVALID_CONTEXT_HANDLE);
    CHECK(HFImageStreamSetFormat(session_as_stream, HF_STREAM_BGR) == HERR_INVALID_IMAGE_STREAM_HANDLE);
    CHECK(HFReleaseImageStream(session_as_stream) == HERR_INVALID_IMAGE_STREAM_HANDLE);
    HFImageBitmapData output = {};
    CHECK(HFImageBitmapGetData(stream_as_bitmap, &output) == HERR_INVALID_IMAGE_BITMAP_HANDLE);
    CHECK(HFReleaseImageBitmap(stream_as_bitmap) == HERR_INVALID_IMAGE_BITMAP_HANDLE);
    HFMultipleFaceData snapshot_data = {};
    CHECK(HFGetFaceResultSnapshotData(session_as_snapshot, &snapshot_data) == HERR_INVALID_PARAM);
    CHECK(HFReleaseFaceResultSnapshot(session_as_snapshot) == HERR_INVALID_PARAM);

    const auto bogus = reinterpret_cast<void*>(static_cast<uintptr_t>(0x55AA55AAu));
    CHECK(HFSessionClearTrackingFace(static_cast<HFSession>(bogus)) == HERR_INVALID_CONTEXT_HANDLE);
    CHECK(HFImageStreamSetRotation(static_cast<HFImageStream>(bogus), HF_CAMERA_ROTATION_0) ==
          HERR_INVALID_IMAGE_STREAM_HANDLE);
    CHECK(HFImageBitmapGetData(static_cast<HFImageBitmap>(bogus), &output) == HERR_INVALID_IMAGE_BITMAP_HANDLE);
    CHECK(HFGetFaceResultSnapshotData(static_cast<HFFaceResultSnapshot>(bogus), &snapshot_data) == HERR_INVALID_PARAM);

    HInt32 preview_size = 0;
    CHECK(HFSessionGetTrackPreviewSize(session.Get(), &preview_size) == HSUCCEED);
    CHECK(HFImageStreamSetFormat(stream.Get(), HF_STREAM_BGR) == HSUCCEED);
    REQUIRE(HFImageBitmapGetData(bitmap.Get(), &output) == HSUCCEED);
    CHECK(output.data != nullptr);
    CHECK(output.width == 4);
    CHECK(output.height == 4);
}

TEST_CASE("C API defers opaque handle destruction while a call is in flight",
          "[api][contract][handle_lifecycle][concurrency]") {
    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/pedestrian.png"));
    REQUIRE_FALSE(image.Empty());

    for (int iteration = 0; iteration < 4; ++iteration) {
        auto session_owner = CreateHandleTestSession();
        auto stream_owner = CreateHandleTestStream(image);
        const HFSession session = session_owner.ReleaseOwnership();
        const HFImageStream stream = stream_owner.ReleaseOwnership();

        std::atomic<bool> started(false);
        HResult execute_status = HERR_UNKNOWN;
        HInt32 detected_num = -1;
        std::thread worker([&] {
            HFMultipleFaceData faces = {};
            started.store(true, std::memory_order_release);
            execute_status = HFExecuteFaceTrack(session, stream, &faces);
            detected_num = faces.detectedNum;
        });
        while (!started.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

        REQUIRE(HFReleaseImageStream(stream) == HSUCCEED);
        REQUIRE(HFReleaseInspireFaceSession(session) == HSUCCEED);
        worker.join();

        CHECK(execute_status == HSUCCEED);
        CHECK(detected_num >= 0);
        CHECK(HFReleaseImageStream(stream) == HERR_INVALID_IMAGE_STREAM_HANDLE);
        CHECK(HFReleaseInspireFaceSession(session) == HERR_INVALID_CONTEXT_HANDLE);
    }
}

TEST_CASE("C API handle validation remains bounded under repeated calls",
          "[api][contract][handle_lifecycle][performance]") {
    auto session = CreateHandleTestSession();
    constexpr int kIterations = 100000;
    HInt32 preview_size = 0;

    const auto started = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < kIterations; ++iteration) {
        REQUIRE(HFSessionGetTrackPreviewSize(session.Get(), &preview_size) == HSUCCEED);
    }
    const auto elapsed = std::chrono::steady_clock::now() - started;
    const double average_microseconds =
      std::chrono::duration<double, std::micro>(elapsed).count() / static_cast<double>(kIterations);
    TEST_PRINT("Opaque handle validation average: {:.3f} us", average_microseconds);
    CHECK(average_microseconds < 10.0);
}
