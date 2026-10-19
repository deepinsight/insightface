#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "inspireface/c_api/inspireface.h"
#include "settings/test_settings.h"
#include "unit/test_helper/c_api_guard.h"
#include "unit/test_helper/test_tools.h"

namespace {

using Clock = std::chrono::steady_clock;
using LegacyFaceTrackFunction = HResult (*)(HFSession, HFImageStream, PHFMultipleFaceData);

static_assert(std::is_same<decltype(&HFExecuteFaceTrack), LegacyFaceTrackFunction>::value,
              "The legacy face tracking signature changed");

class UniqueSnapshot {
public:
    UniqueSnapshot() = default;
    explicit UniqueSnapshot(HFFaceResultSnapshot handle) : handle_(handle) {}
    ~UniqueSnapshot() {
        if (handle_ != nullptr) {
            HFReleaseFaceResultSnapshot(handle_);
        }
    }

    UniqueSnapshot(const UniqueSnapshot&) = delete;
    UniqueSnapshot& operator=(const UniqueSnapshot&) = delete;

    UniqueSnapshot(UniqueSnapshot&& other) noexcept : handle_(other.ReleaseOwnership()) {}

    UniqueSnapshot& operator=(UniqueSnapshot&& other) noexcept {
        if (this != &other) {
            if (handle_ != nullptr) {
                HFReleaseFaceResultSnapshot(handle_);
            }
            handle_ = other.ReleaseOwnership();
        }
        return *this;
    }

    HFFaceResultSnapshot Get() const {
        return handle_;
    }

    HFFaceResultSnapshot ReleaseOwnership() {
        HFFaceResultSnapshot value = handle_;
        handle_ = nullptr;
        return value;
    }

private:
    HFFaceResultSnapshot handle_ = nullptr;
};

inspireface_test::UniqueSession CreateSession(HInt32 max_faces = 5) {
    HFSession handle = nullptr;
    REQUIRE(HFCreateInspireFaceSessionOptional(HF_ENABLE_NONE, HF_DETECT_MODE_ALWAYS_DETECT, max_faces, -1, -1, &handle) == HSUCCEED);
    REQUIRE(handle != nullptr);
    return inspireface_test::UniqueSession(handle);
}

inspireface_test::UniqueImageStream StreamFromImage(const inspirecv::Image& image) {
    HFImageStream handle = nullptr;
    REQUIRE(CVImageToImageStream(image, handle) == HSUCCEED);
    REQUIRE(handle != nullptr);
    return inspireface_test::UniqueImageStream(handle);
}

UniqueSnapshot ExecuteSnapshot(HFSession session, HFImageStream stream) {
    HFFaceResultSnapshot handle = nullptr;
    REQUIRE(HFExecuteFaceTrackSnapshot(session, stream, &handle) == HSUCCEED);
    REQUIRE(handle != nullptr);
    return UniqueSnapshot(handle);
}

void CheckFaceDataEqual(const HFMultipleFaceData& expected, const HFMultipleFaceData& actual) {
    REQUIRE(expected.detectedNum == actual.detectedNum);
    for (HInt32 index = 0; index < expected.detectedNum; ++index) {
        CHECK(expected.rects[index].x == actual.rects[index].x);
        CHECK(expected.rects[index].y == actual.rects[index].y);
        CHECK(expected.rects[index].width == actual.rects[index].width);
        CHECK(expected.rects[index].height == actual.rects[index].height);
        CHECK(expected.trackIds[index] == actual.trackIds[index]);
        CHECK(expected.trackCounts[index] == actual.trackCounts[index]);
        CHECK(expected.detConfidence[index] == Approx(actual.detConfidence[index]).margin(1e-6f));
        CHECK(expected.angles.roll[index] == Approx(actual.angles.roll[index]).margin(1e-6f));
        CHECK(expected.angles.yaw[index] == Approx(actual.angles.yaw[index]).margin(1e-6f));
        CHECK(expected.angles.pitch[index] == Approx(actual.angles.pitch[index]).margin(1e-6f));
        REQUIRE(expected.tokens[index].size == actual.tokens[index].size);
        REQUIRE(expected.tokens[index].size >= 0);
        if (expected.tokens[index].size > 0) {
            REQUIRE(expected.tokens[index].data != nullptr);
            REQUIRE(actual.tokens[index].data != nullptr);
            CHECK(std::memcmp(expected.tokens[index].data, actual.tokens[index].data,
                              static_cast<size_t>(expected.tokens[index].size)) == 0);
        }
    }
}

int64_t Median(std::vector<int64_t> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

template <typename Callable>
int64_t MeasureMicroseconds(Callable&& callable, HResult& status) {
    const auto begin = Clock::now();
    status = callable();
    const auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
}

}  // namespace

TEST_CASE("Face result snapshot validates handles outputs and empty results", "[api][contract][snapshot][boundary]") {
    HFFaceResultSnapshot snapshot = reinterpret_cast<HFFaceResultSnapshot>(static_cast<uintptr_t>(1));
    CHECK(HFExecuteFaceTrackSnapshot(nullptr, nullptr, &snapshot) == HERR_INVALID_CONTEXT_HANDLE);
    CHECK(snapshot == nullptr);
    CHECK(HFExecuteFaceTrackSnapshot(nullptr, nullptr, nullptr) == HERR_INVALID_PARAM);

    HFMultipleFaceData output = {};
    CHECK(HFGetFaceResultSnapshotData(nullptr, &output) == HERR_INVALID_PARAM);
    CHECK(output.detectedNum == 0);
    CHECK(HFGetFaceResultSnapshotData(nullptr, nullptr) == HERR_INVALID_PARAM);
    CHECK(HFReleaseFaceResultSnapshot(nullptr) == HERR_INVALID_PARAM);

    auto session = CreateSession();
    CHECK(HFGetFaceResultSnapshotData(reinterpret_cast<HFFaceResultSnapshot>(session.Get()), &output) == HERR_INVALID_PARAM);
    const auto no_face = inspirecv::Image::Create(GET_DATA("data/crop/no_face.png"));
    REQUIRE(!no_face.Empty());
    auto stream = StreamFromImage(no_face);
    auto empty_snapshot = ExecuteSnapshot(session.Get(), stream.Get());
    REQUIRE(HFGetFaceResultSnapshotData(empty_snapshot.Get(), &output) == HSUCCEED);
    CHECK(output.detectedNum == 0);
    CHECK(output.rects == nullptr);
    CHECK(output.trackIds == nullptr);
    CHECK(output.trackCounts == nullptr);
    CHECK(output.detConfidence == nullptr);
    CHECK(output.angles.roll == nullptr);
    CHECK(output.angles.yaw == nullptr);
    CHECK(output.angles.pitch == nullptr);
    CHECK(output.tokens == nullptr);

    HFFaceResultSnapshot stale = empty_snapshot.ReleaseOwnership();
    REQUIRE(HFReleaseFaceResultSnapshot(stale) == HSUCCEED);
    CHECK(HFGetFaceResultSnapshotData(stale, &output) == HERR_INVALID_PARAM);
    CHECK(output.detectedNum == 0);
    CHECK(HFReleaseFaceResultSnapshot(stale) == HERR_INVALID_PARAM);
}

TEST_CASE("Face result snapshot outlives later frames session and stream", "[api][contract][snapshot][lifecycle][token]") {
    UniqueSnapshot snapshot;
    HFaceRect original_rect = {};
    std::vector<uint8_t> original_token;
    {
        auto session = CreateSession();
        const auto face_image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
        const auto no_face_image = inspirecv::Image::Create(GET_DATA("data/crop/no_face.png"));
        REQUIRE(!face_image.Empty());
        REQUIRE(!no_face_image.Empty());
        auto face_stream = StreamFromImage(face_image);
        auto no_face_stream = StreamFromImage(no_face_image);

        snapshot = ExecuteSnapshot(session.Get(), face_stream.Get());
        HFMultipleFaceData snapshot_data = {};
        REQUIRE(HFGetFaceResultSnapshotData(snapshot.Get(), &snapshot_data) == HSUCCEED);
        REQUIRE(snapshot_data.detectedNum > 0);
        original_rect = snapshot_data.rects[0];
        REQUIRE(snapshot_data.tokens[0].size > 0);
        const auto* token_begin = static_cast<const uint8_t*>(snapshot_data.tokens[0].data);
        original_token.assign(token_begin, token_begin + snapshot_data.tokens[0].size);

        HFMultipleFaceData later_result = {};
        REQUIRE(HFExecuteFaceTrack(session.Get(), no_face_stream.Get(), &later_result) == HSUCCEED);
        REQUIRE(later_result.detectedNum == 0);
        REQUIRE(HFGetFaceResultSnapshotData(snapshot.Get(), &snapshot_data) == HSUCCEED);
        CHECK(snapshot_data.detectedNum > 0);
        CHECK(snapshot_data.rects[0].x == original_rect.x);
        CHECK(snapshot_data.rects[0].y == original_rect.y);
        CHECK(std::memcmp(snapshot_data.tokens[0].data, original_token.data(), original_token.size()) == 0);
    }

    HFMultipleFaceData persistent = {};
    REQUIRE(HFGetFaceResultSnapshotData(snapshot.Get(), &persistent) == HSUCCEED);
    REQUIRE(persistent.detectedNum > 0);
    CHECK(persistent.rects[0].x == original_rect.x);
    CHECK(persistent.rects[0].y == original_rect.y);
    REQUIRE(persistent.tokens[0].size == static_cast<HInt32>(original_token.size()));
    CHECK(std::memcmp(persistent.tokens[0].data, original_token.data(), original_token.size()) == 0);
    std::array<HPoint2f, 5> points = {};
    REQUIRE(HFGetFaceFiveKeyPointsFromFaceToken(persistent.tokens[0], points.data(), points.size()) == HSUCCEED);
    CHECK(std::isfinite(points[0].x));
    CHECK(std::isfinite(points[0].y));
}

TEST_CASE("Borrowed and snapshot face results match across representative images", "[api][snapshot][accuracy][consistency]") {
    auto borrowed_session = CreateSession();
    auto snapshot_session = CreateSession();
    const std::vector<std::string> image_paths = {
      "data/bulk/kun.jpg", "data/bulk/r0.jpg", "data/bulk/woman.png", "data/bulk/pedestrian.png", "data/crop/no_face.png"};

    for (const auto& relative_path : image_paths) {
        DYNAMIC_SECTION(relative_path) {
            const auto image = inspirecv::Image::Create(GET_DATA(relative_path));
            REQUIRE(!image.Empty());
            auto borrowed_stream = StreamFromImage(image);
            auto snapshot_stream = StreamFromImage(image);
            HFMultipleFaceData borrowed = {};
            REQUIRE(HFExecuteFaceTrack(borrowed_session.Get(), borrowed_stream.Get(), &borrowed) == HSUCCEED);
            auto snapshot = ExecuteSnapshot(snapshot_session.Get(), snapshot_stream.Get());
            HFMultipleFaceData owned = {};
            REQUIRE(HFGetFaceResultSnapshotData(snapshot.Get(), &owned) == HSUCCEED);
            CheckFaceDataEqual(borrowed, owned);
            for (HInt32 index = 0; index < owned.detectedNum; ++index) {
                CHECK(borrowed.tokens[index].data != owned.tokens[index].data);
            }
        }
    }
}

TEST_CASE("Owned snapshot latency stays within the borrowed result envelope", "[api][snapshot][performance][latency]") {
    auto borrowed_session = CreateSession(1);
    auto snapshot_session = CreateSession(1);
    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
    REQUIRE(!image.Empty());
    auto borrowed_stream = StreamFromImage(image);
    auto snapshot_stream = StreamFromImage(image);
    HFMultipleFaceData borrowed = {};

    for (int warmup = 0; warmup < 2; ++warmup) {
        REQUIRE(HFExecuteFaceTrack(borrowed_session.Get(), borrowed_stream.Get(), &borrowed) == HSUCCEED);
        auto snapshot = ExecuteSnapshot(snapshot_session.Get(), snapshot_stream.Get());
        HFMultipleFaceData owned = {};
        REQUIRE(HFGetFaceResultSnapshotData(snapshot.Get(), &owned) == HSUCCEED);
        CheckFaceDataEqual(borrowed, owned);
    }

    std::vector<int64_t> borrowed_samples;
    std::vector<int64_t> snapshot_samples;
    for (int iteration = 0; iteration < 9; ++iteration) {
        HResult borrowed_status = HSUCCEED;
        HResult snapshot_status = HSUCCEED;
        auto measure_borrowed = [&] {
            return MeasureMicroseconds(
              [&] { return HFExecuteFaceTrack(borrowed_session.Get(), borrowed_stream.Get(), &borrowed); }, borrowed_status);
        };
        auto measure_snapshot = [&] {
            return MeasureMicroseconds(
              [&] {
                  HFFaceResultSnapshot handle = nullptr;
                  HResult status = HFExecuteFaceTrackSnapshot(snapshot_session.Get(), snapshot_stream.Get(), &handle);
                  if (status == HSUCCEED) {
                      HFMultipleFaceData owned = {};
                      status = HFGetFaceResultSnapshotData(handle, &owned);
                  }
                  if (handle != nullptr) {
                      const HResult release_status = HFReleaseFaceResultSnapshot(handle);
                      if (status == HSUCCEED) {
                          status = release_status;
                      }
                  }
                  return status;
              },
              snapshot_status);
        };
        if (iteration % 2 == 0) {
            borrowed_samples.push_back(measure_borrowed());
            snapshot_samples.push_back(measure_snapshot());
        } else {
            snapshot_samples.push_back(measure_snapshot());
            borrowed_samples.push_back(measure_borrowed());
        }
        REQUIRE(borrowed_status == HSUCCEED);
        REQUIRE(snapshot_status == HSUCCEED);
    }

    const int64_t borrowed_median = Median(borrowed_samples);
    const int64_t snapshot_median = Median(snapshot_samples);
    TEST_PRINT("Borrowed median detection: {} us; owned snapshot: {} us", borrowed_median, snapshot_median);
    CHECK(snapshot_median <= static_cast<int64_t>(borrowed_median * 1.35) + 2000);
}
