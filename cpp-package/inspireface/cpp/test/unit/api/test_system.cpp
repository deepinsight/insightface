#include <algorithm>
#include <vector>

#include "inspireface/c_api/inspireface.h"
#include "settings/test_settings.h"
#include "unit/test_helper/c_api_guard.h"
#include "unit/test_helper/test_tools.h"

using inspireface_test::UniqueImageStream;
using inspireface_test::UniqueSession;

namespace {

std::vector<HFSession> GetTrackedSessions() {
    HInt32 count = -1;
    REQUIRE(HFDeBugGetUnreleasedSessionsCount(&count) == HSUCCEED);
    REQUIRE(count >= 0);
    std::vector<HFSession> handles(static_cast<size_t>(count));
    REQUIRE(HFDeBugGetUnreleasedSessions(handles.data(), count) == HSUCCEED);
    return handles;
}

std::vector<HFImageStream> GetTrackedStreams() {
    HInt32 count = -1;
    REQUIRE(HFDeBugGetUnreleasedStreamsCount(&count) == HSUCCEED);
    REQUIRE(count >= 0);
    std::vector<HFImageStream> handles(static_cast<size_t>(count));
    REQUIRE(HFDeBugGetUnreleasedStreams(handles.data(), count) == HSUCCEED);
    return handles;
}

}  // namespace

TEST_CASE("C API resource registry tracks sessions without order dependence", "[api][contract][resource]") {
    REQUIRE(GetTrackedSessions().empty());

    std::vector<UniqueSession> sessions;
    for (int i = 0; i < 10; ++i) {
        HFSession handle = nullptr;
        REQUIRE(HFCreateInspireFaceSessionOptional(HF_ENABLE_NONE, HF_DETECT_MODE_ALWAYS_DETECT, 3, -1, -1, &handle) == HSUCCEED);
        REQUIRE(handle != nullptr);
        sessions.emplace_back(handle);
    }

    const auto tracked = GetTrackedSessions();
    REQUIRE(tracked.size() == sessions.size());
    for (const auto& session : sessions) {
        CHECK(std::find(tracked.begin(), tracked.end(), session.Get()) != tracked.end());
    }

    for (size_t i = 0; i < sessions.size(); i += 2) {
        REQUIRE(sessions[i].Reset() == HSUCCEED);
    }

    const auto remaining = GetTrackedSessions();
    REQUIRE(remaining.size() == sessions.size() / 2);
    for (size_t i = 0; i < sessions.size(); ++i) {
        const bool found = std::find(remaining.begin(), remaining.end(), sessions[i].Get()) != remaining.end();
        CHECK(found == (i % 2 == 1));
    }

    for (auto& session : sessions) {
        REQUIRE(session.Reset() == HSUCCEED);
    }
    CHECK(GetTrackedSessions().empty());
}

TEST_CASE("C API resource registry tracks streams without order dependence", "[api][contract][resource]") {
    REQUIRE(GetTrackedStreams().empty());

    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/pedestrian.png"));
    REQUIRE(!image.Empty());

    std::vector<UniqueImageStream> streams;
    for (int i = 0; i < 10; ++i) {
        HFImageStream handle = nullptr;
        REQUIRE(CVImageToImageStream(image, handle) == HSUCCEED);
        REQUIRE(handle != nullptr);
        streams.emplace_back(handle);
    }

    const auto tracked = GetTrackedStreams();
    REQUIRE(tracked.size() == streams.size());
    for (const auto& stream : streams) {
        CHECK(std::find(tracked.begin(), tracked.end(), stream.Get()) != tracked.end());
    }

    for (size_t i = 0; i < streams.size(); i += 2) {
        REQUIRE(streams[i].Reset() == HSUCCEED);
    }

    const auto remaining = GetTrackedStreams();
    REQUIRE(remaining.size() == streams.size() / 2);
    for (size_t i = 0; i < streams.size(); ++i) {
        const bool found = std::find(remaining.begin(), remaining.end(), streams[i].Get()) != remaining.end();
        CHECK(found == (i % 2 == 1));
    }

    for (auto& stream : streams) {
        REQUIRE(stream.Reset() == HSUCCEED);
    }
    CHECK(GetTrackedStreams().empty());
}

TEST_CASE("C API resource registry rejects invalid output arguments", "[api][contract][boundary][resource]") {
    CHECK(HFDeBugGetUnreleasedSessionsCount(nullptr) == HERR_INVALID_PARAM);
    CHECK(HFDeBugGetUnreleasedStreamsCount(nullptr) == HERR_INVALID_PARAM);
    CHECK(HFDeBugGetUnreleasedSessions(nullptr, -1) == HERR_INVALID_PARAM);
    CHECK(HFDeBugGetUnreleasedStreams(nullptr, -1) == HERR_INVALID_PARAM);
    CHECK(HFDeBugGetUnreleasedSessions(nullptr, 1) == HERR_INVALID_PARAM);
    CHECK(HFDeBugGetUnreleasedStreams(nullptr, 1) == HERR_INVALID_PARAM);
    CHECK(HFDeBugGetUnreleasedSessions(nullptr, 0) == HSUCCEED);
    CHECK(HFDeBugGetUnreleasedStreams(nullptr, 0) == HSUCCEED);
    CHECK(HFDeBugShowResourceStatistics() == HSUCCEED);
}
