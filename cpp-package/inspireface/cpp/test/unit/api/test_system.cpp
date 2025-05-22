#include <iostream>
#include "settings/test_settings.h"
#include "inspireface/c_api/inspireface.h"
#include "inspireface/include/inspireface/herror.h"
#include "unit/test_helper/test_tools.h"
#include <cstdio>

TEST_CASE("test_System", "[system]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    HResult ret;
    HInt32 status;
    ret = HFQueryInspireFaceLaunchStatus(&status);
    REQUIRE(ret == HSUCCEED);
    if (status == HF_STATUS_ENABLE) {
        // The global TEST environment has been started, so this side needs to be temporarily closed
        // before testing
        HFTerminateInspireFace();
    }

    SECTION("Create a session test when it is not loaded") {
        HFSessionCustomParameter parameter = {0};
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, -1, -1, &session);
        REQUIRE(ret == HERR_ARCHIVE_NOT_LOAD);
        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HERR_INVALID_CONTEXT_HANDLE);
    }

    // Restart and start InspireFace
    ret = HFLaunchInspireFace(GET_RUNTIME_FULLPATH_NAME.c_str());
    REQUIRE(ret == HSUCCEED);

    SECTION("Create a session test when it is reloaded") {
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, -1, -1, &session);
        REQUIRE(ret == HSUCCEED);
        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);
    }
}

TEST_CASE("test_SystemSessionReleaseCase", "[system]") {
    /**
     * @brief Test the release of sessions
     * @details Test the release of sessions and check the unreleased sessions count
     */
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);
    HResult ret;

    SECTION("CreateSessions") {
        /**
         * @brief Create sessions
         * @details Create 10 sessions and check the unreleased sessions count
         */
        HInt32 count;
        ret = HFDeBugGetUnreleasedSessionsCount(&count);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(count == 0);

        HInt32 createCount = 10;
        HFSession sessions[createCount];

        for (int i = 0; i < createCount; ++i) {
            ret = HFCreateInspireFaceSessionOptional(HF_ENABLE_NONE, HF_DETECT_MODE_ALWAYS_DETECT, 3, -1, -1, &sessions[i]);
            REQUIRE(ret == HSUCCEED);
        }

        ret = HFDeBugGetUnreleasedSessionsCount(&count);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(count == createCount);

        HFSession sessionsGet[createCount];
        ret = HFDeBugGetUnreleasedSessions(sessionsGet, createCount);
        REQUIRE(ret == HSUCCEED);
        // The session list obtained from the api is also unordered because it is sorted internally using an unordered dictionary
        for (int i = 0; i < createCount; ++i) {
            bool found = false;
            for (int j = 0; j < createCount; ++j) {
                if (sessions[i] == sessionsGet[j]) {
                    found = true;
                    break;
                }
            }
            REQUIRE(found);
        }
    }

    SECTION("ReleaseSomeSessions") {
        /**
         * @brief Release some sessions
         * @details Release some sessions and check the unreleased sessions count
         */
        HInt32 count;
        ret = HFDeBugGetUnreleasedSessionsCount(&count);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(count == 10);
        auto createCount = count;

        HFSession sessionsGet[createCount];
        ret = HFDeBugGetUnreleasedSessions(sessionsGet, createCount);
        REQUIRE(ret == HSUCCEED);

        std::vector<int32_t> releaseIndex = {0, 2, 4, 6, 8};
        for (int i = 0; i < releaseIndex.size(); ++i) {
            ret = HFReleaseInspireFaceSession(sessionsGet[releaseIndex[i]]);
            REQUIRE(ret == HSUCCEED);
        }

        ret = HFDeBugGetUnreleasedSessionsCount(&count);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(count == createCount - releaseIndex.size());

        HFSession sessionsGet2[count];
        ret = HFDeBugGetUnreleasedSessions(sessionsGet2, count);
        REQUIRE(ret == HSUCCEED);
        for (int i = 0; i < count; ++i) {
            bool found = false;
            for (int j = 0; j < releaseIndex.size(); ++j) {
                if (sessionsGet2[i] == sessionsGet[releaseIndex[j]]) {
                    found = true;
                    break;
                }
            }
            REQUIRE(!found);
        }
    }

    SECTION("ReleaseAllSessions") {
        /**
         * @brief Release all sessions
         * @details Release all sessions and check the unreleased sessions count
         */
        HInt32 count;
        ret = HFDeBugGetUnreleasedSessionsCount(&count);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(count == 5);

        HFSession sessionsGet[count];
        ret = HFDeBugGetUnreleasedSessions(sessionsGet, count);
        REQUIRE(ret == HSUCCEED);

        for (int i = 0; i < count; ++i) {
            ret = HFReleaseInspireFaceSession(sessionsGet[i]);
            REQUIRE(ret == HSUCCEED);
        }

        ret = HFDeBugGetUnreleasedSessionsCount(&count);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(count == 0);
    }
}

TEST_CASE("test_SystemStreamReleaseCase", "[system]") {
    /**
     * @brief Test the release of streams
     * @details Test the release of streams and check the unreleased streams count
     */
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);
    HResult ret;

    SECTION("CreateStreams") {
        /**
         * @brief Create streams
         * @details Create 10 streams and check the unreleased streams count
         */
        HInt32 count;
        ret = HFDeBugGetUnreleasedStreamsCount(&count);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(count == 0);

        HInt32 createCount = 10;
        HFImageStream streams[createCount];

        for (int i = 0; i < createCount; ++i) {
            HFImageStream imgHandle;
            auto image = inspirecv::Image::Create(GET_DATA("data/bulk/pedestrian.png"));
            ret = CVImageToImageStream(image, imgHandle);
            REQUIRE(ret == HSUCCEED);
            streams[i] = imgHandle;
        }

        ret = HFDeBugGetUnreleasedStreamsCount(&count);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(count == createCount);

        HFImageStream streamsGet[createCount];
        ret = HFDeBugGetUnreleasedStreams(streamsGet, createCount);
        REQUIRE(ret == HSUCCEED);
        for (int i = 0; i < createCount; ++i) {
            bool found = false;
            for (int j = 0; j < createCount; ++j) {
                if (streams[i] == streamsGet[j]) {
                    found = true;
                    break;
                }
            }
            REQUIRE(found);
        }
    }

    SECTION("ReleaseSomeStreams") {
        /**
         * @brief Release some streams
         * @details Release some streams and check the unreleased streams count
         */
        HInt32 count;
        ret = HFDeBugGetUnreleasedStreamsCount(&count);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(count == 10);
        auto createCount = count;

        HFImageStream streamsGet[count];
        ret = HFDeBugGetUnreleasedStreams(streamsGet, count);
        REQUIRE(ret == HSUCCEED);

        std::vector<int32_t> releaseIndex = {0, 2, 4, 6, 8};
        for (int i = 0; i < releaseIndex.size(); ++i) {
            ret = HFReleaseImageStream(streamsGet[releaseIndex[i]]);
            REQUIRE(ret == HSUCCEED);
        }

        ret = HFDeBugGetUnreleasedStreamsCount(&count);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(count == createCount - releaseIndex.size());

        HFImageStream streamsGet2[count];
        ret = HFDeBugGetUnreleasedStreams(streamsGet2, count);
        REQUIRE(ret == HSUCCEED);
        for (int i = 0; i < count; ++i) {
            bool found = false;
            for (int j = 0; j < releaseIndex.size(); ++j) {
                if (streamsGet2[i] == streamsGet[releaseIndex[j]]) {
                    found = true;
                    break;
                }
            }
            REQUIRE(!found);
        }
    }

    SECTION("ReleaseAllStreams") {
        /**
         * @brief Release all streams
         * @details Release all streams and check the unreleased streams count
         */
        HInt32 count;
        ret = HFDeBugGetUnreleasedStreamsCount(&count);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(count == 5);

        HFImageStream streamsGet[count];
        ret = HFDeBugGetUnreleasedStreams(streamsGet, count);
        REQUIRE(ret == HSUCCEED);

        for (int i = 0; i < count; ++i) {
            ret = HFReleaseImageStream(streamsGet[i]);
            REQUIRE(ret == HSUCCEED);
        }

        ret = HFDeBugGetUnreleasedStreamsCount(&count);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(count == 0);
    }
}
