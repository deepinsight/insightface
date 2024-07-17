#include <iostream>
#include "settings/test_settings.h"
#include "inspireface/c_api/inspireface.h"
#include "inspireface/herror.h"
#include <cstdio>

TEST_CASE("test_System", "[system]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    // The global TEST environment has been started, so this side needs to be temporarily closed
    // before testing
    HFTerminateInspireFace();

    SECTION("Create a session test when it is not loaded") {
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, -1, -1, &session);
        REQUIRE(ret == HERR_ARCHIVE_NOT_LOAD);
        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HERR_INVALID_CONTEXT_HANDLE);
    }

    // Restart and start InspireFace
    auto ret = HFLaunchInspireFace(GET_RUNTIME_FULLPATH_NAME.c_str());
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