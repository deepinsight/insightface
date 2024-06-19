//
// Created by Tunm-Air13 on 2024/3/20.
//

#include <iostream>
#include "settings/test_settings.h"
#include "../test_helper/test_help.h"

TEST_CASE("test_HelpTools", "[help_tools]") {
        DRAW_SPLIT_LINE
        TEST_PRINT_OUTPUT(true);

    SECTION("Load lfw funneled data") {
#ifdef ISF_ENABLE_USE_LFW_DATA
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        parameter.enable_recognition = 1;
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, -1, -1, &session);
        REQUIRE(ret == HSUCCEED);
        HFFeatureHubConfiguration configuration = {0};
        auto dbPath = GET_SAVE_DATA(".test");
        HString dbPathStr = new char[dbPath.size() + 1];
        std::strcpy(dbPathStr, dbPath.c_str());
        configuration.enablePersistence = 1;
        configuration.dbPath = dbPathStr;
        configuration.featureBlockNum = 20;
        configuration.searchMode = HF_SEARCH_MODE_EXHAUSTIVE;
        configuration.searchThreshold = 0.48f;
        // Delete the previous data before testing
        if (std::remove(configuration.dbPath) != 0) {
            spdlog::trace("Error deleting file");
        }
        ret = HFFeatureHubDataEnable(configuration);
        REQUIRE(ret == HSUCCEED);

        auto lfwDir = getLFWFunneledDir();
        auto dataList = LoadLFWFunneledValidData(lfwDir, getTestLFWFunneledTxt());
        size_t numOfNeedImport = 100;
        auto importStatus = ImportLFWFunneledValidData(session, dataList, numOfNeedImport);
        HFFeatureHubViewDBTable();
        REQUIRE(importStatus);
        HInt32 count;
        ret = HFFeatureHubGetFaceCount(&count);
        REQUIRE(ret == HSUCCEED);
        CHECK(count == numOfNeedImport);

//        ret = HF_ViewFaceDBTable(session);
//        REQUIRE(ret == HSUCCEED);

        // Finish
        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);

        ret = HFFeatureHubDataDisable();
        REQUIRE(ret == HSUCCEED);

        delete []dbPathStr;

#else
        TEST_PRINT("The test case that uses LFW is not enabled, so it will be skipped.");
#endif
    }
}