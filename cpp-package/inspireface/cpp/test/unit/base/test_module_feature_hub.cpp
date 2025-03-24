#include <iostream>
#include "settings/test_settings.h"
#include "inspireface/c_api/inspireface.h"
#include "unit/test_helper/help.h"
#include "feature_hub/feature_hub_db.h"
#include "middleware/costman.h"

using namespace inspire;

TEST_CASE("test_FeatureHubBasic", "[feature_hub") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    // Enable feature hub
    DatabaseConfiguration config;
    config.primary_key_mode = PrimaryKeyMode::AUTO_INCREMENT;
    config.enable_persistence = false;  // memory mode
    int32_t ret;
    ret = FEATURE_HUB_DB->EnableHub(config);
    REQUIRE(ret == HSUCCEED);

    // Check if feature hub is enabled
    ret = FEATURE_HUB_DB->EnableHub(config);
    REQUIRE(ret == HSUCCEED);

    // Check number of features
    int32_t count = 1000;
    std::vector<int64_t> ids;
    std::vector<int64_t> expected_ids;
    for (int32_t i = 0; i < count; i++) {
        auto vec = GenerateRandomFeature(512, false);
        int64_t alloc_id;
        ret = FEATURE_HUB_DB->FaceFeatureInsert(vec, -1, alloc_id);
        REQUIRE(ret == HSUCCEED);
        ids.push_back(alloc_id);
        expected_ids.push_back(i + 1);
    }
    REQUIRE(FEATURE_HUB_DB->GetFaceFeatureCount() == ids.size());
    REQUIRE(ids == expected_ids);

    // Delete data
    std::vector<int64_t> delete_ids = {5, 20, 100};
    for (auto id : delete_ids) {
        FEATURE_HUB_DB->FaceFeatureRemove(id);
    }
    REQUIRE(FEATURE_HUB_DB->GetFaceFeatureCount() == ids.size() - delete_ids.size());

    // Check if the deleted data can be found
    std::vector<float> feature;
    ret = FEATURE_HUB_DB->GetFaceFeature(5, feature);
    REQUIRE(ret == HERR_FT_HUB_NOT_FOUND_FEATURE);

    // Check if the data can be found
    ret = FEATURE_HUB_DB->GetFaceFeature(1, feature);
    REQUIRE(ret == HSUCCEED);
    REQUIRE(feature.size() == 512);

    // Check if the cached data is correct
    ret = FEATURE_HUB_DB->GetFaceFeature(1);
    REQUIRE(ret == HSUCCEED);
    auto cached_feature = FEATURE_HUB_DB->GetFaceFeaturePtrCache();
    for (size_t i = 0; i < cached_feature->dataSize; i++) {
        REQUIRE(feature[i] == cached_feature->data[i]);
    }

    // Update data
    auto update_feature = GenerateRandomFeature(512, false);
    ret = FEATURE_HUB_DB->FaceFeatureUpdate(update_feature, 1);
    REQUIRE(ret == HSUCCEED);

    // Check if the updated data is correct
    ret = FEATURE_HUB_DB->GetFaceFeature(1, feature);
    REQUIRE(ret == HSUCCEED);
    for (size_t i = 0; i < feature.size(); i++) {
        REQUIRE(feature[i] == Approx(update_feature[i]).epsilon(0.0001));
    }

    // Update removed data
    ret = FEATURE_HUB_DB->FaceFeatureUpdate(update_feature, 5);
    REQUIRE(ret == HERR_FT_HUB_NOT_FOUND_FEATURE);

    // Disable feature hub
    FEATURE_HUB_DB->DisableHub();
    REQUIRE(FEATURE_HUB_DB->GetFaceFeatureCount() == 0);

    // Check if the data can be found
    ret = FEATURE_HUB_DB->GetFaceFeature(1, feature);
    REQUIRE(ret == HERR_FT_HUB_DISABLE);

    ret = FEATURE_HUB_DB->EnableHub(config);
    REQUIRE(ret == HSUCCEED);
    // Because the memory mode is turned on, once the data is turned off, it goes back to empty
    REQUIRE(FEATURE_HUB_DB->GetFaceFeatureCount() == 0);

    ret = FEATURE_HUB_DB->DisableHub();
    REQUIRE(ret == HSUCCEED);
}

TEST_CASE("test_PerformanceMemoryMode", "[feature_hub") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    DatabaseConfiguration config;
    config.primary_key_mode = PrimaryKeyMode::AUTO_INCREMENT;
    config.enable_persistence = false;  // memory mode
    int32_t ret;
    ret = FEATURE_HUB_DB->EnableHub(config);
    REQUIRE(ret == HSUCCEED);

    Timer t1;
    int num = 10000;
    for (int i = 0; i < num; i++) {
        auto vec = GenerateRandomFeature(512, false);
        int64_t alloc_id;
        ret = FEATURE_HUB_DB->FaceFeatureInsert(vec, -1, alloc_id);
        REQUIRE(ret == HSUCCEED);
    }
    TEST_PRINT("[Memory Mode]Insert 10000 features cost: {:.2f} ms", t1.GetCostTime());

    Timer t2;
    std::vector<float> feature;
    ret = FEATURE_HUB_DB->GetFaceFeature(1, feature);
    TEST_PRINT("[Memory Mode]Get feature from id cost: {:.2f} ms", t2.GetCostTime());
    REQUIRE(ret == HSUCCEED);

    Timer t3;
    ret = FEATURE_HUB_DB->GetFaceFeature(9998, feature);
    TEST_PRINT("[Memory Mode]Get feature from id cost: {:.2f} ms", t3.GetCostTime());
    REQUIRE(ret == HSUCCEED);
    auto sim_vec = SimulateSimilarVector(feature, false);
    FaceSearchResult search_result;
    Timer t4;
    FEATURE_HUB_DB->SearchFaceFeature(sim_vec, search_result, true);
    TEST_PRINT("[Memory Mode]Search feature cost: {:.2f} ms", t4.GetCostTime());
    REQUIRE(search_result.id == 9998);

    ret = FEATURE_HUB_DB->FaceFeatureRemove(9998);
    REQUIRE(ret == HSUCCEED);

    FEATURE_HUB_DB->DisableHub();
}

TEST_CASE("test_PerformancePersistentMode", "[feature_hub") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    std::string db_path = ".test_db";
    std::remove(db_path.c_str());

    DatabaseConfiguration config;
    config.primary_key_mode = PrimaryKeyMode::AUTO_INCREMENT;
    config.enable_persistence = true;  // persistent mode
    config.persistence_db_path = db_path;

    int32_t ret;
    ret = FEATURE_HUB_DB->EnableHub(config);
    REQUIRE(ret == HSUCCEED);

    Timer t1;
    int num = 10000;
    for (int i = 0; i < num; i++) {
        auto vec = GenerateRandomFeature(512, false);
        int64_t alloc_id;
        ret = FEATURE_HUB_DB->FaceFeatureInsert(vec, -1, alloc_id);
        REQUIRE(ret == HSUCCEED);
    }
    TEST_PRINT("[Persistent Mode]Insert 10000 features cost: {:.2f} ms", t1.GetCostTime());

    Timer t2;
    std::vector<float> feature;
    ret = FEATURE_HUB_DB->GetFaceFeature(1, feature);
    TEST_PRINT("[Persistent Mode]Get feature from id cost: {:.2f} ms", t2.GetCostTime());
    REQUIRE(ret == HSUCCEED);

    Timer t3;
    ret = FEATURE_HUB_DB->GetFaceFeature(9998, feature);
    TEST_PRINT("[Persistent Mode]Get feature from id cost: {:.2f} ms", t3.GetCostTime());
    REQUIRE(ret == HSUCCEED);
    auto sim_vec = SimulateSimilarVector(feature, false);
    FaceSearchResult search_result;
    Timer t4;
    FEATURE_HUB_DB->SearchFaceFeature(sim_vec, search_result, true);
    TEST_PRINT("[Persistent Mode]Search feature cost: {:.2f} ms", t4.GetCostTime());
    REQUIRE(search_result.id == 9998);

    ret = FEATURE_HUB_DB->FaceFeatureRemove(9998);
    REQUIRE(ret == HSUCCEED);

    auto remark_num = FEATURE_HUB_DB->GetFaceFeatureCount();
    REQUIRE(remark_num == num - 1);

    // Verify important of persistence test
    ret = FEATURE_HUB_DB->EnableHub(config);
    REQUIRE(ret == HSUCCEED);
    REQUIRE(FEATURE_HUB_DB->GetFaceFeatureCount() == remark_num);

    FEATURE_HUB_DB->DisableHub();
}
