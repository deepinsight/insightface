#include <iostream>
#include "settings/test_settings.h"
#include "inspireface/c_api/inspireface.h"
#include "unit/test_helper/help.h"
#include <inspireface/include/inspireface/feature_hub_db.h>
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
    ret = INSPIREFACE_FEATURE_HUB->EnableHub(config);
    REQUIRE(ret == HSUCCEED);

    // Check if feature hub is enabled
    ret = INSPIREFACE_FEATURE_HUB->EnableHub(config);
    REQUIRE(ret == HSUCCEED);

    // Check number of features
    int32_t count = 1000;
    std::vector<int64_t> ids;
    std::vector<int64_t> expected_ids;
    for (int32_t i = 0; i < count; i++) {
        auto vec = GenerateRandomFeature(512, false);
        int64_t alloc_id;
        ret = INSPIREFACE_FEATURE_HUB->FaceFeatureInsert(vec, -1, alloc_id);
        REQUIRE(ret == HSUCCEED);
        ids.push_back(alloc_id);
        expected_ids.push_back(i + 1);
    }
    REQUIRE(INSPIREFACE_FEATURE_HUB->GetFaceFeatureCount() == ids.size());
    REQUIRE(ids == expected_ids);

    // Delete data
    std::vector<int64_t> delete_ids = {5, 20, 100};
    for (auto id : delete_ids) {
        INSPIREFACE_FEATURE_HUB->FaceFeatureRemove(id);
    }
    REQUIRE(INSPIREFACE_FEATURE_HUB->GetFaceFeatureCount() == ids.size() - delete_ids.size());

    // Check if the deleted data can be found
    std::vector<float> feature;
    ret = INSPIREFACE_FEATURE_HUB->GetFaceFeature(5, feature);
    REQUIRE(ret == HERR_FT_HUB_NOT_FOUND_FEATURE);

    // Check if the data can be found
    ret = INSPIREFACE_FEATURE_HUB->GetFaceFeature(1, feature);
    REQUIRE(ret == HSUCCEED);
    REQUIRE(feature.size() == 512);

    // Check if the cached data is correct
    ret = INSPIREFACE_FEATURE_HUB->GetFaceFeature(1);
    REQUIRE(ret == HSUCCEED);
    auto cached_feature = INSPIREFACE_FEATURE_HUB->GetFaceFeaturePtrCache();
    for (size_t i = 0; i < cached_feature->dataSize; i++) {
        REQUIRE(feature[i] == cached_feature->data[i]);
    }

    // Update data
    auto update_feature = GenerateRandomFeature(512, false);
    ret = INSPIREFACE_FEATURE_HUB->FaceFeatureUpdate(update_feature, 1);
    REQUIRE(ret == HSUCCEED);

    // Check if the updated data is correct
    ret = INSPIREFACE_FEATURE_HUB->GetFaceFeature(1, feature);
    REQUIRE(ret == HSUCCEED);
    for (size_t i = 0; i < feature.size(); i++) {
        REQUIRE(feature[i] == Approx(update_feature[i]).epsilon(0.0001));
    }

    // Update removed data
    ret = INSPIREFACE_FEATURE_HUB->FaceFeatureUpdate(update_feature, 5);
    REQUIRE(ret == HERR_FT_HUB_NOT_FOUND_FEATURE);

    // Disable feature hub
    INSPIREFACE_FEATURE_HUB->DisableHub();
    REQUIRE(INSPIREFACE_FEATURE_HUB->GetFaceFeatureCount() == 0);

    // Check if the data can be found
    ret = INSPIREFACE_FEATURE_HUB->GetFaceFeature(1, feature);
    REQUIRE(ret == HERR_FT_HUB_DISABLE);

    ret = INSPIREFACE_FEATURE_HUB->EnableHub(config);
    REQUIRE(ret == HSUCCEED);
    // Because the memory mode is turned on, once the data is turned off, it goes back to empty
    REQUIRE(INSPIREFACE_FEATURE_HUB->GetFaceFeatureCount() == 0);

    ret = INSPIREFACE_FEATURE_HUB->DisableHub();
    REQUIRE(ret == HSUCCEED);
}

TEST_CASE("test_PerformanceMemoryMode", "[feature_hub") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    DatabaseConfiguration config;
    config.primary_key_mode = PrimaryKeyMode::AUTO_INCREMENT;
    config.enable_persistence = false;  // memory mode
    int32_t ret;
    ret = INSPIREFACE_FEATURE_HUB->EnableHub(config);
    REQUIRE(ret == HSUCCEED);

    Timer t1;
    int num = 10000;
    for (int i = 0; i < num; i++) {
        auto vec = GenerateRandomFeature(512, false);
        int64_t alloc_id;
        ret = INSPIREFACE_FEATURE_HUB->FaceFeatureInsert(vec, -1, alloc_id);
        REQUIRE(ret == HSUCCEED);
    }
    TEST_PRINT("[Memory Mode]Insert 10000 features cost: {:.2f} ms", t1.GetCostTime());

    Timer t2;
    std::vector<float> feature;
    ret = INSPIREFACE_FEATURE_HUB->GetFaceFeature(1, feature);
    TEST_PRINT("[Memory Mode]Get feature from id cost: {:.2f} ms", t2.GetCostTime());
    REQUIRE(ret == HSUCCEED);

    Timer t3;
    ret = INSPIREFACE_FEATURE_HUB->GetFaceFeature(9998, feature);
    TEST_PRINT("[Memory Mode]Get feature from id cost: {:.2f} ms", t3.GetCostTime());
    REQUIRE(ret == HSUCCEED);
    auto sim_vec = SimulateSimilarVector(feature, false);
    FaceSearchResult search_result;
    Timer t4;
    INSPIREFACE_FEATURE_HUB->SearchFaceFeature(sim_vec, search_result, true);
    TEST_PRINT("[Memory Mode]Search feature cost: {:.2f} ms", t4.GetCostTime());
    REQUIRE(search_result.id == 9998);

    ret = INSPIREFACE_FEATURE_HUB->FaceFeatureRemove(9998);
    REQUIRE(ret == HSUCCEED);

    INSPIREFACE_FEATURE_HUB->DisableHub();
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
    ret = INSPIREFACE_FEATURE_HUB->EnableHub(config);
    REQUIRE(ret == HSUCCEED);

    Timer t1;
    int num = 10000;
    for (int i = 0; i < num; i++) {
        auto vec = GenerateRandomFeature(512, false);
        int64_t alloc_id;
        ret = INSPIREFACE_FEATURE_HUB->FaceFeatureInsert(vec, -1, alloc_id);
        REQUIRE(ret == HSUCCEED);
    }
    TEST_PRINT("[Persistent Mode]Insert 10000 features cost: {:.2f} ms", t1.GetCostTime());

    Timer t2;
    std::vector<float> feature;
    ret = INSPIREFACE_FEATURE_HUB->GetFaceFeature(1, feature);
    TEST_PRINT("[Persistent Mode]Get feature from id cost: {:.2f} ms", t2.GetCostTime());
    REQUIRE(ret == HSUCCEED);

    Timer t3;
    ret = INSPIREFACE_FEATURE_HUB->GetFaceFeature(9998, feature);
    TEST_PRINT("[Persistent Mode]Get feature from id cost: {:.2f} ms", t3.GetCostTime());
    REQUIRE(ret == HSUCCEED);
    auto sim_vec = SimulateSimilarVector(feature, false);
    FaceSearchResult search_result;
    Timer t4;
    INSPIREFACE_FEATURE_HUB->SearchFaceFeature(sim_vec, search_result, true);
    TEST_PRINT("[Persistent Mode]Search feature cost: {:.2f} ms", t4.GetCostTime());
    REQUIRE(search_result.id == 9998);

    ret = INSPIREFACE_FEATURE_HUB->FaceFeatureRemove(9998);
    REQUIRE(ret == HSUCCEED);

    auto remark_num = INSPIREFACE_FEATURE_HUB->GetFaceFeatureCount();
    REQUIRE(remark_num == num - 1);

    // Verify important of persistence test
    ret = INSPIREFACE_FEATURE_HUB->EnableHub(config);
    REQUIRE(ret == HSUCCEED);
    REQUIRE(INSPIREFACE_FEATURE_HUB->GetFaceFeatureCount() == remark_num);

    INSPIREFACE_FEATURE_HUB->DisableHub();
}
