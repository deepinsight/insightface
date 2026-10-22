#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "inspireface/c_api/inspireface.h"
#include "settings/test_settings.h"

namespace {

class FeatureHubReset {
public:
    FeatureHubReset() {
        HFFeatureHubDataDisable();
    }
    ~FeatureHubReset() {
        HFFeatureHubDataDisable();
    }
};

HFFeatureHubConfiguration MemoryConfiguration(HFPKMode primary_key_mode = HF_PK_MANUAL_INPUT) {
    HFFeatureHubConfiguration configuration = {};
    configuration.primaryKeyMode = primary_key_mode;
    configuration.enablePersistence = 0;
    configuration.persistenceDbPath = nullptr;
    configuration.searchThreshold = -1.0f;
    configuration.searchMode = HF_SEARCH_MODE_EXHAUSTIVE;
    return configuration;
}

std::vector<float> UnitFeature(size_t index) {
    std::vector<float> feature(512, 0.0f);
    feature[index % feature.size()] = 1.0f;
    return feature;
}

HFFaceFeature View(std::vector<float>& feature) {
    return {static_cast<HInt32>(feature.size()), feature.data()};
}

}  // namespace

TEST_CASE("C API FeatureHub state machine and CRUD are deterministic", "[api][contract][feature_hub]") {
    FeatureHubReset reset;
    auto first = UnitFeature(0);
    auto second = UnitFeature(1);
    HFFaceFeature first_view = View(first);
    HFFaceFeature second_view = View(second);
    HFFaceFeatureIdentity first_identity = {42, &first_view};

    HFaceId allocated_id = -1;
    CHECK(HFFeatureHubInsertFeature(first_identity, &allocated_id) == HERR_FT_HUB_DISABLE);
    CHECK(HFFeatureHubFaceRemove(42) == HERR_FT_HUB_DISABLE);
    HInt32 disabled_count = -1;
    CHECK(HFFeatureHubGetFaceCount(&disabled_count) == HERR_FT_HUB_DISABLE);
    CHECK(disabled_count == 0);

    REQUIRE(HFFeatureHubDataEnable(MemoryConfiguration()) == HSUCCEED);
    CHECK(HFFeatureHubDataEnable(MemoryConfiguration()) == HSUCCEED);

    HInt32 count = -1;
    REQUIRE(HFFeatureHubGetFaceCount(&count) == HSUCCEED);
    CHECK(count == 0);
    HFFeatureHubExistingIds ids = {};
    REQUIRE(HFFeatureHubGetExistingIds(&ids) == HSUCCEED);
    CHECK(ids.size == 0);
    CHECK(ids.ids == nullptr);

    REQUIRE(HFFeatureHubInsertFeature(first_identity, &allocated_id) == HSUCCEED);
    CHECK(allocated_id == 42);
    CHECK(HFFeatureHubInsertFeature(first_identity, &allocated_id) == HERR_FT_HUB_INSERT_FAILURE);
    REQUIRE(HFFeatureHubGetFaceCount(&count) == HSUCCEED);
    CHECK(count == 1);

    HFFaceFeatureIdentity fetched = {};
    REQUIRE(HFFeatureHubGetFaceIdentity(42, &fetched) == HSUCCEED);
    REQUIRE(fetched.id == 42);
    REQUIRE(fetched.feature != nullptr);
    REQUIRE(fetched.feature->size == 512);
    CHECK(std::equal(first.begin(), first.end(), fetched.feature->data));

    float confidence = -2.0f;
    HFFaceFeatureIdentity match = {};
    REQUIRE(HFFeatureHubFaceSearch(first_view, &confidence, &match) == HSUCCEED);
    CHECK(match.id == 42);
    CHECK(confidence == Approx(1.0f).margin(1e-6f));

    HFFeatureHubSearchResultV2 match_v2 = {};
    REQUIRE(HFFeatureHubFaceSearchV2(first_view, &match_v2) == HSUCCEED);
    REQUIRE(match_v2.found == 1);
    CHECK(match_v2.id == 42);
    CHECK(match_v2.confidence == Approx(1.0f).margin(1e-6f));
    REQUIRE(match_v2.feature.size == 512);
    REQUIRE(match_v2.feature.data != nullptr);
    CHECK(std::equal(first.begin(), first.end(), match_v2.feature.data));

    REQUIRE(HFFeatureHubFaceSearchThresholdSetting(1.0f) == HSUCCEED);
    match = {7, reinterpret_cast<PHFFaceFeature>(static_cast<uintptr_t>(1))};
    confidence = 7.0f;
    REQUIRE(HFFeatureHubFaceSearch(second_view, &confidence, &match) == HSUCCEED);
    CHECK(match.id == -1);
    CHECK(match.feature == nullptr);
    CHECK(confidence == -1.0f);
    match_v2 = {7, 8, 9.0f, {10, reinterpret_cast<HPFloat>(static_cast<uintptr_t>(1))}};
    REQUIRE(HFFeatureHubFaceSearchV2(second_view, &match_v2) == HSUCCEED);
    CHECK(match_v2.found == 0);
    CHECK(match_v2.id == HF_INVALID_FACE_ID);
    CHECK(match_v2.confidence == -1.0f);
    CHECK(match_v2.feature.size == 0);
    CHECK(match_v2.feature.data == nullptr);
    REQUIRE(HFFeatureHubFaceSearchThresholdSetting(-1.0f) == HSUCCEED);

    HFSearchTopKResults top_k = {};
    REQUIRE(HFFeatureHubFaceSearchTopK(first_view, 5, &top_k) == HSUCCEED);
    REQUIRE(top_k.size == 1);
    REQUIRE(top_k.ids != nullptr);
    REQUIRE(top_k.confidence != nullptr);
    CHECK(top_k.ids[0] == 42);

    HFFaceFeatureIdentity update = {42, &second_view};
    REQUIRE(HFFeatureHubFaceUpdate(update) == HSUCCEED);
    REQUIRE(HFFeatureHubGetFaceIdentity(42, &fetched) == HSUCCEED);
    CHECK(std::equal(second.begin(), second.end(), fetched.feature->data));

    REQUIRE(HFFeatureHubGetExistingIds(&ids) == HSUCCEED);
    REQUIRE(ids.size == 1);
    REQUIRE(ids.ids != nullptr);
    CHECK(ids.ids[0] == 42);

    REQUIRE(HFFeatureHubFaceRemove(42) == HSUCCEED);
    CHECK(HFFeatureHubFaceRemove(42) == HERR_FT_HUB_NOT_FOUND_FEATURE);
    CHECK(HFFeatureHubGetFaceIdentity(42, &fetched) == HERR_FT_HUB_NOT_FOUND_FEATURE);
    CHECK(fetched.id == -1);
    CHECK(fetched.feature == nullptr);

    REQUIRE(HFFeatureHubDataDisable() == HSUCCEED);
    CHECK(HFFeatureHubDataDisable() == HSUCCEED);
}

TEST_CASE("C API FeatureHub validates configuration, vector contents, IDs, and outputs", "[api][contract][feature_hub][boundary]") {
    FeatureHubReset reset;

    auto config = MemoryConfiguration();
    config.primaryKeyMode = static_cast<HFPKMode>(99);
    CHECK(HFFeatureHubDataEnable(config) == HERR_INVALID_PARAM);
    config = MemoryConfiguration();
    config.searchMode = static_cast<HFSearchMode>(99);
    CHECK(HFFeatureHubDataEnable(config) == HERR_INVALID_PARAM);
    config = MemoryConfiguration();
    config.enablePersistence = 2;
    CHECK(HFFeatureHubDataEnable(config) == HERR_INVALID_PARAM);
    config = MemoryConfiguration();
    config.searchThreshold = std::numeric_limits<float>::quiet_NaN();
    CHECK(HFFeatureHubDataEnable(config) == HERR_INVALID_PARAM);
    config = MemoryConfiguration();
    config.enablePersistence = 1;
    config.persistenceDbPath = nullptr;
    CHECK(HFFeatureHubDataEnable(config) == HERR_INVALID_PARAM);

    REQUIRE(HFFeatureHubDataEnable(MemoryConfiguration()) == HSUCCEED);
    CHECK(HFFeatureHubGetFaceCount(nullptr) == HERR_INVALID_PARAM);
    CHECK(HFFeatureHubGetExistingIds(nullptr) == HERR_INVALID_PARAM);
    CHECK(HFFeatureHubGetFaceIdentity(1, nullptr) == HERR_INVALID_PARAM);
    CHECK(HFFeatureHubFaceSearchThresholdSetting(-1.01f) == HERR_INVALID_PARAM);
    CHECK(HFFeatureHubFaceSearchThresholdSetting(std::numeric_limits<float>::infinity()) == HERR_INVALID_PARAM);
    CHECK(HFFeatureHubFaceSearchThresholdSetting(0.5f) == HSUCCEED);

    auto valid = UnitFeature(0);
    auto short_feature = valid;
    short_feature.pop_back();
    auto non_finite = valid;
    non_finite[0] = std::numeric_limits<float>::quiet_NaN();
    HFFaceFeature valid_view = View(valid);
    HFFaceFeature short_view = View(short_feature);
    HFFaceFeature non_finite_view = View(non_finite);
    float sentinel = 1.0f;
    HFFaceFeature oversized_view = {std::numeric_limits<HInt32>::max(), &sentinel};
    HFaceId id = -1;

    CHECK(HFFeatureHubInsertFeature({1, &short_view}, &id) == HERR_FT_HUB_INVALID_FEATURE);
    CHECK(HFFeatureHubInsertFeature({1, &non_finite_view}, &id) == HERR_FT_HUB_INVALID_FEATURE);
    CHECK(HFFeatureHubInsertFeature({1, &valid_view}, nullptr) == HERR_INVALID_PARAM);
    id = 77;
    CHECK(HFFeatureHubInsertFeature({1, &oversized_view}, &id) == HERR_FT_HUB_INVALID_FEATURE);
    CHECK(id == -1);
    id = 77;
    CHECK(HFFeatureHubInsertFeature({HF_INVALID_FACE_ID, &valid_view}, &id) == HERR_INVALID_PARAM);
    CHECK(id == HF_INVALID_FACE_ID);

    float confidence = 4.0f;
    HFFaceFeatureIdentity match = {7, nullptr};
    CHECK(HFFeatureHubFaceSearch(short_view, &confidence, &match) == HERR_FT_HUB_INVALID_FEATURE);
    CHECK(HFFeatureHubFaceSearch(oversized_view, &confidence, &match) == HERR_FT_HUB_INVALID_FEATURE);
    CHECK(HFFeatureHubFaceSearch(valid_view, nullptr, &match) == HERR_INVALID_PARAM);
    CHECK(HFFeatureHubFaceSearch(valid_view, &confidence, nullptr) == HERR_INVALID_PARAM);

    HFFeatureHubSearchResultV2 match_v2 = {1, 7, 4.0f, {2, reinterpret_cast<HPFloat>(static_cast<uintptr_t>(1))}};
    CHECK(HFFeatureHubFaceSearchV2(valid_view, nullptr) == HERR_INVALID_PARAM);
    CHECK(HFFeatureHubFaceSearchV2(short_view, &match_v2) == HERR_FT_HUB_INVALID_FEATURE);
    CHECK(match_v2.found == 0);
    CHECK(match_v2.id == HF_INVALID_FACE_ID);
    CHECK(match_v2.confidence == -1.0f);
    CHECK(match_v2.feature.size == 0);
    CHECK(match_v2.feature.data == nullptr);

    HFSearchTopKResults top_k = {7, reinterpret_cast<float*>(1), reinterpret_cast<HFaceId*>(1)};
    CHECK(HFFeatureHubFaceSearchTopK(valid_view, 0, &top_k) == HERR_INVALID_PARAM);
    CHECK(HFFeatureHubFaceSearchTopK(short_view, 1, &top_k) == HERR_FT_HUB_INVALID_FEATURE);
    CHECK(HFFeatureHubFaceSearchTopK(oversized_view, 1, &top_k) == HERR_FT_HUB_INVALID_FEATURE);
    CHECK(top_k.size == 0);
    CHECK(top_k.confidence == nullptr);
    CHECK(top_k.ids == nullptr);
    CHECK(HFFeatureHubFaceUpdate({1, &oversized_view}) == HERR_FT_HUB_INVALID_FEATURE);

    float comparison = 9.0f;
    CHECK(HFFaceComparison(short_view, short_view, &comparison) == HERR_INVALID_FACE_FEATURE);
    CHECK(HFFaceComparison(non_finite_view, non_finite_view, &comparison) == HERR_SESS_REC_CONTRAST_FEAT_ERR);
}

TEST_CASE("C API FeatureHub auto IDs remain unique", "[api][contract][feature_hub]") {
    FeatureHubReset reset;
    REQUIRE(HFFeatureHubDataEnable(MemoryConfiguration(HF_PK_AUTO_INCREMENT)) == HSUCCEED);

    auto first = UnitFeature(2);
    auto second = UnitFeature(3);
    HFFaceFeature first_view = View(first);
    HFFaceFeature second_view = View(second);
    HFaceId first_id = -1;
    HFaceId second_id = -1;
    REQUIRE(HFFeatureHubInsertFeature({HF_INVALID_FACE_ID, &first_view}, &first_id) == HSUCCEED);
    REQUIRE(HFFeatureHubInsertFeature({HF_INVALID_FACE_ID, &second_view}, &second_id) == HSUCCEED);
    CHECK(first_id >= 0);
    CHECK(second_id > first_id);
}

TEST_CASE("C API FeatureHub supports signed 64-bit IDs", "[api][contract][feature_hub][int64]") {
    FeatureHubReset reset;
    REQUIRE(HFFeatureHubDataEnable(MemoryConfiguration()) == HSUCCEED);

    const HFaceId positive_id = static_cast<HFaceId>(std::numeric_limits<int32_t>::max()) + 17;
    const HFaceId negative_id = static_cast<HFaceId>(std::numeric_limits<int32_t>::min()) - 17;
    auto first = UnitFeature(11);
    auto second = UnitFeature(12);
    auto replacement = UnitFeature(13);
    HFFaceFeature first_view = View(first);
    HFFaceFeature second_view = View(second);
    HFFaceFeature replacement_view = View(replacement);
    HFaceId allocated_id = HF_INVALID_FACE_ID;

    REQUIRE(HFFeatureHubInsertFeature({positive_id, &first_view}, &allocated_id) == HSUCCEED);
    CHECK(allocated_id == positive_id);
    REQUIRE(HFFeatureHubInsertFeature({negative_id, &second_view}, &allocated_id) == HSUCCEED);
    CHECK(allocated_id == negative_id);

    HFFaceFeatureIdentity fetched = {};
    REQUIRE(HFFeatureHubGetFaceIdentity(positive_id, &fetched) == HSUCCEED);
    CHECK(fetched.id == positive_id);
    REQUIRE(fetched.feature != nullptr);
    CHECK(std::equal(first.begin(), first.end(), fetched.feature->data));

    HFFeatureHubSearchResultV2 match_v2 = {};
    REQUIRE(HFFeatureHubFaceSearchV2(first_view, &match_v2) == HSUCCEED);
    REQUIRE(match_v2.found == 1);
    CHECK(match_v2.id == positive_id);
    CHECK(match_v2.confidence == Approx(1.0f).margin(1e-6f));

    float confidence = -1.0f;
    HFFaceFeatureIdentity legacy_match = {};
    REQUIRE(HFFeatureHubFaceSearch(first_view, &confidence, &legacy_match) == HSUCCEED);
    CHECK(legacy_match.id == positive_id);
    CHECK(confidence == Approx(match_v2.confidence).margin(1e-6f));

    HFSearchTopKResults top_k = {};
    REQUIRE(HFFeatureHubFaceSearchTopK(second_view, 2, &top_k) == HSUCCEED);
    REQUIRE(top_k.size == 2);
    CHECK(top_k.ids[0] == negative_id);

    REQUIRE(HFFeatureHubFaceUpdate({positive_id, &replacement_view}) == HSUCCEED);
    REQUIRE(HFFeatureHubGetFaceIdentity(positive_id, &fetched) == HSUCCEED);
    CHECK(std::equal(replacement.begin(), replacement.end(), fetched.feature->data));

    HFFeatureHubExistingIds ids = {};
    REQUIRE(HFFeatureHubGetExistingIds(&ids) == HSUCCEED);
    REQUIRE(ids.size == 2);
    std::vector<HFaceId> actual_ids(ids.ids, ids.ids + ids.size);
    std::sort(actual_ids.begin(), actual_ids.end());
    std::vector<HFaceId> expected_ids = {negative_id, positive_id};
    std::sort(expected_ids.begin(), expected_ids.end());
    CHECK(actual_ids == expected_ids);

    CHECK(HFFeatureHubFaceRemove(positive_id) == HSUCCEED);
    CHECK(HFFeatureHubFaceRemove(negative_id) == HSUCCEED);
}
