#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <inspireface/include/inspireface/inspireface.hpp>

#include "settings/test_settings.h"

namespace {

class CppFeatureHubReset {
public:
    CppFeatureHubReset() {
        inspire::FeatureHubDB::GetInstance()->DisableHub();
    }
    ~CppFeatureHubReset() {
        inspire::FeatureHubDB::GetInstance()->DisableHub();
    }
};

inspire::DatabaseConfiguration MemoryConfiguration() {
    inspire::DatabaseConfiguration configuration;
    configuration.primary_key_mode = inspire::PrimaryKeyMode::MANUAL_INPUT;
    configuration.enable_persistence = false;
    configuration.recognition_threshold = -1.0f;
    configuration.search_mode = inspire::SEARCH_MODE_EXHAUSTIVE;
    return configuration;
}

std::vector<float> UnitFeature(size_t index) {
    std::vector<float> feature(512, 0.0f);
    feature[index % feature.size()] = 1.0f;
    return feature;
}

}  // namespace

TEST_CASE("C++ FeatureHub CRUD search and caches remain coherent", "[cpp_api][contract][feature_hub]") {
    CppFeatureHubReset reset;
    const auto hub = inspire::FeatureHubDB::GetInstance();
    REQUIRE(hub == inspire::FeatureHubDB::GetInstance());

    auto first = UnitFeature(0);
    auto second = UnitFeature(1);
    int64_t allocated_id = 99;
    CHECK(hub->FaceFeatureInsert(first, 10, allocated_id) == HERR_FT_HUB_DISABLE);

    REQUIRE(hub->EnableHub(MemoryConfiguration()) == HSUCCEED);
    CHECK(hub->EnableHub(MemoryConfiguration()) == HSUCCEED);
    CHECK(hub->GetFaceFeatureCount() == 0);

    REQUIRE(hub->FaceFeatureInsert(first, 10, allocated_id) == HSUCCEED);
    CHECK(allocated_id == 10);
    REQUIRE(hub->FaceFeatureInsert(second, 20, allocated_id) == HSUCCEED);
    CHECK(allocated_id == 20);
    CHECK(hub->GetFaceFeatureCount() == 2);

    std::vector<int64_t> ids;
    REQUIRE(hub->GetAllIds(ids) == HSUCCEED);
    std::sort(ids.begin(), ids.end());
    CHECK(ids == std::vector<int64_t>{10, 20});
    REQUIRE(hub->GetAllIds() == HSUCCEED);
    auto cached_ids = hub->GetExistingIds();
    std::sort(cached_ids.begin(), cached_ids.end());
    CHECK(cached_ids == ids);

    std::vector<float> fetched;
    REQUIRE(hub->GetFaceFeature(10, fetched) == HSUCCEED);
    CHECK(fetched == first);
    inspire::FaceEmbedding fetched_embedding = {};
    REQUIRE(hub->GetFaceFeature(20, fetched_embedding) == HSUCCEED);
    CHECK(fetched_embedding.embedding == second);
    REQUIRE(hub->GetFaceFeature(10) == HSUCCEED);
    const auto cached_feature = hub->GetFaceFeaturePtrCache();
    REQUIRE(cached_feature);
    REQUIRE(cached_feature->data != nullptr);
    REQUIRE(cached_feature->dataSize == 512);
    CHECK(std::equal(first.begin(), first.end(), cached_feature->data));
    CHECK(hub->GetFaceFeature(999) == HERR_FT_HUB_NOT_FOUND_FEATURE);
    CHECK(cached_feature->data == nullptr);
    CHECK(cached_feature->dataSize == 0);

    inspire::FaceSearchResult best = {};
    REQUIRE(hub->SearchFaceFeature(first, best, true) == HSUCCEED);
    CHECK(best.id == 10);
    CHECK(best.similarity == Approx(1.0).margin(1e-6));
    CHECK(best.feature == first);
    CHECK(hub->GetSearchFaceFeatureCache() == first);
    bool found = false;
    inspire::FaceSearchResult best_v2 = {};
    REQUIRE(hub->SearchFaceFeatureV2(first, best_v2, found, true) == HSUCCEED);
    REQUIRE(found);
    CHECK(best_v2.id == best.id);
    CHECK(best_v2.similarity == Approx(best.similarity).margin(1e-6));
    CHECK(best_v2.feature == best.feature);
    CHECK(hub->SearchFaceFeature(std::vector<float>(511, 0.0f), best, true) == HERR_FT_HUB_INVALID_FEATURE);
    CHECK(cached_feature->data == nullptr);
    CHECK(cached_feature->dataSize == 0);

    std::vector<inspire::FaceSearchResult> top_k;
    REQUIRE(hub->SearchFaceFeatureTopK(first, top_k, 5, true) == HSUCCEED);
    REQUIRE(top_k.size() == 2);
    CHECK(top_k.front().id == 10);
    CHECK(top_k.front().feature == first);
    REQUIRE(hub->SearchFaceFeatureTopKCache(first, 5) == HSUCCEED);
    REQUIRE(hub->GetTopKConfidence().size() == 2);
    REQUIRE(hub->GetTopKCustomIdsCache().size() == 2);
    CHECK(hub->GetTopKCustomIdsCache().front() == 10);
    hub->SetRecognitionThreshold(1.0f);
    hub->SetRecognitionSearchMode(inspire::SEARCH_MODE_EAGER);
    REQUIRE(hub->SearchFaceFeature(first, best, false) == HSUCCEED);
    CHECK(best.id == 10);
    CHECK(best.feature.empty());
    const auto unknown = UnitFeature(7);
    found = true;
    best_v2 = {77, 8.0, first};
    REQUIRE(hub->SearchFaceFeatureV2(unknown, best_v2, found, true) == HSUCCEED);
    CHECK_FALSE(found);
    CHECK(best_v2.id == INSPIRE_INVALID_ID);
    CHECK(best_v2.similarity == Approx(-1.0));
    CHECK(best_v2.feature.empty());
    CHECK(hub->ViewDBTable() == HSUCCEED);

    auto replacement = UnitFeature(2);
    REQUIRE(hub->FaceFeatureUpdate(replacement, 10) == HSUCCEED);
    REQUIRE(hub->GetFaceFeature(10, fetched) == HSUCCEED);
    CHECK(fetched == replacement);
    REQUIRE(hub->FaceFeatureRemove(20) == HSUCCEED);
    CHECK(hub->FaceFeatureRemove(20) == HERR_FT_HUB_NOT_FOUND_FEATURE);
    CHECK(hub->GetFaceFeatureCount() == 1);
}

TEST_CASE("C++ FeatureHub rejects malformed inputs and clears outputs", "[cpp_api][contract][feature_hub][boundary]") {
    CppFeatureHubReset reset;
    const auto hub = inspire::FeatureHubDB::GetInstance();

    auto invalid_configuration = MemoryConfiguration();
    invalid_configuration.primary_key_mode = static_cast<inspire::PrimaryKeyMode>(99);
    CHECK(hub->EnableHub(invalid_configuration) == HERR_INVALID_PARAM);
    invalid_configuration = MemoryConfiguration();
    invalid_configuration.search_mode = static_cast<inspire::SearchMode>(99);
    CHECK(hub->EnableHub(invalid_configuration) == HERR_INVALID_PARAM);
    invalid_configuration = MemoryConfiguration();
    invalid_configuration.enable_persistence = true;
    invalid_configuration.persistence_db_path.clear();
    CHECK(hub->EnableHub(invalid_configuration) == HERR_INVALID_PARAM);
    invalid_configuration = MemoryConfiguration();
    invalid_configuration.recognition_threshold = std::numeric_limits<float>::quiet_NaN();
    CHECK(hub->EnableHub(invalid_configuration) == HERR_INVALID_PARAM);

    REQUIRE(hub->EnableHub(MemoryConfiguration()) == HSUCCEED);
    const auto valid = UnitFeature(0);
    auto short_feature = valid;
    short_feature.pop_back();
    auto non_finite = valid;
    non_finite[0] = std::numeric_limits<float>::quiet_NaN();

    int64_t allocated_id = 123;
    CHECK(hub->FaceFeatureInsert(short_feature, 1, allocated_id) == HERR_FT_HUB_INVALID_FEATURE);
    CHECK(allocated_id == -1);
    CHECK(hub->FaceFeatureInsert(non_finite, 1, allocated_id) == HERR_FT_HUB_INVALID_FEATURE);
    CHECK(hub->FaceFeatureInsert(valid, static_cast<int64_t>(INSPIRE_INVALID_ID), allocated_id) == HERR_INVALID_PARAM);
    CHECK(allocated_id == INSPIRE_INVALID_ID);

    inspire::FaceSearchResult search = {44, 2.0, valid};
    CHECK(hub->SearchFaceFeature(short_feature, search) == HERR_FT_HUB_INVALID_FEATURE);
    CHECK(search.id == -1);
    CHECK(search.similarity == Approx(-1.0));
    CHECK(search.feature.empty());

    std::vector<inspire::FaceSearchResult> top_k(1, {44, 2.0, valid});
    CHECK(hub->SearchFaceFeatureTopK(valid, top_k, 0) == HERR_FT_HUB_INVALID_FEATURE);
    CHECK(top_k.empty());
    CHECK(hub->SearchFaceFeatureTopKCache(valid, 0) == HERR_FT_HUB_INVALID_FEATURE);
    CHECK(hub->GetTopKConfidence().empty());
    CHECK(hub->GetTopKCustomIdsCache().empty());

    float similarity = 7.0f;
    CHECK(inspire::FeatureHubDB::CosineSimilarity(valid, short_feature, similarity, true) == HERR_SESS_REC_CONTRAST_FEAT_ERR);
    CHECK(similarity == 0.0f);
    CHECK(inspire::FeatureHubDB::CosineSimilarity(nullptr, valid.data(), 512, similarity, true) == HERR_SESS_REC_CONTRAST_FEAT_ERR);
    CHECK(inspire::FeatureHubDB::CosineSimilarity(valid.data(), valid.data(), 0, similarity, true) == HERR_SESS_REC_CONTRAST_FEAT_ERR);

    std::vector<float> zero(512, 0.0f);
    CHECK(inspire::FeatureHubDB::CosineSimilarity(zero, zero, similarity, true) == HERR_SESS_REC_CONTRAST_FEAT_ERR);
    CHECK(inspire::FeatureHubDB::CosineSimilarity(valid, valid, similarity, true) == HSUCCEED);
    CHECK(similarity == Approx(1.0f).margin(1e-6f));

    hub->SetRecognitionThreshold(std::numeric_limits<float>::infinity());
    hub->SetRecognitionSearchMode(static_cast<inspire::SearchMode>(99));
}

TEST_CASE("C++ FeatureHub public construction starts disabled", "[cpp_api][contract][feature_hub][lifetime]") {
    CppFeatureHubReset reset;
    inspire::FeatureHubDB local;
    CHECK(local.GetFaceFeatureCount() == 0);
    CHECK(local.DisableHub() == HSUCCEED);
}

TEST_CASE("C++ FeatureHub preserves 32-bit overloads and supports 64-bit IDs", "[cpp_api][contract][feature_hub][abi][int64]") {
    using Hub = inspire::FeatureHubDB;
    using Insert32 = int32_t (Hub::*)(const std::vector<float>&, int32_t, int64_t&);
    using Remove32 = int32_t (Hub::*)(int32_t);
    using Update32 = int32_t (Hub::*)(const std::vector<float>&, int32_t);
    using Get32 = int32_t (Hub::*)(int32_t);
    using GetVector32 = int32_t (Hub::*)(int32_t, std::vector<float>&);
    using GetEmbedding32 = int32_t (Hub::*)(int32_t, inspire::FaceEmbedding&);
    const Insert32 legacy_insert = static_cast<Insert32>(&Hub::FaceFeatureInsert);
    const Remove32 legacy_remove = static_cast<Remove32>(&Hub::FaceFeatureRemove);
    const Update32 legacy_update = static_cast<Update32>(&Hub::FaceFeatureUpdate);
    const Get32 legacy_get = static_cast<Get32>(&Hub::GetFaceFeature);
    const GetVector32 legacy_get_vector = static_cast<GetVector32>(&Hub::GetFaceFeature);
    const GetEmbedding32 legacy_get_embedding = static_cast<GetEmbedding32>(&Hub::GetFaceFeature);
    REQUIRE(legacy_insert != nullptr);
    REQUIRE(legacy_remove != nullptr);
    REQUIRE(legacy_update != nullptr);
    REQUIRE(legacy_get != nullptr);
    REQUIRE(legacy_get_vector != nullptr);
    REQUIRE(legacy_get_embedding != nullptr);

    CppFeatureHubReset reset;
    const auto hub = Hub::GetInstance();
    REQUIRE(hub->EnableHub(MemoryConfiguration()) == HSUCCEED);

    const int64_t wide_id = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 17;
    auto first = UnitFeature(21);
    auto replacement = UnitFeature(22);
    int64_t allocated_id = INSPIRE_INVALID_ID;
    REQUIRE(hub->FaceFeatureInsert(first, wide_id, allocated_id) == HSUCCEED);
    CHECK(allocated_id == wide_id);

    std::vector<float> fetched;
    REQUIRE(hub->GetFaceFeature(wide_id, fetched) == HSUCCEED);
    CHECK(fetched == first);
    REQUIRE(hub->FaceFeatureUpdate(replacement, wide_id) == HSUCCEED);
    REQUIRE(hub->GetFaceFeature(wide_id, fetched) == HSUCCEED);
    CHECK(fetched == replacement);

    inspire::FaceSearchResult match = {};
    bool found = false;
    REQUIRE(hub->SearchFaceFeatureV2(replacement, match, found, false) == HSUCCEED);
    REQUIRE(found);
    CHECK(match.id == wide_id);
    CHECK(match.feature.empty());

    REQUIRE(hub->FaceFeatureRemove(wide_id) == HSUCCEED);
    CHECK(hub->GetFaceFeature(wide_id, fetched) == HERR_FT_HUB_NOT_FOUND_FEATURE);
}
