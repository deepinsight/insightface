#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "inspireface/feature_hub/embedding_db/embedding_db.h"
#include "inspireface/feature_hub_db.h"
#include "inspireface/herror.h"
#include "inspireface.h"

namespace {

constexpr size_t kFeatureSize = 512;

struct Options {
    bool baseline = false;
    double max_cosine_p95_us = 0.0;
    double max_search_p95_us = 0.0;
#if defined(_WIN32)
    std::string database_path = "inspireface_feature_hub_safety_guard.db";
#else
    std::string database_path = "/tmp/inspireface_feature_hub_safety_guard.db";
#endif
};

struct Timings {
    double mean_us = 0.0;
    double p50_us = 0.0;
    double p95_us = 0.0;
};

struct Guard {
    int failures = 0;

    void Expect(bool condition, const std::string &message) {
        if (!condition) {
            ++failures;
            std::cerr << "[FAIL] " << message << '\n';
        }
    }

    void ObserveOrExpect(bool condition, bool baseline, const std::string &message) {
        if (!condition && baseline) {
            std::cout << "[KNOWN-BASELINE-DEFECT] " << message << '\n';
            return;
        }
        Expect(condition, message);
    }
};

double Percentile(std::vector<double> values, double percentile) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(std::ceil(percentile * values.size())) - 1;
    return values[std::min(index, values.size() - 1)];
}

Timings Summarize(const std::vector<double> &values) {
    Timings result;
    if (!values.empty()) {
        result.mean_us = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        result.p50_us = Percentile(values, 0.50);
        result.p95_us = Percentile(values, 0.95);
    }
    return result;
}

std::vector<float> MakeFeature(int seed) {
    std::vector<float> feature(kFeatureSize);
    double squared_norm = 0.0;
    for (size_t i = 0; i < feature.size(); ++i) {
        const double x = static_cast<double>((seed + 3) * (static_cast<int>(i) + 11));
        const float value = static_cast<float>(std::sin(x * 0.0137) + 0.61 * std::cos(x * 0.0079 + seed * 0.11));
        feature[i] = value;
        squared_norm += static_cast<double>(value) * value;
    }
    const double inverse_norm = 1.0 / std::sqrt(squared_norm);
    for (float &value : feature) {
        value = static_cast<float>(value * inverse_norm);
    }
    return feature;
}

std::vector<float> PerturbFeature(const std::vector<float> &feature, int seed) {
    std::vector<float> perturbed = feature;
    double squared_norm = 0.0;
    for (size_t i = 0; i < perturbed.size(); ++i) {
        perturbed[i] += static_cast<float>(0.002 * std::sin((i + 1) * (seed + 1) * 0.17));
        squared_norm += static_cast<double>(perturbed[i]) * perturbed[i];
    }
    const double inverse_norm = 1.0 / std::sqrt(squared_norm);
    for (float &value : perturbed) {
        value = static_cast<float>(value * inverse_norm);
    }
    return perturbed;
}

double ReferenceCosine(const std::vector<float> &left, const std::vector<float> &right) {
    long double dot = 0.0;
    long double left_norm = 0.0;
    long double right_norm = 0.0;
    for (size_t i = 0; i < left.size(); ++i) {
        dot += static_cast<long double>(left[i]) * right[i];
        left_norm += static_cast<long double>(left[i]) * left[i];
        right_norm += static_cast<long double>(right[i]) * right[i];
    }
    return static_cast<double>(dot / std::sqrt(left_norm * right_norm));
}

bool SameFeature(const std::vector<float> &left, const std::vector<float> &right, float tolerance = 1.0e-7f) {
    if (left.size() != right.size()) {
        return false;
    }
    for (size_t i = 0; i < left.size(); ++i) {
        if (std::fabs(left[i] - right[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

void RemoveDatabaseFiles(const std::string &path) {
    std::remove(path.c_str());
    std::remove((path + "-journal").c_str());
    std::remove((path + "-shm").c_str());
    std::remove((path + "-wal").c_str());
}

Options ParseOptions(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--baseline") {
            options.baseline = true;
        } else if (argument.find("--max-cosine-p95-us=") == 0) {
            options.max_cosine_p95_us = std::stod(argument.substr(std::strlen("--max-cosine-p95-us=")));
        } else if (argument.find("--max-search-p95-us=") == 0) {
            options.max_search_p95_us = std::stod(argument.substr(std::strlen("--max-search-p95-us=")));
        } else if (argument.find("--database=") == 0) {
            options.database_path = argument.substr(std::strlen("--database="));
        }
    }
    return options;
}

Timings TestCosineSimilarity(const Options &options, Guard &guard) {
    for (int seed = 0; seed < 24; ++seed) {
        const std::vector<float> left = MakeFeature(seed);
        const std::vector<float> right = PerturbFeature(MakeFeature((seed * 7 + 5) % 31), seed);
        const double expected = ReferenceCosine(left, right);
        float vector_result = -7.0f;
        float pointer_result = -7.0f;
        guard.Expect(inspire::FeatureHubDB::CosineSimilarity(left, right, vector_result, true) == HSUCCEED,
                     "vector cosine rejected valid finite features");
        guard.Expect(inspire::FeatureHubDB::CosineSimilarity(left.data(), right.data(), static_cast<int32_t>(left.size()), pointer_result, true) ==
                       HSUCCEED,
                     "pointer cosine rejected valid finite features");
        guard.Expect(std::isfinite(vector_result) && std::fabs(vector_result - expected) <= 3.0e-5,
                     "vector cosine differs from the long-double reference");
        guard.Expect(std::isfinite(pointer_result) && std::fabs(pointer_result - expected) <= 3.0e-5,
                     "pointer cosine differs from the long-double reference");
        guard.Expect(std::fabs(vector_result - pointer_result) <= 2.0e-6f, "cosine overloads disagree");
    }

    const std::vector<float> scale_source = MakeFeature(88);
    std::vector<float> tiny = scale_source;
    std::vector<float> huge = scale_source;
    for (size_t i = 0; i < scale_source.size(); ++i) {
        tiny[i] *= 1.0e-20f;
        huge[i] *= 1.0e20f;
    }
    float scaled_result = 0.0f;
    const bool tiny_ok = inspire::FeatureHubDB::CosineSimilarity(tiny, tiny, scaled_result, true) == HSUCCEED &&
                         std::isfinite(scaled_result) && std::fabs(scaled_result - 1.0f) <= 3.0e-5f;
    guard.ObserveOrExpect(tiny_ok, options.baseline, "cosine is numerically unstable for small finite features");
    const bool huge_ok = inspire::FeatureHubDB::CosineSimilarity(huge, huge, scaled_result, true) == HSUCCEED &&
                         std::isfinite(scaled_result) && std::fabs(scaled_result - 1.0f) <= 3.0e-5f;
    guard.ObserveOrExpect(huge_ok, options.baseline, "cosine is numerically unstable for large finite features");

    const std::vector<float> zero(kFeatureSize, 0.0f);
    std::vector<float> valid = MakeFeature(91);
    std::vector<float> non_finite = valid;
    non_finite[17] = std::numeric_limits<float>::quiet_NaN();
    float result = 123.0f;
    const int32_t zero_status = inspire::FeatureHubDB::CosineSimilarity(zero, valid, result, true);
    const bool zero_rejected = zero_status != HSUCCEED && std::isfinite(result);
    std::cout << "[NUMERIC] zero_status=" << zero_status << " result_finite=" << std::isfinite(result) << '\n';
    guard.ObserveOrExpect(zero_rejected, options.baseline, "normalized cosine accepted a zero vector or returned NaN");

    result = 123.0f;
    const int32_t nan_status = inspire::FeatureHubDB::CosineSimilarity(non_finite, valid, result, true);
    const bool nan_rejected = nan_status != HSUCCEED && std::isfinite(result);
    std::cout << "[NUMERIC] nan_status=" << nan_status << " result_finite=" << std::isfinite(result) << '\n';
    guard.ObserveOrExpect(nan_rejected, options.baseline, "normalized cosine accepted a NaN feature");

    if (options.baseline) {
        std::cout << "[KNOWN-BASELINE-DEFECT] pointer cosine has no null guard; direct execution is isolated from the baseline process\n";
    } else {
        result = 123.0f;
        const int32_t null_status = inspire::FeatureHubDB::CosineSimilarity(nullptr, valid.data(), static_cast<int32_t>(valid.size()), result, true);
        std::cout << "[NUMERIC] null_status=" << null_status << '\n';
        guard.Expect(null_status != HSUCCEED && std::isfinite(result), "pointer cosine accepted a null input");
    }

    constexpr int kWarmup = 2000;
    constexpr int kIterations = 12000;
    volatile float checksum = 0.0f;
    const std::vector<float> perf_left = MakeFeature(103);
    const std::vector<float> perf_right = MakeFeature(211);
    for (int i = 0; i < kWarmup; ++i) {
        float value = 0.0f;
        inspire::FeatureHubDB::CosineSimilarity(perf_left.data(), perf_right.data(), static_cast<int32_t>(perf_left.size()), value, true);
        checksum += value;
    }

    std::vector<double> samples;
    for (int repeat = 0; repeat < 9; ++repeat) {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < kIterations; ++i) {
            float value = 0.0f;
            inspire::FeatureHubDB::CosineSimilarity(perf_left.data(), perf_right.data(), static_cast<int32_t>(perf_left.size()), value, true);
            checksum += value;
        }
        const auto elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        samples.push_back(elapsed / kIterations);
    }
    const Timings timings = Summarize(samples);
    std::cout << "[PERF] cosine_mean_us=" << timings.mean_us << " cosine_p50_us=" << timings.p50_us
              << " cosine_p95_us=" << timings.p95_us << " checksum=" << checksum << '\n';
    if (options.max_cosine_p95_us > 0.0) {
        guard.Expect(timings.p95_us <= options.max_cosine_p95_us, "cosine p95 latency exceeded the baseline gate");
    }
    return timings;
}

Timings TestFeatureHubAccuracyAndLatency(const Options &options, Guard &guard) {
    const auto hub = inspire::FeatureHubDB::GetInstance();
    hub->DisableHub();
    inspire::DatabaseConfiguration configuration;
    configuration.primary_key_mode = inspire::PrimaryKeyMode::MANUAL_INPUT;
    configuration.recognition_threshold = -1.0f;
    guard.Expect(hub->EnableHub(configuration) == HSUCCEED, "failed to enable the in-memory FeatureHub");

    constexpr int kCorpusSize = 384;
    std::vector<std::vector<float>> corpus;
    corpus.reserve(kCorpusSize);
    for (int i = 0; i < kCorpusSize; ++i) {
        corpus.push_back(MakeFeature(i + 300));
        int64_t allocated_id = -1;
        guard.Expect(hub->FaceFeatureInsert(corpus.back(), 10000 + i, allocated_id) == HSUCCEED && allocated_id == 10000 + i,
                     "failed to insert a deterministic corpus feature");
    }
    guard.Expect(hub->GetFaceFeatureCount() == kCorpusSize, "FeatureHub count differs from inserted corpus size");

    for (int query_index = 0; query_index < 32; ++query_index) {
        const int source_index = (query_index * 37 + 9) % kCorpusSize;
        const std::vector<float> query = PerturbFeature(corpus[source_index], query_index + 700);
        int expected_index = -1;
        double expected_similarity = -2.0;
        for (int i = 0; i < kCorpusSize; ++i) {
            const double similarity = ReferenceCosine(query, corpus[i]);
            if (similarity > expected_similarity) {
                expected_similarity = similarity;
                expected_index = i;
            }
        }

        inspire::FaceSearchResult result{-1, -1.0, {}};
        guard.Expect(hub->SearchFaceFeature(query, result, true) == HSUCCEED, "FeatureHub search failed for a valid query");
        guard.Expect(result.id == 10000 + expected_index, "FeatureHub top-1 ID differs from brute-force cosine search");
        guard.Expect(std::fabs(result.similarity - expected_similarity) <= 3.0e-5,
                     "FeatureHub similarity differs from brute-force cosine search");
        guard.Expect(SameFeature(result.feature, corpus[expected_index]), "FeatureHub returned the wrong stored feature");
        guard.Expect(SameFeature(hub->GetSearchFaceFeatureCache(), corpus[expected_index]), "legacy search cache changed behavior");
        const auto &feature_pointer = hub->GetFaceFeaturePtrCache();
        guard.Expect(feature_pointer && feature_pointer->dataSize == static_cast<int32_t>(kFeatureSize) &&
                       std::equal(feature_pointer->data, feature_pointer->data + kFeatureSize, corpus[expected_index].begin()),
                     "legacy C API feature cache does not match the search result");
    }

    std::vector<inspire::FaceSearchResult> top_k;
    guard.Expect(hub->SearchFaceFeatureTopK(corpus[123], top_k, 7, true) == HSUCCEED && top_k.size() == 7,
                 "top-k search did not return the requested number of results");
    guard.Expect(!top_k.empty() && top_k.front().id == 10123 && std::fabs(top_k.front().similarity - 1.0) <= 3.0e-5,
                 "top-k search changed exact-match behavior");
    for (size_t i = 1; i < top_k.size(); ++i) {
        guard.Expect(top_k[i - 1].similarity >= top_k[i].similarity, "top-k results are not ordered by similarity");
    }
    guard.Expect(hub->SearchFaceFeatureTopKCache(corpus[123], 7) == HSUCCEED && hub->GetTopKConfidence().size() == 7 &&
                   hub->GetTopKCustomIdsCache().size() == 7,
                 "legacy top-k caches changed behavior");

    guard.Expect(hub->GetAllIds() == HSUCCEED, "failed to enumerate FeatureHub IDs");
    std::vector<int64_t> ids = hub->GetExistingIds();
    std::sort(ids.begin(), ids.end());
    guard.Expect(ids.size() == kCorpusSize && ids.front() == 10000 && ids.back() == 10000 + kCorpusSize - 1,
                 "enumerated IDs differ from the inserted corpus");

    if (!options.baseline) {
        std::vector<float> invalid_dimension(kFeatureSize - 1, 0.0f);
        std::vector<float> invalid_number = corpus.front();
        invalid_number[0] = std::numeric_limits<float>::infinity();
        int64_t invalid_id = 123;
        guard.Expect(hub->FaceFeatureInsert(invalid_dimension, 20001, invalid_id) == HERR_FT_HUB_INVALID_FEATURE && invalid_id == -1,
                     "FeatureHub accepted an invalid insert dimension");
        guard.Expect(hub->FaceFeatureInsert(invalid_number, 20002, invalid_id) == HERR_FT_HUB_INVALID_FEATURE && invalid_id == -1,
                     "FeatureHub accepted a non-finite insert");
        inspire::FaceSearchResult invalid_result{123, 123.0, corpus.front()};
        guard.Expect(hub->SearchFaceFeature(invalid_dimension, invalid_result, false) == HERR_FT_HUB_INVALID_FEATURE && invalid_result.id == -1 &&
                       invalid_result.similarity == -1.0 && invalid_result.feature.empty(),
                     "invalid search did not clear output state and return an error");
        std::vector<inspire::FaceSearchResult> invalid_top_k(1);
        guard.Expect(hub->SearchFaceFeatureTopK(corpus.front(), invalid_top_k, 0, false) != HSUCCEED && invalid_top_k.empty(),
                     "zero-sized top-k search was accepted");
        guard.Expect(hub->GetFaceFeatureCount() == kCorpusSize, "invalid operations partially changed the FeatureHub corpus");
    }

    std::vector<double> search_samples;
    for (int warmup = 0; warmup < 32; ++warmup) {
        inspire::FaceSearchResult result{-1, -1.0, {}};
        hub->SearchFaceFeature(corpus[(warmup * 13) % kCorpusSize], result, false);
    }
    for (int i = 0; i < 240; ++i) {
        const auto start = std::chrono::steady_clock::now();
        inspire::FaceSearchResult result{-1, -1.0, {}};
        const int32_t status = hub->SearchFaceFeature(corpus[(i * 17) % kCorpusSize], result, false);
        const double elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        guard.Expect(status == HSUCCEED && result.id >= 10000, "timed search returned an invalid result");
        search_samples.push_back(elapsed);
    }
    const Timings timings = Summarize(search_samples);
    std::cout << "[PERF] search_mean_us=" << timings.mean_us << " search_p50_us=" << timings.p50_us
              << " search_p95_us=" << timings.p95_us << " corpus=" << kCorpusSize << '\n';
    if (options.max_search_p95_us > 0.0) {
        guard.Expect(timings.p95_us <= options.max_search_p95_us, "FeatureHub search p95 latency exceeded the baseline gate");
    }

    std::vector<float> updated = MakeFeature(9001);
    guard.Expect(hub->FaceFeatureUpdate(updated, 10123) == HSUCCEED, "failed to update an existing feature");
    std::vector<float> fetched;
    guard.Expect(hub->GetFaceFeature(10123, fetched) == HSUCCEED && SameFeature(fetched, updated),
                 "updated feature was not read back exactly");
    guard.Expect(hub->FaceFeatureRemove(10123) == HSUCCEED && hub->GetFaceFeature(10123, fetched) == HERR_FT_HUB_NOT_FOUND_FEATURE,
                 "removed feature is still readable");
    guard.Expect(hub->DisableHub() == HSUCCEED, "failed to disable the in-memory FeatureHub");
    return timings;
}

void TestPersistence(const Options &options, Guard &guard) {
    RemoveDatabaseFiles(options.database_path);
    const auto hub = inspire::FeatureHubDB::GetInstance();
    hub->DisableHub();
    inspire::DatabaseConfiguration configuration;
    configuration.primary_key_mode = inspire::PrimaryKeyMode::MANUAL_INPUT;
    configuration.enable_persistence = true;
    configuration.persistence_db_path = options.database_path;
    configuration.recognition_threshold = -1.0f;
    guard.Expect(hub->EnableHub(configuration) == HSUCCEED, "failed to enable persistent FeatureHub");

    const std::vector<float> first = MakeFeature(1201);
    const std::vector<float> second = MakeFeature(1202);
    const std::vector<float> replacement = MakeFeature(1203);
    int64_t allocated_id = -1;
    guard.Expect(hub->FaceFeatureInsert(first, 71, allocated_id) == HSUCCEED && allocated_id == 71, "persistent insert 71 failed");
    guard.Expect(hub->FaceFeatureInsert(second, 72, allocated_id) == HSUCCEED && allocated_id == 72, "persistent insert 72 failed");
    guard.Expect(hub->FaceFeatureUpdate(replacement, 72) == HSUCCEED, "persistent update failed");
    guard.Expect(hub->DisableHub() == HSUCCEED, "persistent disable failed");
    guard.Expect(hub->EnableHub(configuration) == HSUCCEED, "persistent reopen failed");
    guard.Expect(hub->GetFaceFeatureCount() == 2, "persistent reopen changed the row count");

    std::vector<float> fetched;
    guard.Expect(hub->GetFaceFeature(71, fetched) == HSUCCEED && SameFeature(fetched, first), "persistent feature 71 changed after reopen");
    guard.Expect(hub->GetFaceFeature(72, fetched) == HSUCCEED && SameFeature(fetched, replacement),
                 "persistent update was not durable");
    inspire::FaceSearchResult result{-1, -1.0, {}};
    guard.Expect(hub->SearchFaceFeature(replacement, result, true) == HSUCCEED && result.id == 72 &&
                   std::fabs(result.similarity - 1.0) <= 3.0e-5,
                 "persistent search changed after reopen");
    hub->DisableHub();
    RemoveDatabaseFiles(options.database_path);
}

void TestCApiCacheBoundary(const Options &options, Guard &guard) {
    if (options.baseline) {
        std::cout << "[KNOWN-BASELINE-DEFECT] C API cache concurrency and disabled-search checks are enforced after the refactor\n";
        return;
    }

    HFFeatureHubDataDisable();
    HFFeatureHubConfiguration configuration{};
    configuration.primaryKeyMode = HF_PK_MANUAL_INPUT;
    configuration.enablePersistence = 0;
    configuration.persistenceDbPath = nullptr;
    configuration.searchThreshold = -1.0f;
    configuration.searchMode = HF_SEARCH_MODE_EXHAUSTIVE;
    guard.Expect(HFFeatureHubDataEnable(configuration) == HSUCCEED, "C API failed to enable FeatureHub");

    std::vector<std::vector<float>> features = {MakeFeature(1501), MakeFeature(1502), MakeFeature(1503)};
    for (size_t i = 0; i < features.size(); ++i) {
        HFFaceFeature feature{static_cast<HInt32>(features[i].size()), features[i].data()};
        HFFaceFeatureIdentity identity{static_cast<HFaceId>(501 + i), &feature};
        HFaceId allocated_id = -1;
        guard.Expect(HFFeatureHubInsertFeature(identity, &allocated_id) == HSUCCEED && allocated_id == identity.id,
                     "C API insert changed ID or returned an error");
    }

    std::atomic<int> cache_errors(0);
    std::atomic<bool> start(false);
    std::vector<std::thread> workers;
    for (int worker_index = 0; worker_index < 3; ++worker_index) {
        workers.emplace_back([&, worker_index]() {
            HFFaceFeature query{static_cast<HInt32>(features[worker_index].size()), features[worker_index].data()};
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (int iteration = 0; iteration < 80; ++iteration) {
                HFloat confidence = -1.0f;
                HFFaceFeatureIdentity result{-1, nullptr};
                const HResult status = HFFeatureHubFaceSearch(query, &confidence, &result);
                std::this_thread::yield();
                if (status != HSUCCEED || result.id != 501 + worker_index || !result.feature ||
                    result.feature->size != static_cast<HInt32>(kFeatureSize) || !result.feature->data ||
                    !std::equal(result.feature->data, result.feature->data + kFeatureSize, features[worker_index].begin()) ||
                    std::fabs(confidence - 1.0f) > 3.0e-5f) {
                    cache_errors.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (std::thread &worker : workers) {
        worker.join();
    }
    guard.Expect(cache_errors.load(std::memory_order_relaxed) == 0, "C API search cache was overwritten by another caller thread");

    HFFaceFeature query{static_cast<HInt32>(features[1].size()), features[1].data()};
    HFSearchTopKResults top_k{};
    guard.Expect(HFFeatureHubFaceSearchTopK(query, 3, &top_k) == HSUCCEED && top_k.size == 3 && top_k.ids && top_k.confidence &&
                   top_k.ids[0] == 502 && std::fabs(top_k.confidence[0] - 1.0f) <= 3.0e-5f,
                 "C API top-k cache changed result content");

    HFFaceFeatureIdentity fetched{-1, nullptr};
    guard.Expect(HFFeatureHubGetFaceIdentity(503, &fetched) == HSUCCEED && fetched.id == 503 && fetched.feature &&
                   fetched.feature->size == static_cast<HInt32>(kFeatureSize) &&
                   std::equal(fetched.feature->data, fetched.feature->data + kFeatureSize, features[2].begin()),
                 "C API identity cache changed result content");

    HFFeatureHubExistingIds existing_ids{};
    guard.Expect(HFFeatureHubGetExistingIds(&existing_ids) == HSUCCEED && existing_ids.size == 3 && existing_ids.ids &&
                   existing_ids.ids[0] == 501 && existing_ids.ids[1] == 502 && existing_ids.ids[2] == 503,
                 "C API existing-ID cache changed result content");

    HInt32 count = -1;
    guard.Expect(HFFeatureHubGetFaceCount(&count) == HSUCCEED && count == 3, "C API FeatureHub count changed");
    guard.Expect(HFFeatureHubDataDisable() == HSUCCEED, "C API failed to disable FeatureHub");
    HFloat confidence = 123.0f;
    HFFaceFeatureIdentity disabled_result{123, nullptr};
    guard.Expect(HFFeatureHubFaceSearch(query, &confidence, &disabled_result) == HERR_FT_HUB_DISABLE && confidence == -1.0f &&
                   disabled_result.id == -1 && disabled_result.feature == nullptr,
                 "C API disabled search did not fail safely");
}

#if defined(__unix__) || defined(__APPLE__)
bool ChildExitedSuccessfully(pid_t pid) {
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) {
        return false;
    }
    if (WIFSIGNALED(status)) {
        std::cout << " signal=" << WTERMSIG(status);
    } else if (WIFEXITED(status)) {
        std::cout << " exit=" << WEXITSTATUS(status);
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}
#endif

void TestInvalidDimensionIsolation(const Options &options, Guard &guard) {
#if defined(__unix__) || defined(__APPLE__)
    const pid_t pid = fork();
    if (pid == 0) {
        const auto hub = inspire::FeatureHubDB::GetInstance();
        hub->DisableHub();
        inspire::DatabaseConfiguration configuration;
        configuration.primary_key_mode = inspire::PrimaryKeyMode::MANUAL_INPUT;
        if (hub->EnableHub(configuration) != HSUCCEED) {
            _exit(11);
        }
        std::vector<float> invalid(kFeatureSize - 1, 0.1f);
        int64_t allocated_id = -1;
        const int32_t status = hub->FaceFeatureInsert(invalid, 1, allocated_id);
        _exit(status != HSUCCEED ? 0 : 12);
    }
    std::cout << "[ISOLATION] invalid_dimension";
    const bool survived = pid > 0 && ChildExitedSuccessfully(pid);
    std::cout << '\n';
    guard.ObserveOrExpect(survived, options.baseline, "invalid FeatureHub dimension terminated the process");
#else
    std::cout << "[SKIP] invalid-dimension process-isolation check requires POSIX fork\n";
#endif
}

void TestFailedEnableIsolation(const Options &options, Guard &guard) {
#if defined(__unix__) || defined(__APPLE__)
    const std::string wrong_schema_path = options.database_path + ".wrong_schema";
    RemoveDatabaseFiles(wrong_schema_path);
    const pid_t pid = fork();
    if (pid == 0) {
        const auto hub = inspire::FeatureHubDB::GetInstance();
        hub->DisableHub();
        inspire::DatabaseConfiguration bad_configuration;
        bad_configuration.primary_key_mode = inspire::PrimaryKeyMode::MANUAL_INPUT;
        bad_configuration.enable_persistence = true;
        bad_configuration.persistence_db_path = options.database_path + ".missing/feature.db";
        if (hub->EnableHub(bad_configuration) == HSUCCEED) {
            _exit(41);
        }
        bad_configuration.persistence_db_path.clear();
        if (hub->EnableHub(bad_configuration) == HSUCCEED) {
            _exit(42);
        }
        if (!inspire::EmbeddingDB::Init(wrong_schema_path, 4, inspire::IdMode::MANUAL)) {
            _exit(43);
        }
        inspire::EmbeddingDB::Deinit();
        bad_configuration.persistence_db_path = wrong_schema_path;
        if (hub->EnableHub(bad_configuration) == HSUCCEED) {
            _exit(44);
        }
        inspire::FaceSearchResult result{123, 123.0, MakeFeature(1)};
        if (hub->SearchFaceFeature(MakeFeature(1), result, false) != HERR_FT_HUB_DISABLE || result.id != -1 || !result.feature.empty()) {
            _exit(45);
        }
        inspire::DatabaseConfiguration good_configuration;
        good_configuration.primary_key_mode = inspire::PrimaryKeyMode::MANUAL_INPUT;
        if (hub->EnableHub(good_configuration) != HSUCCEED) {
            _exit(46);
        }
        hub->DisableHub();
        _exit(0);
    }
    std::cout << "[ISOLATION] failed_enable_rollback";
    const bool survived = pid > 0 && ChildExitedSuccessfully(pid);
    std::cout << '\n';
    guard.ObserveOrExpect(survived, options.baseline, "failed database enable terminated the process or published partial state");
    RemoveDatabaseFiles(wrong_schema_path);
#else
    std::cout << "[SKIP] failed-enable process-isolation check requires POSIX fork\n";
#endif
}

void TestBatchRollbackIsolation(const Options &options, Guard &guard) {
#if defined(__unix__) || defined(__APPLE__)
    const std::string transaction_path = options.database_path + ".transaction";
    RemoveDatabaseFiles(transaction_path);
    const pid_t pid = fork();
    if (pid == 0) {
        inspire::EmbeddingDB::Deinit();
        inspire::EmbeddingDB::Init(transaction_path, kFeatureSize, inspire::IdMode::MANUAL);
        inspire::EmbeddingDB &database = inspire::EmbeddingDB::GetInstance();
        int64_t allocated_id = -1;
        if (!database.InsertVector(2, MakeFeature(1302), allocated_id)) {
            _exit(21);
        }
        const std::vector<inspire::VectorData> batch = {
          {3, MakeFeature(1303)}, {2, MakeFeature(1304)}, {4, MakeFeature(1305)}};
        const std::vector<int64_t> inserted = database.BatchInsertVectors(batch);
        const bool rolled_back = inserted.empty() && database.GetVectorCount() == 1 && database.GetVector(3).empty() && database.GetVector(4).empty();
        inspire::EmbeddingDB::Deinit();
        _exit(rolled_back ? 0 : 22);
    }
    std::cout << "[ISOLATION] batch_rollback";
    const bool survived = pid > 0 && ChildExitedSuccessfully(pid);
    std::cout << '\n';
    guard.ObserveOrExpect(survived, options.baseline, "failed batch insert terminated the process or committed a partial batch");
    RemoveDatabaseFiles(transaction_path);
#else
    std::cout << "[SKIP] transaction process-isolation check requires POSIX fork\n";
#endif
}

void TestConcurrentLifecycleIsolation(const Options &options, Guard &guard) {
#if defined(__unix__) || defined(__APPLE__)
    const std::string lifecycle_path = options.database_path + ".lifecycle";
    RemoveDatabaseFiles(lifecycle_path);
    const pid_t pid = fork();
    if (pid == 0) {
        const auto hub = inspire::FeatureHubDB::GetInstance();
        hub->DisableHub();
        inspire::DatabaseConfiguration configuration;
        configuration.primary_key_mode = inspire::PrimaryKeyMode::MANUAL_INPUT;
        configuration.enable_persistence = true;
        configuration.persistence_db_path = lifecycle_path;
        configuration.recognition_threshold = -1.0f;
        if (hub->EnableHub(configuration) != HSUCCEED) {
            _exit(31);
        }
        const std::vector<float> query = MakeFeature(1401);
        int64_t allocated_id = -1;
        if (hub->FaceFeatureInsert(query, 42, allocated_id) != HSUCCEED) {
            _exit(32);
        }

        std::atomic<bool> start(false);
        std::atomic<bool> done(false);
        std::atomic<int> errors(0);
        std::vector<std::thread> workers;
        for (int worker_index = 0; worker_index < 4; ++worker_index) {
            workers.emplace_back([&]() {
                while (!start.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                while (!done.load(std::memory_order_acquire)) {
                    inspire::FaceSearchResult result{-999, -999.0, {}};
                    const int32_t status = hub->SearchFaceFeature(query, result, false);
                    if (status == HSUCCEED) {
                        if (result.id != 42 || !std::isfinite(result.similarity) || std::fabs(result.similarity - 1.0) > 3.0e-5) {
                            errors.fetch_add(1, std::memory_order_relaxed);
                        }
                    } else if (status != HERR_FT_HUB_DISABLE) {
                        errors.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            });
        }
        start.store(true, std::memory_order_release);
        for (int iteration = 0; iteration < 80; ++iteration) {
            if (hub->DisableHub() != HSUCCEED || hub->EnableHub(configuration) != HSUCCEED || hub->GetFaceFeatureCount() != 1) {
                errors.fetch_add(1, std::memory_order_relaxed);
            }
        }
        done.store(true, std::memory_order_release);
        for (std::thread &worker : workers) {
            worker.join();
        }
        hub->DisableHub();
        _exit(errors.load(std::memory_order_relaxed) == 0 ? 0 : 33);
    }
    std::cout << "[ISOLATION] concurrent_lifecycle";
    const bool survived = pid > 0 && ChildExitedSuccessfully(pid);
    std::cout << '\n';
    guard.ObserveOrExpect(survived, options.baseline, "concurrent search and lifecycle produced an invalid state or process failure");
    RemoveDatabaseFiles(lifecycle_path);
#else
    std::cout << "[SKIP] concurrent lifecycle process-isolation check requires POSIX fork\n";
#endif
}

}  // namespace

int main(int argc, char **argv) {
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    const Options options = ParseOptions(argc, argv);
    Guard guard;
    std::cout << "FeatureHub safety guard mode=" << (options.baseline ? "baseline" : "gate") << '\n';
    TestCosineSimilarity(options, guard);
    TestFeatureHubAccuracyAndLatency(options, guard);
    TestPersistence(options, guard);
    TestCApiCacheBoundary(options, guard);
    TestInvalidDimensionIsolation(options, guard);
    TestFailedEnableIsolation(options, guard);
    TestBatchRollbackIsolation(options, guard);
    TestConcurrentLifecycleIsolation(options, guard);

    if (guard.failures != 0) {
        std::cerr << "FeatureHub safety guard FAILED failures=" << guard.failures << '\n';
        return 1;
    }
    std::cout << "FeatureHub safety guard " << (options.baseline ? "BASELINE CAPTURED" : "PASSED") << '\n';
    return 0;
}
