#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <inspireface.h>

#include "inspireface/herror.h"
#include "inspireface/similarity_converter.h"
#include "middleware/model_archive/core_archive/microtar/microtar.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    bool baseline = false;
    int iterations = 250000;
    double max_p95_us = 0.0;
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

struct TemporaryFile {
    std::string path;

    ~TemporaryFile() {
        if (!path.empty()) {
            std::remove(path.c_str());
        }
    }
};

Options ParseOptions(int argc, char **argv) {
    Options options;
    for (int index = 2; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--baseline") {
            options.baseline = true;
        } else if (argument.find("--iterations=") == 0) {
            options.iterations = std::max(10000, std::stoi(argument.substr(std::strlen("--iterations="))));
        } else if (argument.find("--max-p95-us=") == 0) {
            options.max_p95_us = std::stod(argument.substr(std::strlen("--max-p95-us=")));
        }
    }
    return options;
}

double Percentile(std::vector<double> samples, double percentile) {
    if (samples.empty()) {
        return 0.0;
    }
    std::sort(samples.begin(), samples.end());
    const size_t index = static_cast<size_t>(std::ceil(percentile * samples.size())) - 1;
    return samples[std::min(index, samples.size() - 1)];
}

HFSimilarityConverterConfig Config(float threshold, float middle_score, float steepness, float output_min, float output_max) {
    HFSimilarityConverterConfig config{};
    config.threshold = threshold;
    config.middleScore = middle_score;
    config.steepness = steepness;
    config.outputMin = output_min;
    config.outputMax = output_max;
    return config;
}

inspire::SimilarityConverterConfig InternalConfig(const HFSimilarityConverterConfig &config) {
    inspire::SimilarityConverterConfig internal;
    internal.threshold = config.threshold;
    internal.middleScore = config.middleScore;
    internal.steepness = config.steepness;
    internal.outputMin = config.outputMin;
    internal.outputMax = config.outputMax;
    return internal;
}

bool NearlyEqual(double lhs, double rhs, double tolerance) {
    return std::isfinite(lhs) && std::isfinite(rhs) && std::abs(lhs - rhs) <= tolerance;
}

bool ConfigEqual(const HFSimilarityConverterConfig &lhs, const HFSimilarityConverterConfig &rhs, float tolerance = 1e-6f) {
    return NearlyEqual(lhs.threshold, rhs.threshold, tolerance) && NearlyEqual(lhs.middleScore, rhs.middleScore, tolerance) &&
           NearlyEqual(lhs.steepness, rhs.steepness, tolerance) && NearlyEqual(lhs.outputMin, rhs.outputMin, tolerance) &&
           NearlyEqual(lhs.outputMax, rhs.outputMax, tolerance);
}

long double ReferenceConvert(const HFSimilarityConverterConfig &config, long double cosine) {
    const long double output_scale = static_cast<long double>(config.outputMax) - config.outputMin;
    const long double bias = -std::log((static_cast<long double>(config.outputMax) - config.middleScore) /
                                       (static_cast<long double>(config.middleScore) - config.outputMin));
    const long double shifted = static_cast<long double>(config.steepness) * (cosine - config.threshold);
    const long double sigmoid = 1.0L / (1.0L + std::exp(-shifted - bias));
    return sigmoid * output_scale + config.outputMin;
}

bool ReadPublicState(HFSimilarityConverterConfig &config, float &recommended_threshold) {
    return HFGetCosineSimilarityConverter(&config) == HSUCCEED &&
           HFGetRecommendedCosineThreshold(&recommended_threshold) == HSUCCEED;
}

bool ReplaceFirst(std::string &text, const std::string &needle, const std::string &replacement, size_t begin, size_t end) {
    const size_t position = text.find(needle, begin);
    if (position == std::string::npos || position >= end) {
        return false;
    }
    text.replace(position, needle.size(), replacement);
    return true;
}

bool MutateManifest(std::string &manifest, bool invalidate_after_config) {
    const size_t converter_begin = manifest.find("similarity_converter:");
    const size_t converter_end = manifest.find("face_detect_pixel_list:", converter_begin);
    if (converter_begin == std::string::npos || converter_end == std::string::npos) {
        return false;
    }
    if (!ReplaceFirst(manifest, "threshold: 0.48", "threshold: 0.27", converter_begin, converter_end) ||
        !ReplaceFirst(manifest, "middle_score: 0.6", "middle_score: 0.73", converter_begin, converter_end) ||
        !ReplaceFirst(manifest, "steepness: 8.0", "steepness: 5.5", converter_begin, converter_end) ||
        !ReplaceFirst(manifest, "output_min: 0.01", "output_min: 0.04", converter_begin, converter_end) ||
        !ReplaceFirst(manifest, "output_max: 1.0", "output_max: 0.96", converter_begin, converter_end)) {
        return false;
    }

    if (!invalidate_after_config) {
        return true;
    }
    const size_t models_begin = manifest.find("face_detect_model_list:");
    const size_t models_end = manifest.find("face_detect_160:", models_begin);
    if (models_begin == std::string::npos || models_end == std::string::npos) {
        return false;
    }
    return ReplaceFirst(manifest, "  - face_detect_640\n", "", models_begin, models_end);
}

bool WriteArchiveEntry(mtar_t &archive, const std::string &name, const std::vector<char> &content) {
    if (mtar_write_file_header(&archive, name.c_str(), static_cast<unsigned>(content.size())) != MTAR_ESUCCESS) {
        return false;
    }
    return content.empty() || mtar_write_data(&archive, content.data(), static_cast<unsigned>(content.size())) == MTAR_ESUCCESS;
}

std::string TemporaryArchivePath() {
    const char *temporary_root = std::getenv("TMPDIR");
    if (temporary_root == nullptr || temporary_root[0] == '\0') {
#if defined(_WIN32)
        temporary_root = ".";
#else
        temporary_root = "/tmp";
#endif
    }
    return std::string(temporary_root) + "/inspireface_similarity_converter_guard_" +
           std::to_string(Clock::now().time_since_epoch().count()) + ".tar";
}

bool CreateReloadFixture(const std::string &source_path, bool invalidate_after_config, TemporaryFile &fixture) {
    mtar_t source{};
    mtar_t destination{};
    fixture.path = TemporaryArchivePath();
    if (mtar_open(&source, source_path.c_str(), "r") != MTAR_ESUCCESS) {
        return false;
    }
    if (mtar_open(&destination, fixture.path.c_str(), "w") != MTAR_ESUCCESS) {
        mtar_close(&source);
        return false;
    }

    bool passed = mtar_rewind(&source) == MTAR_ESUCCESS;
    bool manifest_found = false;
    mtar_header_t header{};
    while (passed && mtar_read_header(&source, &header) == MTAR_ESUCCESS) {
        std::vector<char> content(header.size);
        if (header.size != 0 && mtar_read_data(&source, content.data(), header.size) != MTAR_ESUCCESS) {
            passed = false;
            break;
        }
        if (std::strcmp(header.name, "__inspire__") == 0) {
            manifest_found = true;
            std::string manifest(content.begin(), content.end());
            if (!MutateManifest(manifest, invalidate_after_config)) {
                passed = false;
                break;
            }
            content.assign(manifest.begin(), manifest.end());
        }
        passed = WriteArchiveEntry(destination, header.name, content);
        if (passed && mtar_next(&source) != MTAR_ESUCCESS) {
            passed = false;
        }
    }
    passed = passed && manifest_found;
    if (passed) {
        passed = mtar_finalize(&destination) == MTAR_ESUCCESS;
    }
    mtar_close(&destination);
    mtar_close(&source);
    return passed;
}

void TestPrecision(Guard &guard) {
    const HFSimilarityConverterConfig configs[] = {Config(0.48f, 0.60f, 8.0f, 0.01f, 1.0f),
                                                   Config(0.31f, 0.67f, 4.5f, 0.03f, 0.97f),
                                                   Config(-0.10f, 0.45f, 11.0f, -0.20f, 1.20f)};
    double max_error = 0.0;
    bool monotonic = true;
    for (const auto &config : configs) {
        guard.Expect(HFUpdateCosineSimilarityConverter(config) == HSUCCEED, "valid converter configuration was rejected");
        float previous = -std::numeric_limits<float>::infinity();
        for (int index = 0; index <= 80; ++index) {
            const float cosine = -1.0f + static_cast<float>(index) * 0.025f;
            float actual = std::numeric_limits<float>::quiet_NaN();
            const HResult status = HFCosineSimilarityConvertToPercentage(cosine, &actual);
            const double expected = static_cast<double>(ReferenceConvert(config, cosine));
            max_error = std::max(max_error, std::abs(static_cast<double>(actual) - expected));
            if (status != HSUCCEED || !std::isfinite(actual) || actual + 1e-7f < previous) {
                monotonic = false;
            }
            previous = actual;
        }
    }
    guard.Expect(max_error <= 2e-6 && monotonic, "converter output differs from the long-double reference or is non-monotonic");
    std::cout << "[PRECISION] cases=" << (sizeof(configs) / sizeof(configs[0])) * 81 << " max_abs_error=" << max_error
              << " monotonic=" << monotonic << '\n';
}

void TestInvalidConfiguration(const Options &options, const HFSimilarityConverterConfig &restore_config, Guard &guard) {
    std::vector<HFSimilarityConverterConfig> invalid_configs;
    HFSimilarityConverterConfig invalid = restore_config;
    invalid.middleScore = invalid.outputMax + 0.25f;
    invalid_configs.push_back(invalid);
    invalid = restore_config;
    invalid.middleScore = invalid.outputMin;
    invalid_configs.push_back(invalid);
    invalid = restore_config;
    invalid.steepness = 0.0f;
    invalid_configs.push_back(invalid);
    invalid = restore_config;
    invalid.threshold = std::numeric_limits<float>::quiet_NaN();
    invalid_configs.push_back(invalid);
    invalid = restore_config;
    invalid.outputMax = std::numeric_limits<float>::infinity();
    invalid_configs.push_back(invalid);
    invalid = restore_config;
    invalid.outputMax = invalid.outputMin;
    invalid.middleScore = invalid.outputMin;
    invalid_configs.push_back(invalid);

    int rejected_count = 0;
    HResult first_status = HSUCCEED;
    for (size_t index = 0; index < invalid_configs.size(); ++index) {
        guard.Expect(HFUpdateCosineSimilarityConverter(restore_config) == HSUCCEED,
                     "could not restore converter before invalid-config test");
        const HResult status = HFUpdateCosineSimilarityConverter(invalid_configs[index]);
        if (index == 0) {
            first_status = status;
        }
        HFSimilarityConverterConfig after{};
        float output = std::numeric_limits<float>::quiet_NaN();
        const bool readable = HFGetCosineSimilarityConverter(&after) == HSUCCEED &&
                              HFCosineSimilarityConvertToPercentage(restore_config.threshold, &output) == HSUCCEED;
        const bool rejected = status == HERR_INVALID_PARAM && readable && ConfigEqual(after, restore_config) && std::isfinite(output);
        if (rejected) {
            ++rejected_count;
        } else {
            guard.Expect(HFUpdateCosineSimilarityConverter(restore_config) == HSUCCEED,
                         "baseline invalid config could not be repaired");
        }
    }
    guard.ObserveOrExpect(rejected_count == static_cast<int>(invalid_configs.size()), options.baseline,
                          "one or more invalid converter configs were accepted or poisoned the active curve");

    if (!options.baseline) {
        float result = 123.0f;
        guard.Expect(HFGetCosineSimilarityConverter(nullptr) == HERR_INVALID_PARAM,
                     "null converter-config output was not rejected");
        guard.Expect(HFGetRecommendedCosineThreshold(nullptr) == HERR_INVALID_PARAM,
                     "null recommended-threshold output was not rejected");
        guard.Expect(HFCosineSimilarityConvertToPercentage(0.5f, nullptr) == HERR_INVALID_PARAM,
                     "null conversion output was not rejected");
        guard.Expect(HFCosineSimilarityConvertToPercentage(std::numeric_limits<float>::quiet_NaN(), &result) == HERR_INVALID_PARAM &&
                       result == 0.0f,
                     "non-finite cosine input was not rejected deterministically");
        auto &converter = inspire::SimilarityConverter::getInstance();
        const float threshold_before = converter.getRecommendedCosineThreshold();
        guard.Expect(!converter.setRecommendedCosineThreshold(std::numeric_limits<float>::quiet_NaN()) &&
                       converter.getRecommendedCosineThreshold() == threshold_before,
                     "non-finite recommended threshold changed the active state");
    }
    std::cout << "[VALIDATION] cases=" << invalid_configs.size() << " first_status=" << first_status
              << " rejected=" << rejected_count << '\n';
}

void TestFailedReloadRollback(const std::string &pack_path, const Options &options, Guard &guard,
                              HFSimilarityConverterConfig &loaded_config) {
    guard.Expect(HFLaunchInspireFace(pack_path.c_str()) == HSUCCEED, "could not launch the reference pack");
    float loaded_threshold = 0.0f;
    guard.Expect(ReadPublicState(loaded_config, loaded_threshold), "could not read reference converter state");

    std::vector<float> expected_outputs;
    for (const float cosine : {-0.25f, 0.0f, 0.27f, 0.48f, 0.75f, 1.0f}) {
        float output = 0.0f;
        guard.Expect(HFCosineSimilarityConvertToPercentage(cosine, &output) == HSUCCEED, "reference conversion failed");
        expected_outputs.push_back(output);
    }

    TemporaryFile failed_fixture;
    guard.Expect(CreateReloadFixture(pack_path, true, failed_fixture), "could not create the failed-reload fixture");
    const HResult reload_status = failed_fixture.path.empty() ? HSUCCEED : HFReloadInspireFace(failed_fixture.path.c_str());
    HFSimilarityConverterConfig after{};
    float after_threshold = 0.0f;
    bool unchanged = reload_status != HSUCCEED && ReadPublicState(after, after_threshold) && ConfigEqual(after, loaded_config) &&
                     NearlyEqual(after_threshold, loaded_threshold, 1e-6);
    size_t output_index = 0;
    for (const float cosine : {-0.25f, 0.0f, 0.27f, 0.48f, 0.75f, 1.0f}) {
        float output = 0.0f;
        unchanged = HFCosineSimilarityConvertToPercentage(cosine, &output) == HSUCCEED &&
                    NearlyEqual(output, expected_outputs[output_index++], 1e-7) && unchanged;
    }
    guard.ObserveOrExpect(unchanged, options.baseline,
                          "failed reload changed the active similarity curve or recommended threshold");
    if (!unchanged) {
        guard.Expect(HFUpdateCosineSimilarityConverter(loaded_config) == HSUCCEED,
                     "baseline failed-reload mutation could not be repaired");
    }
    std::cout << "[ROLLBACK] reload_status=" << reload_status << " config_unchanged=" << unchanged
              << " threshold_before=" << loaded_threshold << " threshold_after=" << after_threshold << '\n';

    TemporaryFile valid_fixture;
    const HFSimilarityConverterConfig replacement = Config(0.27f, 0.73f, 5.5f, 0.04f, 0.96f);
    guard.Expect(CreateReloadFixture(pack_path, false, valid_fixture), "could not create the successful-reload fixture");
    HFSimilarityConverterConfig published{};
    float published_threshold = 0.0f;
    const HResult valid_reload_status = valid_fixture.path.empty() ? HERR_ARCHIVE_LOAD_MODEL_FAILURE
                                                                   : HFReloadInspireFace(valid_fixture.path.c_str());
    const bool published_exactly = valid_reload_status == HSUCCEED && ReadPublicState(published, published_threshold) &&
                                   ConfigEqual(published, replacement) && NearlyEqual(published_threshold, replacement.threshold, 1e-6);
    guard.Expect(published_exactly, "successful reload did not publish its complete converter state");
    guard.Expect(HFReloadInspireFace(pack_path.c_str()) == HSUCCEED, "could not restore the reference pack after reload test");
    HFSimilarityConverterConfig restored{};
    float restored_threshold = 0.0f;
    guard.Expect(ReadPublicState(restored, restored_threshold) && ConfigEqual(restored, loaded_config) &&
                   NearlyEqual(restored_threshold, loaded_threshold, 1e-6),
                 "reference pack did not restore its complete converter state");
    std::cout << "[PUBLISH] reload_status=" << valid_reload_status << " config_exact=" << published_exactly
              << " threshold=" << published_threshold << '\n';
}

void TestConcurrentState(const HFSimilarityConverterConfig &restore_config, Guard &guard) {
    auto &converter = inspire::SimilarityConverter::getInstance();
    const inspire::SimilarityConverterConfig first = InternalConfig(Config(0.22f, 0.58f, 4.0f, 0.02f, 0.92f));
    const inspire::SimilarityConverterConfig second = InternalConfig(Config(0.71f, 0.81f, 13.0f, 0.11f, 1.31f));
    const double probe = 0.43;
    inspire::SimilarityConverter first_reference(first);
    inspire::SimilarityConverter second_reference(second);
    const double expected_first = first_reference.convert(probe);
    const double expected_second = second_reference.convert(probe);
#if defined(ISF_SIMILARITY_CONVERTER_COHERENT_STATE)
    converter.updateConfigAndRecommendedThreshold(first, static_cast<float>(first.threshold));
#else
    converter.updateConfig(first);
    converter.setRecommendedCosineThreshold(static_cast<float>(first.threshold));
#endif

    std::atomic<bool> start(false);
    std::atomic<bool> done(false);
    std::atomic<int> errors(0);
    std::thread writer([&]() {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        for (int iteration = 0; iteration < 60000; ++iteration) {
#if defined(ISF_SIMILARITY_CONVERTER_COHERENT_STATE)
            converter.updateConfigAndRecommendedThreshold((iteration & 1) == 0 ? first : second,
                                                          static_cast<float>((iteration & 1) == 0 ? first.threshold : second.threshold));
#else
            const auto &config = (iteration & 1) == 0 ? first : second;
            converter.updateConfig(config);
            converter.setRecommendedCosineThreshold(static_cast<float>(config.threshold));
#endif
        }
        done.store(true, std::memory_order_release);
    });

    std::vector<std::thread> readers;
    for (int worker = 0; worker < 4; ++worker) {
        readers.emplace_back([&]() {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            while (!done.load(std::memory_order_acquire)) {
                const double actual = converter.convert(probe);
                if (!NearlyEqual(actual, expected_first, 1e-12) && !NearlyEqual(actual, expected_second, 1e-12)) {
                    errors.fetch_add(1, std::memory_order_relaxed);
                }
#if defined(ISF_SIMILARITY_CONVERTER_COHERENT_STATE)
                const auto state = converter.getState();
                const bool first_state = NearlyEqual(state.config.threshold, first.threshold, 1e-12) &&
                                         NearlyEqual(state.recommendedCosineThreshold, first.threshold, 1e-6);
                const bool second_state = NearlyEqual(state.config.threshold, second.threshold, 1e-12) &&
                                          NearlyEqual(state.recommendedCosineThreshold, second.threshold, 1e-6);
                if (!first_state && !second_state) {
                    errors.fetch_add(1, std::memory_order_relaxed);
                }
#else
                (void)converter.getRecommendedCosineThreshold();
#endif
            }
        });
    }
    start.store(true, std::memory_order_release);
    writer.join();
    for (auto &reader : readers) {
        reader.join();
    }
    guard.Expect(errors.load(std::memory_order_relaxed) == 0, "concurrent converter readers observed a torn configuration");
#if !defined(ISF_SIMILARITY_CONVERTER_COHERENT_STATE)
    std::cout << "[KNOWN-BASELINE-DEFECT] no coherent config-plus-threshold snapshot is available\n";
#endif
#if defined(ISF_SIMILARITY_CONVERTER_COHERENT_STATE)
    guard.Expect(converter.updateConfigAndRecommendedThreshold(InternalConfig(restore_config), restore_config.threshold),
                 "could not restore converter after concurrency test");
#else
    converter.updateConfig(InternalConfig(restore_config));
    converter.setRecommendedCosineThreshold(restore_config.threshold);
#endif
    std::cout << "[CONCURRENCY] readers=4 updates=60000 errors=" << errors.load(std::memory_order_relaxed) << '\n';
}

void TestPerformance(const Options &options, Guard &guard) {
    auto &converter = inspire::SimilarityConverter::getInstance();
    constexpr int kRepeats = 9;
    std::vector<double> samples;
    samples.reserve(kRepeats);
    volatile double checksum = 0.0;
    for (int repeat = 0; repeat < kRepeats; ++repeat) {
        const auto begin = Clock::now();
        for (int iteration = 0; iteration < options.iterations; ++iteration) {
            checksum += converter.convert((iteration % 2001 - 1000) / 1000.0);
        }
        const double elapsed_us = std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
        samples.push_back(elapsed_us / options.iterations);
    }
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    const double p50 = Percentile(samples, 0.50);
    const double p95 = Percentile(samples, 0.95);
    if (checksum == -1.0) {
        std::cerr << "unreachable checksum=" << checksum << '\n';
    }
    if (options.max_p95_us > 0.0) {
        guard.Expect(p95 <= options.max_p95_us, "converter p95 latency exceeded the pre-change gate");
    }
    std::cout << "[PERF] iterations=" << options.iterations << " mean_us=" << mean << " p50_us=" << p50 << " p95_us=" << p95 << '\n';
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <pack_path> [--baseline] [--iterations=N] [--max-p95-us=N]\n";
        return 2;
    }
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    const Options options = ParseOptions(argc, argv);
    Guard guard;
    HFSetLogLevel(HF_LOG_NONE);
    std::cout << "SimilarityConverter guard mode=" << (options.baseline ? "baseline" : "gate") << '\n';

    HFSimilarityConverterConfig loaded_config{};
    TestFailedReloadRollback(argv[1], options, guard, loaded_config);
    TestPrecision(guard);
    TestInvalidConfiguration(options, loaded_config, guard);
    TestConcurrentState(loaded_config, guard);
    TestPerformance(options, guard);
    guard.Expect(HFTerminateInspireFace() == HSUCCEED, "could not terminate InspireFace after converter guard");

    if (guard.failures != 0) {
        std::cerr << "SimilarityConverter guard FAILED failures=" << guard.failures << '\n';
        return 1;
    }
    std::cout << "SimilarityConverter guard " << (options.baseline ? "BASELINE CAPTURED" : "PASSED") << '\n';
    return 0;
}
