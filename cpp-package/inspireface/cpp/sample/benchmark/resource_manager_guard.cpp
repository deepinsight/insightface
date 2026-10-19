#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <fstream>
#include <unistd.h>
#endif

#if defined(__unix__) || defined(__APPLE__)
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "inspireface.h"
#include "inspireface/herror.h"
#include "inspireface/runtime_module/resource_manage.h"

namespace {

struct Options {
    bool baseline = false;
    size_t churn_iterations = 300000;
    double max_late_p95_us = 0.0;
    uint64_t max_rss_growth_bytes = 0;
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

Options ParseOptions(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--baseline") {
            options.baseline = true;
        } else if (argument.find("--churn=") == 0) {
            options.churn_iterations = std::max<size_t>(10000, std::stoull(argument.substr(std::strlen("--churn="))));
        } else if (argument.find("--max-late-p95-us=") == 0) {
            options.max_late_p95_us = std::stod(argument.substr(std::strlen("--max-late-p95-us=")));
        } else if (argument.find("--max-rss-growth-bytes=") == 0) {
            options.max_rss_growth_bytes = std::stoull(argument.substr(std::strlen("--max-rss-growth-bytes=")));
        }
    }
    return options;
}

double Percentile(std::vector<double> values, double percentile) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(std::ceil(percentile * values.size())) - 1;
    return values[std::min(index, values.size() - 1)];
}

Timings Summarize(const std::vector<double> &values) {
    Timings timings;
    if (!values.empty()) {
        timings.mean_us = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        timings.p50_us = Percentile(values, 0.50);
        timings.p95_us = Percentile(values, 0.95);
    }
    return timings;
}

uint64_t ResidentBytes() {
#if defined(__APPLE__)
    mach_task_basic_info_data_t information{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&information), &count) != KERN_SUCCESS) {
        return 0;
    }
    return static_cast<uint64_t>(information.resident_size);
#elif defined(__linux__)
    std::ifstream statistics("/proc/self/statm");
    uint64_t total_pages = 0;
    uint64_t resident_pages = 0;
    if (!(statistics >> total_pages >> resident_pages)) {
        return 0;
    }
    return resident_pages * static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
#else
    return 0;
#endif
}

template <typename Values>
bool ContainsHandle(const Values &values, uintptr_t handle) {
    return std::any_of(values.begin(), values.end(), [handle](const typename Values::value_type value) {
        return static_cast<uintptr_t>(value) == handle;
    });
}

void TestManagerSemantics(const Options &options, Guard &guard) {
    auto *manager = RESOURCE_MANAGE;
    const uintptr_t low_handle = 0x13579u;
    const uintptr_t high_handle = sizeof(uintptr_t) > 4 ? (uintptr_t(1) << 40) | low_handle : 0x24680u;

    const inspire::ResourceStatistics before = manager->getResourceStatistics();
    guard.Expect(!manager->createSession(0), "null synthetic session registration succeeded");
    guard.Expect(manager->createSession(low_handle), "first synthetic session registration failed");
    guard.Expect(!manager->createSession(low_handle), "duplicate live synthetic session registration succeeded");
    guard.Expect(manager->createSession(high_handle), "high-bit synthetic session registration failed");
    const auto sessions = manager->getUnreleasedSessions();
    const bool width_safe = sessions.size() == 2 && ContainsHandle(sessions, low_handle) && ContainsHandle(sessions, high_handle);
    std::cout << "[HANDLE] pointer_bytes=" << sizeof(void *) << " long_bytes=" << sizeof(long) << " distinct=" << width_safe << '\n';
    guard.ObserveOrExpect(width_safe, options.baseline, "resource handles lose their upper pointer bits");
    const inspire::ResourceStatistics registered = manager->getResourceStatistics();
    guard.Expect(registered.sessions.total_created == before.sessions.total_created + 2 &&
                   registered.sessions.total_released == before.sessions.total_released && registered.sessions.live == 2,
                 "session lifecycle counters changed on a rejected registration");
    guard.Expect(manager->releaseSession(low_handle), "first release of a live synthetic session failed");
    guard.Expect(manager->releaseSession(high_handle) == width_safe, "high-bit synthetic session release disagrees with registration");
    guard.Expect(!manager->releaseSession(low_handle), "double release of a synthetic session succeeded");

    manager->createSession(low_handle);
    guard.Expect(manager->releaseSession(low_handle), "address reuse could not be registered and released");

    manager->createStream(low_handle + 1);
    manager->createImageBitmap(low_handle + 2);
    manager->createFaceFeature(low_handle + 3);
    guard.Expect(ContainsHandle(manager->getUnreleasedStreams(), low_handle + 1), "stream snapshot lost a live handle");
    guard.Expect(ContainsHandle(manager->getUnreleasedImageBitmaps(), low_handle + 2), "bitmap snapshot lost a live handle");
    guard.Expect(ContainsHandle(manager->getUnreleasedFaceFeatures(), low_handle + 3), "feature snapshot lost a live handle");
    guard.Expect(manager->releaseStream(low_handle + 1), "synthetic stream release failed");
    guard.Expect(manager->releaseImageBitmap(low_handle + 2), "synthetic bitmap release failed");
    guard.Expect(manager->releaseFaceFeature(low_handle + 3), "synthetic feature release failed");

    const uintptr_t owned_handle = high_handle + 0x100u;
    int destruction_count = 0;
    std::shared_ptr<void> owner(new uint8_t(0), [&](void *object) {
        delete static_cast<uint8_t *>(object);
        ++destruction_count;
    });
    guard.Expect(manager->createSession(owned_handle, owner), "owned session registration failed");
    owner.reset();
    auto lease = manager->acquireSession(owned_handle);
    guard.Expect(static_cast<bool>(lease), "live owned session could not be leased");
    guard.Expect(!manager->acquireStream(owned_handle), "session handle acquired through the stream registry");
    guard.Expect(manager->releaseSession(owned_handle), "leased session release failed");
    guard.Expect(!manager->isSessionLive(owned_handle), "released leased session remained publicly live");
    guard.Expect(destruction_count == 0, "release destroyed an object while a call lease was active");
    lease = {};
    guard.Expect(destruction_count == 1, "final lease did not destroy the released object exactly once");
}

void TestCApiBasicLifecycle(Guard &guard) {
    HFSessionCustomParameter parameters{};
    guard.Expect(HFCreateInspireFaceSession(parameters, HF_DETECT_MODE_ALWAYS_DETECT, 1, 320, -1, nullptr) == HERR_INVALID_PARAM,
                 "session creation accepted a null output handle");
    guard.Expect(HFCreateInspireFaceSessionOptional(HF_ENABLE_NONE, HF_DETECT_MODE_ALWAYS_DETECT, 1, 320, -1, nullptr) ==
                   HERR_INVALID_PARAM,
                 "optional session creation accepted a null output handle");

    HFImageStream stream = nullptr;
    guard.Expect(HFCreateImageStreamEmpty(&stream) == HSUCCEED && stream != nullptr, "C API stream creation failed");
    guard.Expect(HFReleaseImageStream(stream) == HSUCCEED, "C API stream release failed");
    guard.Expect(HFReleaseImageStream(stream) == HERR_INVALID_IMAGE_STREAM_HANDLE, "C API stream double release succeeded");

    uint8_t pixels[3] = {1, 2, 3};
    HFImageBitmapData bitmap_data{};
    bitmap_data.data = pixels;
    bitmap_data.width = 1;
    bitmap_data.height = 1;
    bitmap_data.channels = 3;
    HFImageBitmap bitmap = nullptr;
    guard.Expect(HFCreateImageBitmap(&bitmap_data, &bitmap) == HSUCCEED && bitmap != nullptr, "C API bitmap creation failed");
    guard.Expect(HFReleaseImageBitmap(bitmap) == HSUCCEED, "C API bitmap release failed");
    guard.Expect(HFReleaseImageBitmap(bitmap) == HERR_INVALID_IMAGE_BITMAP_HANDLE, "C API bitmap double release succeeded");

    HFImageData invalid_image{};
    invalid_image.data = pixels;
    invalid_image.width = 1;
    invalid_image.height = 1;
    invalid_image.format = static_cast<HFImageFormat>(-1);
    HFImageStream invalid_stream = reinterpret_cast<HFImageStream>(uintptr_t(0x55AAu));
    guard.Expect(HFCreateImageStream(&invalid_image, &invalid_stream) == HERR_INVALID_IMAGE_STREAM_PARAM && invalid_stream == nullptr,
                 "invalid image format published a stream handle");

    HInt32 stream_count = -1;
    guard.Expect(HFDeBugGetUnreleasedStreamsCount(&stream_count) == HSUCCEED && stream_count == 0,
                 "released C API streams remain in the live snapshot");
}

void TestConcurrentManager(Guard &guard) {
    auto *manager = RESOURCE_MANAGE;
    constexpr int kWorkers = 8;
    constexpr int kIterations = 12000;
    std::atomic<bool> start(false);
    std::atomic<bool> done(false);
    std::atomic<int> errors(0);
    std::thread observer([&]() {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        while (!done.load(std::memory_order_acquire)) {
            const auto sessions = manager->getUnreleasedSessions();
            const auto streams = manager->getUnreleasedStreams();
            if (sessions.size() > static_cast<size_t>(kWorkers) || streams.size() > static_cast<size_t>(kWorkers)) {
                errors.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    std::vector<std::thread> workers;
    for (int worker = 0; worker < kWorkers; ++worker) {
        workers.emplace_back([&, worker]() {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (int iteration = 0; iteration < kIterations; ++iteration) {
                const uintptr_t handle = (uintptr_t(0x200000) << 20) | (static_cast<uintptr_t>(worker) << 32) |
                                         static_cast<uintptr_t>(iteration + 1);
                if ((iteration & 1) == 0) {
                    manager->createSession(handle);
                    if (!manager->releaseSession(handle)) {
                        errors.fetch_add(1, std::memory_order_relaxed);
                    }
                } else {
                    manager->createStream(handle);
                    if (!manager->releaseStream(handle)) {
                        errors.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (std::thread &worker : workers) {
        worker.join();
    }
    done.store(true, std::memory_order_release);
    observer.join();
    guard.Expect(errors.load(std::memory_order_relaxed) == 0, "concurrent ResourceManager operations produced an inconsistent result");
    guard.Expect(manager->getUnreleasedSessions().empty() && manager->getUnreleasedStreams().empty(),
                 "concurrent ResourceManager operations left live handles behind");
}

void TestConcurrentCApiStreams(Guard &guard) {
    constexpr int kWorkers = 4;
    constexpr int kIterations = 1500;
    std::atomic<int> errors(0);
    std::vector<std::thread> workers;
    for (int worker = 0; worker < kWorkers; ++worker) {
        workers.emplace_back([&]() {
            for (int iteration = 0; iteration < kIterations; ++iteration) {
                HFImageStream stream = nullptr;
                if (HFCreateImageStreamEmpty(&stream) != HSUCCEED || !stream || HFReleaseImageStream(stream) != HSUCCEED) {
                    errors.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (std::thread &worker : workers) {
        worker.join();
    }
    HInt32 count = -1;
    guard.Expect(errors.load(std::memory_order_relaxed) == 0 && HFDeBugGetUnreleasedStreamsCount(&count) == HSUCCEED && count == 0,
                 "concurrent C API stream lifecycle did not return to zero live handles");
}

#if defined(__unix__) || defined(__APPLE__)
bool ChildExitedSuccessfully(pid_t pid) {
    int status = 0;
    if (pid <= 0 || waitpid(pid, &status, 0) != pid) {
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

void TestFaceFeatureDoubleReleaseIsolation(const Options &options, Guard &guard) {
#if defined(__unix__) || defined(__APPLE__)
    const pid_t pid = fork();
    if (pid == 0) {
        HFFaceFeature feature{};
        if (HFCreateFaceFeature(&feature) != HSUCCEED || !feature.data || feature.size != 512) {
            _exit(11);
        }
        HFloat *const original_data = feature.data;
        if (!options.baseline &&
            (HFCreateFaceFeature(&feature) != HERR_INVALID_FACE_FEATURE || feature.data != original_data || feature.size != 512)) {
            _exit(15);
        }
        if (HFReleaseFaceFeature(&feature) != HSUCCEED) {
            _exit(12);
        }
        if (!options.baseline && (feature.data != nullptr || feature.size != 0)) {
            _exit(13);
        }
        if (HFReleaseFaceFeature(&feature) != HERR_INVALID_FACE_FEATURE) {
            _exit(14);
        }
        _exit(0);
    }
    std::cout << "[ISOLATION] face_feature_double_release";
    const bool safe = ChildExitedSuccessfully(pid);
    std::cout << '\n';
    guard.ObserveOrExpect(safe, options.baseline, "face feature double release corrupted or terminated the process");
#else
    std::cout << "[SKIP] face-feature process-isolation check requires POSIX fork\n";
#endif
}

void TestDebugCapacityIsolation(const Options &options, Guard &guard) {
#if defined(__unix__) || defined(__APPLE__)
    const pid_t pid = fork();
    if (pid == 0) {
        HFImageStream first = nullptr;
        HFImageStream second = nullptr;
        if (HFCreateImageStreamEmpty(&first) != HSUCCEED || HFCreateImageStreamEmpty(&second) != HSUCCEED) {
            _exit(21);
        }
        const HFImageStream sentinel = reinterpret_cast<HFImageStream>(uintptr_t(0x55AA55AAu));
        HFImageStream output[5] = {sentinel, sentinel, sentinel, sentinel, sentinel};
        if (HFDeBugGetUnreleasedStreams(output, 5) != HSUCCEED || output[0] == sentinel || output[1] == sentinel || output[2] != sentinel ||
            output[3] != sentinel || output[4] != sentinel) {
            _exit(22);
        }
        if (HFReleaseImageStream(first) != HSUCCEED || HFReleaseImageStream(second) != HSUCCEED) {
            _exit(23);
        }
        if (HFDeBugGetUnreleasedStreams(nullptr, 1) != HERR_INVALID_PARAM || HFDeBugGetUnreleasedStreams(nullptr, -1) != HERR_INVALID_PARAM ||
            HFDeBugGetUnreleasedStreams(nullptr, 0) != HSUCCEED || HFDeBugGetUnreleasedStreamsCount(nullptr) != HERR_INVALID_PARAM ||
            HFDeBugGetUnreleasedSessions(nullptr, 1) != HERR_INVALID_PARAM || HFDeBugGetUnreleasedSessionsCount(nullptr) != HERR_INVALID_PARAM) {
            _exit(24);
        }
        _exit(0);
    }
    std::cout << "[ISOLATION] debug_capacity";
    const bool safe = ChildExitedSuccessfully(pid);
    std::cout << '\n';
    guard.ObserveOrExpect(safe, options.baseline, "debug snapshot capacity or null validation caused an out-of-bounds access");
#else
    std::cout << "[SKIP] debug-capacity process-isolation check requires POSIX fork\n";
#endif
}

Timings MeasureSessionChurn(uintptr_t first_handle, int iterations, int repeats, Guard &guard) {
    std::vector<double> samples;
    samples.reserve(repeats);
    volatile uintptr_t checksum = 0;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const uintptr_t repeat_base = first_handle + static_cast<uintptr_t>(repeat) * static_cast<uintptr_t>(iterations + 1);
        const auto start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const uintptr_t handle = repeat_base + static_cast<uintptr_t>(iteration + 1);
            RESOURCE_MANAGE->createSession(handle);
            if (!RESOURCE_MANAGE->releaseSession(handle)) {
                guard.Expect(false, "timed session churn failed to release a registered handle");
                break;
            }
            checksum ^= handle;
        }
        const double elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        samples.push_back(elapsed / iterations);
    }
    if (checksum == 0xFFFFFFFFu) {
        std::cerr << "unreachable checksum=" << checksum << '\n';
    }
    return Summarize(samples);
}

void GrowHistoricalRegistries(size_t iterations, Guard &guard) {
    auto *manager = RESOURCE_MANAGE;
    const uintptr_t base = sizeof(uintptr_t) > 4 ? uintptr_t(0x40000000000ULL) : uintptr_t(0x40000000u);
    for (size_t i = 0; i < iterations; ++i) {
        const uintptr_t handle = base + i + 1;
        switch (i & 3u) {
            case 0:
                manager->createSession(handle);
                guard.Expect(manager->releaseSession(handle), "growth churn failed for a session");
                break;
            case 1:
                manager->createStream(handle);
                guard.Expect(manager->releaseStream(handle), "growth churn failed for a stream");
                break;
            case 2:
                manager->createImageBitmap(handle);
                guard.Expect(manager->releaseImageBitmap(handle), "growth churn failed for a bitmap");
                break;
            default:
                manager->createFaceFeature(handle);
                guard.Expect(manager->releaseFaceFeature(handle), "growth churn failed for a face feature");
                break;
        }
    }
}

void TestGrowthAndPerformance(const Options &options, Guard &guard) {
    const uint64_t resident_before = ResidentBytes();
    const Timings early = MeasureSessionChurn(uintptr_t(0x50000000000ULL), 12000, 7, guard);
    GrowHistoricalRegistries(options.churn_iterations, guard);
    const Timings late = MeasureSessionChurn(uintptr_t(0x60000000000ULL), 12000, 7, guard);
    const uint64_t resident_after = ResidentBytes();
    const uint64_t resident_growth = resident_after > resident_before ? resident_after - resident_before : 0;

    std::cout << "[PERF] early_mean_us=" << early.mean_us << " early_p50_us=" << early.p50_us << " early_p95_us=" << early.p95_us
              << " late_mean_us=" << late.mean_us << " late_p50_us=" << late.p50_us << " late_p95_us=" << late.p95_us << '\n';
    std::cout << "[MEMORY] churn=" << options.churn_iterations << " rss_before=" << resident_before << " rss_after=" << resident_after
              << " rss_growth=" << resident_growth << '\n';

    guard.Expect(RESOURCE_MANAGE->getUnreleasedSessions().empty() && RESOURCE_MANAGE->getUnreleasedStreams().empty() &&
                   RESOURCE_MANAGE->getUnreleasedImageBitmaps().empty() && RESOURCE_MANAGE->getUnreleasedFaceFeatures().empty(),
                 "sequential churn left live resources behind");
    if (options.max_late_p95_us > 0.0) {
        guard.Expect(late.p95_us <= options.max_late_p95_us, "late ResourceManager latency exceeded the baseline gate");
    }
    if (options.max_rss_growth_bytes > 0 && resident_before != 0 && resident_after != 0) {
        guard.Expect(resident_growth <= options.max_rss_growth_bytes, "ResourceManager RSS growth exceeded the baseline gate");
    }
}

}  // namespace

int main(int argc, char **argv) {
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    const Options options = ParseOptions(argc, argv);
    Guard guard;
    std::cout << "ResourceManager guard mode=" << (options.baseline ? "baseline" : "gate") << '\n';
    TestManagerSemantics(options, guard);
    TestCApiBasicLifecycle(guard);
    TestConcurrentManager(guard);
    TestConcurrentCApiStreams(guard);
    TestFaceFeatureDoubleReleaseIsolation(options, guard);
    TestDebugCapacityIsolation(options, guard);
    TestGrowthAndPerformance(options, guard);

    if (guard.failures != 0) {
        std::cerr << "ResourceManager guard FAILED failures=" << guard.failures << '\n';
        return 1;
    }
    std::cout << "ResourceManager guard " << (options.baseline ? "BASELINE CAPTURED" : "PASSED") << '\n';
    return 0;
}
