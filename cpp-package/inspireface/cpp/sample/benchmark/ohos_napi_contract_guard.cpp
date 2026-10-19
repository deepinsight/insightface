#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

#include "platform/ohos/napi/ohos_napi_contract.h"

namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool condition, const char* message) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::cerr << "[OHOS-NAPI][FAIL] " << message << '\n';
    }
}

void CheckSize(HFImageFormat format, int32_t width, int32_t height, size_t expected) {
    const inspire::ohos::ImageByteSizeResult result = inspire::ohos::CalculateImageByteSize(format, width, height);
    Check(result.status == inspire::ohos::ImageByteSizeStatus::kOk, "valid format and dimensions must be accepted");
    Check(result.bytes == expected, "calculated image byte size must match the exact layout");
}

void TestImageLayouts() {
    CheckSize(HF_STREAM_RGB, 13, 17, 13U * 17U * 3U);
    CheckSize(HF_STREAM_BGR, 19, 23, 19U * 23U * 3U);
    CheckSize(HF_STREAM_RGBA, 640, 480, 640U * 480U * 4U);
    CheckSize(HF_STREAM_BGRA, 2, 4, 2U * 4U * 4U);
    CheckSize(HF_STREAM_GRAY, 31, 19, 31U * 19U);
    CheckSize(HF_STREAM_YUV_NV12, 640, 480, 640U * 480U * 3U / 2U);
    CheckSize(HF_STREAM_YUV_NV21, 128, 96, 128U * 96U * 3U / 2U);
    CheckSize(HF_STREAM_I420, 8, 6, 8U * 6U * 3U / 2U);

    Check(inspire::ohos::CalculateImageByteSize(HF_STREAM_RGB, 0, 10).status ==
            inspire::ohos::ImageByteSizeStatus::kInvalidDimensions,
          "zero width must be rejected");
    Check(inspire::ohos::CalculateImageByteSize(HF_STREAM_RGB, 10, -1).status ==
            inspire::ohos::ImageByteSizeStatus::kInvalidDimensions,
          "negative height must be rejected");
    Check(inspire::ohos::CalculateImageByteSize(HF_STREAM_YUV_NV12, 7, 8).status ==
            inspire::ohos::ImageByteSizeStatus::kOddYuvDimensions,
          "odd NV12 width must be rejected");
    Check(inspire::ohos::CalculateImageByteSize(HF_STREAM_I420, 8, 7).status ==
            inspire::ohos::ImageByteSizeStatus::kOddYuvDimensions,
          "odd I420 height must be rejected");
    Check(inspire::ohos::CalculateImageByteSize(static_cast<HFImageFormat>(99), 8, 8).status ==
            inspire::ohos::ImageByteSizeStatus::kUnsupportedFormat,
          "unknown image format must be rejected");
}

void TestCopyOwnershipAndBounds() {
    std::vector<uint8_t> source(13U * 17U * 3U);
    for (size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<uint8_t>((index * 37U + 11U) & 0xffU);
    }
    const std::vector<uint8_t> original = source;
    std::vector<uint8_t> copied;
    Check(inspire::ohos::CopyExactImageBytes(source.data(), source.size(), HF_STREAM_RGB, 13, 17, &copied),
          "exact RGB buffer must be copied");
    source.assign(source.size(), 0U);
    Check(copied == original, "native image ownership must be independent from the caller buffer");
    Check(!inspire::ohos::CopyExactImageBytes(original.data(), original.size() - 1U, HF_STREAM_RGB, 13, 17, &copied),
          "short input must be rejected");
    Check(!inspire::ohos::CopyExactImageBytes(original.data(), original.size() + 1U, HF_STREAM_RGB, 13, 17, &copied),
          "oversized input must be rejected");
    Check(!inspire::ohos::CopyExactImageBytes(nullptr, original.size(), HF_STREAM_RGB, 13, 17, &copied),
          "null input must be rejected");
    Check(!inspire::ohos::CopyExactImageBytes(original.data(), original.size(), HF_STREAM_RGB, 13, 17, nullptr),
          "null destination must be rejected");
}

void TestConcurrentCopies() {
    constexpr int kThreadCount = 4;
    constexpr int kIterations = 100;
    constexpr int32_t kWidth = 160;
    constexpr int32_t kHeight = 120;
    constexpr size_t kBytes = static_cast<size_t>(kWidth) * kHeight * 3U;
    std::atomic<int> failures{0};
    std::vector<std::thread> workers;
    workers.reserve(kThreadCount);
    for (int thread_index = 0; thread_index < kThreadCount; ++thread_index) {
        workers.emplace_back([thread_index, &failures]() {
            std::vector<uint8_t> source(kBytes, static_cast<uint8_t>(thread_index + 1));
            for (int iteration = 0; iteration < kIterations; ++iteration) {
                std::vector<uint8_t> copied;
                if (!inspire::ohos::CopyExactImageBytes(source.data(), source.size(), HF_STREAM_RGB, kWidth, kHeight,
                                                        &copied) ||
                    copied.size() != kBytes || copied.front() != static_cast<uint8_t>(thread_index + 1) ||
                    copied.back() != static_cast<uint8_t>(thread_index + 1)) {
                    ++failures;
                }
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    Check(failures.load() == 0, "parallel image copies must remain isolated and deterministic");
}

void TestCopyPerformance() {
    constexpr int32_t kWidth = 640;
    constexpr int32_t kHeight = 480;
    constexpr int kWarmupIterations = 10;
    constexpr int kIterations = 200;
    constexpr double kP95LimitMicroseconds = 20000.0;
    const size_t bytes = static_cast<size_t>(kWidth) * kHeight * 4U;
    std::vector<uint8_t> source(bytes, 0x5aU);
    std::vector<uint8_t> copied;
    for (int index = 0; index < kWarmupIterations; ++index) {
        Check(inspire::ohos::CopyExactImageBytes(source.data(), source.size(), HF_STREAM_RGBA, kWidth, kHeight, &copied),
              "performance warmup copy must succeed");
    }

    std::vector<double> samples;
    samples.reserve(kIterations);
    for (int index = 0; index < kIterations; ++index) {
        source[static_cast<size_t>(index) % source.size()] ^= static_cast<uint8_t>(index);
        const auto begin = std::chrono::steady_clock::now();
        const bool copied_ok =
          inspire::ohos::CopyExactImageBytes(source.data(), source.size(), HF_STREAM_RGBA, kWidth, kHeight, &copied);
        const auto end = std::chrono::steady_clock::now();
        Check(copied_ok, "measured image copy must succeed");
        Check(copied == source, "measured image copy content must remain exact");
        samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
    }

    std::sort(samples.begin(), samples.end());
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    const double p50 = samples[samples.size() / 2U];
    const size_t p95_index = static_cast<size_t>(std::ceil(samples.size() * 0.95)) - 1U;
    const double p95 = samples[p95_index];
    const double throughput_mib = (static_cast<double>(bytes) / (1024.0 * 1024.0)) / (mean / 1000000.0);
    std::cout << "[OHOS-NAPI][PERFORMANCE] bytes=" << bytes << " iterations=" << kIterations
              << " mean_us=" << mean << " p50_us=" << p50 << " p95_us=" << p95
              << " throughput_mib_s=" << throughput_mib << '\n';
    Check(p95 < kP95LimitMicroseconds, "RGBA bridge-copy p95 must stay below the regression limit");
}

}  // namespace

int main() {
    TestImageLayouts();
    TestCopyOwnershipAndBounds();
    TestConcurrentCopies();
    TestCopyPerformance();
    if (g_failures != 0) {
        std::cerr << "[OHOS-NAPI][RESULT] failed=" << g_failures << " checks=" << g_checks << '\n';
        return 1;
    }
    std::cout << "[OHOS-NAPI][RESULT] passed checks=" << g_checks << '\n';
    return 0;
}
