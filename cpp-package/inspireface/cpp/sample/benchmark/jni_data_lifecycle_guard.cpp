#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

#include "../../inspireface/platform/jni/common/jni_data_utils.h"

namespace {

void Require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

void TestImageSizes() {
    size_t size = 0;
    Require(inspire::jni::CheckedImageByteSize(HF_STREAM_RGB, 4, 2, &size) && size == 24, "RGB size must be exact");
    Require(inspire::jni::CheckedImageByteSize(HF_STREAM_RGBA, 4, 2, &size) && size == 32, "RGBA size must be exact");
    Require(inspire::jni::CheckedImageByteSize(HF_STREAM_GRAY, 4, 2, &size) && size == 8, "GRAY size must be exact");
    Require(inspire::jni::CheckedImageByteSize(HF_STREAM_YUV_NV21, 4, 2, &size) && size == 12, "YUV size must be exact");
    Require(!inspire::jni::CheckedImageByteSize(HF_STREAM_YUV_NV12, 3, 2, &size), "odd YUV width must fail");
    Require(!inspire::jni::CheckedImageByteSize(HF_STREAM_RGB, 0, 2, &size), "zero width must fail");
    Require(!inspire::jni::CheckedImageByteSize(999, 4, 2, &size), "unknown format must fail");
}

void TestStrideRemoval() {
    const std::vector<uint8_t> padded = {
      1, 2, 3, 4, 5, 6, 7, 8, 99, 98,
      9, 10, 11, 12, 13, 14, 15, 16, 97, 96,
    };
    std::vector<uint8_t> packed;
    Require(inspire::jni::CopyPackedRows(padded.data(), 10, 8, 2, &packed), "padded RGBA rows must copy");
    Require(packed == std::vector<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}),
            "row padding must not enter native image data");
    Require(!inspire::jni::CopyPackedRows(padded.data(), 7, 8, 2, &packed), "short stride must fail");
}

void TestRgb565Conversion() {
    const uint16_t pixels[] = {0xF800, 0x07E0, 0x001F, 0xFFFF};
    std::vector<uint8_t> source(12, 0xA5);
    std::memcpy(source.data(), pixels, sizeof(uint16_t) * 2);
    std::memcpy(source.data() + 6, pixels + 2, sizeof(uint16_t) * 2);

    std::vector<uint8_t> rgb;
    Require(inspire::jni::ConvertRgb565RowsToRgb(source.data(), 6, 2, 2, &rgb), "RGB565 conversion must succeed");
    const std::vector<uint8_t> expected = {
      255, 0, 0, 0, 255, 0,
      0, 0, 255, 255, 255, 255,
    };
    Require(rgb == expected, "RGB565 conversion must preserve canonical colors and ignore stride padding");
}

void TestPerFaceAngles() {
    float rolls[] = {1.0f, 2.0f, 3.0f};
    float yaws[] = {-1.0f, -2.0f, -3.0f};
    float pitches[] = {0.25f, 0.5f, 0.75f};
    HFFaceEulerAngle angles{rolls, yaws, pitches};
    for (int index = 0; index < 3; ++index) {
        float roll = 0.0f;
        float yaw = 0.0f;
        float pitch = 0.0f;
        Require(inspire::jni::ReadEulerAngle(angles, index, &roll, &yaw, &pitch), "angle read must succeed");
        Require(roll == rolls[index] && yaw == yaws[index] && pitch == pitches[index], "each face must use its own angles");
    }
    angles.roll = nullptr;
    float value = 0.0f;
    Require(!inspire::jni::ReadEulerAngle(angles, 0, &value, &value, &value), "missing angle storage must fail safely");
}

void TestOwnedBufferLifetimeAndConcurrency() {
    inspire::jni::OwnedImageBufferRegistry registry;
    auto buffer = std::make_shared<std::vector<uint8_t>>(16, 7);
    std::weak_ptr<std::vector<uint8_t>> weak = buffer;
    HFImageStream stream = reinterpret_cast<HFImageStream>(static_cast<uintptr_t>(1));
    Require(registry.Store(stream, buffer), "owned buffer must register");
    buffer.reset();
    Require(!weak.expired(), "registry must keep JNI image storage alive");
    Require(registry.Find(stream) && registry.Find(stream)->at(0) == 7, "registered image bytes must remain readable");
    Require(registry.Erase(stream), "registered buffer must erase");
    Require(weak.expired(), "buffer must release with its native stream");

    constexpr int threadCount = 4;
    constexpr int itemsPerThread = 1000;
    std::vector<std::thread> workers;
    for (int thread = 0; thread < threadCount; ++thread) {
        workers.emplace_back([thread, &registry]() {
            for (int item = 0; item < itemsPerThread; ++item) {
                const uintptr_t key = static_cast<uintptr_t>(2 + thread * itemsPerThread + item);
                HFImageStream handle = reinterpret_cast<HFImageStream>(key);
                auto bytes = std::make_shared<std::vector<uint8_t>>(8, static_cast<uint8_t>(thread));
                Require(registry.Store(handle, bytes), "concurrent store must succeed");
                Require(registry.Find(handle) != nullptr, "concurrent find must succeed");
                Require(registry.Erase(handle), "concurrent erase must succeed");
            }
        });
    }
    for (auto &worker : workers) {
        worker.join();
    }
    Require(registry.Size() == 0, "concurrent registry operations must leave no entries");
}

void RunCopyPerformanceGate() {
    constexpr int width = 640;
    constexpr int height = 480;
    constexpr size_t rowBytes = width * 4;
    constexpr size_t stride = rowBytes + 16;
    constexpr int iterations = 300;
    std::vector<uint8_t> source(stride * height);
    for (size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<uint8_t>((index * 17 + 3) & 0xFF);
    }
    std::vector<uint8_t> destination;
    std::vector<double> samples;
    samples.reserve(iterations);
    uint64_t digest = 1469598103934665603ULL;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        auto start = std::chrono::steady_clock::now();
        Require(inspire::jni::CopyPackedRows(source.data(), stride, rowBytes, height, &destination), "benchmark copy must succeed");
        auto end = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        digest = (digest ^ destination[static_cast<size_t>(iteration) % destination.size()]) * 1099511628211ULL;
    }
    std::sort(samples.begin(), samples.end());
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    const double p50 = samples[samples.size() / 2];
    const double p95 = samples[samples.size() * 95 / 100];
    Require(p95 < 20000.0, "640x480 ownership copy p95 must stay below 20 ms");
    std::cout << "JNI image ownership copy: iterations=" << iterations << " bytes=" << destination.size() << " digest=0x" << std::hex
              << digest << std::dec << " mean_us=" << mean << " p50_us=" << p50 << " p95_us=" << p95 << std::endl;
}

}  // namespace

int main() {
    TestImageSizes();
    TestStrideRemoval();
    TestRgb565Conversion();
    TestPerFaceAngles();
    TestOwnedBufferLifetimeAndConcurrency();
    RunCopyPerformanceGate();
    std::cout << "JNI image lifetime, layout, per-face angle, concurrency, and performance gates: PASS" << std::endl;
    return 0;
}
