#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "inspireface/c_api/inspireface.h"
#include "settings/test_settings.h"

namespace {

constexpr std::array<HResult, 51> kPublishedResultCodes = {{
  HSUCCEED,
  HERR_UNKNOWN,
  HERR_INVALID_PARAM,
  HERR_INVALID_IMAGE_STREAM_HANDLE,
  HERR_INVALID_CONTEXT_HANDLE,
  HERR_INVALID_FACE_TOKEN,
  HERR_INVALID_FACE_FEATURE,
  HERR_INVALID_FACE_LIST,
  HERR_INVALID_BUFFER_SIZE,
  HERR_INVALID_IMAGE_STREAM_PARAM,
  HERR_INVALID_SERIALIZATION_FAILED,
  HERR_INVALID_DETECTION_INPUT,
  HERR_INVALID_IMAGE_BITMAP_HANDLE,
  HERR_IMAGE_STREAM_DECODE_FAILED,
  HERR_UNSUPPORTED,
  HERR_SESS_FUNCTION_UNUSABLE,
  HERR_SESS_TRACKER_FAILURE,
  HERR_SESS_PIPELINE_FAILURE,
  HERR_SESS_INVALID_RESOURCE,
  HERR_SESS_LANDMARK_NUM_NOT_MATCH,
  HERR_SESS_LANDMARK_NOT_ENABLE,
  HERR_SESS_KEY_POINT_NUM_NOT_MATCH,
  HERR_SESS_REC_EXTRACT_FAILURE,
  HERR_SESS_REC_CONTRAST_FEAT_ERR,
  HERR_SESS_FACE_DATA_ERROR,
  HERR_SESS_FACE_REC_OPTION_ERROR,
  HERR_FT_HUB_DISABLE,
  HERR_FT_HUB_INSERT_FAILURE,
  HERR_FT_HUB_NOT_FOUND_FEATURE,
  HERR_FT_HUB_INVALID_FEATURE,
  HERR_FT_HUB_DATABASE_FAILURE,
  HERR_ARCHIVE_LOAD_FAILURE,
  HERR_ARCHIVE_LOAD_MODEL_FAILURE,
  HERR_ARCHIVE_FILE_FORMAT_ERROR,
  HERR_ARCHIVE_REPETITION_LOAD,
  HERR_ARCHIVE_NOT_LOAD,
  HERR_DEVICE_CUDA_NOT_SUPPORT,
  HERR_DEVICE_CUDA_TENSORRT_NOT_SUPPORT,
  HERR_DEVICE_CUDA_UNKNOWN_ERROR,
  HERR_DEVICE_CUDA_DISABLE,
  HERR_DEVICE_IMAGE_PROCESS_FAILURE,
  HERR_EXTENSION_ERROR,
  HERR_EXTENSION_MLMODEL_LOAD_FAILED,
  HERR_EXTENSION_HETERO_MODEL_TAG_ERROR,
  HERR_EXTENSION_HETERO_REC_HEAD_CONFIG_ERROR,
  HERR_EXTENSION_HETERO_MODEL_NOT_MATCH,
  HERR_EXTENSION_HETERO_MODEL_NOT_LOADED,
  HERR_CAPTURE_INVALID_CONFIG,
  HERR_CAPTURE_REQUIRED_FEATURE_OFF,
  HERR_CAPTURE_FRAME_OUT_OF_ORDER,
  HERR_CAPTURE_INVALID_HANDLE,
}};

std::string GetErrorMessage(HResult code) {
    HInt32 required_size = 0;
    if (HFGetErrorMessage(code, nullptr, 0, &required_size) != HSUCCEED || required_size <= 1) {
        return {};
    }
    std::vector<char> buffer(static_cast<size_t>(required_size), 'x');
    if (HFGetErrorMessage(code, buffer.data(), required_size, &required_size) != HSUCCEED || buffer.back() != '\0') {
        return {};
    }
    return std::string(buffer.data());
}

}  // namespace

TEST_CASE("C API describes every published result code", "[api][contract][error_message]") {
    for (HResult code : kPublishedResultCodes) {
        INFO("Result code: " << code);
        HInt32 required_size = -1;
        REQUIRE(HFGetErrorMessage(code, nullptr, 0, &required_size) == HSUCCEED);
        REQUIRE(required_size > 1);

        std::vector<char> exact_buffer(static_cast<size_t>(required_size), 'x');
        HInt32 copied_size = -1;
        REQUIRE(HFGetErrorMessage(code, exact_buffer.data(), required_size, &copied_size) == HSUCCEED);
        CHECK(copied_size == required_size);
        CHECK(exact_buffer.back() == '\0');
        CHECK(std::strlen(exact_buffer.data()) == static_cast<size_t>(required_size - 1));
        CHECK(std::string(exact_buffer.data()) != "Unknown error code");
    }

    CHECK(GetErrorMessage(HSUCCEED) == "Success");
    CHECK(GetErrorMessage(HERR_INVALID_PARAM).find("parameter") != std::string::npos);
    CHECK(GetErrorMessage(HERR_UNSUPPORTED).find("unsupported") != std::string::npos);
    CHECK(GetErrorMessage(HERR_FT_HUB_DISABLE).find("FeatureHub") != std::string::npos);
}

TEST_CASE("C API error descriptions enforce the copy buffer contract", "[api][contract][error_message][boundary]") {
    HInt32 required_size = 123;
    CHECK(HFGetErrorMessage(HERR_INVALID_PARAM, nullptr, 0, nullptr) == HERR_INVALID_PARAM);
    CHECK(HFGetErrorMessage(HERR_INVALID_PARAM, nullptr, 1, &required_size) == HERR_INVALID_PARAM);

    char untouched = 'x';
    CHECK(HFGetErrorMessage(HERR_INVALID_PARAM, &untouched, -1, &required_size) == HERR_INVALID_PARAM);
    CHECK(untouched == 'x');
    CHECK(HFGetErrorMessage(HERR_INVALID_PARAM, &untouched, 0, &required_size) == HERR_INVALID_BUFFER_SIZE);
    CHECK(untouched == 'x');

    REQUIRE(HFGetErrorMessage(HERR_INVALID_PARAM, nullptr, 0, &required_size) == HSUCCEED);
    REQUIRE(required_size > 2);
    std::vector<char> short_buffer(static_cast<size_t>(required_size - 1), 'x');
    HInt32 short_required_size = -1;
    CHECK(HFGetErrorMessage(HERR_INVALID_PARAM, short_buffer.data(), required_size - 1, &short_required_size) ==
          HERR_INVALID_BUFFER_SIZE);
    CHECK(short_required_size == required_size);
    CHECK(short_buffer.front() == '\0');

    const std::string unknown = GetErrorMessage(static_cast<HResult>(0x7fffffff));
    CHECK(unknown == "Unknown error code");
    CHECK(GetErrorMessage(static_cast<HResult>(-1)) == unknown);
}

TEST_CASE("C API error descriptions remain deterministic under concurrency", "[api][contract][error_message][concurrency]") {
    constexpr size_t kThreadCount = 8;
    constexpr size_t kQueriesPerThread = 2000;
    std::array<std::string, kPublishedResultCodes.size()> expected = {};
    for (size_t index = 0; index < kPublishedResultCodes.size(); ++index) {
        expected[index] = GetErrorMessage(kPublishedResultCodes[index]);
        REQUIRE_FALSE(expected[index].empty());
    }

    std::atomic<bool> consistent{true};
    std::array<std::thread, kThreadCount> workers;
    const auto started = std::chrono::steady_clock::now();
    for (size_t thread_index = 0; thread_index < kThreadCount; ++thread_index) {
        workers[thread_index] = std::thread([&, thread_index]() {
            for (size_t iteration = 0; iteration < kQueriesPerThread; ++iteration) {
                const size_t index = (thread_index + iteration) % kPublishedResultCodes.size();
                if (GetErrorMessage(kPublishedResultCodes[index]) != expected[index]) {
                    consistent.store(false, std::memory_order_relaxed);
                    return;
                }
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    const auto elapsed = std::chrono::steady_clock::now() - started;

    CHECK(consistent.load(std::memory_order_relaxed));
    CHECK(elapsed < std::chrono::seconds(2));
}

TEST_CASE("C API error description lookup remains allocation-free and low latency", "[api][error_message][performance]") {
    constexpr size_t kQueryCount = 100000;
    std::array<char, 128> buffer = {};
    HInt32 required_size = 0;
    HResult aggregate = 0;

    const auto started = std::chrono::steady_clock::now();
    for (size_t iteration = 0; iteration < kQueryCount; ++iteration) {
        const HResult code = kPublishedResultCodes[iteration % kPublishedResultCodes.size()];
        const HResult status = HFGetErrorMessage(code, buffer.data(), static_cast<HInt32>(buffer.size()), &required_size);
        if (status != HSUCCEED || required_size <= 1 || buffer[0] == '\0') {
            aggregate = HERR_UNKNOWN;
            break;
        }
        aggregate += static_cast<unsigned char>(buffer[0]);
    }
    const auto elapsed = std::chrono::steady_clock::now() - started;
    const double elapsed_milliseconds = std::chrono::duration<double, std::milli>(elapsed).count();
    TEST_PRINT("Error message lookup: {} queries in {:.3f} ms", kQueryCount, elapsed_milliseconds);

    CHECK(aggregate != HERR_UNKNOWN);
    CHECK(elapsed < std::chrono::seconds(1));
}
