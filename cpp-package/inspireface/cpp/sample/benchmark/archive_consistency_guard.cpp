#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "middleware/model_archive/core_archive/core_archive.h"
#include "middleware/model_archive/core_archive/microtar/microtar.h"
#include "middleware/model_archive/inspire_archive.h"

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

struct TemporaryArchive {
    std::string path;

    ~TemporaryArchive() {
        if (!path.empty()) {
            std::remove(path.c_str());
        }
    }
};

struct Entry {
    std::string name;
    std::vector<char> content;
};

uint64_t Hash(const std::vector<char>& data) {
    uint64_t digest = kFnvOffset;
    for (char value : data) {
        digest ^= static_cast<unsigned char>(value);
        digest *= kFnvPrime;
    }
    return digest;
}

std::vector<char> Bytes(const std::string& value) {
    return std::vector<char>(value.begin(), value.end());
}

std::string TemporaryPath(const std::string& suffix) {
    const char* root = std::getenv("TMPDIR");
    if (root == nullptr || root[0] == '\0') {
#if defined(_WIN32)
        root = ".";
#else
        root = "/tmp";
#endif
    }
    std::string path(root);
    if (!path.empty() && path.back() != '/') {
        path.push_back('/');
    }
    path += "inspireface_archive_consistency_";
    path += suffix;
    path += "_" + std::to_string(Clock::now().time_since_epoch().count()) + ".tar";
    return path;
}

bool WriteArchive(const std::vector<Entry>& entries, const std::string& suffix, TemporaryArchive& archive) {
    archive.path = TemporaryPath(suffix);
    mtar_t tar = {};
    if (mtar_open(&tar, archive.path.c_str(), "w") != MTAR_ESUCCESS) {
        return false;
    }
    bool passed = true;
    for (const auto& entry : entries) {
        if (mtar_write_file_header(&tar, entry.name.c_str(), static_cast<unsigned>(entry.content.size())) != MTAR_ESUCCESS ||
            (!entry.content.empty() &&
             mtar_write_data(&tar, entry.content.data(), static_cast<unsigned>(entry.content.size())) != MTAR_ESUCCESS)) {
            passed = false;
            break;
        }
    }
    if (passed) {
        passed = mtar_finalize(&tar) == MTAR_ESUCCESS;
    }
    mtar_close(&tar);
    return passed;
}

std::string ValidManifest(const std::string& version, const std::string& model_type = "MNN") {
    return "tag: ArchiveGuard\n"
           "version: " + version + "\n"
           "major: transactional\n"
           "release: synthetic\n"
           "face_detect_pixel_list: [160]\n"
           "face_detect_model_list: [guard_model]\n"
           "guard_model:\n"
           "  name: detector\n"
           "  fullname: detector.bin\n"
           "  version: " + version + "\n"
           "  model_type: " + model_type + "\n"
           "  infer_engine: MNN\n"
           "  infer_device: MNN\n"
           "  infer_backend: CPU\n"
           "  data_type: image\n"
           "  input_tensor_type: float32\n"
           "  output_tensor_type: float32\n"
           "  input_size: [16, 16]\n"
           "  mean: [0.0, 0.0, 0.0]\n"
           "  norm: [1.0, 1.0, 1.0]\n"
           "  outputs_layers: [output]\n";
}

std::vector<Entry> ValidEntries(const std::vector<char>& payload, const std::string& version,
                                const std::string& model_type = "MNN") {
    // The decoy intentionally precedes the real model. Historical substring
    // lookup selected this entry for the query "detector".
    return {{"__inspire__", Bytes(ValidManifest(version, model_type))},
            {"models/detector_extra", Bytes("DECOY_MODEL_BYTES")},
            {"models/detector", payload},
            {"assets/metadata", Bytes("metadata")}};
}

bool Equal(const std::shared_ptr<const std::vector<char>>& actual, const std::vector<char>& expected) {
    return actual && *actual == expected;
}

double Percentile(std::vector<double> values, double fraction) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(fraction * static_cast<double>(values.size() - 1));
    return values[index];
}

std::vector<char> ReadFile(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        return {};
    }
    const std::streamsize size = stream.tellg();
    if (size <= 0) {
        return {};
    }
    std::vector<char> content(static_cast<size_t>(size));
    stream.seekg(0, std::ios::beg);
    if (!stream.read(content.data(), size)) {
        return {};
    }
    return content;
}

void RewriteTarChecksum(std::vector<char>& archive) {
    constexpr size_t kChecksumOffset = 148;
    constexpr size_t kTypeOffset = 156;
    unsigned checksum = 256;
    for (size_t index = 0; index < kChecksumOffset; ++index) {
        checksum += static_cast<unsigned char>(archive[index]);
    }
    for (size_t index = kTypeOffset; index < 512; ++index) {
        checksum += static_cast<unsigned char>(archive[index]);
    }
    std::memset(archive.data() + kChecksumOffset, 0, 8);
    std::snprintf(archive.data() + kChecksumOffset, 8, "%06o", checksum);
    archive[kChecksumOffset + 7] = ' ';
}

bool MicrotarBoundaryGate(const TemporaryArchive& source, const std::vector<char>& expected_payload) {
    bool passed = true;
    std::vector<char> archive_bytes = ReadFile(source.path);
    if (archive_bytes.empty()) {
        return false;
    }

    mtar_t memory_archive = {};
    passed = passed && mtar_open_memory(&memory_archive, archive_bytes.data(), archive_bytes.size()) == MTAR_ESUCCESS;
    mtar_header_t header = {};
    passed = passed && mtar_find(&memory_archive, "models/detector", &header) == MTAR_ESUCCESS;
    std::vector<char> chunked(header.size);
    const unsigned first_chunk = std::min<unsigned>(17, header.size);
    passed = passed && mtar_read_data(&memory_archive, chunked.data(), first_chunk) == MTAR_ESUCCESS;
    passed = passed && mtar_read_data(&memory_archive, chunked.data() + first_chunk, header.size - first_chunk) == MTAR_ESUCCESS;
    passed = passed && chunked == expected_payload;

    passed = passed && mtar_find(&memory_archive, "models/detector", &header) == MTAR_ESUCCESS;
    passed = passed && mtar_read_data(&memory_archive, chunked.data(), header.size + 1) == MTAR_EREADFAIL;
    passed = passed && mtar_read_data(&memory_archive, chunked.data(), header.size) == MTAR_ESUCCESS;
    passed = passed && chunked == expected_payload;
    passed = passed && mtar_close(&memory_archive) == MTAR_ESUCCESS;

    std::vector<char> unterminated_name = archive_bytes;
    std::fill(unterminated_name.begin(), unterminated_name.begin() + 100, 'x');
    RewriteTarChecksum(unterminated_name);
    mtar_t malformed_archive = {};
    passed = passed && mtar_open_memory(&malformed_archive, unterminated_name.data(), unterminated_name.size()) == MTAR_EFAILURE;

    std::vector<char> invalid_size = archive_bytes;
    std::fill(invalid_size.begin() + 124, invalid_size.begin() + 136, '9');
    RewriteTarChecksum(invalid_size);
    passed = passed && mtar_open_memory(&malformed_archive, invalid_size.data(), invalid_size.size()) == MTAR_EFAILURE;
    passed = passed && mtar_open_memory(nullptr, archive_bytes.data(), archive_bytes.size()) == MTAR_EOPENFAIL;
    passed = passed && mtar_open_memory(&malformed_archive, archive_bytes.data(), 511) == MTAR_EOPENFAIL;

    TemporaryArchive write_boundary;
    write_boundary.path = TemporaryPath("microtar_boundary");
    mtar_t writer = {};
    passed = passed && mtar_open(&writer, write_boundary.path.c_str(), "w") == MTAR_ESUCCESS;
    const std::string maximum_name(99, 'a');
    const std::string oversized_name(100, 'b');
    passed = passed && mtar_write_file_header(&writer, maximum_name.c_str(), 0) == MTAR_ESUCCESS;
    passed = passed && mtar_write_file_header(&writer, oversized_name.c_str(), 0) == MTAR_EWRITEFAIL;
    passed = passed && mtar_write_file_header(&writer, "bounded", 4) == MTAR_ESUCCESS;
    const std::array<char, 5> content = {{'a', 'b', 'c', 'd', 'e'}};
    passed = passed && mtar_write_data(&writer, content.data(), content.size()) == MTAR_EWRITEFAIL;
    passed = passed && mtar_write_data(&writer, content.data(), 4) == MTAR_ESUCCESS;
    passed = passed && mtar_finalize(&writer) == MTAR_ESUCCESS;
    passed = passed && mtar_close(&writer) == MTAR_ESUCCESS;

    std::cout << "MICROTAR_BOUNDARY,chunked_read=YES,oversized_read_rejected=YES,malformed_header_rejected=YES,"
                 "bounded_write=YES,status="
              << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool ExactLookupGate(const TemporaryArchive& source, const std::vector<char>& payload) {
    inspire::CoreArchive archive(source.path);
    if (archive.QueryLoadStatus() != inspire::SARC_SUCCESS ||
        !Equal(archive.GetFileContentShared("models/detector"), payload) ||
        !Equal(archive.GetFileContentShared("detector"), payload) ||
        !Equal(archive.GetFileContentShared("models/detector_extra"), Bytes("DECOY_MODEL_BYTES")) ||
        !archive.GetFileContentShared("detect")->empty() || !archive.GetFileContentShared("detector_ex")->empty() ||
        !archive.GetFileContentShared("missing")->empty()) {
        return false;
    }

    TemporaryArchive ambiguous;
    if (!WriteArchive({{"one/shared", Bytes("one")}, {"two/shared", Bytes("two")}}, "ambiguous", ambiguous)) {
        return false;
    }
    inspire::CoreArchive ambiguous_archive(ambiguous.path);
    TemporaryArchive duplicate;
    if (!WriteArchive({{"duplicate", Bytes("first")}, {"duplicate", Bytes("second")}}, "duplicate", duplicate)) {
        return false;
    }
    inspire::CoreArchive duplicate_archive(duplicate.path);
    const bool passed = ambiguous_archive.QueryLoadStatus() == inspire::SARC_SUCCESS &&
                        ambiguous_archive.GetFileContentShared("shared")->empty() &&
                        *ambiguous_archive.GetFileContentShared("one/shared") == Bytes("one") &&
                        *ambiguous_archive.GetFileContentShared("two/shared") == Bytes("two") &&
                        duplicate_archive.QueryLoadStatus() == inspire::SARC_SUCCESS &&
                        duplicate_archive.GetFileContentShared("duplicate")->empty();
    std::cout << "ARCHIVE_EXACT_LOOKUP,decoy_before_target=YES,substring_rejected=YES,ambiguous_basename_rejected=YES,"
                 "duplicate_exact_rejected=YES,status="
              << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool ConcurrentFirstReadGate(const TemporaryArchive& source, const std::vector<char>& payload) {
    inspire::CoreArchive archive(source.path);
    constexpr int kThreads = 8;
    constexpr int kIterations = 5000;
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::atomic<bool> passed{true};
    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    const auto begin = Clock::now();
    for (int thread_index = 0; thread_index < kThreads; ++thread_index) {
        workers.emplace_back([&, thread_index]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (int iteration = 0; iteration < kIterations; ++iteration) {
                const char* query = ((iteration + thread_index) & 1) == 0 ? "detector" : "models/detector";
                if (!Equal(archive.GetFileContentShared(query), payload)) {
                    passed.store(false, std::memory_order_relaxed);
                    return;
                }
            }
        });
    }
    while (ready.load(std::memory_order_acquire) != kThreads) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (auto& worker : workers) {
        worker.join();
    }
    const double elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    const double operations_per_second = static_cast<double>(kThreads * kIterations) * 1000.0 / elapsed_ms;
    std::cout << "ARCHIVE_CONCURRENT_FIRST_READ,threads=" << kThreads << ",operations=" << (kThreads * kIterations)
              << ",ops_per_second=" << operations_per_second << ",status=" << (passed.load() ? "PASS" : "FAIL") << '\n';
    return passed.load();
}

bool ConcurrentResetGate(const TemporaryArchive& first, const TemporaryArchive& second, const std::vector<char>& first_payload,
                         const std::vector<char>& second_payload) {
    inspire::CoreArchive archive(first.path);
    std::atomic<bool> stop{false};
    std::atomic<bool> passed{true};
    std::vector<std::thread> readers;
    for (int index = 0; index < 4; ++index) {
        readers.emplace_back([&]() {
            while (!stop.load(std::memory_order_acquire)) {
                const auto value = archive.GetFileContentShared("detector");
                if (!value || (*value != first_payload && *value != second_payload)) {
                    passed.store(false, std::memory_order_relaxed);
                    return;
                }
            }
        });
    }
    for (int iteration = 0; iteration < 30 && passed.load(); ++iteration) {
        const std::string& path = (iteration & 1) == 0 ? second.path : first.path;
        if (archive.Reset(path) != inspire::SARC_SUCCESS) {
            passed.store(false);
            break;
        }
    }
    stop.store(true, std::memory_order_release);
    for (auto& reader : readers) {
        reader.join();
    }

    const auto pinned = archive.GetFileContentShared("detector");
    const std::vector<char> pinned_copy = *pinned;
    archive.Close();
    const bool lifetime_passed = *pinned == pinned_copy && archive.QueryLoadStatus() == inspire::SARC_NOT_LOAD;
    passed.store(passed.load() && lifetime_passed);
    std::cout << "ARCHIVE_SNAPSHOT_LIFETIME,resets=30,readers=4,pinned_after_close=" << (lifetime_passed ? "YES" : "NO")
              << ",status=" << (passed.load() ? "PASS" : "FAIL") << '\n';
    return passed.load();
}

bool TransactionGate(const TemporaryArchive& first, const TemporaryArchive& second, const std::vector<char>& first_payload,
                     const std::vector<char>& second_payload) {
    inspire::InspireArchive archive(first.path);
    inspire::InspireModel old_model;
    if (archive.QueryStatus() != inspire::SARC_SUCCESS || archive.LoadModel("guard_model", old_model) != inspire::SARC_SUCCESS ||
        old_model.bufferSize != first_payload.size() ||
        !std::equal(first_payload.begin(), first_payload.end(), old_model.buffer)) {
        return false;
    }

    TemporaryArchive invalid;
    const std::string invalid_manifest =
      "tag: Broken\nversion: 9\nface_detect_pixel_list: [160, 320]\nface_detect_model_list: [guard_model]\n";
    if (!WriteArchive({{"__inspire__", Bytes(invalid_manifest)}, {"models/detector", Bytes("BROKEN")}}, "invalid", invalid)) {
        return false;
    }
    const int invalid_status = archive.ReLoad(invalid.path);
    inspire::InspireModel after_failure;
    const bool rollback_passed = invalid_status == inspire::FORMAT_ERROR && archive.QueryStatus() == inspire::SARC_SUCCESS &&
                                 archive.LoadModel("guard_model", after_failure) == inspire::SARC_SUCCESS &&
                                 after_failure.bufferSize == first_payload.size() &&
                                 std::equal(first_payload.begin(), first_payload.end(), after_failure.buffer);

    const int replacement_status = archive.ReLoad(second.path);
    inspire::InspireModel replacement_model;
    const bool replacement_passed = replacement_status == inspire::SARC_SUCCESS &&
                                    archive.LoadModel("guard_model", replacement_model) == inspire::SARC_SUCCESS &&
                                    replacement_model.bufferSize == second_payload.size() &&
                                    std::equal(second_payload.begin(), second_payload.end(), replacement_model.buffer);
    const bool old_owner_passed = old_model.bufferSize == first_payload.size() &&
                                  std::equal(first_payload.begin(), first_payload.end(), old_model.buffer);

    inspire::InspireArchive copied(archive);
    copied.Release();
    inspire::InspireModel after_copy_release;
    const bool copy_isolation_passed = archive.LoadModel("guard_model", after_copy_release) == inspire::SARC_SUCCESS &&
                                       after_copy_release.bufferSize == second_payload.size() &&
                                       std::equal(second_payload.begin(), second_payload.end(), after_copy_release.buffer);
    const bool passed = rollback_passed && replacement_passed && old_owner_passed && copy_isolation_passed;
    std::cout << "ARCHIVE_TRANSACTION,failed_reload_rollback=" << (rollback_passed ? "PASS" : "FAIL")
              << ",replacement=" << (replacement_passed ? "PASS" : "FAIL")
              << ",old_model_owner=" << (old_owner_passed ? "PASS" : "FAIL")
              << ",copy_release_isolation=" << (copy_isolation_passed ? "PASS" : "FAIL")
              << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool ModelMetadataGate(const TemporaryArchive& invalid_model_archive) {
    inspire::InspireModel defaults;
    if (defaults.buffer != nullptr || defaults.bufferSize != 0 || defaults.modelType != InferenceWrapper::INFER_MNN ||
        defaults.inferEngine != InferenceWrapper::INFER_MNN || defaults.inferDevice != inspire::InspireInferEngineMNN ||
        defaults.inferBackend != inspire::InspireInferBackendCPU) {
        return false;
    }

    inspire::InspireArchive invalid_archive(invalid_model_archive.path);
    inspire::InspireModel model;
    const int status = invalid_archive.LoadModel("guard_model", model);

    const YAML::Node valid = YAML::Load("name: stable\nmodel_type: MNN\ninfer_backend: CPU\ninput_tensor_type: float32\n");
    const YAML::Node invalid = YAML::Load("name: changed\nmodel_type: TYPO\ninfer_backend: CUDA\n");
    std::vector<char> owned = Bytes("stable-buffer");
    const bool initialized = model.Reset(valid) == 0;
    model.SetBuffer(owned, owned.size());
    char* old_buffer = model.buffer;
    const size_t old_size = model.bufferSize;
    const bool rollback = model.Reset(invalid) != 0 && model.name == "stable" && model.modelType == InferenceWrapper::INFER_MNN &&
                          model.inferBackend == inspire::InspireInferBackendCPU && model.buffer == old_buffer && model.bufferSize == old_size;
    const bool passed = status != inspire::SARC_SUCCESS && initialized && rollback;
    std::cout << "ARCHIVE_MODEL_METADATA,unknown_enum_rejected=" << (status != inspire::SARC_SUCCESS ? "YES" : "NO")
              << ",reset_rollback=" << (rollback ? "PASS" : "FAIL") << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool PerformanceGate(const TemporaryArchive& source, const std::vector<char>& payload) {
    inspire::CoreArchive archive(source.path);
    if (!Equal(archive.GetFileContentShared("detector"), payload)) {
        return false;
    }
    constexpr int kIterations = 50000;
    std::vector<double> samples_us;
    samples_us.reserve(kIterations);
    uint64_t digest = kFnvOffset;
    for (int iteration = 0; iteration < kIterations; ++iteration) {
        const auto begin = Clock::now();
        const auto content = archive.GetFileContentShared((iteration & 1) == 0 ? "detector" : "models/detector");
        const auto end = Clock::now();
        if (!Equal(content, payload)) {
            return false;
        }
        samples_us.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
        digest ^= static_cast<uint64_t>(content->size() + iteration);
        digest *= kFnvPrime;
    }
    const double p50_us = Percentile(samples_us, 0.50);
    const double p95_us = Percentile(samples_us, 0.95);
    const bool passed = p95_us < 1000.0;
    std::cout << "ARCHIVE_CACHED_LOOKUP_TIMING,iterations=" << kIterations << ",p50_us=" << p50_us << ",p95_us=" << p95_us
              << ",payload_digest=0x" << std::hex << Hash(payload) << ",loop_digest=0x" << digest << std::dec
              << ",budget_us=1000,status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

}  // namespace

int main() {
    std::vector<char> first_payload(4096);
    std::vector<char> second_payload(6144);
    for (size_t i = 0; i < first_payload.size(); ++i) {
        first_payload[i] = static_cast<char>((i * 17 + 3) & 0xff);
    }
    for (size_t i = 0; i < second_payload.size(); ++i) {
        second_payload[i] = static_cast<char>((i * 29 + 11) & 0xff);
    }

    TemporaryArchive first;
    TemporaryArchive second;
    TemporaryArchive invalid_model;
    if (!WriteArchive(ValidEntries(first_payload, "1.0"), "first", first) ||
        !WriteArchive(ValidEntries(second_payload, "2.0"), "second", second) ||
        !WriteArchive(ValidEntries(first_payload, "3.0", "UNKNOWN_ENGINE"), "invalid_model", invalid_model)) {
        std::cerr << "Failed to create archive guard fixtures\n";
        return 1;
    }

    bool passed = true;
    passed = MicrotarBoundaryGate(first, first_payload) && passed;
    passed = ExactLookupGate(first, first_payload) && passed;
    passed = ConcurrentFirstReadGate(first, first_payload) && passed;
    passed = ConcurrentResetGate(first, second, first_payload, second_payload) && passed;
    passed = TransactionGate(first, second, first_payload, second_payload) && passed;
    passed = ModelMetadataGate(invalid_model) && passed;
    passed = PerformanceGate(first, first_payload) && passed;
    std::cout << "ARCHIVE_CONSISTENCY_GUARD,status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}
