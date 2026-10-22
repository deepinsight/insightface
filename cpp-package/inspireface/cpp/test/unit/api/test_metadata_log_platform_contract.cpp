#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "inspireface/c_api/inspireface.h"
#include "settings/test_settings.h"

TEST_CASE("C API metadata outputs are initialized and null terminated", "[api][contract][metadata]") {
    HFInspireFaceVersion version = {-1, -1, -1};
    REQUIRE(HFQueryInspireFaceVersion(&version) == HSUCCEED);
    CHECK(version.major >= 0);
    CHECK(version.minor >= 0);
    CHECK(version.patch >= 0);
    CHECK(HFQueryInspireFaceVersion(nullptr) == HERR_INVALID_PARAM);

    HFInspireFaceExtendedInformation information;
    std::fill(std::begin(information.information), std::end(information.information), static_cast<char>(0x7f));
    REQUIRE(HFQueryInspireFaceExtendedInformation(&information) == HSUCCEED);
    CHECK(std::memchr(information.information, '\0', sizeof(information.information)) != nullptr);
    CHECK(std::strlen(information.information) > 0);
    CHECK(HFQueryInspireFaceExtendedInformation(nullptr) == HERR_INVALID_PARAM);
}

TEST_CASE("C API component versions expose structured and copyable diagnostics", "[api][contract][metadata]") {
    static const std::array<const char*, HF_COMPONENT_COUNT> component_names = {{
      "mnn", "inspirecv", "eigen", "sqlite", "sqlite_vec", "nlohmann_json",
      "opencv", "tensorrt", "cuda", "rknn", "rga", "coreml",
    }};

    HInt32 required_size = -1;
    REQUIRE(HFQueryInspireFaceComponentVersions(nullptr, 0, &required_size) == HSUCCEED);
    REQUIRE(required_size > 1);
    CHECK(HFQueryInspireFaceComponentVersions(nullptr, 1, &required_size) == HERR_INVALID_PARAM);
    CHECK(HFQueryInspireFaceComponentVersions(nullptr, 0, nullptr) == HERR_INVALID_PARAM);
    char zero_capacity_buffer = 'x';
    CHECK(HFQueryInspireFaceComponentVersions(&zero_capacity_buffer, 0, &required_size) == HERR_INVALID_BUFFER_SIZE);
    CHECK(zero_capacity_buffer == 'x');
    CHECK(HFQueryInspireFaceComponentVersions(&zero_capacity_buffer, -1, &required_size) == HERR_INVALID_PARAM);

    std::array<char, 4> short_buffer = {{'x', 'x', 'x', '\0'}};
    CHECK(HFQueryInspireFaceComponentVersions(short_buffer.data(), short_buffer.size(), &required_size) == HERR_INVALID_BUFFER_SIZE);
    CHECK(short_buffer[0] == '\0');

    std::vector<char> buffer(static_cast<size_t>(required_size), 'x');
    HInt32 copied_size = -1;
    REQUIRE(HFQueryInspireFaceComponentVersions(buffer.data(), static_cast<HInt32>(buffer.size()), &copied_size) == HSUCCEED);
    REQUIRE(copied_size == required_size);
    REQUIRE(buffer.back() == '\0');
    const std::string diagnostics(buffer.data());
    CHECK(diagnostics.find("inspireface=") == 0);

    HInt32 diagnostic_size = 0;
    REQUIRE(HFQueryInspireFaceDiagnosticInformation(nullptr, 0, &diagnostic_size) == HSUCCEED);
    REQUIRE(diagnostic_size > required_size);
    std::vector<char> diagnostic_buffer(static_cast<size_t>(diagnostic_size));
    REQUIRE(HFQueryInspireFaceDiagnosticInformation(diagnostic_buffer.data(), diagnostic_size, &diagnostic_size) == HSUCCEED);
    const std::string full_diagnostics(diagnostic_buffer.data());
    CHECK(full_diagnostics.find("InspireFace SDK ") == 0);
    CHECK(full_diagnostics.find("\nComponents: " + diagnostics) != std::string::npos);
    CHECK(HFQueryInspireFaceDiagnosticInformation(nullptr, 0, nullptr) == HERR_INVALID_PARAM);

    for (int index = 0; index < HF_COMPONENT_COUNT; ++index) {
        HFComponentVersion component_version = {-1, -1, -1, static_cast<HFComponentVersionState>(-1)};
        REQUIRE(HFQueryInspireFaceComponentVersion(static_cast<HFComponentType>(index), &component_version) == HSUCCEED);
        REQUIRE((component_version.state == HF_COMPONENT_VERSION_DISABLED || component_version.state == HF_COMPONENT_VERSION_KNOWN ||
                 component_version.state == HF_COMPONENT_VERSION_UNKNOWN));
        const std::string rendered = component_version.state == HF_COMPONENT_VERSION_DISABLED
                                       ? "disabled"
                                       : component_version.state == HF_COMPONENT_VERSION_UNKNOWN
                                           ? "unknown"
                                           : std::to_string(component_version.major) + "." + std::to_string(component_version.minor) + "." +
                                               std::to_string(component_version.patch);
        CHECK(diagnostics.find(std::string(component_names[static_cast<size_t>(index)]) + "=" + rendered) != std::string::npos);
        if (index <= HF_COMPONENT_NLOHMANN_JSON) {
            CHECK(component_version.state == HF_COMPONENT_VERSION_KNOWN);
        }
        if (component_version.state == HF_COMPONENT_VERSION_KNOWN) {
            CHECK(component_version.major >= 0);
            CHECK(component_version.minor >= 0);
            CHECK(component_version.patch >= 0);
        } else {
            CHECK(component_version.major == 0);
            CHECK(component_version.minor == 0);
            CHECK(component_version.patch == 0);
        }
    }

    HFComponentVersion version = {};
    CHECK(HFQueryInspireFaceComponentVersion(static_cast<HFComponentType>(-1), &version) == HERR_INVALID_PARAM);
    CHECK(HFQueryInspireFaceComponentVersion(HF_COMPONENT_COUNT, &version) == HERR_INVALID_PARAM);
    CHECK(HFQueryInspireFaceComponentVersion(HF_COMPONENT_MNN, nullptr) == HERR_INVALID_PARAM);
}

TEST_CASE("C API logging validates levels and safely truncates long messages", "[api][contract][logging]") {
    CHECK(HFSetLogLevel(static_cast<HFLogLevel>(-1)) == HERR_INVALID_PARAM);
    CHECK(HFSetLogLevel(static_cast<HFLogLevel>(99)) == HERR_INVALID_PARAM);
    REQUIRE(HFSetLogLevel(HF_LOG_ERROR) == HSUCCEED);
    CHECK(HFLogPrint(HF_LOG_INFO, "filtered message") == HSUCCEED);
    CHECK(HFLogPrint(static_cast<HFLogLevel>(99), "invalid") == HERR_INVALID_PARAM);
    CHECK(HFLogPrint(HF_LOG_ERROR, nullptr) == HERR_INVALID_PARAM);

    const std::string long_message(4096, 'x');
    CHECK(HFLogPrint(HF_LOG_ERROR, "%s", long_message.c_str()) == HSUCCEED);
    CHECK(HFLogDisable() == HSUCCEED);
    CHECK(HFLogPrint(HF_LOG_ERROR, "disabled message") == HSUCCEED);
    CHECK(HFSetLogLevel(HF_LOG_INFO) == HSUCCEED);
}

TEST_CASE("C API platform configuration reports supported or disabled behavior", "[api][contract][platform]") {
    HInt32 rga_enabled = -1;
    REQUIRE(HFQueryExpansiveHardwareRGACompileOption(&rga_enabled) == HSUCCEED);
    CHECK((rga_enabled == 0 || rga_enabled == 1));
    CHECK(HFQueryExpansiveHardwareRGACompileOption(nullptr) == HERR_INVALID_PARAM);
    CHECK(HFSwitchLandmarkEngine(HF_LANDMARK_HYPLMV2_0_25) == HSUCCEED);
    CHECK(HFSwitchLandmarkEngine(static_cast<HFSessionLandmarkEngine>(99)) == HERR_INVALID_PARAM);

    std::array<char, 1024> original_path = {};
    REQUIRE(HFQueryExpansiveHardwareRockchipDmaHeapPath(original_path.data()) == HSUCCEED);
    REQUIRE(HFSetExpansiveHardwareRockchipDmaHeapPath("/dev/dma_heap/contract-test") == HSUCCEED);
    std::array<char, 4> short_path = {{'x', 'x', 'x', '\0'}};
    CHECK(HFQueryExpansiveHardwareRockchipDmaHeapPathWithSize(short_path.data(), short_path.size()) == HERR_INVALID_BUFFER_SIZE);
    CHECK(short_path[0] == '\0');
    CHECK(HFQueryExpansiveHardwareRockchipDmaHeapPathWithSize(nullptr, 256) == HERR_INVALID_PARAM);
    CHECK(HFQueryExpansiveHardwareRockchipDmaHeapPathWithSize(original_path.data(), 0) == HERR_INVALID_PARAM);
    std::array<char, 1024> actual_path = {};
    REQUIRE(HFQueryExpansiveHardwareRockchipDmaHeapPathWithSize(actual_path.data(), actual_path.size()) == HSUCCEED);
    CHECK(std::string(actual_path.data()) == "/dev/dma_heap/contract-test");
    CHECK(HFSetExpansiveHardwareRockchipDmaHeapPath(nullptr) == HERR_INVALID_PARAM);
    CHECK(HFSetExpansiveHardwareRockchipDmaHeapPath("") == HSUCCEED);
    const std::string oversized_path(256, 'x');
    CHECK(HFSetExpansiveHardwareRockchipDmaHeapPath(oversized_path.c_str()) == HERR_INVALID_PARAM);
    CHECK(HFQueryExpansiveHardwareRockchipDmaHeapPath(nullptr) == HERR_INVALID_PARAM);
    REQUIRE(HFSetExpansiveHardwareRockchipDmaHeapPath(original_path.data()) == HSUCCEED);

    CHECK(HFSetAppleCoreMLInferenceMode(static_cast<HFAppleCoreMLInferenceMode>(99)) == HERR_INVALID_PARAM);
    CHECK(HFSwitchImageProcessingBackend(static_cast<HFImageProcessingBackend>(99)) == HERR_INVALID_PARAM);
    CHECK(HFSetImageProcessAlignedWidth(0) == HERR_INVALID_PARAM);
    CHECK(HFSetCudaDeviceId(-1) == HERR_INVALID_PARAM);
    CHECK(HFGetCudaDeviceId(nullptr) == HERR_INVALID_PARAM);
    CHECK(HFGetNumCudaDevices(nullptr) == HERR_INVALID_PARAM);
    CHECK(HFCheckCudaDeviceSupport(nullptr) == HERR_INVALID_PARAM);

    HInt32 value = -1;
    const HResult count_result = HFGetNumCudaDevices(&value);
    CHECK((count_result == HSUCCEED || count_result == HERR_DEVICE_CUDA_DISABLE));
    const HResult support_result = HFCheckCudaDeviceSupport(&value);
    CHECK((support_result == HSUCCEED || support_result == HERR_DEVICE_CUDA_DISABLE));
    const HResult print_result = HFPrintCudaDeviceInfo();
    CHECK((print_result == HSUCCEED || print_result == HERR_DEVICE_CUDA_DISABLE));
}
