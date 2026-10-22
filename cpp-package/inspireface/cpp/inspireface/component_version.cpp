#include "include/inspireface/component_version.h"

#include <array>
#include <cstdio>
#include <sstream>

#include <Eigen/Core>
#include <MNN/MNNDefine.h>
#include <inspirecv/version.h>
#include <sqlite-vec.h>
#include <sqlite3.h>

#include "include/inspireface/meta.h"
#include "middleware/nlohmann/json.hpp"

#if defined(ISF_ENABLE_OPENCV)
#include <opencv2/core/version.hpp>
#endif

#if defined(ISF_ENABLE_TENSORRT)
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>
#endif

#if defined(ISF_GLOBAL_INFERENCE_BACKEND_USE_MNN_CUDA) && !defined(ISF_ENABLE_TENSORRT)
#include <cuda_runtime_api.h>
#endif

#if defined(ISF_ENABLE_RGA)
#include <im2d_version.h>
#endif

namespace inspire {
namespace {

struct ComponentEntry {
    const char* name;
    ComponentVersion version;
};

ComponentVersion DisabledVersion() {
    return {0, 0, 0, ComponentVersionState::DISABLED};
}

ComponentVersion UnknownVersion() {
    return {0, 0, 0, ComponentVersionState::UNKNOWN};
}

ComponentVersion KnownVersion(int major, int minor, int patch) {
    return {major, minor, patch, ComponentVersionState::KNOWN};
}

ComponentVersion ParseEmbeddedVersion(const char* text) {
    if (text == nullptr) {
        return UnknownVersion();
    }
    while (*text != '\0' && (*text < '0' || *text > '9')) {
        ++text;
    }
    int major = 0;
    int minor = 0;
    int patch = 0;
    if (*text == '\0' || std::sscanf(text, "%d.%d.%d", &major, &minor, &patch) != 3 || major < 0 || minor < 0 || patch < 0) {
        return UnknownVersion();
    }
    return KnownVersion(major, minor, patch);
}

ComponentVersion MnnVersion() {
#if defined(MNN_VERSION_MAJOR) && defined(MNN_VERSION_MINOR) && defined(MNN_VERSION_PATCH)
    return KnownVersion(MNN_VERSION_MAJOR, MNN_VERSION_MINOR, MNN_VERSION_PATCH);
#else
    return UnknownVersion();
#endif
}

ComponentVersion OpenCVVersion() {
#if defined(ISF_ENABLE_OPENCV)
#if defined(CV_VERSION_MAJOR) && defined(CV_VERSION_MINOR) && defined(CV_VERSION_REVISION)
    return KnownVersion(CV_VERSION_MAJOR, CV_VERSION_MINOR, CV_VERSION_REVISION);
#else
    return UnknownVersion();
#endif
#else
    return DisabledVersion();
#endif
}

ComponentVersion TensorRTVersion() {
#if defined(ISF_ENABLE_TENSORRT)
#if defined(NV_TENSORRT_MAJOR) && defined(NV_TENSORRT_MINOR) && defined(NV_TENSORRT_PATCH)
    return KnownVersion(NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);
#else
    return UnknownVersion();
#endif
#else
    return DisabledVersion();
#endif
}

ComponentVersion CudaVersion() {
#if defined(ISF_ENABLE_TENSORRT) || defined(ISF_GLOBAL_INFERENCE_BACKEND_USE_MNN_CUDA)
#if defined(CUDART_VERSION)
    return KnownVersion(CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10, CUDART_VERSION % 10);
#else
    return UnknownVersion();
#endif
#else
    return DisabledVersion();
#endif
}

ComponentVersion RknnVersion() {
#if defined(ISF_ENABLE_RKNN)
    return UnknownVersion();
#else
    return DisabledVersion();
#endif
}

ComponentVersion RgaVersion() {
#if defined(ISF_ENABLE_RGA)
#if defined(RGA_API_MAJOR_VERSION) && defined(RGA_API_MINOR_VERSION) && defined(RGA_API_REVISION_VERSION)
    return KnownVersion(RGA_API_MAJOR_VERSION, RGA_API_MINOR_VERSION, RGA_API_REVISION_VERSION);
#else
    return UnknownVersion();
#endif
#else
    return DisabledVersion();
#endif
}

ComponentVersion CoreMLVersion() {
#if defined(ISF_ENABLE_APPLE_EXTENSION) || (defined(TARGET_OS_IOS) && TARGET_OS_IOS)
    return UnknownVersion();
#else
    return DisabledVersion();
#endif
}

const std::array<ComponentEntry, static_cast<size_t>(ComponentType::COUNT)>& ComponentEntries() {
    static const std::array<ComponentEntry, static_cast<size_t>(ComponentType::COUNT)> entries = {{
      {"mnn", MnnVersion()},
      {"inspirecv", ParseEmbeddedVersion(inspirecv::GetVersion())},
      {"eigen", KnownVersion(EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION)},
      {"sqlite", KnownVersion(SQLITE_VERSION_NUMBER / 1000000, (SQLITE_VERSION_NUMBER / 1000) % 1000, SQLITE_VERSION_NUMBER % 1000)},
      {"sqlite_vec", KnownVersion(SQLITE_VEC_VERSION_MAJOR, SQLITE_VEC_VERSION_MINOR, SQLITE_VEC_VERSION_PATCH)},
      {"nlohmann_json", KnownVersion(NLOHMANN_JSON_VERSION_MAJOR, NLOHMANN_JSON_VERSION_MINOR, NLOHMANN_JSON_VERSION_PATCH)},
      {"opencv", OpenCVVersion()},
      {"tensorrt", TensorRTVersion()},
      {"cuda", CudaVersion()},
      {"rknn", RknnVersion()},
      {"rga", RgaVersion()},
      {"coreml", CoreMLVersion()},
    }};
    return entries;
}

bool IsValidComponent(ComponentType component) {
    const int index = static_cast<int>(component);
    return index >= 0 && index < static_cast<int>(ComponentType::COUNT);
}

}  // namespace

bool ComponentVersion::IsEnabled() const {
    return state != ComponentVersionState::DISABLED;
}

bool ComponentVersion::IsVersionKnown() const {
    return state == ComponentVersionState::KNOWN;
}

std::string ComponentVersion::GetVersionString() const {
    if (state == ComponentVersionState::DISABLED) {
        return "disabled";
    }
    if (state == ComponentVersionState::UNKNOWN) {
        return "unknown";
    }
    return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
}

const char* GetComponentName(ComponentType component) {
    if (!IsValidComponent(component)) {
        return nullptr;
    }
    return ComponentEntries()[static_cast<size_t>(component)].name;
}

ComponentVersion GetComponentVersion(ComponentType component) {
    if (!IsValidComponent(component)) {
        return DisabledVersion();
    }
    return ComponentEntries()[static_cast<size_t>(component)].version;
}

std::string GetComponentVersionsString() {
    std::ostringstream stream;
    stream << "inspireface=" << GetSDKInfo().GetVersionString();
    for (const auto& entry : ComponentEntries()) {
        stream << ';' << entry.name << '=' << entry.version.GetVersionString();
    }
    return stream.str();
}

std::string GetDiagnosticInfo() {
    return GetSDKInfo().GetFullVersionInfo() + "\nComponents: " + GetComponentVersionsString();
}

}  // namespace inspire
