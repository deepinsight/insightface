#ifndef INSPIREFACE_COMPONENT_VERSION_H
#define INSPIREFACE_COMPONENT_VERSION_H

#include <string>

#include "data_type.h"

namespace inspire {

enum class ComponentType {
    MNN = 0,
    INSPIRECV,
    EIGEN,
    SQLITE,
    SQLITE_VEC,
    NLOHMANN_JSON,
    OPENCV,
    TENSORRT,
    CUDA,
    RKNN,
    RGA,
    COREML,
    COUNT,
};

enum class ComponentVersionState {
    DISABLED = 0,
    KNOWN = 1,
    UNKNOWN = 2,
};

struct INSPIRE_API_EXPORT ComponentVersion {
    int major;
    int minor;
    int patch;
    ComponentVersionState state;

    bool IsEnabled() const;
    bool IsVersionKnown() const;
    std::string GetVersionString() const;
};

INSPIRE_API_EXPORT const char* GetComponentName(ComponentType component);

INSPIRE_API_EXPORT ComponentVersion GetComponentVersion(ComponentType component);

INSPIRE_API_EXPORT std::string GetComponentVersionsString();

INSPIRE_API_EXPORT std::string GetDiagnosticInfo();

}  // namespace inspire

#endif  // INSPIREFACE_COMPONENT_VERSION_H
