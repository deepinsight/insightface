#ifndef INSPIRE_FACE_META_H
#define INSPIRE_FACE_META_H

#include <iostream>
#include <string>
#include "data_type.h"

namespace inspire {

/**
 * @brief SDK meta information
 */
struct INSPIRE_API_EXPORT SDKInfo {
    // version
    int version_major;
    int version_minor;
    int version_patch;

    // series name
    std::string series;

    // build info
    std::string build_date;
    std::string build_time;
    std::string compiler;
    std::string platform;
    std::string architecture;

    // backend info
    std::string inference_backend;
    std::string inspirecv_backend;
    bool rga_backend_enabled;

    // description
    std::string description;
    
    std::string GetFullVersionInfo() const;
    
    // auxiliary methods: return the string form of the version number
    std::string GetVersionMajorStr() const;
    std::string GetVersionMinorStr() const;
    std::string GetVersionPatchStr() const;
    std::string GetVersionString() const;
};

/**
 * @brief Get the SDK info
 * @return The constant reference of the SDK info
 */
INSPIRE_API_EXPORT const SDKInfo& GetSDKInfo();

}   // namespace inspire

#endif  // INSPIRE_FACE_META_H