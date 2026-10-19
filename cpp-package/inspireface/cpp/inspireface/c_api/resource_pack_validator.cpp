#include "inspireface.h"

#include <cstring>
#include <limits>

#include "middleware/model_archive/inspire_archive.h"
#include "middleware/system.h"

namespace {

template <size_t Capacity>
bool CopyMetadataText(const std::string& source, HChar (&destination)[Capacity]) {
    if (source.size() >= Capacity || source.find('\0') != std::string::npos) {
        return false;
    }
    std::memcpy(destination, source.c_str(), source.size() + 1);
    return true;
}

HFStatus MapArchiveValidationStatus(int32_t status) {
    if (status == inspire::SARC_SUCCESS) {
        return HSUCCEED;
    }
    if (status == inspire::UNSUPPORTED_MODEL_ENGINE) {
        return HERR_UNSUPPORTED;
    }
    if (status == inspire::NOT_MATCH_MODEL || status == inspire::ERROR_MODEL_BUFFER) {
        return HERR_ARCHIVE_LOAD_MODEL_FAILURE;
    }
    return HERR_ARCHIVE_FILE_FORMAT_ERROR;
}

}  // namespace

HFStatus HFValidateResourcePack(HPath resourcePath, PHFResourcePackInfo info) {
    if (resourcePath == nullptr || resourcePath[0] == '\0') {
        return HERR_INVALID_PARAM;
    }
    if (info != nullptr &&
        (info->structSize < sizeof(HFResourcePackInfo) || info->structVersion != HF_RESOURCE_PACK_INFO_VERSION)) {
        return HERR_INVALID_PARAM;
    }

    const std::string path(resourcePath);
    if (!inspire::os::IsExists(path) || !inspire::os::IsFile(path)) {
        return HERR_ARCHIVE_LOAD_FAILURE;
    }

    try {
        inspire::InspireArchive archive;
        const int32_t load_status = archive.ReLoad(path);
        if (load_status != inspire::SARC_SUCCESS) {
            return MapArchiveValidationStatus(load_status);
        }

        inspire::ResourcePackMetadata metadata;
        const int32_t validation_status = archive.ValidateContents(metadata);
        if (validation_status != inspire::SARC_SUCCESS) {
            return MapArchiveValidationStatus(validation_status);
        }
        if (metadata.archive_file_count > std::numeric_limits<HFUInt32>::max() ||
            metadata.model_count > std::numeric_limits<HFUInt32>::max()) {
            return HERR_ARCHIVE_FILE_FORMAT_ERROR;
        }

        if (info != nullptr) {
            HFResourcePackInfo output = {};
            output.structSize = sizeof(HFResourcePackInfo);
            output.structVersion = HF_RESOURCE_PACK_INFO_VERSION;
            output.archiveFileCount = static_cast<HFUInt32>(metadata.archive_file_count);
            output.modelCount = static_cast<HFUInt32>(metadata.model_count);
            if (!CopyMetadataText(metadata.tag, output.tag) || !CopyMetadataText(metadata.version, output.version) ||
                !CopyMetadataText(metadata.major, output.major) || !CopyMetadataText(metadata.release_time, output.releaseDate)) {
                return HERR_ARCHIVE_FILE_FORMAT_ERROR;
            }
            std::memcpy(info, &output, sizeof(output));
        }
        return HSUCCEED;
    } catch (const std::exception& error) {
        INSPIRE_LOGE("Resource-pack validation failed: %s", error.what());
        return HERR_ARCHIVE_FILE_FORMAT_ERROR;
    } catch (...) {
        INSPIRE_LOGE("Resource-pack validation failed with an unknown exception");
        return HERR_ARCHIVE_FILE_FORMAT_ERROR;
    }
}
