//
// Created by tunm on 2024/3/30.
//

#ifndef MODELLOADERTAR_INSPIREARCHIVE_H
#define MODELLOADERTAR_INSPIREARCHIVE_H
#include "simple_archive.h"
#include "inspire_model/inspire_model.h"
#include "yaml-cpp/yaml.h"
#include "fstream"

namespace inspire {

enum {
    MISS_MANIFEST = -11,
    FORMAT_ERROR = -12,
    NOT_MATCH_MODEL = -13,
    ERROR_MODEL_BUFFER = -14,
    NOT_READ = -15,
};

class INSPIRE_API InspireArchive: SimpleArchive {
public:
    InspireArchive() : SimpleArchive() {
        m_status_ = NOT_READ;
    }

    explicit InspireArchive(const std::string& archiveFile) : SimpleArchive(archiveFile) {
        m_status_ = QueryStatus();
        if (m_status_ == SARC_SUCCESS) {
            m_status_ = loadManifestFile();
        }
    }

    int32_t ReLoad(const std::string& archiveFile) {
        auto ret = Reset(archiveFile);
        if (ret != SARC_SUCCESS) {
            return ret;
        }
        m_status_ = loadManifestFile();
        return m_status_;
    }

    int32_t QueryStatus() const {
        return m_status_;
    }

    int32_t LoadModel(const std::string &name, InspireModel &model) {
        if (m_config_[name]) {
            auto ret = model.Reset(m_config_[name]);
            if (ret != 0) {
                return ret;
            }
            auto &buffer = GetFileContent(model.name);
            if (buffer.empty()) {
                return ERROR_MODEL_BUFFER;
            }
            model.SetBuffer(buffer, buffer.size());
            return SARC_SUCCESS;
        } else {
            return NOT_MATCH_MODEL;
        }
    }

    void PublicPrintSubFiles() {
        PrintSubFiles();
    }

    void Release() {
        m_status_ = NOT_READ;
        Close();
    }

private:

    int32_t loadManifestFile() {
        if (QueryLoadStatus() == SARC_SUCCESS) {
            auto configBuffer = GetFileContent(MANIFEST_FILE);
            configBuffer.push_back('\0');
            if (configBuffer.empty()) {
                return MISS_MANIFEST;
            }
            m_config_ = YAML::Load(configBuffer.data());
            if (!m_config_["tag"] || !m_config_["version"]) {
                return FORMAT_ERROR;
            }
            m_tag_ = m_config_["tag"].as<std::string>();
            m_version_ = m_config_["version"].as<std::string>();
            INSPIRE_LOGI("== %s %s ==", m_tag_.c_str(), m_version_.c_str());
        }
        return 0;
    }

private:
    YAML::Node m_config_;

    int32_t m_status_;

    const std::string MANIFEST_FILE = "__inspire__";

    std::string m_tag_;
    std::string m_version_;

};

}   // namespace inspire

#endif //MODELLOADERTAR_INSPIREARCHIVE_H
