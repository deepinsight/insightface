/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#ifndef MODELLOADERTAR_INSPIREARCHIVE_H
#define MODELLOADERTAR_INSPIREARCHIVE_H

#include "core_archive/core_archive.h"
#include "inspire_model/inspire_model.h"
#include "yaml-cpp/yaml.h"
#include "fstream"
#include "similarity_converter.h"
#include "launch.h"
#include "track_module/landmark/landmark_param.h"

namespace inspire {

enum {
    MISS_MANIFEST = -11,
    FORMAT_ERROR = -12,
    NOT_MATCH_MODEL = -13,
    ERROR_MODEL_BUFFER = -14,
    NOT_READ = -15,
};

class INSPIRE_API InspireArchive {
public:
    InspireArchive() : m_archive_(std::make_shared<CoreArchive>()) {
        m_status_ = NOT_READ;
    }

    explicit InspireArchive(const std::string& archiveFile) : m_archive_(std::make_shared<CoreArchive>(archiveFile)) {
        m_status_ = m_archive_->QueryLoadStatus();
        if (m_status_ == SARC_SUCCESS) {
            m_status_ = loadManifestFile();
        }
    }

    InspireArchive(const InspireArchive& other)
    : m_archive_(other.m_archive_),
      m_config_(other.m_config_),
      m_status_(other.m_status_),
      m_tag_(other.m_tag_),
      m_version_(other.m_version_),
      m_major_(other.m_major_),
      m_release_time_(other.m_release_time_) {}

    InspireArchive& operator=(const InspireArchive& other) {
        if (this != &other) {
            m_archive_ = other.m_archive_;
            m_config_ = other.m_config_;
            m_status_ = other.m_status_;
            m_tag_ = other.m_tag_;
            m_version_ = other.m_version_;
            m_major_ = other.m_major_;
            m_release_time_ = other.m_release_time_;
        }
        return *this;
    }

    int32_t ReLoad(const std::string& archiveFile) {
        auto ret = m_archive_->Reset(archiveFile);
        if (ret != SARC_SUCCESS) {
            m_archive_->Close();
            m_status_ = ret;
            return ret;
        }
        m_status_ = loadManifestFile();
        return m_status_;
    }

    int32_t QueryStatus() const {
        return m_status_;
    }

    int32_t LoadModel(const std::string& name, InspireModel& model) {
        if (m_config_[name]) {
            auto ret = model.Reset(m_config_[name]);
            if (ret != 0) {
                return ret;
            }
            if (model.loadFilePath) {
                // No model files are loaded, only configuration files are loaded for extension modules such as CoreML.
                return SARC_SUCCESS;
            }
            auto& buffer = m_archive_->GetFileContent(model.name);
            if (buffer.empty()) {
                return ERROR_MODEL_BUFFER;
            }
            model.SetBuffer(buffer, buffer.size());
            return SARC_SUCCESS;
        } else {
            return NOT_MATCH_MODEL;
        }
    }

    void PrintSubFiles() {
        m_archive_->PrintSubFiles();
    }

    const std::vector<std::string>& GetSubfilesNames() const {
        return m_archive_->GetSubfilesNames();
    }

    void Release() {
        m_status_ = NOT_READ;
        m_archive_->Close();
    }

    std::vector<char>& GetFileContent(const std::string& filename) {
        return m_archive_->GetFileContent(filename);
    }

    const std::vector<int>& GetFaceDetectPixelList() const {
        return m_face_detect_pixel_list_;
    }

    const std::vector<std::string>& GetFaceDetectModelList() const {
        return m_face_detect_model_list_;
    }

    const std::shared_ptr<LandmarkParam>& GetLandmarkParam() const {
        return m_landmark_param_;
    }
private:
    int32_t loadManifestFile() {
        if (m_archive_->QueryLoadStatus() == SARC_SUCCESS) {
            auto configBuffer = m_archive_->GetFileContent(MANIFEST_FILE);
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
            if (m_config_["major"]) {
                m_major_ = m_config_["major"].as<std::string>();
            } else {
                m_major_ = "unknown";
            }
            if (m_config_["release"]) {
                m_release_time_ = m_config_["release"].as<std::string>();
            } else {
                m_release_time_ = "unknown";
            }
            INSPIRE_LOGI("== Load %s-%s, Version: %s, Release: %s ==", m_tag_.c_str(), m_major_.c_str(), m_version_.c_str(), m_release_time_.c_str());
            // Load similarity converter config
            if (m_config_["similarity_converter"]) {
                SimilarityConverterConfig config;
                config.threshold = m_config_["similarity_converter"]["threshold"].as<double>();
                config.middleScore = m_config_["similarity_converter"]["middle_score"].as<double>();
                config.steepness = m_config_["similarity_converter"]["steepness"].as<double>();
                config.outputMin = m_config_["similarity_converter"]["output_min"].as<double>();
                config.outputMax = m_config_["similarity_converter"]["output_max"].as<double>();
                SIMILARITY_CONVERTER_UPDATE_CONFIG(config);
                INSPIRE_LOGI(
                  "Successfully loaded similarity converter config: \n \t threshold: %f \n \t middle_score: %f \n \t steepness: %f \n \t output_min: "
                  "%f \n \t output_max: %f",
                  config.threshold, config.middleScore, config.steepness, config.outputMin, config.outputMax);
                SIMILARITY_CONVERTER_SET_RECOMMENDED_COSINE_THRESHOLD(config.threshold);
            } else {
                INSPIRE_LOGW("No similarity converter config found, use default config: ");
                auto config = SIMILARITY_CONVERTER_GET_CONFIG();
                INSPIRE_LOGI("threshold: %f \n \t middle_score: %f \n \t steepness: %f \n \t output_min: %f \n \t output_max: %f", config.threshold,
                             config.middleScore, config.steepness, config.outputMin, config.outputMax);
                SIMILARITY_CONVERTER_SET_RECOMMENDED_COSINE_THRESHOLD(config.threshold);
            }
            // Load face detect model
            if (m_config_["face_detect_pixel_list"] && m_config_["face_detect_model_list"]) {
                auto node_face_detect_pixel_list = m_config_["face_detect_pixel_list"];
                for (std::size_t i = 0; i < node_face_detect_pixel_list.size(); ++i) {
                    m_face_detect_pixel_list_.push_back(node_face_detect_pixel_list[i].as<int>());
                }
                auto node_face_detect_model_list = m_config_["face_detect_model_list"];
                for (std::size_t i = 0; i < node_face_detect_model_list.size(); ++i) {
                    m_face_detect_model_list_.push_back(node_face_detect_model_list[i].as<std::string>());
                }
                if (m_face_detect_pixel_list_.size() != m_face_detect_model_list_.size()) {
                    return FORMAT_ERROR;
                }
            } else {
                m_face_detect_pixel_list_ = {160, 320, 640};
                m_face_detect_model_list_ = {"face_detect_160", "face_detect_320", "face_detect_640"};
            }
            m_landmark_param_ = std::make_shared<LandmarkParam>(m_config_["landmark_table"]);
        }
        return 0;
    }


private:
    std::shared_ptr<CoreArchive> m_archive_;
    YAML::Node m_config_;

    int32_t m_status_;

    const std::string MANIFEST_FILE = "__inspire__";

    std::string m_tag_;
    std::string m_version_;
    std::string m_major_;
    std::string m_release_time_;

    std::vector<int> m_face_detect_pixel_list_;
    std::vector<std::string> m_face_detect_model_list_;

    std::shared_ptr<LandmarkParam> m_landmark_param_;
};

}  // namespace inspire

#endif  // MODELLOADERTAR_INSPIREARCHIVE_H