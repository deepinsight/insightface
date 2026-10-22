/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#ifndef MODELLOADERTAR_INSPIREARCHIVE_H
#define MODELLOADERTAR_INSPIREARCHIVE_H

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "core_archive/core_archive.h"
#include "inspire_model/inspire_model.h"
#include "yaml-cpp/yaml.h"
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
    UNSUPPORTED_MODEL_ENGINE = -16,
};

struct ResourcePackMetadata {
    std::string tag;
    std::string version;
    std::string major;
    std::string release_time;
    size_t archive_file_count{0};
    size_t model_count{0};
};

class INSPIRE_API InspireArchive {
public:
    InspireArchive() : m_archive_(std::make_shared<CoreArchive>()) {}

    explicit InspireArchive(const std::string& archiveFile) : InspireArchive() {
        ReLoad(archiveFile);
    }

    InspireArchive(const InspireArchive& other)
    : m_archive_(other.m_archive_),
      m_config_(other.m_config_),
      m_status_(other.m_status_),
      m_tag_(other.m_tag_),
      m_version_(other.m_version_),
      m_major_(other.m_major_),
      m_release_time_(other.m_release_time_),
      m_face_detect_pixel_list_(other.m_face_detect_pixel_list_),
      m_face_detect_model_list_(other.m_face_detect_model_list_),
      m_landmark_param_(other.m_landmark_param_ ? std::make_shared<LandmarkParam>(*other.m_landmark_param_) : nullptr),
      m_similarity_converter_config_(other.m_similarity_converter_config_) {}

    InspireArchive& operator=(const InspireArchive& other) {
        if (this != &other) {
            m_archive_ = other.m_archive_;
            // YAML nodes are immutable after a manifest is published. Rebind
            // the handle instead of deep-cloning the full manifest; all
            // replacement/release paths below also use reset(), so they do not
            // mutate the shared YAML graph.
            m_config_.reset(other.m_config_);
            m_status_ = other.m_status_;
            m_tag_ = other.m_tag_;
            m_version_ = other.m_version_;
            m_major_ = other.m_major_;
            m_release_time_ = other.m_release_time_;
            m_face_detect_pixel_list_ = other.m_face_detect_pixel_list_;
            m_face_detect_model_list_ = other.m_face_detect_model_list_;
            m_landmark_param_ = other.m_landmark_param_ ? std::make_shared<LandmarkParam>(*other.m_landmark_param_) : nullptr;
            m_similarity_converter_config_ = other.m_similarity_converter_config_;
        }
        return *this;
    }

    // Build the reader and parse every manifest field before publishing the
    // replacement. A failed reload leaves a previously valid archive intact.
    int32_t ReLoad(const std::string& archiveFile) {
        auto replacement = std::make_shared<CoreArchive>(archiveFile);
        int32_t status = replacement->QueryLoadStatus();
        if (status != SARC_SUCCESS) {
            RecordInitialFailure(status);
            return status;
        }

        ManifestData manifest;
        status = ParseManifest(replacement, manifest);
        if (status != SARC_SUCCESS) {
            RecordInitialFailure(status);
            return status;
        }

        m_archive_.swap(replacement);
        m_config_.reset(manifest.config);
        m_tag_.swap(manifest.tag);
        m_version_.swap(manifest.version);
        m_major_.swap(manifest.major);
        m_release_time_.swap(manifest.release_time);
        m_face_detect_pixel_list_.swap(manifest.face_detect_pixel_list);
        m_face_detect_model_list_.swap(manifest.face_detect_model_list);
        m_landmark_param_.swap(manifest.landmark_param);
        m_similarity_converter_config_ = manifest.similarity_converter_config;
        m_status_ = SARC_SUCCESS;
        return SARC_SUCCESS;
    }

    int32_t QueryStatus() const {
        return m_status_;
    }

    int32_t LoadModel(const std::string& name, InspireModel& model) {
        if (m_status_ != SARC_SUCCESS || name.empty()) {
            return m_status_ == SARC_SUCCESS ? NOT_MATCH_MODEL : NOT_READ;
        }
        const YAML::Node& root_config = m_config_;
        const YAML::Node config = root_config[name];
        if (!config || !config.IsMap()) {
            return NOT_MATCH_MODEL;
        }
        const int32_t status = model.Reset(config);
        if (status != SARC_SUCCESS) {
            return status;
        }
        if (model.loadFilePath) {
            // Extension modules such as CoreML load from a bundle path.
            return SARC_SUCCESS;
        }
        const auto buffer = m_archive_->GetFileContentShared(model.name);
        if (!buffer || buffer->empty()) {
            return ERROR_MODEL_BUFFER;
        }
        model.SetBuffer(buffer);
        return SARC_SUCCESS;
    }

    void PrintSubFiles() {
        m_archive_->PrintSubFiles();
    }

    const std::vector<std::string>& GetSubfilesNames() const {
        return m_archive_->GetSubfilesNames();
    }

    void Release() {
        m_archive_ = std::make_shared<CoreArchive>();
        m_config_.reset();
        m_status_ = NOT_READ;
        m_tag_.clear();
        m_version_.clear();
        m_major_.clear();
        m_release_time_.clear();
        m_face_detect_pixel_list_.clear();
        m_face_detect_model_list_.clear();
        m_landmark_param_.reset();
        m_similarity_converter_config_ = SimilarityConverterConfig();
    }

    std::shared_ptr<const std::vector<char>> GetFileContentShared(const std::string& filename) const {
        return m_archive_->GetFileContentShared(filename);
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

    bool SwitchLandmarkEngine(const std::string& name) {
        if (!m_landmark_param_) {
            return false;
        }
        auto replacement = std::make_shared<LandmarkParam>(*m_landmark_param_);
        if (!replacement->ReLoad(name)) {
            return false;
        }
        m_landmark_param_.swap(replacement);
        return true;
    }

    const SimilarityConverterConfig& GetSimilarityConverterConfig() const {
        return m_similarity_converter_config_;
    }

    /**
     * Validate all required and declared models without constructing an
     * inference engine or publishing any global SDK state.
     */
    int32_t ValidateContents(ResourcePackMetadata& metadata) {
        if (m_status_ != SARC_SUCCESS || !m_config_ || !m_config_.IsMap()) {
            return m_status_ == SARC_SUCCESS ? FORMAT_ERROR : m_status_;
        }

        std::set<std::string> validated_models;
        auto validate_model = [&](const std::string& key) -> int32_t {
            if (key.empty() || validated_models.find(key) != validated_models.end()) {
                return key.empty() ? NOT_MATCH_MODEL : SARC_SUCCESS;
            }
            InspireModel model;
            const int32_t status = LoadModel(key, model);
            if (status != SARC_SUCCESS) {
                return status;
            }
            if (!IsInferenceEngineSupported(model.inferEngine) || !IsInferenceEngineSupported(model.modelType)) {
                return UNSUPPORTED_MODEL_ENGINE;
            }
            if (model.loadFilePath && model.fullname.empty()) {
                return ERROR_MODEL_BUFFER;
            }
            validated_models.emplace(key);
            return SARC_SUCCESS;
        };

        for (const auto& detector : m_face_detect_model_list_) {
            const int32_t status = validate_model(detector);
            if (status != SARC_SUCCESS) {
                return status;
            }
        }
        if (!m_landmark_param_) {
            return FORMAT_ERROR;
        }
        int32_t status = validate_model(m_landmark_param_->landmark_engine_name);
        if (status != SARC_SUCCESS) {
            return status;
        }
        status = validate_model("refine_net");
        if (status != SARC_SUCCESS) {
            return status;
        }

        try {
            for (const auto& entry : m_config_) {
                if (!entry.first.IsScalar() || !entry.second.IsMap() || !entry.second["name"]) {
                    continue;
                }
                status = validate_model(entry.first.as<std::string>());
                if (status != SARC_SUCCESS) {
                    return status;
                }
            }
        } catch (const std::exception& error) {
            INSPIRE_LOGE("Failed to validate archive model entries: %s", error.what());
            return FORMAT_ERROR;
        }

        metadata.tag = m_tag_;
        metadata.version = m_version_;
        metadata.major = m_major_;
        metadata.release_time = m_release_time_;
        metadata.archive_file_count = m_archive_->GetSubfilesNames().size();
        metadata.model_count = validated_models.size();
        return SARC_SUCCESS;
    }

private:
    struct ManifestData {
        YAML::Node config;
        std::string tag;
        std::string version;
        std::string major;
        std::string release_time;
        std::vector<int> face_detect_pixel_list;
        std::vector<std::string> face_detect_model_list;
        std::shared_ptr<LandmarkParam> landmark_param;
        SimilarityConverterConfig similarity_converter_config;
    };

    void RecordInitialFailure(int32_t status) {
        if (m_status_ != SARC_SUCCESS) {
            m_status_ = status;
        }
    }

    static int32_t ParseManifest(const std::shared_ptr<CoreArchive>& archive, ManifestData& result) {
        const auto config_buffer = archive->GetFileContentShared("__inspire__");
        if (!config_buffer || config_buffer->empty()) {
            return MISS_MANIFEST;
        }

        try {
            const std::string manifest_text(config_buffer->data(), config_buffer->size());
            const YAML::Node config = YAML::Load(manifest_text);
            if (!config || !config.IsMap() || !config["tag"] || !config["version"]) {
                return FORMAT_ERROR;
            }

            ManifestData next;
            next.config.reset(config);
            next.tag = config["tag"].as<std::string>();
            next.version = config["version"].as<std::string>();
            next.major = config["major"] ? config["major"].as<std::string>() : "unknown";
            next.release_time = config["release"] ? config["release"].as<std::string>() : "unknown";
            if (next.tag.empty() || next.version.empty()) {
                return FORMAT_ERROR;
            }

            if (config["similarity_converter"]) {
                const YAML::Node converter = config["similarity_converter"];
                if (!converter.IsMap() || !converter["threshold"] || !converter["middle_score"] || !converter["steepness"] ||
                    !converter["output_min"] || !converter["output_max"]) {
                    return FORMAT_ERROR;
                }
                SimilarityConverterConfig converter_config;
                converter_config.threshold = converter["threshold"].as<double>();
                converter_config.middleScore = converter["middle_score"].as<double>();
                converter_config.steepness = converter["steepness"].as<double>();
                converter_config.outputMin = converter["output_min"].as<double>();
                converter_config.outputMax = converter["output_max"].as<double>();
                if (!SimilarityConverter::IsConfigValid(converter_config)) {
                    INSPIRE_LOGE("Invalid similarity converter config in the resource manifest");
                    return FORMAT_ERROR;
                }
                next.similarity_converter_config = converter_config;
            }

            const bool has_pixel_list = static_cast<bool>(config["face_detect_pixel_list"]);
            const bool has_model_list = static_cast<bool>(config["face_detect_model_list"]);
            if (has_pixel_list != has_model_list) {
                return FORMAT_ERROR;
            }
            if (has_pixel_list) {
                const YAML::Node pixels = config["face_detect_pixel_list"];
                const YAML::Node models = config["face_detect_model_list"];
                if (!pixels.IsSequence() || !models.IsSequence() || pixels.size() == 0 || pixels.size() != models.size()) {
                    return FORMAT_ERROR;
                }
                std::set<int> unique_pixels;
                for (std::size_t i = 0; i < pixels.size(); ++i) {
                    const int pixel = pixels[i].as<int>();
                    const std::string model_name = models[i].as<std::string>();
                    if (pixel <= 0 || model_name.empty() || !unique_pixels.emplace(pixel).second || !config[model_name] ||
                        !config[model_name].IsMap()) {
                        return FORMAT_ERROR;
                    }
                    next.face_detect_pixel_list.push_back(pixel);
                    next.face_detect_model_list.push_back(model_name);
                }
            } else {
                next.face_detect_pixel_list = {160, 320, 640};
                next.face_detect_model_list = {"face_detect_160", "face_detect_320", "face_detect_640"};
            }

            const YAML::Node landmark_table = config["landmark_table"];
            if (landmark_table && !landmark_table.IsNull() && !landmark_table.IsMap()) {
                return FORMAT_ERROR;
            }
            next.landmark_param = landmark_table ? std::make_shared<LandmarkParam>(landmark_table)
                                                 : std::make_shared<LandmarkParam>(YAML::Node());
            result = std::move(next);
            INSPIRE_LOGI("== Load %s-%s, Version: %s, Release: %s ==", result.tag.c_str(), result.major.c_str(),
                         result.version.c_str(), result.release_time.c_str());
            return SARC_SUCCESS;
        } catch (const std::exception& error) {
            INSPIRE_LOGE("Failed to parse archive manifest: %s", error.what());
            return FORMAT_ERROR;
        }
    }

    static bool IsInferenceEngineSupported(int engine) {
        switch (engine) {
            case InferenceWrapper::INFER_MNN:
                return true;
            case InferenceWrapper::INFER_RKNN:
#if defined(ISF_ENABLE_RKNN)
                return true;
#else
                return false;
#endif
            case InferenceWrapper::INFER_COREML:
#if defined(ISF_ENABLE_APPLE_EXTENSION)
                return true;
#else
                return false;
#endif
            case InferenceWrapper::INFER_TENSORRT:
#if defined(ISF_ENABLE_TENSORRT)
                return true;
#else
                return false;
#endif
            default:
                return false;
        }
    }

    std::shared_ptr<CoreArchive> m_archive_;
    YAML::Node m_config_;
    int32_t m_status_{NOT_READ};

    std::string m_tag_;
    std::string m_version_;
    std::string m_major_;
    std::string m_release_time_;

    std::vector<int> m_face_detect_pixel_list_;
    std::vector<std::string> m_face_detect_model_list_;
    std::shared_ptr<LandmarkParam> m_landmark_param_;
    SimilarityConverterConfig m_similarity_converter_config_;
};

}  // namespace inspire

#endif  // MODELLOADERTAR_INSPIREARCHIVE_H
