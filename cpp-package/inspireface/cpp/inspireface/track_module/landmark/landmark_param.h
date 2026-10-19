#ifndef INSPIRE_LANDMARK_PARAM_H
#define INSPIRE_LANDMARK_PARAM_H

#include "data_type.h"
#include "yaml-cpp/yaml.h"
#include "mean_shape.h"
#include "log.h"
#include "landmark_tools.h"
#include "order_of_hyper_landmark.h"
#include <cmath>

namespace inspire {

struct SemanticIndex {
    int32_t left_eye_center = 67;
    int32_t right_eye_center = 68;
    int32_t nose_corner = 100;
    int32_t mouth_left_corner = 104;
    int32_t mouth_right_corner = 105;
    int32_t mouth_lower = 84;
    int32_t mouth_upper = 87;
    std::vector<int32_t> left_eye_region = HLMK_LEFT_EYE_POINTS_INDEX;
    std::vector<int32_t> right_eye_region = HLMK_RIGHT_EYE_POINTS_INDEX;
};

class INSPIRE_API LandmarkParam {
public:
    LandmarkParam(const YAML::Node &config) {
        LoadDefaultMeshShape();
        // yaml-cpp represents both a default node and an explicit YAML null as
        // defined Null nodes. They mean that this legacy pack has no landmark
        // parameter table, so engine selection must keep using defaults.
        m_is_available_ = static_cast<bool>(config) && !config.IsNull();
        m_table_ = config;
    }

    void LoadDefaultMeshShape() {
        num_of_landmark = 106;
        mean_shape_points = DefaultMeshShape();
    }

    bool ReLoad(const std::string &name) {
        if (!m_is_available_) {
            landmark_engine_name = name;
            return true;
        }
        try {
            const auto landmark_table = m_table_[name];
            if (!landmark_table) {
                INSPIRE_LOGE("landmark config not found: %s", name.c_str());
                return false;
            }

            const int next_num_landmarks = landmark_table["num_of_landmark"].as<int>();
            const float next_expansion_scale = landmark_table["expansion_scale"].as<float>();
            const int next_input_size = landmark_table["input_size"].as<int>();
            std::string next_normalization_mode = landmark_table["normalization_mode"].as<std::string>();
            std::string next_engine_name = name;
            if (next_num_landmarks <= 0 || next_input_size <= 0 || !std::isfinite(next_expansion_scale) || next_expansion_scale <= 0.0f ||
                (next_normalization_mode != "MinMax" && next_normalization_mode != "CenterScaling")) {
                INSPIRE_LOGE("invalid landmark scalar config: %s", name.c_str());
                return false;
            }

            const auto semanticNode = landmark_table["semantic_index"];
            if (!semanticNode) {
                INSPIRE_LOGE("semantic_index not found: %s", name.c_str());
                return false;
            }
            SemanticIndex next_semantic_index;
            next_semantic_index.left_eye_center = semanticNode["left_eye_center"].as<int>();
            next_semantic_index.right_eye_center = semanticNode["right_eye_center"].as<int>();
            next_semantic_index.nose_corner = semanticNode["nose_corner"].as<int>();
            next_semantic_index.mouth_left_corner = semanticNode["mouth_left_corner"].as<int>();
            next_semantic_index.mouth_right_corner = semanticNode["mouth_right_corner"].as<int>();
            next_semantic_index.mouth_lower = semanticNode["mouth_lower"].as<int>();
            next_semantic_index.mouth_upper = semanticNode["mouth_upper"].as<int>();
            if (semanticNode["left_eye_region"]) {
                next_semantic_index.left_eye_region = semanticNode["left_eye_region"].as<std::vector<int>>();
            }
            if (semanticNode["right_eye_region"]) {
                next_semantic_index.right_eye_region = semanticNode["right_eye_region"].as<std::vector<int>>();
            }
            if (!IsSemanticIndexValid(next_semantic_index, next_num_landmarks)) {
                INSPIRE_LOGE("landmark semantic index is out of range: %s", name.c_str());
                return false;
            }

            std::vector<inspirecv::Point2f> next_mean_shape;
            const auto meshShape = landmark_table["mesh_shape"];
            if (meshShape && meshShape.size() > 0) {
                const std::vector<float> meshShapeData = meshShape.as<std::vector<float>>();
                if (meshShapeData.size() != static_cast<size_t>(next_num_landmarks) * 2) {
                    INSPIRE_LOGE("mesh_shape size is not equal to num_of_landmark * 2: %s", name.c_str());
                    return false;
                }
                next_mean_shape.resize(next_num_landmarks);
                for (int index = 0; index < next_num_landmarks; ++index) {
                    if (!std::isfinite(meshShapeData[index * 2]) || !std::isfinite(meshShapeData[index * 2 + 1])) {
                        INSPIRE_LOGE("mesh_shape contains a non-finite value: %s", name.c_str());
                        return false;
                    }
                    next_mean_shape[index].SetX(meshShapeData[index * 2]);
                    next_mean_shape[index].SetY(meshShapeData[index * 2 + 1]);
                }
                next_mean_shape = LandmarkCropped(next_mean_shape);
            } else {
                if (next_num_landmarks != 106) {
                    INSPIRE_LOGE("non-default landmark count requires mesh_shape: %s", name.c_str());
                    return false;
                }
                next_mean_shape = DefaultMeshShape();
            }

            num_of_landmark = next_num_landmarks;
            expansion_scale = next_expansion_scale;
            input_size = next_input_size;
            std::swap(semantic_index, next_semantic_index);
            mean_shape_points.swap(next_mean_shape);
            normalization_mode.swap(next_normalization_mode);
            landmark_engine_name.swap(next_engine_name);
            return true;
        } catch (const std::exception &error) {
            INSPIRE_LOGE("failed to parse landmark config %s: %s", name.c_str(), error.what());
            return false;
        }
    }

public:
    int num_of_landmark{106};
    float expansion_scale{1.1f};
    int input_size{112};
    std::vector<inspirecv::Point2f> mean_shape_points;
    SemanticIndex semantic_index;
    std::string landmark_engine_name{"landmark"};
    std::string normalization_mode{"MinMax"};

private:
    static bool IsIndexValid(int32_t index, int32_t count) {
        return index >= 0 && index < count;
    }

    static bool IsIndexListValid(const std::vector<int32_t> &indices, int32_t count) {
        if (indices.empty()) {
            return false;
        }
        for (int32_t index : indices) {
            if (!IsIndexValid(index, count)) {
                return false;
            }
        }
        return true;
    }

    static bool IsSemanticIndexValid(const SemanticIndex &index, int32_t count) {
        return IsIndexValid(index.left_eye_center, count) && IsIndexValid(index.right_eye_center, count) &&
               IsIndexValid(index.nose_corner, count) && IsIndexValid(index.mouth_left_corner, count) &&
               IsIndexValid(index.mouth_right_corner, count) && IsIndexValid(index.mouth_lower, count) &&
               IsIndexValid(index.mouth_upper, count) && IsIndexListValid(index.left_eye_region, count) &&
               IsIndexListValid(index.right_eye_region, count);
    }

    static std::vector<inspirecv::Point2f> DefaultMeshShape() {
        std::vector<inspirecv::Point2f> points(106);
        for (int index = 0; index < 106; ++index) {
            points[index].SetX(HYPLMK_MESH_SHAPE[index * 2]);
            points[index].SetY(HYPLMK_MESH_SHAPE[index * 2 + 1]);
        }
        return points;
    }

    YAML::Node m_table_;
    bool m_is_available_{false};
};

}  // namespace inspire

#endif  // INSPIRE_LANDMARK_PARAM_H
