#ifndef INSPIRE_LANDMARK_PARAM_H
#define INSPIRE_LANDMARK_PARAM_H

#include "data_type.h"
#include "yaml-cpp/yaml.h"
#include "mean_shape.h"
#include "log.h"
#include "landmark_tools.h"
#include "order_of_hyper_landmark.h"

namespace inspire {

typedef struct {
    int32_t left_eye_center = 67;
    int32_t right_eye_center = 68;
    int32_t nose_corner = 100;
    int32_t mouth_left_corner = 104;
    int32_t mouth_right_corner = 105;
    int32_t mouth_lower = 84;
    int32_t mouth_upper = 87;
    std::vector<int32_t> left_eye_region = HLMK_LEFT_EYE_POINTS_INDEX;
    std::vector<int32_t> right_eye_region = HLMK_RIGHT_EYE_POINTS_INDEX;
} SemanticIndex;

class INSPIRE_API LandmarkParam {
public:
    LandmarkParam(const YAML::Node &config) {
        LoadDefaultMeshShape();
        if (!config) {
        } else {
            // TODO: parse config
            m_is_available_ = true;
        }
        m_table_ = config;
    }

    void LoadDefaultMeshShape() {
        mean_shape_points.clear();
        mean_shape_points.resize(num_of_landmark);
        for (int k = 0; k < num_of_landmark; k++) {
            mean_shape_points[k].SetX(HYPLMK_MESH_SHAPE[k * 2]);
            mean_shape_points[k].SetY(HYPLMK_MESH_SHAPE[k * 2 + 1]);
        }
    }

    bool ReLoad(const std::string &name) {
        if (!m_is_available_) {
            landmark_engine_name = name;
            return true;
        }
        // parse config
        auto landmark_table = m_table_[name];
        if (!landmark_table) {
            INSPIRE_LOGE("landmark config not found: %s", name.c_str());
            return false;
        }
        num_of_landmark = landmark_table["num_of_landmark"].as<int>();
        expansion_scale = landmark_table["expansion_scale"].as<float>();
        input_size = landmark_table["input_size"].as<int>();
        auto semanic_index = landmark_table["semantic_index"];
        if (!semanic_index) {
            INSPIRE_LOGE("semantic_index not found: %s", name.c_str());
            return false;
        }
        semantic_index.left_eye_center = semanic_index["left_eye_center"].as<int>();
        semantic_index.right_eye_center = semanic_index["right_eye_center"].as<int>();
        semantic_index.nose_corner = semanic_index["nose_corner"].as<int>();
        semantic_index.mouth_left_corner = semanic_index["mouth_left_corner"].as<int>();
        semantic_index.mouth_right_corner = semanic_index["mouth_right_corner"].as<int>();
        semantic_index.mouth_lower = semanic_index["mouth_lower"].as<int>();
        semantic_index.mouth_upper = semanic_index["mouth_upper"].as<int>();
        auto left_eye_region = semanic_index["left_eye_region"];
        if (left_eye_region) {
            semantic_index.left_eye_region = left_eye_region.as<std::vector<int>>();
        }
        auto right_eye_region = semanic_index["right_eye_region"];
        if (right_eye_region) {
            semantic_index.right_eye_region = right_eye_region.as<std::vector<int>>();
        }
        auto mesh_shape = landmark_table["mesh_shape"];
        if (mesh_shape.size() > 0) {
            std::vector<float> mesh_shape_data = mesh_shape.as<std::vector<float>>();
            mean_shape_points.clear();
            mean_shape_points.resize(num_of_landmark);
            if (mesh_shape_data.size() == num_of_landmark * 2) {
                for (int i = 0; i < num_of_landmark; i++) {
                    mean_shape_points[i].SetX(mesh_shape_data[i * 2]);
                    mean_shape_points[i].SetY(mesh_shape_data[i * 2 + 1]);
                }
                mean_shape_points = LandmarkCropped(mean_shape_points);
                // auto img = inspirecv::Image::Create(192, 192, 3);
                // img.Fill(0);
                // for (int i = 0; i < num_of_landmark; i++) {
                //     auto point = mean_shape_points[i];
                //     img.DrawCircle(inspirecv::Point2i(point.GetX(), point.GetY()), 2, inspirecv::Color::Red);
                // }
                // img.Show("mean_shape");
            } else {
                INSPIRE_LOGE("norm_track_index_from_112x size is not equal to num_of_landmark: %s", name.c_str());
                return false;
            }
        } else {
            LoadDefaultMeshShape();
        }
        normalization_mode = landmark_table["normalization_mode"].as<std::string>();
        landmark_engine_name = name;

        return true;
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
    YAML::Node m_table_;
    bool m_is_available_{false};
};

}  // namespace inspire

#endif  // INSPIRE_LANDMARK_PARAM_H
