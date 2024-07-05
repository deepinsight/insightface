//
// Created by Tunm-Air13 on 2023/9/8.
//
#pragma once
#ifndef HYPERFACEREPO_GENDERPREDICT_H
#define HYPERFACEREPO_GENDERPREDICT_H
#include "data_type.h"
#include "middleware/any_net.h"

namespace inspire {

/**
 * @class FaceAttributePredict
 * @brief According to the face image, three classification information of age, gender and race were extracted.
 *
 * This class inherits from AnyNet and provides methods for performing face attribute prediction.
 */
class INSPIRE_API FaceAttributePredict : public AnyNet { 
public:
    /**
     * @brief Constructor for FaceAttributePredict class.
     */
    FaceAttributePredict();

    /**
     * @brief Exec infer.
     *
     * @param bgr_affine The BGR affine matrix to perform mask prediction on.
     * @return The multi-list attribute prediction result.
     */
    std::vector<int> operator()(const Matrix& bgr_affine);

private:
    // Define primitive tag
    const std::vector<std::string> m_original_labels_ = {
        "Black", "East Asian", "Indian", "Latino_Hispanic", "Middle Eastern", "Southeast Asian", "White"
    };

    // Define simplified labels
    const std::vector<std::string> m_simplified_labels_ = {
        "Black", "Asian", "Latino/Hispanic", "Middle Eastern", "White"
    };

    // Define the mapping from the original tag to the simplified tag
    const std::unordered_map<std::string, std::string> m_label_map_ = {
        {"Black", "Black"},
        {"East Asian", "Asian"},
        {"Indian", "Asian"},
        {"Latino_Hispanic", "Latino/Hispanic"},
        {"Middle Eastern", "Middle Eastern"},
        {"Southeast Asian", "Asian"},
        {"White", "White"}
    };

    // Define index maps for simplified labels
    const std::unordered_map<std::string, int> m_simplified_label_index_ = {
        {"Black", 0},
        {"Asian", 1},
        {"Latino/Hispanic", 2},
        {"Middle Eastern", 3},
        {"White", 4}
    };

};


}   // namespace hyper

#endif //HYPERFACEREPO_GENDERPREDICT_H
