/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#pragma once
#ifndef INSPIRE_FACE_ATTRIBUTE_ADAPT_H
#define INSPIRE_FACE_ATTRIBUTE_ADAPT_H
#include "data_type.h"
#include "middleware/any_net_adapter.h"

namespace inspire {

/**
 * @class FaceAttributePredict
 * @brief According to the face image, three classification information of age, gender and race were extracted.
 *
 * This class inherits from AnyNet and provides methods for performing face attribute prediction.
 */
class INSPIRE_API FaceAttributePredictAdapt : public AnyNetAdapter {
public:
    int32_t INPUT_WIDTH = 112;
    int32_t INPUT_HEIGHT = 112;

    /**
     * @brief Constructor for FaceAttributePredict class.
     */
    FaceAttributePredictAdapt();

    /**
     * @brief Exec infer.
     *
     * @param bgr_affine The BGR affine matrix to perform mask prediction on.
     * @return The multi-list attribute prediction result.
     */
    std::vector<int> operator()(const inspirecv::Image& bgr_affine);

private:
    // Define primitive tag
    const std::vector<std::string> m_original_labels_ = {"Black",          "East Asian",      "Indian", "Latino_Hispanic",
                                                         "Middle Eastern", "Southeast Asian", "White"};

    // Define simplified labels
    const std::vector<std::string> m_simplified_labels_ = {"Black", "Asian", "Latino/Hispanic", "Middle Eastern", "White"};

    // Define the mapping from the original tag to the simplified tag
    const std::unordered_map<std::string, std::string> m_label_map_ = {{"Black", "Black"},
                                                                       {"East Asian", "Asian"},
                                                                       {"Indian", "Asian"},
                                                                       {"Latino_Hispanic", "Latino/Hispanic"},
                                                                       {"Middle Eastern", "Middle Eastern"},
                                                                       {"Southeast Asian", "Asian"},
                                                                       {"White", "White"}};

    // Define index maps for simplified labels
    const std::unordered_map<std::string, int> m_simplified_label_index_ = {
      {"Black", 0}, {"Asian", 1}, {"Latino/Hispanic", 2}, {"Middle Eastern", 3}, {"White", 4}};
};

}  // namespace inspire

#endif  // INSPIRE_FACE_ATTRIBUTE_ADAPT_H
