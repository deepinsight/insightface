//
// Created by Tunm-Air13 on 2023/9/11.
//

#include "feature_block.h"
#include "log.h"
#include "feature_hub/features_block/implement/feature_block_none.h"

#ifdef FEATURE_BLOCK_ENABLE_OPENCV
#include "feature_hub/features_block/implement/feature_block_opencv.h"
#endif

namespace inspire {


FeatureBlock *FeatureBlock::Create(const MatrixCore crop_type, int32_t features_max, int32_t feature_length) {
    FeatureBlock* p = nullptr;
    switch (crop_type) {
#ifdef FEATURE_BLOCK_ENABLE_OPENCV
        case MC_OPENCV:
            p = new FeatureBlockOpenCV(features_max, feature_length);
            break;
#endif
#ifdef FEATURE_BLOCK_ENABLE_EIGEN
        case MC_EIGEN:
            LOGD("Not Implement");
            break;
#endif
        case MC_NONE:
            INSPIRE_LOGD("Not Implement");
            break;
    }

    if (p != nullptr) {
        p->m_matrix_core_ = crop_type;
        p->m_features_max_ = features_max;          // Number of facial features
        p->m_feature_length_ = feature_length;      // Face feature length (default: 512)
        p->m_feature_state_.resize(features_max, FEATURE_STATE::IDLE);
        p->m_tag_list_.resize(features_max, "None");
        p->m_custom_id_list_.resize(features_max, -1);
    } else {
        INSPIRE_LOGE("Create FeatureBlock error.");
    }

    return p;
}

}   // namespace hyper