#ifndef INSPIREFACE_DEST_CONST_H
#define INSPIREFACE_DEST_CONST_H

#include <inspirecv/inspirecv.h>

namespace inspire {

// face feature dimension
const int32_t FACE_FEATURE_DIM = 512;
// face crop size
const int32_t FACE_CROP_SIZE = 112;  
// similarity transform destination points
const std::vector<inspirecv::Point2f> SIMILARITY_TRANSFORM_DEST = {{38.2946, 51.6963},
            {73.5318, 51.5014},
            {56.0252, 71.7366},
            {41.5493, 92.3655},
            {70.7299, 92.2041}};

} // namespace inspire

#endif //INSPIREFACE_DEST_CONST_H
