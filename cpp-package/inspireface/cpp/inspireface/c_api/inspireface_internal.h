/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#ifndef INSPIREFACE_INTERNAL_H
#define INSPIREFACE_INTERNAL_H

#include "engine/face_session.h"

typedef struct HF_FaceAlgorithmSession {
    inspire::FaceSession impl;  ///< Implementation of the face context.
} HF_FaceAlgorithmSession;      ///< Handle for managing face context.

typedef struct HF_CameraStream {
    inspirecv::FrameProcess impl;  ///< Implementation of the camera stream.
} HF_CameraStream;                 ///< Handle for managing camera stream.

typedef struct HF_ImageBitmap {
    inspirecv::Image impl;  ///< Implementation of the image bitmap.
} HF_ImageBitmap;           ///< Handle for managing image bitmap.

#endif  // INSPIREFACE_INTERNAL_H
