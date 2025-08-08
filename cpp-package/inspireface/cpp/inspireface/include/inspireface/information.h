/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#include "data_type.h"

#ifndef INSPIRE_FACE_INFORMATION_H
#define INSPIRE_FACE_INFORMATION_H

#ifdef __cplusplus
extern "C" {
#endif

// C-style function declarations, using API export macro
INSPIRE_API_EXPORT const char* GetInspireFaceVersionMajorStr();
INSPIRE_API_EXPORT const char* GetInspireFaceVersionMinorStr();
INSPIRE_API_EXPORT const char* GetInspireFaceVersionPatchStr();
INSPIRE_API_EXPORT const char* GetInspireFaceExtendedInformation();

#ifdef __cplusplus
}
#endif

// C-style macro definitions
#define INSPIRE_FACE_VERSION_MAJOR_STR GetInspireFaceVersionMajorStr()
#define INSPIRE_FACE_VERSION_MINOR_STR GetInspireFaceVersionMinorStr()
#define INSPIRE_FACE_VERSION_PATCH_STR GetInspireFaceVersionPatchStr()
#define INSPIRE_FACE_EXTENDED_INFORMATION GetInspireFaceExtendedInformation()

#endif  // INSPIRE_FACE_INFORMATION_H
