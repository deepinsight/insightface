#ifndef INSPIRE_FACE_CHECK_H
#define INSPIRE_FACE_CHECK_H
#include "log.h"
#include "herror.h"

#define INSPIREFACE_RETURN_IF_ERROR(...)             \
    do {                                             \
        const int32_t _status = (__VA_ARGS__);       \
        if (_status != HSUCCEED) {                   \
            INSPIRE_LOGE("Error code: %d", _status); \
            return _status;                          \
        }                                            \
    } while (0)

#define INSPIREFACE_LOG_IF(severity, condition) \
    if (condition)                              \
    INSPIRE_LOG##severity

#define INSPIREFACE_CHECK(condition)                        \
    do {                                                    \
        if (!(condition)) {                                 \
            INSPIRE_LOGF("Check failed: (%s)", #condition); \
        }                                                   \
    } while (0)

#define INSPIREFACE_CHECK_MSG(condition, message)                       \
    do {                                                                \
        if (!(condition)) {                                             \
            INSPIRE_LOGF("Check failed: (%s) %s", #condition, message); \
        }                                                               \
    } while (0)

#endif  // INSPIRE_FACE_CHECK_H
