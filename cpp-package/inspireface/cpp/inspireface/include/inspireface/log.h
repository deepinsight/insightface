#ifndef INSPIRE_FACE_LOG_H
#define INSPIRE_FACE_LOG_H

#include <memory>
#include <string>
#include <cstring>
#include "data_type.h"

// Macro to extract the filename from the full path
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#ifdef ANDROID
// Android platform log macros
#define INSPIRE_ANDROID_LOG_TAG "InspireFace"
#define INSPIRE_LOGD(...) inspire::LogManager::getInstance()->logAndroid(inspire::LogLevel::ISF_LOG_DEBUG, INSPIRE_ANDROID_LOG_TAG, __VA_ARGS__)
#define INSPIRE_LOGI(...) inspire::LogManager::getInstance()->logAndroid(inspire::LogLevel::ISF_LOG_INFO, INSPIRE_ANDROID_LOG_TAG, __VA_ARGS__)
#define INSPIRE_LOGW(...) inspire::LogManager::getInstance()->logAndroid(inspire::LogLevel::ISF_LOG_WARN, INSPIRE_ANDROID_LOG_TAG, __VA_ARGS__)
#define INSPIRE_LOGE(...) inspire::LogManager::getInstance()->logAndroid(inspire::LogLevel::ISF_LOG_ERROR, INSPIRE_ANDROID_LOG_TAG, __VA_ARGS__)
#define INSPIRE_LOGF(...) inspire::LogManager::getInstance()->logAndroid(inspire::LogLevel::ISF_LOG_FATAL, INSPIRE_ANDROID_LOG_TAG, __VA_ARGS__)
#else
// Standard platform log macros
#define INSPIRE_LOGD(...) \
    inspire::LogManager::getInstance()->logStandard(inspire::LogLevel::ISF_LOG_DEBUG, __FILENAME__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define INSPIRE_LOGI(...) inspire::LogManager::getInstance()->logStandard(inspire::LogLevel::ISF_LOG_INFO, "", "", -1, __VA_ARGS__)
#define INSPIRE_LOGW(...) inspire::LogManager::getInstance()->logStandard(inspire::LogLevel::ISF_LOG_WARN, "", "", -1, __VA_ARGS__)
#define INSPIRE_LOGE(...) inspire::LogManager::getInstance()->logStandard(inspire::LogLevel::ISF_LOG_ERROR, "", "", -1, __VA_ARGS__)
#define INSPIRE_LOGF(...) inspire::LogManager::getInstance()->logStandard(inspire::LogLevel::ISF_LOG_FATAL, "", "", -1, __VA_ARGS__)
#endif

// Macro to set the global log level
#define INSPIRE_SET_LOG_LEVEL(level) inspire::LogManager::getInstance()->setLogLevel(level)

namespace inspire {

// Log levels
enum LogLevel { ISF_LOG_NONE = 0, ISF_LOG_DEBUG, ISF_LOG_INFO, ISF_LOG_WARN, ISF_LOG_ERROR, ISF_LOG_FATAL };

/**
 * @class LogManager
 * @brief A singleton class for logging messages to the console or Android logcat.
 *
 * This class provides methods to log messages of different severity levels (DEBUG, INFO, WARN, ERROR, FATAL)
 * to the console or Android logcat based on the current log level setting.
 *
 * Implementation details are hidden using the PIMPL (Pointer to Implementation) pattern.
 */
class INSPIRE_API_EXPORT LogManager {
public:
    // Get the singleton instance
    static LogManager* getInstance();

    // Destructor
    ~LogManager();

    // Set the log level
    void setLogLevel(LogLevel level);

    // Get the current log level
    LogLevel getLogLevel() const;

#ifdef ANDROID
    // Method for logging on the Android platform
    void logAndroid(LogLevel level, const char* tag, const char* format, ...) const;
#else
    // Method for standard platform logging
    void logStandard(LogLevel level, const char* filename, const char* function, int line, const char* format, ...) const;
#endif

private:
    // Private constructor for singleton pattern
    LogManager();

    // Disable copy construction and assignment
    LogManager(const LogManager&) = delete;
    LogManager& operator=(const LogManager&) = delete;

    // Forward declaration of the implementation class
    class Impl;

    // Pointer to implementation
    std::unique_ptr<Impl> pImpl;

    // Static instance for singleton pattern
    static LogManager* instance;
};

}  // namespace inspire

#endif  // INSPIRE_FACE_LOG_H