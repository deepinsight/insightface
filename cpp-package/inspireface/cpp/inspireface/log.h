#ifndef LOG_MANAGER_H
#define LOG_MANAGER_H

#include <mutex>
#include <string>
#include <cstdarg>
#include <cstring>
#include <iostream>

#ifndef INSPIRE_API
#define INSPIRE_API
#endif

// Macro to extract the filename from the full path
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#ifdef ANDROID
// Android platform log macros
#include <android/log.h>
#define INSPIRE_ANDROID_LOG_TAG "InspireFace"
#define INSPIRE_LOGD(...) inspire::LogManager::getInstance()->logAndroid(inspire::ISF_LOG_DEBUG, INSPIRE_ANDROID_LOG_TAG, __VA_ARGS__)
#define INSPIRE_LOGI(...) inspire::LogManager::getInstance()->logAndroid(inspire::ISF_LOG_INFO, INSPIRE_ANDROID_LOG_TAG, __VA_ARGS__)
#define INSPIRE_LOGW(...) inspire::LogManager::getInstance()->logAndroid(inspire::ISF_LOG_WARN, INSPIRE_ANDROID_LOG_TAG, __VA_ARGS__)
#define INSPIRE_LOGE(...) inspire::LogManager::getInstance()->logAndroid(inspire::ISF_LOG_ERROR, INSPIRE_ANDROID_LOG_TAG, __VA_ARGS__)
#define INSPIRE_LOGF(...) inspire::LogManager::getInstance()->logAndroid(inspire::ISF_LOG_FATAL, INSPIRE_ANDROID_LOG_TAG, __VA_ARGS__)
#else
// Standard platform log macros
#define INSPIRE_LOGD(...) inspire::LogManager::getInstance()->logStandard(inspire::ISF_LOG_DEBUG, __FILENAME__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define INSPIRE_LOGI(...) inspire::LogManager::getInstance()->logStandard(inspire::ISF_LOG_INFO, "", "", -1, __VA_ARGS__)
#define INSPIRE_LOGW(...) inspire::LogManager::getInstance()->logStandard(inspire::ISF_LOG_WARN, __FILENAME__, "", __LINE__, __VA_ARGS__)
#define INSPIRE_LOGE(...) inspire::LogManager::getInstance()->logStandard(inspire::ISF_LOG_ERROR, __FILENAME__, "", __LINE__, __VA_ARGS__)
#define INSPIRE_LOGF(...) inspire::LogManager::getInstance()->logStandard(inspire::ISF_LOG_FATAL, __FILENAME__, __FUNCTION__, __LINE__, __VA_ARGS__)
#endif


// Macro to set the global log level
#define INSPIRE_SET_LOG_LEVEL(level) inspire::LogManager::getInstance()->setLogLevel(level)

namespace inspire {

// Log levels
enum LogLevel {
    ISF_LOG_NONE = 0,
    ISF_LOG_DEBUG,
    ISF_LOG_INFO,
    ISF_LOG_WARN,
    ISF_LOG_ERROR,
    ISF_LOG_FATAL
};

class INSPIRE_API LogManager {
private:
    LogLevel currentLevel;
    static LogManager* instance;
    static std::mutex mutex;

    // Private constructor
    LogManager() : currentLevel(ISF_LOG_DEBUG) {}  // Default log level is DEBUG

public:
    // Disable copy construction and assignment
    LogManager(const LogManager&) = delete;
    LogManager& operator=(const LogManager&) = delete;

    // Get the singleton instance
    static LogManager* getInstance() {
        std::lock_guard<std::mutex> lock(mutex);
        if (instance == nullptr) {
            instance = new LogManager();
        }
        return instance;
    }

    // Set the log level
    void setLogLevel(LogLevel level) {
        currentLevel = level;
    }

    // Get the current log level
    LogLevel getLogLevel() const {
        return currentLevel;
    }

#ifdef ANDROID
    // Method for logging on the Android platform
    void logAndroid(LogLevel level, const char* tag, const char* format, ...) const {
        if (currentLevel == ISF_LOG_NONE || level < currentLevel) return;

        int androidLevel;
        switch (level) {
            case ISF_LOG_DEBUG: androidLevel = ANDROID_LOG_DEBUG; break;
            case ISF_LOG_INFO: androidLevel = ANDROID_LOG_INFO; break;
            case ISF_LOG_WARN: androidLevel = ANDROID_LOG_WARN; break;
            case ISF_LOG_ERROR: androidLevel = ANDROID_LOG_ERROR; break;
            case ISF_LOG_FATAL: androidLevel = ANDROID_LOG_FATAL; break;
            default: androidLevel = ANDROID_LOG_DEFAULT;
        }

        va_list args;
        va_start(args, format);
        __android_log_vprint(androidLevel, tag, format, args);
        va_end(args);
    }
#else
    // Method for standard platform logging
    void logStandard(LogLevel level, const char* filename, const char* function, int line, const char* format, ...) const {
        // Check whether the current level is LOG NONE or the log level is not enough to log
        if (currentLevel == ISF_LOG_NONE || level < currentLevel) return;

        // Build log prefix dynamically based on available data
        bool hasPrintedPrefix = false;
        if (filename && strlen(filename) > 0) {
            printf("[%s]", filename);
            hasPrintedPrefix = true;
        }
        if (function && strlen(function) > 0) {
            printf("[%s]", function);
            hasPrintedPrefix = true;
        }
        if (line != -1) {
            printf("[%d]", line);
            hasPrintedPrefix = true;
        }

        // Only add colon and space if any prefix was printed
        if (hasPrintedPrefix) {
            printf(": ");
        }

        // Set text color for different log levels
        if (level == ISF_LOG_ERROR || level == ISF_LOG_FATAL) {
            printf("\033[1;31m");  // Red color for errors and fatal issues
        } else if (level == ISF_LOG_WARN) {
            printf("\033[1;33m");  // Yellow color for warnings
        }

        // Print the actual log message
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);

        // Reset text color if needed
        if (level == ISF_LOG_ERROR || level == ISF_LOG_WARN || level == ISF_LOG_FATAL) {
            printf("\033[0m");  // Reset color
        }

        printf("\n"); // New line after log message
    }


#endif

};

}   // namespace inspire

#endif // LOG_MANAGER_H
