/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#include "log.h"
#include <mutex>
#include <cstdarg>
#include <cstring>
#include <iostream>

#ifdef ANDROID
#include <android/log.h>
#endif

namespace inspire {

// Implementation class for LogManager
class LogManager::Impl {
public:
    Impl() : currentLevel(LogLevel::ISF_LOG_INFO) {}

    LogLevel currentLevel;
    static std::mutex mutex;
};

// Static initialization
std::mutex LogManager::Impl::mutex;
LogManager* LogManager::instance = nullptr;

// Constructor
LogManager::LogManager() : pImpl(std::make_unique<Impl>()) {}

// Destructor
LogManager::~LogManager() = default;

// Get singleton instance
LogManager* LogManager::getInstance() {
    std::lock_guard<std::mutex> lock(Impl::mutex);
    if (instance == nullptr) {
        instance = new LogManager();
    }
    return instance;
}

// Set log level
void LogManager::setLogLevel(LogLevel level) {
    pImpl->currentLevel = level;
}

// Get log level
LogLevel LogManager::getLogLevel() const {
    return pImpl->currentLevel;
}

#ifdef ANDROID
// Android logging implementation
void LogManager::logAndroid(LogLevel level, const char* tag, const char* format, ...) const {
    if (pImpl->currentLevel == LogLevel::ISF_LOG_NONE || level < pImpl->currentLevel)
        return;

    int androidLevel;
    switch (level) {
        case LogLevel::ISF_LOG_DEBUG:
            androidLevel = ANDROID_LOG_DEBUG;
            break;
        case LogLevel::ISF_LOG_INFO:
            androidLevel = ANDROID_LOG_INFO;
            break;
        case LogLevel::ISF_LOG_WARN:
            androidLevel = ANDROID_LOG_WARN;
            break;
        case LogLevel::ISF_LOG_ERROR:
            androidLevel = ANDROID_LOG_ERROR;
            break;
        case LogLevel::ISF_LOG_FATAL:
            androidLevel = ANDROID_LOG_FATAL;
            break;
        default:
            androidLevel = ANDROID_LOG_DEFAULT;
    }

    va_list args;
    va_start(args, format);
    __android_log_vprint(androidLevel, tag, format, args);
    va_end(args);

    // If the log level is fatal, flush the error stream and abort the program
    if (level == LogLevel::ISF_LOG_FATAL) {
        std::flush(std::cerr);
        abort();
    }
}
#else
// Standard logging implementation
void LogManager::logStandard(LogLevel level, const char* filename, const char* function, int line, const char* format, ...) const {
    // Check whether the current level is LOG NONE or the log level is not enough to log
    if (pImpl->currentLevel == LogLevel::ISF_LOG_NONE || level < pImpl->currentLevel)
        return;

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

    // Set text color for different log levels, but only if not on iOS
#ifndef TARGET_OS_IOS
    if (level == LogLevel::ISF_LOG_ERROR || level == LogLevel::ISF_LOG_FATAL) {
        printf("\033[1;31m");  // Red color for errors and fatal issues
    } else if (level == LogLevel::ISF_LOG_WARN) {
        printf("\033[1;33m");  // Yellow color for warnings
    }
#endif

    // Print the actual log message
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);

    // Reset text color if needed, but only if not on iOS
#ifndef TARGET_OS_IOS
    if (level == LogLevel::ISF_LOG_ERROR || level == LogLevel::ISF_LOG_WARN || level == LogLevel::ISF_LOG_FATAL) {
        printf("\033[0m");  // Reset color
    }
#endif

    printf("\n");  // New line after log message

    // If the log level is fatal, flush the error stream and abort the program
    if (level == LogLevel::ISF_LOG_FATAL) {
        std::flush(std::cerr);
        abort();
    }
}
#endif

}  // namespace inspire