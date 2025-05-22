/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#pragma once
#ifndef INSPIREFACE_RESOURCE_MANAGE_H
#define INSPIREFACE_RESOURCE_MANAGE_H
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <memory>
#include <iomanip>  // For std::setw and std::left
#include <vector>
#include "log.h"
#ifndef INSPIRE_API
#define INSPIRE_API
#endif

#define RESOURCE_MANAGE inspire::ResourceManager::getInstance()

namespace inspire {

/**
 * @brief ResourceManager is a singleton class that manages the creation and release of sessions and image streams.
 * It uses hash tables to store session and image stream handles, and provides methods to create, release, and query these resources.
 * The ResourceManager class is designed to be used in a multi-threaded environment, and it uses a mutex to synchronize access to its data structures.
 */
class INSPIRE_API ResourceManager {
private:
    // Private static instance pointer
    static std::unique_ptr<ResourceManager> instance;
    static std::mutex mutex;

    // Use hash tables to store session and image stream handles
    std::unordered_map<long, bool> sessionMap;
    std::unordered_map<long, bool> streamMap;
    std::unordered_map<long, bool> imageBitmapMap;
    std::unordered_map<long, bool> faceFeatureMap;

    // The private constructor guarantees singletons
    ResourceManager() {}

public:
    // Remove copy constructors and assignment operators
    ResourceManager(const ResourceManager&) = delete;
    ResourceManager& operator=(const ResourceManager&) = delete;

    // Method of obtaining singleton instance
    static ResourceManager* getInstance() {
        std::lock_guard<std::mutex> lock(mutex);
        if (!instance) {
            instance.reset(new ResourceManager());
        }
        return instance.get();
    }

    // Method of obtaining singleton instance
    void createSession(long handle) {
        std::lock_guard<std::mutex> lock(mutex);
        sessionMap[handle] = false;  // false indicates that it is not released
    }

    // Release session
    bool releaseSession(long handle) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = sessionMap.find(handle);
        if (it != sessionMap.end() && !it->second) {
            it->second = true;  // Mark as released
            return true;
        }
        return false;  // Release failed, possibly because the handle could not be found or was
                       // released
    }

    // Create and record image streams
    void createStream(long handle) {
        std::lock_guard<std::mutex> lock(mutex);
        streamMap[handle] = false;  // false indicates that it is not released
    }

    // Release image stream
    bool releaseStream(long handle) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = streamMap.find(handle);
        if (it != streamMap.end() && !it->second) {
            it->second = true;  // Mark as released
            return true;
        }
        return false;  // Release failed, possibly because the handle could not be found or was
                       // released
    }

    // Create and record image bitmaps
    void createImageBitmap(long handle) {
        std::lock_guard<std::mutex> lock(mutex);
        imageBitmapMap[handle] = false;  // false indicates that it is not released
    }

    // Release image bitmap
    bool releaseImageBitmap(long handle) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = imageBitmapMap.find(handle);
        if (it != imageBitmapMap.end() && !it->second) {
            it->second = true;  // Mark as released
            return true;
        }
        return false;  // Release failed, possibly because the handle could not be found or was released
    }

    // Create and record face features
    void createFaceFeature(long handle) {
        std::lock_guard<std::mutex> lock(mutex);
        faceFeatureMap[handle] = false;  // false indicates that it is not released
    }

    // Release face feature
    bool releaseFaceFeature(long handle) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = faceFeatureMap.find(handle);
        if (it != faceFeatureMap.end() && !it->second) {
            it->second = true;  // Mark as released
            return true;
        }
        return false;  // Release failed, possibly because the handle could not be found or was released
    }

    // Gets a list of unreleased session handles
    std::vector<long> getUnreleasedSessions() {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<long> unreleasedSessions;
        for (const auto& entry : sessionMap) {
            if (!entry.second) {
                unreleasedSessions.push_back(entry.first);
            }
        }
        return unreleasedSessions;
    }

    // Gets a list of unreleased image stream handles
    std::vector<long> getUnreleasedStreams() {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<long> unreleasedStreams;
        for (const auto& entry : streamMap) {
            if (!entry.second) {
                unreleasedStreams.push_back(entry.first);
            }
        }
        return unreleasedStreams;
    }

    // Gets a list of unreleased image bitmap handles
    std::vector<long> getUnreleasedImageBitmaps() {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<long> unreleasedImageBitmaps;
        for (const auto& entry : imageBitmapMap) {
            if (!entry.second) {
                unreleasedImageBitmaps.push_back(entry.first);
            }
        }
        return unreleasedImageBitmaps;
    }

    // Gets a list of unreleased face feature handles
    std::vector<long> getUnreleasedFaceFeatures() {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<long> unreleasedFaceFeatures;
        for (const auto& entry : faceFeatureMap) {
            if (!entry.second) {
                unreleasedFaceFeatures.push_back(entry.first);
            }
        }
        return unreleasedFaceFeatures;
    }

    // Method to print resource management statistics
    void printResourceStatistics() {
        std::lock_guard<std::mutex> lock(mutex);
        INSPIRE_LOGI("================================================================");
        INSPIRE_LOGI("%-15s%-15s%-15s%-15s", "Resource Name", "Total Created", "Total Released", "Not Released");
        INSPIRE_LOGI("----------------------------------------------------------------");

        // Print session statistics
        int totalSessionsCreated = sessionMap.size();
        int totalSessionsReleased = 0;
        int sessionsNotReleased = 0;
        for (const auto& entry : sessionMap) {
            if (entry.second)
                ++totalSessionsReleased;
            if (!entry.second)
                ++sessionsNotReleased;
        }
        INSPIRE_LOGI("%-15s%-15d%-15d%-15d", "Session", totalSessionsCreated, totalSessionsReleased, sessionsNotReleased);

        // Print stream statistics
        int totalStreamsCreated = streamMap.size();
        int totalStreamsReleased = 0;
        int streamsNotReleased = 0;
        for (const auto& entry : streamMap) {
            if (entry.second)
                ++totalStreamsReleased;
            if (!entry.second)
                ++streamsNotReleased;
        }
        INSPIRE_LOGI("%-15s%-15d%-15d%-15d", "Stream", totalStreamsCreated, totalStreamsReleased, streamsNotReleased);

        // Print bitmap statistics
        int totalBitmapsCreated = imageBitmapMap.size();
        int totalBitmapsReleased = 0;
        int bitmapsNotReleased = 0;
        for (const auto& entry : imageBitmapMap) {
            if (entry.second)
                ++totalBitmapsReleased;
            if (!entry.second)
                ++bitmapsNotReleased;
        }
        INSPIRE_LOGI("%-15s%-15d%-15d%-15d", "Bitmap", totalBitmapsCreated, totalBitmapsReleased, bitmapsNotReleased);

        // Print face feature statistics
        int totalFeaturesCreated = faceFeatureMap.size();
        int totalFeaturesReleased = 0;
        int featuresNotReleased = 0;
        for (const auto& entry : faceFeatureMap) {
            if (entry.second)
                ++totalFeaturesReleased;
            if (!entry.second)
                ++featuresNotReleased;
        }
        INSPIRE_LOGI("%-15s%-15d%-15d%-15d", "FaceFeature", totalFeaturesCreated, totalFeaturesReleased, featuresNotReleased);
        INSPIRE_LOGI("================================================================");
    }
};

}  // namespace inspire

#endif  // INSPIREFACE_RESOURCE_MANAGE_H