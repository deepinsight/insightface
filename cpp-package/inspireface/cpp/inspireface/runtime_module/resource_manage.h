/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#pragma once
#ifndef INSPIREFACE_RESOURCE_MANAGE_H
#define INSPIREFACE_RESOURCE_MANAGE_H

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "log.h"

#ifndef INSPIRE_API
#define INSPIRE_API
#endif

#define RESOURCE_MANAGE inspire::ResourceManager::getInstance()

namespace inspire {

using ResourceHandle = uintptr_t;
static_assert(sizeof(ResourceHandle) >= sizeof(void *), "ResourceHandle must preserve every pointer bit");

struct ResourceCounts {
    uint64_t total_created = 0;
    uint64_t total_released = 0;
    uint64_t live = 0;
};

struct ResourceStatistics {
    ResourceCounts sessions;
    ResourceCounts streams;
    ResourceCounts image_bitmaps;
    ResourceCounts face_features;
    ResourceCounts face_result_snapshots;
    ResourceCounts face_capture_sessions;
};

/**
 * @brief Tracks the live C API handles and cumulative lifecycle counters.
 *
 * Only live handles are retained, so memory use is bounded by peak concurrent
 * resources rather than the number of resources ever created.
 */
class INSPIRE_API ResourceManager {
public:
    /**
     * Pins the object backing a validated opaque handle for the duration of a
     * C API call. Releasing the public handle removes it from the registry
     * immediately, while destruction is deferred until the final lease ends.
     */
    class ResourceLease {
    public:
        ResourceLease() = default;

        explicit operator bool() const {
            return acquired_;
        }

    private:
        friend class ResourceManager;

        ResourceLease(bool acquired, std::shared_ptr<void> owner) : acquired_(acquired), owner_(std::move(owner)) {}

        bool acquired_{false};
        std::shared_ptr<void> owner_;
    };

    ResourceManager(const ResourceManager &) = delete;
    ResourceManager &operator=(const ResourceManager &) = delete;

    static ResourceManager *getInstance() {
        static ResourceManager instance;
        return &instance;
    }

    bool createSession(ResourceHandle handle, std::shared_ptr<void> owner = {}) {
        std::lock_guard<std::mutex> lock(mutex_);
        return Register(session_registry_, handle, std::move(owner));
    }

    bool releaseSession(ResourceHandle handle) {
        std::shared_ptr<void> released_owner;
        bool released = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released = Release(session_registry_, handle, released_owner);
        }
        return released;
    }

    bool createStream(ResourceHandle handle, std::shared_ptr<void> owner = {}) {
        std::lock_guard<std::mutex> lock(mutex_);
        return Register(stream_registry_, handle, std::move(owner));
    }

    bool releaseStream(ResourceHandle handle) {
        std::shared_ptr<void> released_owner;
        bool released = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released = Release(stream_registry_, handle, released_owner);
        }
        return released;
    }

    bool createImageBitmap(ResourceHandle handle, std::shared_ptr<void> owner = {}) {
        std::lock_guard<std::mutex> lock(mutex_);
        return Register(image_bitmap_registry_, handle, std::move(owner));
    }

    bool releaseImageBitmap(ResourceHandle handle) {
        std::shared_ptr<void> released_owner;
        bool released = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released = Release(image_bitmap_registry_, handle, released_owner);
        }
        return released;
    }

    bool createFaceFeature(ResourceHandle handle, std::shared_ptr<void> owner = {}) {
        std::lock_guard<std::mutex> lock(mutex_);
        return Register(face_feature_registry_, handle, std::move(owner));
    }

    bool releaseFaceFeature(ResourceHandle handle) {
        std::shared_ptr<void> released_owner;
        bool released = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released = Release(face_feature_registry_, handle, released_owner);
        }
        return released;
    }

    bool createFaceResultSnapshot(ResourceHandle handle, std::shared_ptr<void> owner = {}) {
        std::lock_guard<std::mutex> lock(mutex_);
        return Register(face_result_snapshot_registry_, handle, std::move(owner));
    }

    bool releaseFaceResultSnapshot(ResourceHandle handle) {
        std::shared_ptr<void> released_owner;
        bool released = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released = Release(face_result_snapshot_registry_, handle, released_owner);
        }
        return released;
    }

    bool createFaceCaptureSession(ResourceHandle handle, std::shared_ptr<void> owner = {}) {
        std::lock_guard<std::mutex> lock(mutex_);
        return Register(face_capture_session_registry_, handle, std::move(owner));
    }

    bool releaseFaceCaptureSession(ResourceHandle handle) {
        std::shared_ptr<void> released_owner;
        bool released = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released = Release(face_capture_session_registry_, handle, released_owner);
        }
        return released;
    }

    ResourceLease acquireSession(ResourceHandle handle) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return Acquire(session_registry_, handle);
    }

    ResourceLease acquireStream(ResourceHandle handle) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return Acquire(stream_registry_, handle);
    }

    ResourceLease acquireImageBitmap(ResourceHandle handle) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return Acquire(image_bitmap_registry_, handle);
    }

    ResourceLease acquireFaceResultSnapshot(ResourceHandle handle) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return Acquire(face_result_snapshot_registry_, handle);
    }

    ResourceLease acquireFaceCaptureSession(ResourceHandle handle) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return Acquire(face_capture_session_registry_, handle);
    }

    bool isSessionLive(ResourceHandle handle) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return Contains(session_registry_, handle);
    }

    bool isStreamLive(ResourceHandle handle) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return Contains(stream_registry_, handle);
    }

    bool isImageBitmapLive(ResourceHandle handle) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return Contains(image_bitmap_registry_, handle);
    }

    bool isFaceFeatureLive(ResourceHandle handle) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return Contains(face_feature_registry_, handle);
    }

    bool isFaceResultSnapshotLive(ResourceHandle handle) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return Contains(face_result_snapshot_registry_, handle);
    }

    bool isFaceCaptureSessionLive(ResourceHandle handle) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return Contains(face_capture_session_registry_, handle);
    }

    std::vector<ResourceHandle> getUnreleasedSessions() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return Snapshot(session_registry_);
    }

    std::vector<ResourceHandle> getUnreleasedStreams() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return Snapshot(stream_registry_);
    }

    std::vector<ResourceHandle> getUnreleasedImageBitmaps() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return Snapshot(image_bitmap_registry_);
    }

    std::vector<ResourceHandle> getUnreleasedFaceFeatures() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return Snapshot(face_feature_registry_);
    }

    std::vector<ResourceHandle> getUnreleasedFaceResultSnapshots() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return Snapshot(face_result_snapshot_registry_);
    }

    ResourceStatistics getResourceStatistics() const {
        std::lock_guard<std::mutex> lock(mutex_);
        ResourceStatistics statistics;
        statistics.sessions = Counts(session_registry_);
        statistics.streams = Counts(stream_registry_);
        statistics.image_bitmaps = Counts(image_bitmap_registry_);
        statistics.face_features = Counts(face_feature_registry_);
        statistics.face_result_snapshots = Counts(face_result_snapshot_registry_);
        statistics.face_capture_sessions = Counts(face_capture_session_registry_);
        return statistics;
    }

    void printResourceStatistics() const {
        const ResourceStatistics statistics = getResourceStatistics();
        INSPIRE_LOGI("================================================================");
        INSPIRE_LOGI("%-15s%-15s%-15s%-15s", "Resource Name", "Total Created", "Total Released", "Not Released");
        INSPIRE_LOGI("----------------------------------------------------------------");
        PrintCounts("Session", statistics.sessions);
        PrintCounts("Stream", statistics.streams);
        PrintCounts("Bitmap", statistics.image_bitmaps);
        PrintCounts("FaceFeature", statistics.face_features);
        PrintCounts("FaceSnapshot", statistics.face_result_snapshots);
        PrintCounts("FaceCapture", statistics.face_capture_sessions);
        INSPIRE_LOGI("================================================================");
    }

private:
    struct Registry {
        std::unordered_map<ResourceHandle, std::shared_ptr<void>> live_handles;
        uint64_t total_created = 0;
        uint64_t total_released = 0;
    };

    ResourceManager() = default;

    static bool Register(Registry &registry, ResourceHandle handle, std::shared_ptr<void> owner) {
        if (handle == 0) {
            return false;
        }
        const bool inserted = registry.live_handles.emplace(handle, std::move(owner)).second;
        if (inserted) {
            ++registry.total_created;
        }
        return inserted;
    }

    static bool Release(Registry &registry, ResourceHandle handle, std::shared_ptr<void>& released_owner) {
        const auto position = registry.live_handles.find(handle);
        if (handle == 0 || position == registry.live_handles.end()) {
            return false;
        }
        released_owner = std::move(position->second);
        registry.live_handles.erase(position);
        ++registry.total_released;
        return true;
    }

    static ResourceLease Acquire(const Registry &registry, ResourceHandle handle) {
        if (handle == 0) {
            return {};
        }
        const auto position = registry.live_handles.find(handle);
        if (position == registry.live_handles.end()) {
            return {};
        }
        return ResourceLease(true, position->second);
    }

    static bool Contains(const Registry &registry, ResourceHandle handle) {
        return handle != 0 && registry.live_handles.find(handle) != registry.live_handles.end();
    }

    static std::vector<ResourceHandle> Snapshot(const Registry &registry) {
        std::vector<ResourceHandle> handles;
        handles.reserve(registry.live_handles.size());
        for (const auto& entry : registry.live_handles) {
            handles.push_back(entry.first);
        }
        std::sort(handles.begin(), handles.end());
        return handles;
    }

    static ResourceCounts Counts(const Registry &registry) {
        ResourceCounts counts;
        counts.total_created = registry.total_created;
        counts.total_released = registry.total_released;
        counts.live = registry.live_handles.size();
        return counts;
    }

    static void PrintCounts(const char *name, const ResourceCounts &counts) {
        INSPIRE_LOGI("%-15s%-15llu%-15llu%-15llu", name, static_cast<unsigned long long>(counts.total_created),
                     static_cast<unsigned long long>(counts.total_released), static_cast<unsigned long long>(counts.live));
    }

    mutable std::mutex mutex_;
    Registry session_registry_;
    Registry stream_registry_;
    Registry image_bitmap_registry_;
    Registry face_feature_registry_;
    Registry face_result_snapshot_registry_;
    Registry face_capture_session_registry_;
};

}  // namespace inspire

#endif  // INSPIREFACE_RESOURCE_MANAGE_H
