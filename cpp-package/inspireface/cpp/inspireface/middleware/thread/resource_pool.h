#ifndef INSPIRE_RESOURCE_POOL_H
#define INSPIRE_RESOURCE_POOL_H

#include <iostream>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <memory>
#include <functional>

namespace inspire {
namespace parallel {

/**
 * @brief ResourcePool is a thread-safe resource pool that can be used to manage resources in a multi-threaded environment.
 * @tparam Resource The type of the resource to be managed.
 */
template <typename Resource>
class ResourcePool {
public:
    using ResourceDeleter = std::function<void(Resource&)>;

    class ResourceGuard {
    public:
        ResourceGuard(Resource& resource, ResourcePool& pool) : m_resource(resource), m_pool(pool), m_valid(true) {}

        // Move constructor
        ResourceGuard(ResourceGuard&& other) noexcept : m_resource(other.m_resource), m_pool(other.m_pool), m_valid(other.m_valid) {
            other.m_valid = false;
        }

        // Disable copy
        ResourceGuard(const ResourceGuard&) = delete;
        ResourceGuard& operator=(const ResourceGuard&) = delete;

        ~ResourceGuard() {
            if (m_valid) {
                m_pool.ReturnResource(std::move(m_resource));
            }
        }

        Resource* operator->() {
            return &m_resource;
        }

        Resource& operator*() {
            return m_resource;
        }

    private:
        Resource& m_resource;
        ResourcePool& m_pool;
        bool m_valid;
    };

    explicit ResourcePool(size_t size, ResourceDeleter deleter = nullptr) : m_deleter(deleter) {
        m_resources.reserve(size);
    }

    ~ResourcePool() {
        if (m_deleter) {
            std::lock_guard<std::mutex> lock(m_mutex);
            for (auto& resource : m_resources) {
                m_deleter(resource);
            }
        }
    }

    void AddResource(Resource&& resource) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_resources.push_back(std::move(resource));
        m_available_resources.push(&m_resources.back());
        m_cv.notify_one();
    }

    // Acquire resource (blocking mode)
    ResourceGuard AcquireResource() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [this] { return !m_available_resources.empty(); });

        Resource* resource = m_available_resources.front();
        m_available_resources.pop();

        return ResourceGuard(*resource, *this);
    }

    // Try to acquire resource (non-blocking mode), returns nullptr if no resource available
    std::unique_ptr<ResourceGuard> TryAcquireResource() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_available_resources.empty()) {
            return nullptr;
        }

        Resource* resource = m_available_resources.front();
        m_available_resources.pop();

        return std::unique_ptr<ResourceGuard>(new ResourceGuard(*resource, *this));
    }

    // Acquire resource with timeout, returns nullptr if timeout
    std::unique_ptr<ResourceGuard> AcquireResource(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (!m_cv.wait_for(lock, timeout, [this] { return !m_available_resources.empty(); })) {
            return nullptr;
        }

        Resource* resource = m_available_resources.front();
        m_available_resources.pop();

        return std::unique_ptr<ResourceGuard>(new ResourceGuard(*resource, *this));
    }

    size_t AvailableCount() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_available_resources.size();
    }

    size_t TotalCount() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_resources.size();
    }

private:
    void ReturnResource(Resource&& resource) {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (auto& stored_resource : m_resources) {
            if (&stored_resource == &resource) {
                m_available_resources.push(&stored_resource);
                m_cv.notify_one();
                break;
            }
        }
    }

private:
    mutable std::mutex m_mutex;
    std::condition_variable m_cv;
    std::vector<Resource> m_resources;            // Store actual resources
    std::queue<Resource*> m_available_resources;  // Queue of available resources
    ResourceDeleter m_deleter;                    // Resource cleanup callback

    friend class ResourceGuard;
};

}  // namespace parallel
}  // namespace inspire

#endif  // INSPIRE_RESOURCE_POOL_H