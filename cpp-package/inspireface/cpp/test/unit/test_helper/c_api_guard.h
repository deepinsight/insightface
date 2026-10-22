#ifndef INSPIREFACE_TEST_C_API_GUARD_H
#define INSPIREFACE_TEST_C_API_GUARD_H

#include "inspireface/c_api/inspireface.h"

namespace inspireface_test {

template <typename Handle, HResult (*Release)(Handle)>
class UniqueHandle {
public:
    UniqueHandle() = default;
    explicit UniqueHandle(Handle handle) : handle_(handle) {}
    ~UniqueHandle() {
        Reset();
    }

    UniqueHandle(const UniqueHandle&) = delete;
    UniqueHandle& operator=(const UniqueHandle&) = delete;

    UniqueHandle(UniqueHandle&& other) noexcept : handle_(other.ReleaseOwnership()) {}

    UniqueHandle& operator=(UniqueHandle&& other) noexcept {
        if (this != &other) {
            Reset();
            handle_ = other.ReleaseOwnership();
        }
        return *this;
    }

    Handle Get() const {
        return handle_;
    }

    Handle* Put() {
        Reset();
        return &handle_;
    }

    Handle ReleaseOwnership() {
        Handle value = handle_;
        handle_ = nullptr;
        return value;
    }

    HResult Reset(Handle replacement = nullptr) {
        HResult result = HSUCCEED;
        if (handle_ != nullptr) {
            result = Release(handle_);
        }
        handle_ = replacement;
        return result;
    }

    explicit operator bool() const {
        return handle_ != nullptr;
    }

private:
    Handle handle_ = nullptr;
};

using UniqueSession = UniqueHandle<HFSession, HFReleaseInspireFaceSession>;
using UniqueImageStream = UniqueHandle<HFImageStream, HFReleaseImageStream>;
using UniqueImageBitmap = UniqueHandle<HFImageBitmap, HFReleaseImageBitmap>;
using UniqueFaceResultSnapshot = UniqueHandle<HFFaceResultSnapshot, HFReleaseFaceResultSnapshot>;
using UniqueFaceCaptureSession = UniqueHandle<HFFaceCaptureSession, HFReleaseFaceCaptureSession>;

class UniqueFaceFeature {
public:
    UniqueFaceFeature() = default;
    ~UniqueFaceFeature() {
        Reset();
    }

    UniqueFaceFeature(const UniqueFaceFeature&) = delete;
    UniqueFaceFeature& operator=(const UniqueFaceFeature&) = delete;

    HFFaceFeature* Put() {
        Reset();
        return &feature_;
    }

    HFFaceFeature& Get() {
        return feature_;
    }

    const HFFaceFeature& Get() const {
        return feature_;
    }

    HResult Reset() {
        if (feature_.data == nullptr) {
            feature_.size = 0;
            return HSUCCEED;
        }
        return HFReleaseFaceFeature(&feature_);
    }

private:
    HFFaceFeature feature_ = {0, nullptr};
};

class ScopedSdkTermination {
public:
    ScopedSdkTermination() {
        TerminateIfNeeded();
    }

    ~ScopedSdkTermination() {
        TerminateIfNeeded();
    }

    ScopedSdkTermination(const ScopedSdkTermination&) = delete;
    ScopedSdkTermination& operator=(const ScopedSdkTermination&) = delete;

private:
    static void TerminateIfNeeded() {
        HInt32 status = HF_STATUS_DISABLE;
        if (HFQueryInspireFaceLaunchStatus(&status) == HSUCCEED && status == HF_STATUS_ENABLE) {
            HFTerminateInspireFace();
        }
    }
};

}  // namespace inspireface_test

#endif  // INSPIREFACE_TEST_C_API_GUARD_H
