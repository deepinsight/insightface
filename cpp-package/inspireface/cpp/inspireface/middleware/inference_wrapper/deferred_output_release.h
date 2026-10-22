#ifndef INSPIREFACE_DEFERRED_OUTPUT_RELEASE_H
#define INSPIREFACE_DEFERRED_OUTPUT_RELEASE_H

namespace inference_wrapper_detail {

// Keeps runtime-owned output buffers alive until the next inference starts.
// This matches the lifetime of OutputTensorInfo views without adding a copy to
// every inference. The owner must call Release() before destroying the runtime.
class DeferredOutputRelease {
public:
    bool HasAcquiredOutputs() const {
        return has_acquired_outputs_;
    }

    void MarkAcquired() {
        has_acquired_outputs_ = true;
    }

    template <typename ReleaseFunction>
    bool Release(ReleaseFunction&& release_function) {
        if (!has_acquired_outputs_) {
            return true;
        }
        if (release_function() != 0) {
            return false;
        }
        has_acquired_outputs_ = false;
        return true;
    }

    // Use only after the runtime has already been destroyed and therefore owns
    // no output buffers that can still be released.
    void ResetAfterRuntimeDestroyed() {
        has_acquired_outputs_ = false;
    }

private:
    bool has_acquired_outputs_{false};
};

// Releases the previous output lease, runs inference, and retains the newly
// acquired outputs when population succeeds. A population failure releases the
// new outputs immediately.
template <typename ReleaseFunction, typename RunFunction, typename PopulateFunction>
bool RunWithDeferredOutputRelease(DeferredOutputRelease& output_release,
                                  ReleaseFunction&& release_function,
                                  RunFunction&& run_function,
                                  PopulateFunction&& populate_function) {
    if (!output_release.Release(release_function)) {
        return false;
    }
    if (run_function() != 0) {
        return false;
    }

    output_release.MarkAcquired();
    bool populated = false;
    try {
        populated = populate_function();
    } catch (...) {
        output_release.Release(release_function);
        throw;
    }
    if (!populated) {
        output_release.Release(release_function);
        return false;
    }
    return true;
}

}  // namespace inference_wrapper_detail

#endif  // INSPIREFACE_DEFERRED_OUTPUT_RELEASE_H
