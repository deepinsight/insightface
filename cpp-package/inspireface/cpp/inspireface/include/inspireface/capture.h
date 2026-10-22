#ifndef INSPIRE_FACE_CAPTURE_H
#define INSPIRE_FACE_CAPTURE_H

#include <cstdint>
#include <memory>
#include <vector>

#include "data_type.h"
#include "face_wrapper.h"
#include "frame_process.h"

namespace inspire {

enum FaceCaptureFilter : uint64_t {
    CAPTURE_FILTER_NONE = 0,
    CAPTURE_FILTER_FACE_COUNT = UINT64_C(1) << 0,
    CAPTURE_FILTER_FACE_SIZE = UINT64_C(1) << 1,
    CAPTURE_FILTER_FACE_POSITION = UINT64_C(1) << 2,
    CAPTURE_FILTER_FACE_BOUNDARY = UINT64_C(1) << 3,
    CAPTURE_FILTER_STABILITY = UINT64_C(1) << 4,
    CAPTURE_FILTER_POSE = UINT64_C(1) << 5,
    CAPTURE_FILTER_QUALITY = UINT64_C(1) << 6,
    CAPTURE_FILTER_SHARPNESS = UINT64_C(1) << 7,
    CAPTURE_FILTER_BRIGHTNESS = UINT64_C(1) << 8,
    CAPTURE_FILTER_TRACK_COUNT = UINT64_C(1) << 9,
};

enum FaceCaptureRejectReason : uint64_t {
    CAPTURE_REJECT_NONE = 0,
    CAPTURE_REJECT_NO_FACE = UINT64_C(1) << 0,
    CAPTURE_REJECT_MULTIPLE_FACES = UINT64_C(1) << 1,
    CAPTURE_REJECT_FACE_TOO_SMALL = UINT64_C(1) << 2,
    CAPTURE_REJECT_FACE_TOO_LARGE = UINT64_C(1) << 3,
    CAPTURE_REJECT_FACE_OFF_CENTER = UINT64_C(1) << 4,
    CAPTURE_REJECT_FACE_OUT_OF_BOUNDS = UINT64_C(1) << 5,
    CAPTURE_REJECT_UNSTABLE = UINT64_C(1) << 6,
    CAPTURE_REJECT_POSE = UINT64_C(1) << 7,
    CAPTURE_REJECT_QUALITY = UINT64_C(1) << 8,
    CAPTURE_REJECT_SHARPNESS = UINT64_C(1) << 9,
    CAPTURE_REJECT_BRIGHTNESS = UINT64_C(1) << 10,
    CAPTURE_REJECT_TRACK_COUNT_TOO_LOW = UINT64_C(1) << 11,
};

enum FaceCaptureState {
    CAPTURE_STATE_IDLE = 0,
    CAPTURE_STATE_STABILIZING = 1,
    CAPTURE_STATE_COLLECTING = 2,
    CAPTURE_STATE_READY = 3,
    CAPTURE_STATE_FINISHED = 4,
    CAPTURE_STATE_TRACK_LOST = 5,
};

struct INSPIRE_API_EXPORT FaceCaptureConfig {
    uint64_t filterMask = CAPTURE_FILTER_FACE_COUNT | CAPTURE_FILTER_FACE_SIZE |
                          CAPTURE_FILTER_FACE_POSITION | CAPTURE_FILTER_FACE_BOUNDARY |
                          CAPTURE_FILTER_STABILITY | CAPTURE_FILTER_TRACK_COUNT;
    uint32_t outputCount = 1;
    uint32_t minTrackCount = 5;
    uint64_t stableDurationMs = 300;
    uint64_t collectDurationMs = 800;
    uint64_t maxCollectDurationMs = 3000;
    uint64_t trackLostGraceMs = 300;
    uint64_t minCandidateIntervalMs = 150;

    float minFaceWidthRatio = 0.12f;
    float maxFaceWidthRatio = 0.75f;
    float maxCenterOffsetX = 0.25f;
    float maxCenterOffsetY = 0.25f;
    float boundaryMarginRatio = 0.02f;
    float maxCenterMotionRatio = 0.025f;
    float maxSizeChangeRatio = 0.08f;
    float maxAbsYaw = 25.0f;
    float maxAbsPitch = 25.0f;
    float maxAbsRoll = 20.0f;
    float minQualityScore = 0.60f;
    float minSharpnessScore = 0.03f;
    float minBrightnessScore = 0.15f;
    float maxBrightnessScore = 0.90f;
};

struct INSPIRE_API_EXPORT FaceCaptureMetrics {
    uint64_t availableMetrics = 0;
    float faceWidthRatio = -1.0f;
    float centerOffsetX = -1.0f;
    float centerOffsetY = -1.0f;
    float stabilityScore = -1.0f;
    float poseScore = -1.0f;
    float qualityScore = -1.0f;
    float sharpnessScore = -1.0f;
    float brightnessScore = -1.0f;
};

struct INSPIRE_API_EXPORT FaceCaptureCandidate {
    uint64_t frameId = 0;
    uint64_t timestampMs = 0;
    int32_t trackId = -1;
    float score = 0.0f;
    FaceTrackWrap face{};
    FaceCaptureMetrics metrics{};
};

struct INSPIRE_API_EXPORT FaceCaptureUpdate {
    FaceCaptureState state = CAPTURE_STATE_IDLE;
    uint64_t frameId = 0;
    uint64_t timestampMs = 0;
    int32_t trackId = -1;
    int32_t trackCount = 0;
    uint64_t evaluatedFilters = 0;
    uint64_t rejectReasons = CAPTURE_REJECT_NONE;
    float progress = 0.0f;
    float currentScore = 0.0f;
    uint32_t candidateCount = 0;
    FaceCaptureMetrics metrics{};
};

/**
 * Stateful, synchronous face-capture policy. It never owns a camera, thread,
 * or inference session. Callers may feed results from Session::FaceDetectAndTrack.
 * A selector instance serializes its own calls; different instances are independent.
 */
class INSPIRE_API_EXPORT FaceCaptureSelector {
public:
    FaceCaptureSelector();
    ~FaceCaptureSelector();

    FaceCaptureSelector(FaceCaptureSelector&&) noexcept;
    FaceCaptureSelector& operator=(FaceCaptureSelector&&) noexcept;
    FaceCaptureSelector(const FaceCaptureSelector&) = delete;
    FaceCaptureSelector& operator=(const FaceCaptureSelector&) = delete;

    int32_t Configure(const FaceCaptureConfig& config, const CustomPipelineParameter& enabledFeatures = {});

    int32_t Update(int imageWidth, int imageHeight, const std::vector<FaceTrackWrap>& faces,
                   uint64_t frameId, uint64_t timestampMs, FaceCaptureUpdate& update);

    int32_t Update(inspirecv::FrameProcess& process, const std::vector<FaceTrackWrap>& faces,
                   uint64_t frameId, uint64_t timestampMs, FaceCaptureUpdate& update);

    int32_t Finish(FaceCaptureUpdate& update);
    void Reset();
    std::vector<FaceCaptureCandidate> GetResults() const;
    int32_t GetStatus() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

}  // namespace inspire

#endif  // INSPIRE_FACE_CAPTURE_H
