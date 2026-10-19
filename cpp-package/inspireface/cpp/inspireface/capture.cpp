#include "capture.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>

#include "herror.h"

namespace inspire {
namespace {

constexpr uint64_t kSupportedFilters =
  CAPTURE_FILTER_FACE_COUNT | CAPTURE_FILTER_FACE_SIZE | CAPTURE_FILTER_FACE_POSITION |
  CAPTURE_FILTER_FACE_BOUNDARY | CAPTURE_FILTER_STABILITY | CAPTURE_FILTER_POSE |
  CAPTURE_FILTER_QUALITY | CAPTURE_FILTER_SHARPNESS | CAPTURE_FILTER_BRIGHTNESS |
  CAPTURE_FILTER_TRACK_COUNT;
constexpr uint32_t kMaxOutputCount = 8;

float Clamp01(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

bool IsFinite(float value) {
    return std::isfinite(value);
}

bool IsUnitRange(float value) {
    return IsFinite(value) && value >= 0.0f && value <= 1.0f;
}

bool IsNonNegative(float value) {
    return IsFinite(value) && value >= 0.0f;
}

bool ValidateConfig(const FaceCaptureConfig& config) {
    if ((config.filterMask & ~kSupportedFilters) != 0 || config.outputCount == 0 ||
        config.outputCount > kMaxOutputCount || config.maxCollectDurationMs < config.collectDurationMs ||
        ((config.filterMask & CAPTURE_FILTER_TRACK_COUNT) != 0 &&
         (config.minTrackCount == 0 ||
          config.minTrackCount > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())))) {
        return false;
    }
    if (!IsUnitRange(config.minFaceWidthRatio) || !IsUnitRange(config.maxFaceWidthRatio) ||
        config.minFaceWidthRatio > config.maxFaceWidthRatio || !IsUnitRange(config.maxCenterOffsetX) ||
        !IsUnitRange(config.maxCenterOffsetY) || !IsUnitRange(config.boundaryMarginRatio) ||
        config.boundaryMarginRatio >= 0.5f || !IsUnitRange(config.maxCenterMotionRatio) ||
        !IsUnitRange(config.maxSizeChangeRatio) || !IsNonNegative(config.maxAbsYaw) ||
        !IsNonNegative(config.maxAbsPitch) || !IsNonNegative(config.maxAbsRoll) ||
        !IsUnitRange(config.minQualityScore) || !IsUnitRange(config.minSharpnessScore) ||
        !IsUnitRange(config.minBrightnessScore) || !IsUnitRange(config.maxBrightnessScore) ||
        config.minBrightnessScore > config.maxBrightnessScore) {
        return false;
    }
    return true;
}

float FaceQualityScore(const FaceTrackWrap& face) {
    float sum = 0.0f;
    for (float value : face.quality) {
        if (!IsFinite(value) || value < 0.0f) {
            return -1.0f;
        }
        sum += value;
    }
    return Clamp01(1.0f - sum / 5.0f);
}

uint8_t Luminance(const uint8_t* pixel, int channels) {
    if (channels == 1) {
        return pixel[0];
    }
    // FrameProcess produces BGR unless explicitly configured otherwise.
    const int value = 29 * pixel[0] + 150 * pixel[1] + 77 * pixel[2];
    return static_cast<uint8_t>((value + 128) >> 8);
}

bool ComputePixelMetrics(const inspirecv::Image& image, const FaceRect& faceRect,
                         float* sharpness, float* brightness) {
    if (sharpness == nullptr && brightness == nullptr) {
        return true;
    }
    if (image.Empty() || image.Data() == nullptr || image.Width() <= 0 || image.Height() <= 0 ||
        (image.Channels() != 1 && image.Channels() < 3)) {
        return false;
    }
    const int left = std::max(0, faceRect.x);
    const int top = std::max(0, faceRect.y);
    const int right = std::min(image.Width(), faceRect.x + faceRect.width);
    const int bottom = std::min(image.Height(), faceRect.y + faceRect.height);
    if (right - left < 3 || bottom - top < 3) {
        return false;
    }

    const int stepX = std::max(1, (right - left) / 64);
    const int stepY = std::max(1, (bottom - top) / 64);
    const int channels = image.Channels();
    const size_t stride = static_cast<size_t>(image.Width()) * static_cast<size_t>(channels);
    double luminanceSum = 0.0;
    double gradientSum = 0.0;
    size_t sampleCount = 0;
    size_t gradientCount = 0;
    const uint8_t* data = image.Data();
    for (int y = top; y < bottom; y += stepY) {
        for (int x = left; x < right; x += stepX) {
            const uint8_t current = Luminance(data + static_cast<size_t>(y) * stride + static_cast<size_t>(x) * channels, channels);
            if (brightness != nullptr) {
                luminanceSum += current;
                ++sampleCount;
            }
            if (sharpness != nullptr && x + stepX < right) {
                const uint8_t adjacent = Luminance(
                  data + static_cast<size_t>(y) * stride + static_cast<size_t>(x + stepX) * channels, channels);
                gradientSum += std::abs(static_cast<int>(current) - static_cast<int>(adjacent));
                ++gradientCount;
            }
            if (sharpness != nullptr && y + stepY < bottom) {
                const uint8_t adjacent = Luminance(
                  data + static_cast<size_t>(y + stepY) * stride + static_cast<size_t>(x) * channels, channels);
                gradientSum += std::abs(static_cast<int>(current) - static_cast<int>(adjacent));
                ++gradientCount;
            }
        }
    }
    if ((brightness != nullptr && sampleCount == 0) ||
        (sharpness != nullptr && gradientCount == 0)) {
        return false;
    }
    if (brightness != nullptr) {
        *brightness = Clamp01(static_cast<float>(luminanceSum / (255.0 * sampleCount)));
    }
    if (sharpness != nullptr) {
        *sharpness = Clamp01(static_cast<float>(gradientSum / (255.0 * gradientCount)));
    }
    return true;
}

}  // namespace

class FaceCaptureSelector::Impl {
public:
    int32_t Configure(const FaceCaptureConfig& config, const CustomPipelineParameter& enabledFeatures) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ValidateConfig(config)) {
            status_ = HERR_CAPTURE_INVALID_CONFIG;
            ResetState();
            return status_;
        }
        if ((config.filterMask & CAPTURE_FILTER_POSE) != 0 && !enabledFeatures.enable_face_pose) {
            status_ = HERR_CAPTURE_REQUIRED_FEATURE_OFF;
            ResetState();
            return status_;
        }
        if ((config.filterMask & CAPTURE_FILTER_QUALITY) != 0 && !enabledFeatures.enable_face_quality) {
            status_ = HERR_CAPTURE_REQUIRED_FEATURE_OFF;
            ResetState();
            return status_;
        }
        config_ = config;
        status_ = HSUCCEED;
        ResetState();
        return HSUCCEED;
    }

    int32_t Update(int imageWidth, int imageHeight, const inspirecv::Image* image,
                   const std::vector<FaceTrackWrap>& faces, uint64_t frameId,
                   uint64_t timestampMs, FaceCaptureUpdate& update) {
        std::lock_guard<std::mutex> lock(mutex_);
        update = {};
        if (status_ != HSUCCEED) {
            return status_;
        }
        if (finished_) {
            update = LastUpdate(frameId, timestampMs);
            return HSUCCEED;
        }
        if (imageWidth <= 0 || imageHeight <= 0 ||
            (hasInput_ && (frameId <= lastFrameId_ || timestampMs <= lastTimestampMs_))) {
            return imageWidth <= 0 || imageHeight <= 0 ? HERR_INVALID_PARAM : HERR_CAPTURE_FRAME_OUT_OF_ORDER;
        }
        if ((config_.filterMask & (CAPTURE_FILTER_SHARPNESS | CAPTURE_FILTER_BRIGHTNESS)) != 0 && image == nullptr) {
            return HERR_INVALID_PARAM;
        }

        hasInput_ = true;
        lastFrameId_ = frameId;
        lastTimestampMs_ = timestampMs;
        update.frameId = frameId;
        update.timestampMs = timestampMs;
        update.evaluatedFilters = config_.filterMask;

        if (faces.empty()) {
            update.rejectReasons |= CAPTURE_REJECT_NO_FACE;
            if (activeTrackId_ >= 0 && timestampMs - lastSeenTimestampMs_ <= config_.trackLostGraceMs) {
                update.state = CAPTURE_STATE_TRACK_LOST;
                update.trackId = activeTrackId_;
            } else {
                ResetTrackState();
                update.state = CAPTURE_STATE_IDLE;
            }
            CompleteUpdate(update);
            return HSUCCEED;
        }
        if ((config_.filterMask & CAPTURE_FILTER_FACE_COUNT) != 0 && faces.size() != 1) {
            update.rejectReasons |= CAPTURE_REJECT_MULTIPLE_FACES;
        }

        const FaceTrackWrap* face = &faces.front();
        for (const auto& candidate : faces) {
            const int64_t area = static_cast<int64_t>(candidate.rect.width) * candidate.rect.height;
            const int64_t selectedArea = static_cast<int64_t>(face->rect.width) * face->rect.height;
            if (area > selectedArea || (area == selectedArea && candidate.trackId < face->trackId)) {
                face = &candidate;
            }
        }
        update.trackId = face->trackId;
        update.trackCount = face->trackCount;
        if ((config_.filterMask & CAPTURE_FILTER_TRACK_COUNT) != 0 &&
            (face->trackCount < 0 || static_cast<uint32_t>(face->trackCount) < config_.minTrackCount)) {
            update.rejectReasons |= CAPTURE_REJECT_TRACK_COUNT_TOO_LOW;
        }
        const FaceRect& rect = face->rect;
        if (rect.width <= 0 || rect.height <= 0) {
            update.rejectReasons |= CAPTURE_REJECT_FACE_OUT_OF_BOUNDS;
            CompleteUpdate(update);
            return HSUCCEED;
        }

        float widthRatio = -1.0f;
        float offsetX = -1.0f;
        float offsetY = -1.0f;

        if ((config_.filterMask & CAPTURE_FILTER_FACE_SIZE) != 0) {
            widthRatio = static_cast<float>(rect.width) / imageWidth;
            update.metrics.faceWidthRatio = widthRatio;
            update.metrics.availableMetrics |= CAPTURE_FILTER_FACE_SIZE;
            if (widthRatio < config_.minFaceWidthRatio) update.rejectReasons |= CAPTURE_REJECT_FACE_TOO_SMALL;
            if (widthRatio > config_.maxFaceWidthRatio) update.rejectReasons |= CAPTURE_REJECT_FACE_TOO_LARGE;
        }
        if ((config_.filterMask & CAPTURE_FILTER_FACE_POSITION) != 0) {
            const float centerX = rect.x + rect.width * 0.5f;
            const float centerY = rect.y + rect.height * 0.5f;
            offsetX = std::abs(centerX - imageWidth * 0.5f) / imageWidth;
            offsetY = std::abs(centerY - imageHeight * 0.5f) / imageHeight;
            update.metrics.centerOffsetX = offsetX;
            update.metrics.centerOffsetY = offsetY;
            update.metrics.availableMetrics |= CAPTURE_FILTER_FACE_POSITION;
            if (offsetX > config_.maxCenterOffsetX || offsetY > config_.maxCenterOffsetY) {
                update.rejectReasons |= CAPTURE_REJECT_FACE_OFF_CENTER;
            }
        }
        if ((config_.filterMask & CAPTURE_FILTER_FACE_BOUNDARY) != 0) {
            update.metrics.availableMetrics |= CAPTURE_FILTER_FACE_BOUNDARY;
            const float marginX = config_.boundaryMarginRatio * imageWidth;
            const float marginY = config_.boundaryMarginRatio * imageHeight;
            if (rect.x < marginX || rect.y < marginY || rect.x + rect.width > imageWidth - marginX ||
                rect.y + rect.height > imageHeight - marginY) {
                update.rejectReasons |= CAPTURE_REJECT_FACE_OUT_OF_BOUNDS;
            }
        }

        UpdateStability(*face, imageWidth, imageHeight, timestampMs, update);

        if ((config_.filterMask & CAPTURE_FILTER_POSE) != 0) {
            update.metrics.availableMetrics |= CAPTURE_FILTER_POSE;
            const float yaw = std::abs(face->face3DAngle.yaw);
            const float pitch = std::abs(face->face3DAngle.pitch);
            const float roll = std::abs(face->face3DAngle.roll);
            if (!IsFinite(yaw) || !IsFinite(pitch) || !IsFinite(roll) || yaw > config_.maxAbsYaw ||
                pitch > config_.maxAbsPitch || roll > config_.maxAbsRoll) {
                update.rejectReasons |= CAPTURE_REJECT_POSE;
                update.metrics.poseScore = 0.0f;
            } else {
                const float yawScore = config_.maxAbsYaw == 0.0f ? (yaw == 0.0f ? 1.0f : 0.0f) : 1.0f - yaw / config_.maxAbsYaw;
                const float pitchScore = config_.maxAbsPitch == 0.0f ? (pitch == 0.0f ? 1.0f : 0.0f) : 1.0f - pitch / config_.maxAbsPitch;
                const float rollScore = config_.maxAbsRoll == 0.0f ? (roll == 0.0f ? 1.0f : 0.0f) : 1.0f - roll / config_.maxAbsRoll;
                update.metrics.poseScore = Clamp01(std::min(yawScore, std::min(pitchScore, rollScore)));
            }
        }

        if ((config_.filterMask & CAPTURE_FILTER_QUALITY) != 0) {
            update.metrics.availableMetrics |= CAPTURE_FILTER_QUALITY;
            update.metrics.qualityScore = FaceQualityScore(*face);
            if (update.metrics.qualityScore < config_.minQualityScore) {
                update.rejectReasons |= CAPTURE_REJECT_QUALITY;
            }
        }

        if ((config_.filterMask & (CAPTURE_FILTER_SHARPNESS | CAPTURE_FILTER_BRIGHTNESS)) != 0) {
            float sharpness = -1.0f;
            float brightness = -1.0f;
            float* sharpnessOutput = (config_.filterMask & CAPTURE_FILTER_SHARPNESS) != 0
                                       ? &sharpness
                                       : nullptr;
            float* brightnessOutput = (config_.filterMask & CAPTURE_FILTER_BRIGHTNESS) != 0
                                        ? &brightness
                                        : nullptr;
            if (!ComputePixelMetrics(*image, rect, sharpnessOutput, brightnessOutput)) {
                if ((config_.filterMask & CAPTURE_FILTER_SHARPNESS) != 0) update.rejectReasons |= CAPTURE_REJECT_SHARPNESS;
                if ((config_.filterMask & CAPTURE_FILTER_BRIGHTNESS) != 0) update.rejectReasons |= CAPTURE_REJECT_BRIGHTNESS;
            } else {
                if ((config_.filterMask & CAPTURE_FILTER_SHARPNESS) != 0) {
                    update.metrics.availableMetrics |= CAPTURE_FILTER_SHARPNESS;
                    update.metrics.sharpnessScore = sharpness;
                    if (sharpness < config_.minSharpnessScore) update.rejectReasons |= CAPTURE_REJECT_SHARPNESS;
                }
                if ((config_.filterMask & CAPTURE_FILTER_BRIGHTNESS) != 0) {
                    update.metrics.availableMetrics |= CAPTURE_FILTER_BRIGHTNESS;
                    update.metrics.brightnessScore = brightness;
                    if (brightness < config_.minBrightnessScore || brightness > config_.maxBrightnessScore) {
                        update.rejectReasons |= CAPTURE_REJECT_BRIGHTNESS;
                    }
                }
            }
        }

        update.currentScore = Score(update.metrics, update.trackCount);
        if (update.rejectReasons == CAPTURE_REJECT_NONE) {
            AddCandidate(*face, update, frameId, timestampMs);
            if (!collecting_) {
                collecting_ = true;
                collectionStartTimestampMs_ = timestampMs;
            }
            const uint64_t elapsed = timestampMs - collectionStartTimestampMs_;
            update.state = results_.size() >= config_.outputCount && elapsed >= config_.collectDurationMs
                             ? CAPTURE_STATE_READY
                             : CAPTURE_STATE_COLLECTING;
            if (elapsed >= config_.maxCollectDurationMs) {
                finished_ = true;
                update.state = CAPTURE_STATE_FINISHED;
            }
        } else {
            update.state = CAPTURE_STATE_STABILIZING;
            if (collecting_ && timestampMs - collectionStartTimestampMs_ >= config_.maxCollectDurationMs) {
                finished_ = true;
                update.state = CAPTURE_STATE_FINISHED;
            }
        }
        CompleteUpdate(update);
        return HSUCCEED;
    }

    int32_t Finish(FaceCaptureUpdate& update) {
        std::lock_guard<std::mutex> lock(mutex_);
        update = {};
        if (status_ != HSUCCEED) return status_;
        finished_ = true;
        update = LastUpdate(lastFrameId_, lastTimestampMs_);
        return HSUCCEED;
    }

    void Reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        ResetState();
    }

    std::vector<FaceCaptureCandidate> GetResults() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return results_;
    }

    int32_t GetStatus() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return status_;
    }

    bool NeedsPixels() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return status_ == HSUCCEED &&
               (config_.filterMask & (CAPTURE_FILTER_SHARPNESS | CAPTURE_FILTER_BRIGHTNESS)) != 0;
    }

private:
    void ResetTrackState() {
        activeTrackId_ = -1;
        stableSinceTimestampMs_ = 0;
        lastSeenTimestampMs_ = 0;
        hasLastRect_ = false;
        lastRect_ = {};
    }

    void ResetState() {
        hasInput_ = false;
        finished_ = false;
        collecting_ = false;
        lastFrameId_ = 0;
        lastTimestampMs_ = 0;
        collectionStartTimestampMs_ = 0;
        results_.clear();
        lastUpdate_ = {};
        ResetTrackState();
    }

    void UpdateStability(const FaceTrackWrap& face, int imageWidth, int imageHeight,
                         uint64_t timestampMs, FaceCaptureUpdate& update) {
        const bool switchedTrack = activeTrackId_ != face.trackId;
        if (switchedTrack) {
            activeTrackId_ = face.trackId;
            stableSinceTimestampMs_ = timestampMs;
            hasLastRect_ = false;
        }
        lastSeenTimestampMs_ = timestampMs;
        if ((config_.filterMask & CAPTURE_FILTER_STABILITY) == 0) {
            return;
        }
        bool moved = false;
        if (hasLastRect_ && !switchedTrack) {
            const float previousCenterX = lastRect_.x + lastRect_.width * 0.5f;
            const float previousCenterY = lastRect_.y + lastRect_.height * 0.5f;
            const float currentCenterX = face.rect.x + face.rect.width * 0.5f;
            const float currentCenterY = face.rect.y + face.rect.height * 0.5f;
            const float dx = (currentCenterX - previousCenterX) / imageWidth;
            const float dy = (currentCenterY - previousCenterY) / imageHeight;
            const float motion = std::sqrt(dx * dx + dy * dy);
            const float previousArea = static_cast<float>(lastRect_.width) * lastRect_.height;
            const float currentArea = static_cast<float>(face.rect.width) * face.rect.height;
            const float sizeChange = previousArea > 0.0f ? std::abs(currentArea - previousArea) / previousArea
                                                         : std::numeric_limits<float>::infinity();
            moved = motion > config_.maxCenterMotionRatio || sizeChange > config_.maxSizeChangeRatio;
        }
        if (moved) stableSinceTimestampMs_ = timestampMs;
        lastRect_ = face.rect;
        hasLastRect_ = true;
        const uint64_t stableElapsed = timestampMs - stableSinceTimestampMs_;
        update.metrics.availableMetrics |= CAPTURE_FILTER_STABILITY;
        update.metrics.stabilityScore = config_.stableDurationMs == 0
                                          ? 1.0f
                                          : Clamp01(static_cast<float>(stableElapsed) / config_.stableDurationMs);
        if ((config_.filterMask & CAPTURE_FILTER_STABILITY) != 0 && stableElapsed < config_.stableDurationMs) {
            update.rejectReasons |= CAPTURE_REJECT_UNSTABLE;
        }
    }

    float Score(const FaceCaptureMetrics& metrics, int32_t trackCount) const {
        float total = 0.0f;
        uint32_t count = 0;
        const auto add = [&](float value, float& sum, uint32_t& number) {
            sum += Clamp01(value);
            ++number;
        };
        if ((config_.filterMask & CAPTURE_FILTER_FACE_COUNT) != 0) add(1.0f, total, count);
        if ((config_.filterMask & CAPTURE_FILTER_FACE_SIZE) != 0) {
            const float ideal = (config_.minFaceWidthRatio + config_.maxFaceWidthRatio) * 0.5f;
            const float radius = std::max(0.0001f, (config_.maxFaceWidthRatio - config_.minFaceWidthRatio) * 0.5f);
            add(1.0f - std::abs(metrics.faceWidthRatio - ideal) / radius, total, count);
        }
        if ((config_.filterMask & CAPTURE_FILTER_FACE_POSITION) != 0) {
            const float x = config_.maxCenterOffsetX == 0.0f ? (metrics.centerOffsetX == 0.0f ? 1.0f : 0.0f)
                                                             : 1.0f - metrics.centerOffsetX / config_.maxCenterOffsetX;
            const float y = config_.maxCenterOffsetY == 0.0f ? (metrics.centerOffsetY == 0.0f ? 1.0f : 0.0f)
                                                             : 1.0f - metrics.centerOffsetY / config_.maxCenterOffsetY;
            add(std::min(x, y), total, count);
        }
        if ((config_.filterMask & CAPTURE_FILTER_FACE_BOUNDARY) != 0) add(1.0f, total, count);
        if ((config_.filterMask & CAPTURE_FILTER_TRACK_COUNT) != 0) {
            const float trackScore = config_.minTrackCount == 0
                                       ? 1.0f
                                       : static_cast<float>(std::max(0, trackCount)) / config_.minTrackCount;
            add(trackScore, total, count);
        }
        if ((config_.filterMask & CAPTURE_FILTER_STABILITY) != 0) add(metrics.stabilityScore, total, count);
        if ((config_.filterMask & CAPTURE_FILTER_POSE) != 0) add(metrics.poseScore, total, count);
        if ((config_.filterMask & CAPTURE_FILTER_QUALITY) != 0) add(metrics.qualityScore, total, count);
        if ((config_.filterMask & CAPTURE_FILTER_SHARPNESS) != 0) add(metrics.sharpnessScore, total, count);
        if ((config_.filterMask & CAPTURE_FILTER_BRIGHTNESS) != 0) {
            const float ideal = (config_.minBrightnessScore + config_.maxBrightnessScore) * 0.5f;
            const float radius = std::max(0.0001f, (config_.maxBrightnessScore - config_.minBrightnessScore) * 0.5f);
            add(1.0f - std::abs(metrics.brightnessScore - ideal) / radius, total, count);
        }
        return count == 0 ? 1.0f : total / count;
    }

    void AddCandidate(const FaceTrackWrap& face, const FaceCaptureUpdate& update,
                      uint64_t frameId, uint64_t timestampMs) {
        FaceCaptureCandidate candidate;
        candidate.frameId = frameId;
        candidate.timestampMs = timestampMs;
        candidate.trackId = face.trackId;
        candidate.score = update.currentScore;
        candidate.face = face;
        candidate.metrics = update.metrics;

        auto duplicate = results_.end();
        if (config_.minCandidateIntervalMs > 0) {
            duplicate = std::find_if(results_.begin(), results_.end(), [&](const FaceCaptureCandidate& existing) {
                const uint64_t delta = existing.timestampMs > timestampMs ? existing.timestampMs - timestampMs
                                                                           : timestampMs - existing.timestampMs;
                return existing.trackId == candidate.trackId && delta < config_.minCandidateIntervalMs;
            });
        }
        if (duplicate != results_.end()) {
            if (candidate.score > duplicate->score) *duplicate = candidate;
        } else {
            results_.push_back(candidate);
        }
        std::sort(results_.begin(), results_.end(), [](const FaceCaptureCandidate& lhs, const FaceCaptureCandidate& rhs) {
            if (lhs.score != rhs.score) return lhs.score > rhs.score;
            if (lhs.timestampMs != rhs.timestampMs) return lhs.timestampMs < rhs.timestampMs;
            return lhs.frameId < rhs.frameId;
        });
        if (results_.size() > config_.outputCount) results_.resize(config_.outputCount);
    }

    FaceCaptureUpdate LastUpdate(uint64_t frameId, uint64_t timestampMs) const {
        FaceCaptureUpdate update = lastUpdate_;
        update.frameId = frameId;
        update.timestampMs = timestampMs;
        update.state = CAPTURE_STATE_FINISHED;
        update.progress = 1.0f;
        update.candidateCount = static_cast<uint32_t>(results_.size());
        return update;
    }

    void CompleteUpdate(FaceCaptureUpdate& update) {
        update.candidateCount = static_cast<uint32_t>(results_.size());
        if (update.state == CAPTURE_STATE_READY || update.state == CAPTURE_STATE_FINISHED) {
            update.progress = 1.0f;
        } else if (collecting_) {
            const uint64_t elapsed = update.timestampMs - collectionStartTimestampMs_;
            update.progress = config_.collectDurationMs == 0
                                ? 1.0f
                                : 0.5f + 0.5f * Clamp01(static_cast<float>(elapsed) / config_.collectDurationMs);
        } else {
            bool hasGateProgress = false;
            float gateProgress = 1.0f;
            if ((config_.filterMask & CAPTURE_FILTER_STABILITY) != 0 &&
                update.metrics.stabilityScore >= 0.0f) {
                hasGateProgress = true;
                gateProgress = std::min(gateProgress, update.metrics.stabilityScore);
            }
            if ((config_.filterMask & CAPTURE_FILTER_TRACK_COUNT) != 0 && config_.minTrackCount > 0) {
                hasGateProgress = true;
                const float trackProgress = static_cast<float>(std::max(0, update.trackCount)) /
                                            config_.minTrackCount;
                gateProgress = std::min(gateProgress, Clamp01(trackProgress));
            }
            update.progress = hasGateProgress ? 0.5f * gateProgress : 0.0f;
        }
        lastUpdate_ = update;
    }

    mutable std::mutex mutex_;
    FaceCaptureConfig config_{};
    int32_t status_{HERR_CAPTURE_INVALID_CONFIG};
    bool hasInput_{false};
    bool finished_{false};
    bool collecting_{false};
    uint64_t lastFrameId_{0};
    uint64_t lastTimestampMs_{0};
    uint64_t collectionStartTimestampMs_{0};
    int32_t activeTrackId_{-1};
    uint64_t stableSinceTimestampMs_{0};
    uint64_t lastSeenTimestampMs_{0};
    bool hasLastRect_{false};
    FaceRect lastRect_{};
    std::vector<FaceCaptureCandidate> results_;
    FaceCaptureUpdate lastUpdate_{};
};

FaceCaptureSelector::FaceCaptureSelector() : pImpl(new Impl()) {}
FaceCaptureSelector::~FaceCaptureSelector() = default;
FaceCaptureSelector::FaceCaptureSelector(FaceCaptureSelector&&) noexcept = default;
FaceCaptureSelector& FaceCaptureSelector::operator=(FaceCaptureSelector&&) noexcept = default;

int32_t FaceCaptureSelector::Configure(const FaceCaptureConfig& config,
                                       const CustomPipelineParameter& enabledFeatures) {
    return pImpl->Configure(config, enabledFeatures);
}

int32_t FaceCaptureSelector::Update(int imageWidth, int imageHeight,
                                    const std::vector<FaceTrackWrap>& faces, uint64_t frameId,
                                    uint64_t timestampMs, FaceCaptureUpdate& update) {
    return pImpl->Update(imageWidth, imageHeight, nullptr, faces, frameId, timestampMs, update);
}

int32_t FaceCaptureSelector::Update(inspirecv::FrameProcess& process,
                                    const std::vector<FaceTrackWrap>& faces, uint64_t frameId,
                                    uint64_t timestampMs, FaceCaptureUpdate& update) {
    inspirecv::Image image;
    const bool needsPixels = pImpl->NeedsPixels();
    // Pixel conversion is deferred until Update verifies that a pixel filter is active.
    // A full-resolution conversion keeps face coordinates identical to tracking output.
    if (needsPixels) {
        // Work on a copy so caller-owned output-format state remains unchanged and
        // luminance calculations always receive a known channel order.
        auto metricProcess = process;
        metricProcess.SetDestFormat(inspirecv::BGR);
        image = metricProcess.ExecuteImageScaleProcessing(1.0f, true);
    }
    const inspirecv::Image* imagePtr = image.Empty() ? nullptr : &image;
    const int width = imagePtr == nullptr ? process.GetWidth() : image.Width();
    const int height = imagePtr == nullptr ? process.GetHeight() : image.Height();
    return pImpl->Update(width, height, imagePtr, faces, frameId, timestampMs, update);
}

int32_t FaceCaptureSelector::Finish(FaceCaptureUpdate& update) {
    return pImpl->Finish(update);
}

void FaceCaptureSelector::Reset() {
    pImpl->Reset();
}

std::vector<FaceCaptureCandidate> FaceCaptureSelector::GetResults() const {
    return pImpl->GetResults();
}

int32_t FaceCaptureSelector::GetStatus() const {
    return pImpl->GetStatus();
}

}  // namespace inspire
