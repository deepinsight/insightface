#pragma once
#include <array>
#include "kalmanFilter.h"

using namespace std;

enum TrackState { New = 0, Tracked, Lost, Removed };

using TrackBox = std::array<float, 4>;

class STrack {
public:
    STrack(const TrackBox &tlwh_, float score);
    ~STrack() = default;
    STrack(const STrack &) = default;
    STrack &operator=(const STrack &) = default;
    STrack(STrack &&) noexcept = default;
    STrack &operator=(STrack &&) noexcept = default;
    TrackBox static tlbr_to_tlwh(TrackBox tlbr);
    void static multi_predict(vector<STrack *> &stracks, byte_kalman::KalmanFilter &kalman_filter);
    void static_tlwh();
    void static_tlbr();
    TrackBox tlwh_to_xyah(TrackBox tlwh_tmp);
    TrackBox to_xyah();
    void mark_lost();
    void mark_removed();
    int end_frame();

    void activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id, int track_id);
    void re_activate(STrack &new_track, int frame_id, int new_track_id = 0);
    void update(STrack &new_track, int frame_id);

public:
    bool is_activated;
    int track_id;
    int state;

    TrackBox _tlwh;
    TrackBox tlwh;
    TrackBox tlbr;
    int frame_id;
    int tracklet_len;
    int start_frame;

    KAL_MEAN mean;
    KAL_COVA covariance;
    float score;

private:
    byte_kalman::KalmanFilter kalman_filter;
};
