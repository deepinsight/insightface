#include "STrack.h"

STrack::STrack(const TrackBox &tlwh_, float score) : _tlwh(tlwh_), tlwh{}, tlbr{} {

    is_activated = false;
    track_id = 0;
    state = TrackState::New;

    static_tlwh();
    static_tlbr();
    frame_id = 0;
    tracklet_len = 0;
    this->score = score;
    start_frame = 0;
}

void STrack::activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id, int track_id) {
    this->kalman_filter = kalman_filter;
    this->track_id = track_id;

    TrackBox xyah = tlwh_to_xyah(this->_tlwh);
    DETECTBOX xyah_box;
    xyah_box[0] = xyah[0];
    xyah_box[1] = xyah[1];
    xyah_box[2] = xyah[2];
    xyah_box[3] = xyah[3];
    auto mc = this->kalman_filter.initiate(xyah_box);
    this->mean = mc.first;
    this->covariance = mc.second;

    static_tlwh();
    static_tlbr();

    this->tracklet_len = 0;
    this->state = TrackState::Tracked;
    if (frame_id == 1) {
        this->is_activated = true;
    }
    // this->is_activated = true;
    this->frame_id = frame_id;
    this->start_frame = frame_id;
}

void STrack::re_activate(STrack &new_track, int frame_id, int new_track_id) {
    TrackBox xyah = tlwh_to_xyah(new_track.tlwh);
    DETECTBOX xyah_box;
    xyah_box[0] = xyah[0];
    xyah_box[1] = xyah[1];
    xyah_box[2] = xyah[2];
    xyah_box[3] = xyah[3];
    auto mc = this->kalman_filter.update(this->mean, this->covariance, xyah_box);
    this->mean = mc.first;
    this->covariance = mc.second;

    static_tlwh();
    static_tlbr();

    this->tracklet_len = 0;
    this->state = TrackState::Tracked;
    this->is_activated = true;
    this->frame_id = frame_id;
    this->score = new_track.score;
    if (new_track_id > 0)
        this->track_id = new_track_id;
}

void STrack::update(STrack &new_track, int frame_id) {
    this->frame_id = frame_id;
    this->tracklet_len++;

    TrackBox xyah = tlwh_to_xyah(new_track.tlwh);
    DETECTBOX xyah_box;
    xyah_box[0] = xyah[0];
    xyah_box[1] = xyah[1];
    xyah_box[2] = xyah[2];
    xyah_box[3] = xyah[3];

    auto mc = this->kalman_filter.update(this->mean, this->covariance, xyah_box);
    this->mean = mc.first;
    this->covariance = mc.second;

    static_tlwh();
    static_tlbr();

    this->state = TrackState::Tracked;
    this->is_activated = true;

    this->score = new_track.score;
}

void STrack::static_tlwh() {
    if (this->state == TrackState::New) {
        tlwh[0] = _tlwh[0];
        tlwh[1] = _tlwh[1];
        tlwh[2] = _tlwh[2];
        tlwh[3] = _tlwh[3];
        return;
    }

    tlwh[0] = mean[0];
    tlwh[1] = mean[1];
    tlwh[2] = mean[2];
    tlwh[3] = mean[3];

    tlwh[2] *= tlwh[3];
    tlwh[0] -= tlwh[2] / 2;
    tlwh[1] -= tlwh[3] / 2;
}

void STrack::static_tlbr() {
    tlbr = tlwh;
    tlbr[2] += tlbr[0];
    tlbr[3] += tlbr[1];
}

TrackBox STrack::tlwh_to_xyah(TrackBox tlwh_tmp) {
    TrackBox tlwh_output = tlwh_tmp;
    tlwh_output[0] += tlwh_output[2] / 2;
    tlwh_output[1] += tlwh_output[3] / 2;
    tlwh_output[2] /= tlwh_output[3];
    return tlwh_output;
}

TrackBox STrack::to_xyah() {
    return tlwh_to_xyah(tlwh);
}

TrackBox STrack::tlbr_to_tlwh(TrackBox tlbr) {
    tlbr[2] -= tlbr[0];
    tlbr[3] -= tlbr[1];
    return tlbr;
}

void STrack::mark_lost() {
    state = TrackState::Lost;
}

void STrack::mark_removed() {
    state = TrackState::Removed;
}

int STrack::end_frame() {
    return this->frame_id;
}

void STrack::multi_predict(vector<STrack *> &stracks, byte_kalman::KalmanFilter &kalman_filter) {
    for (int i = 0; i < stracks.size(); i++) {
        if (stracks[i]->state != TrackState::Tracked) {
            stracks[i]->mean[7] = 0;
        }
        kalman_filter.predict(stracks[i]->mean, stracks[i]->covariance);
    }
}
