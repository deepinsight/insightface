#include "BYTETracker.h"
#include "lapjv.h"
#include <map>
#include <iostream>
#include <algorithm>

vector<STrack *> BYTETracker::joint_stracks(vector<STrack *> &tlista, vector<STrack> &tlistb) {
    std::map<int, int> exists;
    vector<STrack *> res;
    res.reserve(tlista.size() + tlistb.size());
    for (int i = 0; i < tlista.size(); i++) {
        exists.insert(pair<int, int>(tlista[i]->track_id, 1));
        res.push_back(tlista[i]);
    }
    for (int i = 0; i < tlistb.size(); i++) {
        int tid = tlistb[i].track_id;
        if (exists.find(tid) == exists.end()) {
            exists[tid] = 1;
            res.push_back(&tlistb[i]);
        }
    }
    return res;
}

vector<STrack> BYTETracker::joint_stracks(vector<STrack> &tlista, vector<STrack> &tlistb) {
    map<int, int> exists;
    vector<STrack> res;
    res.reserve(tlista.size() + tlistb.size());
    for (int i = 0; i < tlista.size(); i++) {
        exists.insert(pair<int, int>(tlista[i].track_id, 1));
        res.push_back(tlista[i]);
    }
    for (int i = 0; i < tlistb.size(); i++) {
        int tid = tlistb[i].track_id;
        if (exists.find(tid) == exists.end()) {
            exists[tid] = 1;
            res.push_back(tlistb[i]);
        }
    }
    return res;
}

vector<STrack> BYTETracker::sub_stracks(vector<STrack> &tlista, vector<STrack> &tlistb) {
    map<int, const STrack *> stracks;
    for (int i = 0; i < tlista.size(); i++) {
        stracks.insert(pair<int, const STrack *>(tlista[i].track_id, &tlista[i]));
    }
    for (int i = 0; i < tlistb.size(); i++) {
        int tid = tlistb[i].track_id;
        if (stracks.count(tid) != 0) {
            stracks.erase(tid);
        }
    }

    vector<STrack> res;
    res.reserve(stracks.size());
    std::map<int, const STrack *>::iterator it;
    for (it = stracks.begin(); it != stracks.end(); ++it) {
        res.push_back(*it->second);
    }

    return res;
}

void BYTETracker::remove_duplicate_stracks(vector<STrack> &resa, vector<STrack> &resb, vector<STrack> &stracksa, vector<STrack> &stracksb) {
    vector<vector<float> > pdist = iou_distance(stracksa, stracksb);
    vector<pair<int, int> > pairs;
    for (int i = 0; i < pdist.size(); i++) {
        for (int j = 0; j < pdist[i].size(); j++) {
            if (pdist[i][j] < 0.15) {
                pairs.push_back(pair<int, int>(i, j));
            }
        }
    }

    vector<bool> dupa(stracksa.size(), false);
    vector<bool> dupb(stracksb.size(), false);
    for (int i = 0; i < pairs.size(); i++) {
        int timep = stracksa[pairs[i].first].frame_id - stracksa[pairs[i].first].start_frame;
        int timeq = stracksb[pairs[i].second].frame_id - stracksb[pairs[i].second].start_frame;
        if (timep > timeq)
            dupb[pairs[i].second] = true;
        else
            dupa[pairs[i].first] = true;
    }

    for (int i = 0; i < stracksa.size(); i++) {
        if (!dupa[i]) {
            resa.push_back(stracksa[i]);
        }
    }

    for (int i = 0; i < stracksb.size(); i++) {
        if (!dupb[i]) {
            resb.push_back(stracksb[i]);
        }
    }
}

void BYTETracker::linear_assignment(vector<vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
                                    vector<TrackMatch> &matches, vector<int> &unmatched_a, vector<int> &unmatched_b) {
    if (cost_matrix.size() == 0) {
        for (int i = 0; i < cost_matrix_size; i++) {
            unmatched_a.push_back(i);
        }
        for (int i = 0; i < cost_matrix_size_size; i++) {
            unmatched_b.push_back(i);
        }
        return;
    }

    vector<int> rowsol;
    vector<int> colsol;
    float c = lapjv(cost_matrix, rowsol, colsol, true, thresh);
    for (int i = 0; i < rowsol.size(); i++) {
        if (rowsol[i] >= 0) {
            matches.push_back({i, rowsol[i]});
        } else {
            unmatched_a.push_back(i);
        }
    }

    for (int i = 0; i < colsol.size(); i++) {
        if (colsol[i] < 0) {
            unmatched_b.push_back(i);
        }
    }
}

vector<vector<float> > BYTETracker::iou_distance(vector<STrack *> &atracks, vector<STrack> &btracks, int &dist_size, int &dist_size_size) {
    vector<vector<float> > cost_matrix;
    if (atracks.size() * btracks.size() == 0) {
        dist_size = atracks.size();
        dist_size_size = btracks.size();
        return cost_matrix;
    }
    dist_size = atracks.size();
    dist_size_size = btracks.size();
    cost_matrix.assign(atracks.size(), vector<float>(btracks.size()));
    for (size_t k = 0; k < btracks.size(); ++k) {
        const auto &box_b = btracks[k].tlbr;
        const float box_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1);
        for (size_t n = 0; n < atracks.size(); ++n) {
            const auto &box_a = atracks[n]->tlbr;
            float iou = 0.0f;
            const float iw = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]) + 1;
            if (iw > 0) {
                const float ih = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]) + 1;
                if (ih > 0) {
                    const float ua = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1) + box_area - iw * ih;
                    iou = iw * ih / ua;
                }
            }
            cost_matrix[n][k] = 1 - iou;
        }
    }

    return cost_matrix;
}

vector<vector<float> > BYTETracker::iou_distance(vector<STrack> &atracks, vector<STrack> &btracks) {
    vector<vector<float> > cost_matrix;
    if (atracks.empty() || btracks.empty()) {
        return cost_matrix;
    }
    cost_matrix.assign(atracks.size(), vector<float>(btracks.size()));
    for (size_t k = 0; k < btracks.size(); ++k) {
        const auto &box_b = btracks[k].tlbr;
        const float box_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1);
        for (size_t n = 0; n < atracks.size(); ++n) {
            const auto &box_a = atracks[n].tlbr;
            float iou = 0.0f;
            const float iw = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]) + 1;
            if (iw > 0) {
                const float ih = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]) + 1;
                if (ih > 0) {
                    const float ua = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1) + box_area - iw * ih;
                    iou = iw * ih / ua;
                }
            }
            cost_matrix[n][k] = 1 - iou;
        }
    }

    return cost_matrix;
}

double BYTETracker::lapjv(const vector<vector<float> > &cost, vector<int> &rowsol, vector<int> &colsol, bool extend_cost, float cost_limit,
                          bool return_cost) {
    vector<vector<float> > cost_c;
    cost_c.assign(cost.begin(), cost.end());

    vector<vector<float> > cost_c_extended;

    int n_rows = cost.size();
    int n_cols = cost[0].size();
    rowsol.resize(n_rows);
    colsol.resize(n_cols);

    int n = 0;
    if (n_rows == n_cols) {
        n = n_rows;
    } else {
        if (!extend_cost) {
            std::cout << "set extend_cost=True" << std::endl;
            // system("pause");
            exit(0);
        }
    }

    if (extend_cost || cost_limit < LONG_MAX) {
        n = n_rows + n_cols;
        cost_c_extended.resize(n);
        for (int i = 0; i < cost_c_extended.size(); i++)
            cost_c_extended[i].resize(n);

        if (cost_limit < LONG_MAX) {
            for (int i = 0; i < cost_c_extended.size(); i++) {
                for (int j = 0; j < cost_c_extended[i].size(); j++) {
                    cost_c_extended[i][j] = cost_limit / 2.0;
                }
            }
        } else {
            float cost_max = -1;
            for (int i = 0; i < cost_c.size(); i++) {
                for (int j = 0; j < cost_c[i].size(); j++) {
                    if (cost_c[i][j] > cost_max)
                        cost_max = cost_c[i][j];
                }
            }
            for (int i = 0; i < cost_c_extended.size(); i++) {
                for (int j = 0; j < cost_c_extended[i].size(); j++) {
                    cost_c_extended[i][j] = cost_max + 1;
                }
            }
        }

        for (int i = n_rows; i < cost_c_extended.size(); i++) {
            for (int j = n_cols; j < cost_c_extended[i].size(); j++) {
                cost_c_extended[i][j] = 0;
            }
        }
        for (int i = 0; i < n_rows; i++) {
            for (int j = 0; j < n_cols; j++) {
                cost_c_extended[i][j] = cost_c[i][j];
            }
        }

        cost_c.clear();
        cost_c.assign(cost_c_extended.begin(), cost_c_extended.end());
    }

    vector<double> cost_storage(static_cast<size_t>(n) * n);
    vector<double *> cost_ptr(n);
    for (int i = 0; i < n; i++)
        cost_ptr[i] = cost_storage.data() + static_cast<size_t>(i) * n;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cost_ptr[i][j] = cost_c[i][j];
        }
    }

    vector<int> x_c(n);
    vector<int> y_c(n);

    int ret = lapjv_internal(n, cost_ptr.data(), x_c.data(), y_c.data());
    if (ret != 0) {
        cout << "Calculate Wrong!" << endl;
        // system("pause");
        exit(0);
    }

    double opt = 0.0;

    if (n != n_rows) {
        for (int i = 0; i < n; i++) {
            if (x_c[i] >= n_cols)
                x_c[i] = -1;
            if (y_c[i] >= n_rows)
                y_c[i] = -1;
        }
        for (int i = 0; i < n_rows; i++) {
            rowsol[i] = x_c[i];
        }
        for (int i = 0; i < n_cols; i++) {
            colsol[i] = y_c[i];
        }

        if (return_cost) {
            for (int i = 0; i < rowsol.size(); i++) {
                if (rowsol[i] != -1) {
                    // cout << i << "\t" << rowsol[i] << "\t" << cost_ptr[i][rowsol[i]] << endl;
                    opt += cost_ptr[i][rowsol[i]];
                }
            }
        }
    } else if (return_cost) {
        for (int i = 0; i < rowsol.size(); i++) {
            opt += cost_ptr[i][rowsol[i]];
        }
    }

    return opt;
}

inspirecv::Vec3i BYTETracker::get_color(int idx) {
    idx += 3;
    return inspirecv::Vec3i{37 * idx % 255, 17 * idx % 255, 29 * idx % 255};
}
