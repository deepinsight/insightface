#ifndef INSPIRSE_FACE_FACE_ACTION_DATA_H
#define INSPIRSE_FACE_FACE_ACTION_DATA_H

#include <iostream>
#include <inspirecv/inspirecv.h>
#include "middleware/utils.h"
#include "data_type.h"
#include "track_module/landmark/face_landmark_adapt.h"

namespace inspire {

enum FACE_ACTIONS { ACT_NORMAL = 0, ACT_SHAKE = 0, ACT_BLINK = 1, ACT_JAW_OPEN = 2, ACT_RAISE_HEAD = 3 };

typedef struct FaceActionList {
    int normal = 0;
    int shake = 0;
    int blink = 0;
    int jawOpen = 0;
    int raiseHead = 0;
} FaceActionList;

class INSPIRE_API FaceActionPredictor {
public:
    FaceActionPredictor(int record_list_length) {
        record_list.resize(record_list_length);
        record_list_euler.resize(record_list_length);
        record_list_eyes.resize(record_list_length);
        record_size = record_list_length;
        index = 0;
    }

    void RecordActionFrame(const std::vector<inspirecv::Point2f> &landmark, const inspirecv::Vec3f &euler_angle,
                           const inspirecv::Vec2f &eyes_status) {
        MoveRecordList();
        record_list[0] = landmark;
        record_list_euler[0] = euler_angle;
        record_list_eyes[0] = eyes_status;
        index += 1;
    }

    void Reset() {
        record_list.clear();
        record_list.resize(record_size);
        record_list_euler.clear();
        record_list_euler.resize(record_size);
        record_list_eyes.clear();
        record_list_eyes.resize(record_size);
        index = 0;
    }

    FaceActionList AnalysisFaceAction(const SemanticIndex& semantic_index) {
        FaceActionList actionRecord;
        actions.clear();
        eye_state_list.clear();
        if (index < record_list.size()) {
            actions.push_back(ACT_NORMAL);
            actionRecord.normal = 1;
        } else {
            for (int i = 0; i < record_list_eyes.size(); i++) {
                const auto &eye = record_list_eyes[i];
                std::pair<float, float> eye_state(eye[0], eye[1]);
                eye_state_list.push_back(eye_state);
            }

            // count mouth aspect ratio

            float mouth_widthwise_d = record_list[0][semantic_index.mouth_left_corner].Distance(record_list[0][semantic_index.mouth_right_corner]);
            float mouth_heightwise_d = record_list[0][semantic_index.mouth_upper].Distance(record_list[0][semantic_index.mouth_lower]);
            float mouth_aspect_ratio = mouth_heightwise_d / mouth_widthwise_d;
            if (mouth_aspect_ratio > 0.3) {
                actions.push_back(ACT_JAW_OPEN);
                actionRecord.jawOpen = 1;
            }

            int counter_eye_open = 0;
            int counter_eye_close = 0;
            for (auto &e : eye_state_list) {
                if (e.first < 0.5 || e.second < 0.5) {
                    counter_eye_close += 1;
                }
                if (e.first > 0.5 || e.second > 0.5) {
                    counter_eye_open += 1;
                }
            }
            if (counter_eye_close > 0 && counter_eye_open > 2 && record_list_euler[0][1] > -6 && record_list_euler[0][0] < 6) {
                actions.push_back(ACT_BLINK);
                actionRecord.blink = 1;
                Reset();
            }

            bool counter_head_shake_left = false;
            bool counter_head_shake_right = false;
            for (auto &e : record_list_euler) {
                if (e[1] < -6) {
                    counter_head_shake_left = true;
                }
                if (e[1] > 6) {
                    counter_head_shake_right = true;
                }
            }
            if (counter_head_shake_left && counter_head_shake_right) {
                actions.push_back(ACT_SHAKE);
                actionRecord.shake = 1;
            }

            if (record_list_euler[0][0] > 10) {
                actions.push_back(ACT_RAISE_HEAD);
                actionRecord.raiseHead = 1;
            }
        }
        return actionRecord;
    }

    std::vector<FACE_ACTIONS> GetActions() const {
        return actions;
    }

private:
    void MoveRecordList() {
        // for(int i = 0 ; i < record_list.size() - 1 ; i++){
        //    record_list[i+1] = record_list[i];
        //    record_list_euler[i+1] = record_list_euler[i];
        //}
        for (int i = record_list.size() - 1; i > 0; i--) {
            record_list[i] = record_list[i - 1];
            record_list_euler[i] = record_list_euler[i - 1];
            record_list_eyes[i] = record_list_eyes[i - 1];
        }
    }

    std::vector<std::vector<inspirecv::Point2f>> record_list;
    std::vector<inspirecv::Vec3f> record_list_euler;
    std::vector<inspirecv::Vec2f> record_list_eyes;
    std::vector<std::pair<float, float>> eye_state_list;  // pair  left right
    std::vector<float> mouth_state_list;
    std::vector<FACE_ACTIONS> actions;
    int record_size;
    int index;
};

}  // namespace inspire

#endif  // INSPIRSE_FACE_FACE_ACTION_DATA_H