//
// Created by Tunm-Air13 on 2024/3/26.
//

#include <iostream>
#include "settings/test_settings.h"
#include "inspireface/c_api/inspireface.h"
#include "opencv2/opencv.hpp"
#include "unit/test_helper/simple_csv_writer.h"
#include "unit/test_helper/test_help.h"
#include "unit/test_helper/test_tools.h"

TEST_CASE("test_Evaluation", "[face_evaluation") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    SECTION("Test compare tools") {
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        parameter.enable_recognition = 1;
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 5, -1, -1, &session);
        REQUIRE(ret == HSUCCEED);

        float mostSim = -1.0f;
        auto succ = FindMostSimilarScoreFromTwoPic(session,
                                                   GET_DATA("data/bulk/jntm.jpg"),
                                                   GET_DATA("data/bulk/kun.jpg"),
                                                   mostSim);
        CHECK(succ);
        TEST_PRINT("kun v kun :{}", mostSim);

        succ = FindMostSimilarScoreFromTwoPic(session,
                                              GET_DATA("data/bulk/jntm.jpg"),
                                              GET_DATA("data/bulk/Rob_Lowe_0001.jpg"),
                                              mostSim);
        CHECK(succ);
        TEST_PRINT("kun v other :{}", mostSim);

        succ = FindMostSimilarScoreFromTwoPic(session,
                                              GET_DATA("data/bulk/kun.jpg"),
                                              GET_DATA("data/bulk/view.jpg"),
                                              mostSim);
        CHECK(!succ);
        TEST_PRINT("kun v other :{}", mostSim);

        // finish
        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);
    }


    SECTION("Test LFW evaluation") {
#ifdef ISF_ENABLE_TEST_EVALUATION
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        parameter.enable_recognition = 1;
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 5, -1, -1, &session);
        REQUIRE(ret == HSUCCEED);
        std::vector<int> labels;
        std::vector<float> confidences;
        auto pairs = ReadPairs(getTestLFWFunneledEvaTxt());
        // Hide cursor
        show_console_cursor(false);
        BlockProgressBar bar{
                option::BarWidth{60},
                option::Start{"["},
                option::End{"]"},
                option::PostfixText{"Extracting face features"},
                option::ForegroundColor{Color::white}  ,
                option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}
        };
        auto progress = 0.0f;

        for (int i = 0; i < pairs.size(); ++i) {
            bar.set_progress(progress);

            auto &pair = pairs[i];
            std::string person1, person2;
            int imgNum1, imgNum2;
            std::string imgPath1, imgPath2;
            int match;
            if (pair.size() == 3) {
                person1 = pair[0];
                imgNum1 = std::stoi(pair[1]);
                imgNum2 = std::stoi(pair[2]);
                imgPath1 = PathJoin(PathJoin(getLFWFunneledDir(), person1),
                                            person1 + "_" + zfill(imgNum1, 4) + ".jpg");
                imgPath2 = PathJoin(PathJoin(getLFWFunneledDir(), person1),
                                            person1 + "_" + zfill(imgNum2, 4) + ".jpg");
                match = 1;
            } else {
                person1 = pair[0];
                imgNum1 = std::stoi(pair[1]);
                person2 = pair[2];
                imgNum2 = std::stoi(pair[3]);
                imgPath1 = PathJoin(PathJoin(getLFWFunneledDir(), person1),
                                            person1 + "_" + zfill(imgNum1, 4) + ".jpg");
                imgPath2 = PathJoin(PathJoin(getLFWFunneledDir(), person2),
                                            person2 + "_" + zfill(imgNum2, 4) + ".jpg");
                match = 0;
            }

            float mostSim;
            auto succ = FindMostSimilarScoreFromTwoPic(session, imgPath1, imgPath2, mostSim);
            if (!succ) {
                continue;
            }

            labels.push_back(match);
            confidences.push_back(mostSim);

            // Update progress
            progress = 100.0f * (float)(i + 1) / pairs.size();
        }
        // Show cursor
        show_console_cursor(true);

        REQUIRE(labels.size() == confidences.size());
        TEST_PRINT("scan pair: {}", labels.size());
        bar.set_progress(100.0f);

        auto result = FindBestThreshold(confidences, labels);
        TEST_PRINT("Best Threshold: {}, Best Accuracy: {}", result.first, result.second);
        EvaluationRecord record(getEvaluationRecordFile());
        record.insertEvaluationData(TEST_MODEL_FILE, "LFW", result.second, result.first);
        // finish
        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);
#endif
    }

}