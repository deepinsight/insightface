/**
 * Created by Jingyu Yan
 * @date 2025-01-20
 */
#include <iostream>
#include "settings/test_settings.h"
#include "unit/test_helper/test_help.h"
#include "inspireface/include/inspireface/similarity_converter.h"

TEST_CASE("test_similarity_converter", "[similarity_converter]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    SECTION("test_similarity_converter_0.42") {
        inspire::SimilarityConverterConfig config;
        config.threshold = 0.42;
        config.middleScore = 0.6;
        config.steepness = 8.0;
        config.outputMin = 0.01;
        config.outputMax = 1.0;
        inspire::SimilarityConverter similarity_converter(config);
        std::vector<double> test_points = {-0.80, -0.20, 0.02, 0.10, 0.25, 0.30, 0.48, 0.70, 0.80, 0.90, 1.00};
        std::vector<double> expected_scores = {0.0101, 0.0201, 0.0661, 0.1113, 0.2819, 0.3673, 0.7074, 0.9334, 0.9689, 0.9858, 0.9936};
        REQUIRE(test_points.size() == expected_scores.size());

        for (size_t i = 0; i < test_points.size(); ++i) {
            double cosine = test_points[i];
            double similarity = similarity_converter.convert(cosine);
            REQUIRE(similarity == Approx(expected_scores[i]).epsilon(0.01));
        }
    }

    SECTION("test_similarity_converter_0.32") {
        inspire::SimilarityConverterConfig config;
        config.threshold = 0.32;
        config.middleScore = 0.6;
        config.steepness = 10.0;
        config.outputMin = 0.02;
        config.outputMax = 1.0;
        inspire::SimilarityConverter similarity_converter(config);
        std::vector<double> test_points = {-0.80, -0.20, 0.02, 0.10, 0.25, 0.32, 0.50, 0.70, 0.80, 0.90, 1.00};
        std::vector<double> expected_scores = {0.0200, 0.0278, 0.0860, 0.1557, 0.4302, 0.6000, 0.8997, 0.9851, 0.9945, 0.9980, 0.9992};
        REQUIRE(test_points.size() == expected_scores.size());

        for (size_t i = 0; i < test_points.size(); ++i) {
            double cosine = test_points[i];
            double similarity = similarity_converter.convert(cosine);
            REQUIRE(similarity == Approx(expected_scores[i]).epsilon(0.01));
        }
    }
}