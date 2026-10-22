#include <cmath>
#include <limits>

#include "inspireface/c_api/inspireface.h"
#include "settings/test_settings.h"

namespace {

class SimilarityConfigReset {
public:
    SimilarityConfigReset() {
        REQUIRE(HFGetCosineSimilarityConverter(&original_) == HSUCCEED);
    }
    ~SimilarityConfigReset() {
        HFUpdateCosineSimilarityConverter(original_);
    }

private:
    HFSimilarityConverterConfig original_ = {};
};

}  // namespace

TEST_CASE("C API similarity converter is monotonic and round-trips configuration", "[api][contract][similarity_converter]") {
    SimilarityConfigReset reset;
    HFSimilarityConverterConfig config = {};
    REQUIRE(HFGetCosineSimilarityConverter(&config) == HSUCCEED);
    CHECK(HFGetCosineSimilarityConverter(nullptr) == HERR_INVALID_PARAM);

    float recommended = 0.0f;
    REQUIRE(HFGetRecommendedCosineThreshold(&recommended) == HSUCCEED);
    CHECK(std::isfinite(recommended));

    float low = 0.0f;
    float middle = 0.0f;
    float high = 0.0f;
    REQUIRE(HFCosineSimilarityConvertToPercentage(-1.0f, &low) == HSUCCEED);
    REQUIRE(HFCosineSimilarityConvertToPercentage(config.threshold, &middle) == HSUCCEED);
    REQUIRE(HFCosineSimilarityConvertToPercentage(1.0f, &high) == HSUCCEED);
    CHECK(low < middle);
    CHECK(middle < high);
    CHECK(low >= config.outputMin);
    CHECK(high <= config.outputMax);

    HFSimilarityConverterConfig replacement = {0.35f, 0.55f, 6.0f, 0.02f, 0.98f};
    REQUIRE(HFUpdateCosineSimilarityConverter(replacement) == HSUCCEED);
    HFSimilarityConverterConfig actual = {};
    REQUIRE(HFGetCosineSimilarityConverter(&actual) == HSUCCEED);
    CHECK(actual.threshold == Approx(replacement.threshold));
    CHECK(actual.middleScore == Approx(replacement.middleScore));
    CHECK(actual.steepness == Approx(replacement.steepness));
    CHECK(actual.outputMin == Approx(replacement.outputMin));
    CHECK(actual.outputMax == Approx(replacement.outputMax));
}

TEST_CASE("C API similarity converter rejects non-finite inputs and invalid ranges", "[api][contract][similarity_converter][boundary]") {
    SimilarityConfigReset reset;
    float output = 12.0f;
    CHECK(HFCosineSimilarityConvertToPercentage(0.5f, nullptr) == HERR_INVALID_PARAM);
    CHECK(HFCosineSimilarityConvertToPercentage(std::numeric_limits<float>::quiet_NaN(), &output) == HERR_INVALID_PARAM);
    CHECK(output == 0.0f);
    CHECK(HFCosineSimilarityConvertToPercentage(std::numeric_limits<float>::infinity(), &output) == HERR_INVALID_PARAM);

    HFSimilarityConverterConfig invalid = {0.4f, 0.6f, 0.0f, 0.01f, 1.0f};
    CHECK(HFUpdateCosineSimilarityConverter(invalid) == HERR_INVALID_PARAM);
    invalid = {0.4f, 0.01f, 8.0f, 0.01f, 1.0f};
    CHECK(HFUpdateCosineSimilarityConverter(invalid) == HERR_INVALID_PARAM);
    invalid = {0.4f, 0.6f, 8.0f, 1.0f, 1.0f};
    CHECK(HFUpdateCosineSimilarityConverter(invalid) == HERR_INVALID_PARAM);
    invalid.threshold = std::numeric_limits<float>::quiet_NaN();
    CHECK(HFUpdateCosineSimilarityConverter(invalid) == HERR_INVALID_PARAM);
}
