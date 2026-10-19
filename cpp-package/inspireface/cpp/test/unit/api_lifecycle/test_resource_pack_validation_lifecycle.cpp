#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "inspireface/c_api/inspireface.h"
#include "settings/test_settings.h"
#include "unit/test_helper/c_api_guard.h"

using inspireface_test::ScopedSdkTermination;

namespace {

class ScopedArchiveFile {
public:
    explicit ScopedArchiveFile(std::string path) : path_(std::move(path)) {
        std::remove(path_.c_str());
    }

    ~ScopedArchiveFile() {
        std::remove(path_.c_str());
    }

    const std::string& Path() const {
        return path_;
    }

private:
    std::string path_;
};

void WriteOctal(char* destination, size_t capacity, size_t value) {
    std::snprintf(destination, capacity, "%0*lo", static_cast<int>(capacity - 1), static_cast<unsigned long>(value));
}

bool WriteTarArchive(const std::string& path, const std::vector<std::pair<std::string, std::string>>& files) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        return false;
    }

    for (const auto& file : files) {
        if (file.first.empty() || file.first.size() >= 100) {
            return false;
        }
        std::array<char, 512> header = {};
        std::memcpy(header.data(), file.first.data(), file.first.size());
        WriteOctal(header.data() + 100, 8, 0644);
        WriteOctal(header.data() + 108, 8, 0);
        WriteOctal(header.data() + 116, 8, 0);
        WriteOctal(header.data() + 124, 12, file.second.size());
        WriteOctal(header.data() + 136, 12, 0);
        std::memset(header.data() + 148, ' ', 8);
        header[156] = '0';
        std::memcpy(header.data() + 257, "ustar", 5);
        header[263] = '0';
        header[264] = '0';

        unsigned checksum = 0;
        for (unsigned char byte : header) {
            checksum += byte;
        }
        std::snprintf(header.data() + 148, 8, "%06o", checksum);
        header[154] = '\0';
        header[155] = ' ';

        output.write(header.data(), header.size());
        output.write(file.second.data(), static_cast<std::streamsize>(file.second.size()));
        const size_t padding = (512 - (file.second.size() % 512)) % 512;
        const std::array<char, 512> zeros = {};
        output.write(zeros.data(), static_cast<std::streamsize>(padding));
    }
    const std::array<char, 1024> end_blocks = {};
    output.write(end_blocks.data(), end_blocks.size());
    return output.good();
}

HFResourcePackInfo ResourcePackInfo() {
    HFResourcePackInfo info = {};
    info.structSize = sizeof(info);
    info.structVersion = HF_RESOURCE_PACK_INFO_VERSION;
    return info;
}

}  // namespace

static_assert(sizeof(HFResourcePackInfo) == 336, "Resource-pack metadata layout must remain fixed");

TEST_CASE("Resource-pack validation releases the input file before returning",
          "[api][contract][lifecycle][resource_pack]") {
    ScopedArchiveFile archive(GET_SAVE_DATA("validation_releases_model.tar"));
    {
        std::ifstream input(GET_RUNTIME_FULLPATH_NAME, std::ios::binary);
        REQUIRE(input.good());
        std::ofstream output(archive.Path(), std::ios::binary | std::ios::trunc);
        REQUIRE(output.good());
        output << input.rdbuf();
        REQUIRE(output.good());
    }
    auto info = ResourcePackInfo();
    REQUIRE(HFValidateResourcePack(archive.Path().c_str(), &info) == HSUCCEED);
    // Windows refuses this while any FILE opened by the validator remains
    // alive. No subsequent archive call may be needed to release that handle.
    REQUIRE(std::remove(archive.Path().c_str()) == 0);
}

TEST_CASE("C API validates resource-pack metadata without launching the SDK",
          "[api][contract][lifecycle][resource_pack]") {
    ScopedSdkTermination cleanup;
    HInt32 launch_status = -1;
    REQUIRE(HFQueryInspireFaceLaunchStatus(&launch_status) == HSUCCEED);
    REQUIRE(launch_status == HF_STATUS_DISABLE);

    HFResourcePackInfo info = ResourcePackInfo();
    const auto started = std::chrono::steady_clock::now();
    REQUIRE(HFValidateResourcePack(GET_RUNTIME_FULLPATH_NAME.c_str(), &info) == HSUCCEED);
    const auto elapsed = std::chrono::steady_clock::now() - started;
    const double elapsed_ms = std::chrono::duration<double, std::milli>(elapsed).count();
    TEST_PRINT("Resource-pack validation: {:.3f} ms", elapsed_ms);

    CHECK(info.structSize == sizeof(info));
    CHECK(info.structVersion == HF_RESOURCE_PACK_INFO_VERSION);
    CHECK(info.archiveFileCount > 1);
    CHECK(info.modelCount > 0);
    CHECK(std::string(info.tag) == "Pikachu");
    CHECK(std::string(info.version) == "4.0");
    CHECK(std::strlen(info.major) > 0);
    CHECK(std::strlen(info.releaseDate) > 0);
    CHECK(std::all_of(std::begin(info.reserved), std::end(info.reserved), [](HFUInt64 value) { return value == 0; }));
    CHECK(elapsed < std::chrono::seconds(2));
    CHECK(HFValidateResourcePack(GET_RUNTIME_FULLPATH_NAME.c_str(), nullptr) == HSUCCEED);

    REQUIRE(HFQueryInspireFaceLaunchStatus(&launch_status) == HSUCCEED);
    CHECK(launch_status == HF_STATUS_DISABLE);
}

TEST_CASE("C API resource-pack validation rejects malformed inputs transactionally",
          "[api][contract][lifecycle][resource_pack][boundary]") {
    ScopedSdkTermination cleanup;
    HFResourcePackInfo info = ResourcePackInfo();
    std::memset(info.tag, 'x', sizeof(info.tag));
    info.tag[sizeof(info.tag) - 1] = '\0';

    CHECK(HFValidateResourcePack(nullptr, &info) == HERR_INVALID_PARAM);
    CHECK(HFValidateResourcePack("", &info) == HERR_INVALID_PARAM);
    CHECK(HFValidateResourcePack((GET_RUNTIME_FULLPATH_NAME + ".missing").c_str(), &info) == HERR_ARCHIVE_LOAD_FAILURE);
    CHECK(HFValidateResourcePack(GET_DIR.c_str(), &info) == HERR_ARCHIVE_LOAD_FAILURE);
    CHECK(info.tag[0] == 'x');

    HFResourcePackInfo short_info = ResourcePackInfo();
    short_info.structSize = sizeof(short_info) - 1;
    CHECK(HFValidateResourcePack(GET_RUNTIME_FULLPATH_NAME.c_str(), &short_info) == HERR_INVALID_PARAM);
    HFResourcePackInfo future_info = ResourcePackInfo();
    future_info.structVersion = HF_RESOURCE_PACK_INFO_VERSION + 1;
    CHECK(HFValidateResourcePack(GET_RUNTIME_FULLPATH_NAME.c_str(), &future_info) == HERR_INVALID_PARAM);

    ScopedArchiveFile plain_file(GET_SAVE_DATA(".resource-pack-plain.bin"));
    {
        std::ofstream output(plain_file.Path(), std::ios::binary | std::ios::trunc);
        REQUIRE(output.good());
        output << "not a resource pack";
    }
    CHECK(HFValidateResourcePack(plain_file.Path().c_str(), &info) == HERR_ARCHIVE_FILE_FORMAT_ERROR);
    CHECK(info.tag[0] == 'x');

    ScopedArchiveFile malformed_manifest(GET_SAVE_DATA(".resource-pack-malformed.tar"));
    REQUIRE(WriteTarArchive(malformed_manifest.Path(), {{"__inspire__", "tag: Broken\n"}}));
    CHECK(HFValidateResourcePack(malformed_manifest.Path().c_str(), &info) == HERR_ARCHIVE_FILE_FORMAT_ERROR);
    CHECK(info.tag[0] == 'x');

    const std::string missing_model_manifest =
      "tag: MissingModel\n"
      "version: 1.0\n"
      "face_detect_pixel_list: [160]\n"
      "face_detect_model_list: [face_detect_160]\n"
      "face_detect_160:\n"
      "  name: missing_detector\n"
      "  model_type: MNN\n"
      "  infer_engine: MNN\n";
    ScopedArchiveFile missing_model(GET_SAVE_DATA(".resource-pack-missing-model.tar"));
    REQUIRE(WriteTarArchive(missing_model.Path(), {{"__inspire__", missing_model_manifest}}));
    CHECK(HFValidateResourcePack(missing_model.Path().c_str(), &info) == HERR_ARCHIVE_LOAD_MODEL_FAILURE);
    CHECK(info.tag[0] == 'x');

    const std::string unsupported_engine_manifest =
      "tag: UnsupportedEngine\n"
      "version: 1.0\n"
      "face_detect_pixel_list: [160]\n"
      "face_detect_model_list: [face_detect_160]\n"
      "face_detect_160:\n"
      "  name: detector_blob\n"
      "  model_type: RKNN\n"
      "  infer_engine: RKNN\n";
    ScopedArchiveFile unsupported_engine(GET_SAVE_DATA(".resource-pack-unsupported-engine.tar"));
    REQUIRE(WriteTarArchive(unsupported_engine.Path(),
                            {{"__inspire__", unsupported_engine_manifest}, {"detector_blob", "model"}}));
#if defined(ISF_ENABLE_RKNN)
    CHECK(HFValidateResourcePack(unsupported_engine.Path().c_str(), &info) == HERR_ARCHIVE_LOAD_MODEL_FAILURE);
#else
    CHECK(HFValidateResourcePack(unsupported_engine.Path().c_str(), &info) == HERR_UNSUPPORTED);
#endif
    CHECK(info.tag[0] == 'x');

    HInt32 launch_status = -1;
    REQUIRE(HFQueryInspireFaceLaunchStatus(&launch_status) == HSUCCEED);
    CHECK(launch_status == HF_STATUS_DISABLE);
}

TEST_CASE("Failed resource-pack validation preserves an active SDK", "[api][contract][lifecycle][resource_pack]") {
    ScopedSdkTermination cleanup;
    REQUIRE(HFLaunchInspireFace(GET_RUNTIME_FULLPATH_NAME.c_str()) == HSUCCEED);

    HFSessionCustomParameter parameter = {};
    HFSession session = nullptr;
    REQUIRE(HFCreateInspireFaceSession(parameter, HF_DETECT_MODE_ALWAYS_DETECT, 1, -1, -1, &session) == HSUCCEED);
    inspireface_test::UniqueSession owned_session(session);

    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
    REQUIRE_FALSE(image.Empty());
    HFImageData image_data = {};
    image_data.data = const_cast<HUInt8*>(image.Data());
    image_data.width = image.Width();
    image_data.height = image.Height();
    image_data.format = HF_STREAM_BGR;
    image_data.rotation = HF_CAMERA_ROTATION_0;
    inspireface_test::UniqueImageStream stream;
    REQUIRE(HFCreateImageStream(&image_data, stream.Put()) == HSUCCEED);

    HFMultipleFaceData before = {};
    REQUIRE(HFExecuteFaceTrack(owned_session.Get(), stream.Get(), &before) == HSUCCEED);
    REQUIRE(before.detectedNum == 1);

    HFResourcePackInfo info = ResourcePackInfo();
    CHECK(HFValidateResourcePack((GET_RUNTIME_FULLPATH_NAME + ".missing").c_str(), &info) == HERR_ARCHIVE_LOAD_FAILURE);
    REQUIRE(HFValidateResourcePack(GET_RUNTIME_FULLPATH_NAME.c_str(), &info) == HSUCCEED);

    HInt32 launch_status = -1;
    REQUIRE(HFQueryInspireFaceLaunchStatus(&launch_status) == HSUCCEED);
    CHECK(launch_status == HF_STATUS_ENABLE);
    HFMultipleFaceData after = {};
    REQUIRE(HFExecuteFaceTrack(owned_session.Get(), stream.Get(), &after) == HSUCCEED);
    CHECK(after.detectedNum == before.detectedNum);
}
