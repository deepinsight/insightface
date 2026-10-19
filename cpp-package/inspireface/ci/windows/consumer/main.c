#include <inspireface.h>

#include <stdio.h>
#include <string.h>

#if defined(ISF_EXPORTS)
#error "A consumer must not inherit the SDK producer's ISF_EXPORTS"
#endif
#if defined(_WIN32)
#if ISF_CONSUMER_EXPECT_SHARED && !defined(ISF_BUILD_SHARED_LIBS)
#error "The shared package must propagate ISF_BUILD_SHARED_LIBS"
#elif !ISF_CONSUMER_EXPECT_SHARED && defined(ISF_BUILD_SHARED_LIBS)
#error "The static package must not propagate DLL import declarations"
#endif
#endif

static int check_status(const char* operation, HResult status) {
    if (status != HSUCCEED) {
        fprintf(stderr, "%s failed: %d\n", operation, (int)status);
        return 1;
    }
    return 0;
}

static int check_bitmap(void) {
    HUInt8 pixels[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    HFImageBitmapData input = {pixels, 2, 2, 3};
    HFImageBitmapData output = {0};
    HFImageBitmap bitmap = NULL;
    int failed;
    if (check_status("HFCreateImageBitmap", HFCreateImageBitmap(&input, &bitmap))) return 1;
    failed = check_status("HFImageBitmapGetData", HFImageBitmapGetData(bitmap, &output));
    if (!failed && (output.width != 2 || output.height != 2 || output.channels != 3 ||
                    !output.data || memcmp(output.data, pixels, sizeof(pixels)) != 0)) {
        fprintf(stderr, "The installed C API changed bitmap dimensions or pixels\n");
        failed = 1;
    }
    if (check_status("HFReleaseImageBitmap", HFReleaseImageBitmap(bitmap))) failed = 1;
    return failed;
}

static int check_model(const char* path) {
    HUInt8 pixels[64 * 64 * 3] = {0};
    HFImageData image = {pixels, 64, 64, HF_STREAM_BGR, HF_CAMERA_ROTATION_0};
    HFMultipleFaceData faces = {0};
    HFSession session = NULL;
    HFImageStream stream = NULL;
    int failed = 1;
    if (check_status("HFLaunchInspireFace", HFLaunchInspireFace(path))) return 1;
    if (check_status("HFCreateInspireFaceSessionOptional",
                     HFCreateInspireFaceSessionOptional(0, HF_DETECT_MODE_ALWAYS_DETECT,
                                                        1, -1, -1, &session))) goto cleanup;
    if (check_status("HFCreateImageStream", HFCreateImageStream(&image, &stream))) goto cleanup;
    if (check_status("HFExecuteFaceTrack", HFExecuteFaceTrack(session, stream, &faces))) goto cleanup;
    if (faces.detectedNum < 0 || faces.detectedNum > 1) {
        fprintf(stderr, "Invalid installed C API detection count: %d\n", (int)faces.detectedNum);
        goto cleanup;
    }
    failed = 0;
cleanup:
    if (stream && check_status("HFReleaseImageStream", HFReleaseImageStream(stream))) failed = 1;
    if (session && check_status("HFReleaseInspireFaceSession", HFReleaseInspireFaceSession(session))) failed = 1;
    if (check_status("HFTerminateInspireFace", HFTerminateInspireFace())) failed = 1;
    return failed;
}

int main(int argc, char** argv) {
    HFInspireFaceVersion version = {0};
    HFUInt32 api_level = 0;
    if (argc > 2) {
        fprintf(stderr, "Usage: %s [resource-pack]\n", argv[0]);
        return 2;
    }
    if (check_status("HFQueryInspireFaceVersion", HFQueryInspireFaceVersion(&version))) return 1;
    if (check_status("HFQueryCAPILevel", HFQueryCAPILevel(&api_level))) return 1;
    if (version.major < 1 || api_level == 0 || check_bitmap()) return 1;
    if (argc == 2 && check_model(argv[1])) return 1;
    printf("Installed C consumer passed: %d.%d.%d, API level %u, model=%s\n",
           (int)version.major, (int)version.minor, (int)version.patch,
           (unsigned)api_level, argc == 2 ? "checked" : "not requested");
    return 0;
}
