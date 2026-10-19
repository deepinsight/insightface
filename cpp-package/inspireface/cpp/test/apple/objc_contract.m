#import <InspireFace/InspireFaceApple.h>
#import <dispatch/dispatch.h>
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "allocation_probe.h"

static _Atomic unsigned checks = 0;
#define CHECK(...)                                                                 \
    do {                                                                           \
        ++checks;                                                                  \
        if (!(__VA_ARGS__)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #__VA_ARGS__); \
            exit(1);                                                               \
        }                                                                          \
    } while (0)
#define OK(...)               \
    do {                      \
        error = nil;          \
        CHECK((__VA_ARGS__)); \
        CHECK(error == nil);  \
    } while (0)
#define BAD(expected, ...)                                   \
    do {                                                     \
        error = nil;                                         \
        CHECK(!(__VA_ARGS__));                               \
        CHECK(error.code == (expected));                     \
        CHECK([error.domain isEqualToString:IFErrorDomain]); \
    } while (0)

static void Metadata(void) {
    NSError *error = nil;
    HFInspireFaceVersion version = {0}, baseline = {0};
    OK([IFDiagnostics getVersion:&version error:&error]);
    CHECK(HFQueryInspireFaceVersion(&baseline) == HSUCCEED);
    CHECK(memcmp(&version, &baseline, sizeof(version)) == 0);
    HFUInt32 level = 0;
    OK([IFDiagnostics getCAPILevel:&level error:&error]);
    CHECK(level == HF_C_API_LEVEL);
    HFInspireFaceExtendedInformation info = {0};
    OK([IFDiagnostics getExtendedInformation:&info error:&error]);
    for (int i = 0; i < HF_COMPONENT_COUNT; ++i) {
        HFComponentVersion value = {0};
        OK([IFDiagnostics getComponentVersion:(HFComponentType)i result:&value error:&error]);
    }
    HFComponentVersion coreml = {0};
    OK([IFDiagnostics getComponentVersion:HF_COMPONENT_COREML result:&coreml error:&error]);
#ifdef IF_TEST_COREML
    CHECK(coreml.state != HF_COMPONENT_VERSION_DISABLED);
#else
    CHECK(coreml.state == HF_COMPONENT_VERSION_DISABLED);
#endif
    char buffer[8192];
    HInt32 size = 0;
    OK([IFDiagnostics getComponentVersions:NULL capacity:0 requiredSize:&size error:&error]);
    CHECK(size > 1);
    OK([IFDiagnostics getComponentVersions:buffer capacity:sizeof(buffer) requiredSize:&size error:&error]);
    CHECK(strstr(buffer, "mnn") != NULL || strstr(buffer, "MNN") != NULL);
    OK([IFDiagnostics getDiagnosticInformation:buffer capacity:sizeof(buffer) requiredSize:&size error:&error]);
    OK([IFDiagnostics getErrorMessage:HERR_INVALID_PARAM
                                 into:buffer
                             capacity:sizeof(buffer)
                         requiredSize:&size
                                error:&error]);
    CHECK(size > 1);
    BAD(HERR_INVALID_BUFFER_SIZE, [IFDiagnostics getErrorMessage:HERR_INVALID_PARAM
                                                            into:buffer
                                                        capacity:1
                                                    requiredSize:&size
                                                           error:&error]);
    OK([IFDiagnostics setLogLevel:HF_LOG_NONE error:&error]);
    OK([IFDiagnostics logMessage:@"literal %s %n %@" level:HF_LOG_INFO error:&error]);
    OK([IFDiagnostics disableLoggingWithError:&error]);
    OK([IFDiagnostics printResourceStatisticsWithError:&error]);
    BAD(HERR_INVALID_PARAM, [IFRuntime setImageProcessingAlignedWidth:0 error:&error]);
    OK([IFRuntime setImageProcessingAlignedWidth:16 error:&error]);
    OK([IFRuntime setImageProcessingBackend:HF_IMAGE_PROCESSING_CPU error:&error]);
    OK([IFRuntime setCoreMLInferenceMode:HF_APPLE_COREML_INFERENCE_MODE_CPU error:&error]);
    BAD(HERR_INVALID_PARAM, [IFRuntime setCoreMLInferenceMode:(HFAppleCoreMLInferenceMode)99 error:&error]);
    HFFaceDetectPixelList pixels = {0};
    OK([IFRuntime getSupportedDetectionPixelLevels:&pixels error:&error]);
}

static void Images(NSString *output) {
    NSError *error = nil;
    HInt32 before = -1, after = -1;
    OK([IFDiagnostics getLiveStreamCount:&before error:&error]);
    @autoreleasepool {
        uint8_t pixels[8 * 8 * 4];
        memset(pixels, 127, sizeof(pixels));
        for (int format = 0; format < 8; ++format) {
            for (int rotation = 0; rotation < 4; ++rotation) {
                HFImageData input = {pixels, 8, 8, (HFImageFormat)format, (HFRotation)rotation};
                IFImageStream *stream = [[IFImageStream alloc] initWithBorrowedData:input error:&error];
                CHECK(stream);
                OK([stream closeWithError:&error]);
                BAD(HERR_INVALID_IMAGE_STREAM_HANDLE, [stream closeWithError:&error]);
            }
        }
        IFImageStream *stream = [[IFImageStream alloc] initEmptyWithError:&error];
        CHECK(stream);
        OK([stream setFormat:HF_STREAM_BGR error:&error]);
        OK([stream setBorrowedBuffer:pixels width:8 height:8 error:&error]);
        OK([stream setRotation:HF_CAMERA_ROTATION_0 error:&error]);
        IFImageBitmap *processed = [stream processedBitmapWithRotation:NO scale:1 error:&error];
        CHECK(processed);
        HFImageBitmapData data = {0};
        OK([processed getBorrowedData:&data error:&error]);
        CHECK(data.width == 8);
        HFImageBitmapData source = {pixels, 8, 8, 3};
        IFImageBitmap *bitmap = [[IFImageBitmap alloc] initCopyingData:source error:&error];
        CHECK(bitmap);
        OK([bitmap getBorrowedData:&data error:&error]);
        CHECK(data.data != pixels && memcmp(data.data, pixels, 192) == 0);
        HFImageBitmapData cdata = {0};
        CHECK(HFImageBitmapGetData(bitmap.nativeHandle, &cdata) == HSUCCEED);
        CHECK(cdata.data == data.data);
        IFImageBitmap *copy = [bitmap copyBitmapWithError:&error];
        CHECK(copy);
        IFImageStream *snapshot = [bitmap snapshotStreamWithRotation:HF_CAMERA_ROTATION_0 error:&error];
        CHECK(snapshot);
        HColor color = {1, 0, 0};
        HFaceRect rect = {1, 1, 3, 3};
        HPoint2f point = {3, 3};
        HPoint2i ipoint = {4, 4};
        OK([bitmap drawRect:rect color:color thickness:1 error:&error]);
        OK([bitmap drawCircle:point radius:1 color:color thickness:1 error:&error]);
        OK([bitmap drawIntegerCircle:ipoint radius:1 color:color thickness:1 error:&error]);
        NSString *path = [output stringByAppendingPathComponent:@"objc-bitmap.bmp"];
        OK([bitmap writeToFile:path error:&error]);
        IFImageBitmap *loaded = [[IFImageBitmap alloc] initWithContentsOfFile:path channels:3 error:&error];
        CHECK(loaded);
        OK([stream saveDebugImageAtPath:[output stringByAppendingPathComponent:@"objc-debug.bmp"] error:&error]);
        OK([bitmap closeWithError:&error]);
        BAD(HERR_INVALID_IMAGE_BITMAP_HANDLE, [bitmap showWithTitle:@"closed" delay:1 error:&error]);
        OK([stream closeWithError:&error]);
        [stream showDebugImage];  // invalid handle: never starts a GUI
        CHECK([snapshot processedBitmapWithRotation:NO scale:1 error:&error] != nil);
        __weak IFImageStream *weakStream;
        @autoreleasepool {
            IFImageStream *temporary = [[IFImageStream alloc] initEmptyWithError:&error];
            weakStream = temporary;
        }
        CHECK(weakStream == nil);
    }
    OK([IFDiagnostics getLiveStreamCount:&after error:&error]);
    CHECK(before == after);
}

static void ReleasePixels(void *context, const void *bytes) {
    ++*(int *)context;
    free((void *)bytes);
}

static void ReleasePlanes(void *context, const void *data, size_t size, size_t count, const void *planes[]) {
    ++*(int *)context;
    free((void *)data);
}

static void PixelBuffers(void) {
    NSError *error = nil;
    int released = 0;
    void *pixels = calloc(16 * 8, 4);
    CVPixelBufferRef buffer = NULL;
    CHECK(CVPixelBufferCreateWithBytes(NULL, 16, 8, kCVPixelFormatType_32BGRA, pixels, 64, ReleasePixels, &released,
                                       NULL, &buffer) == kCVReturnSuccess);
    IFImageStream *stream = [[IFImageStream alloc] initWithPixelBuffer:buffer
                                                              rotation:HF_CAMERA_ROTATION_0
                                                                 error:&error];
    CHECK(stream);
    CVPixelBufferRelease(buffer);
    CHECK(released == 0);
    IFImageBitmap *bitmap = [stream processedBitmapWithRotation:NO scale:1 error:&error];
    CHECK(bitmap);
    OK([stream closeWithError:&error]);
    CHECK(released == 1);
    pixels = calloc(16 * 8, 8);
    buffer = NULL;
    CHECK(CVPixelBufferCreateWithBytes(NULL, 16, 8, kCVPixelFormatType_32BGRA, pixels, 128, ReleasePixels, &released,
                                       NULL, &buffer) == kCVReturnSuccess);
    CHECK([[IFImageStream alloc] initWithPixelBuffer:buffer rotation:HF_CAMERA_ROTATION_0 error:&error] == nil);
    CHECK(error.code == HERR_INVALID_IMAGE_STREAM_PARAM);
    CVPixelBufferRelease(buffer);
    CHECK(released == 2);

    // Tight NV12 storage is borrowed; a gap between Y and UV must never be hidden by a copy.
    for (size_t gap = 0; gap <= 16; gap += 16) {
        size_t size = 16 * 8 * 3 / 2 + gap;
        uint8_t *data = calloc(size, 1);
        void *planes[] = {data, data + 16 * 8 + gap};
        size_t widths[] = {16, 8}, heights[] = {8, 4}, strides[] = {16, 16};
        buffer = NULL;
        CHECK(CVPixelBufferCreateWithPlanarBytes(NULL, 16, 8, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
            data, size, 2, planes, widths, heights, strides, ReleasePlanes, &released, NULL, &buffer) == kCVReturnSuccess);
        error = nil;
        stream = [[IFImageStream alloc] initWithPixelBuffer:buffer rotation:HF_CAMERA_ROTATION_0 error:&error];
        if (gap) {
            CHECK(stream == nil && error.code == HERR_INVALID_IMAGE_STREAM_PARAM);
        } else {
            CHECK(stream != nil && error == nil);
            CVPixelBufferRelease(buffer);
            buffer = NULL;
            CHECK(released == 2);
            OK([stream closeWithError:&error]);
            CHECK(released == 3);
        }
        if (buffer) CVPixelBufferRelease(buffer);
    }
    CHECK(released == 4);
}

static void HotPaths(void) {
    NSError *error = nil;
    uint8_t pixels[192] = {0};
    HFImageBitmapData input = {pixels, 8, 8, 3};
    IFImageBitmap *bitmap = [[IFImageBitmap alloc] initCopyingData:input error:&error];
    CHECK(bitmap);
    IFImageStream *stream = [[IFImageStream alloc] initEmptyWithError:&error];
    CHECK(stream);
    OK([stream setFormat:HF_STREAM_BGR error:&error]);
    OK([stream setBorrowedBuffer:pixels width:8 height:8 error:&error]);
    HFImageBitmapData view = {0};
    HFUInt32 level = 0;
    HInt32 length = 0;
    for (int i = 0; i < 100; ++i) {
        [IFDiagnostics getCAPILevel:&level error:NULL];
        [IFFeatureBuffer getLength:&length error:NULL];
        [bitmap getBorrowedData:&view error:NULL];
        [stream setRotation:HF_CAMERA_ROTATION_0 error:NULL];
    }
    IFTestBeginAllocations();
    void *allocation = malloc(777);
    IFTestConsumeAllocation(allocation);
    uint64_t control = IFTestEndAllocations();
    CHECK(allocation && (!IFTestAllocationObservationSupported() || control > 0));
    free(allocation);
    HFImageBitmap handle = bitmap.nativeHandle;
    HFImageStream streamHandle = stream.nativeHandle;
    clock_t start = clock();
    IFTestBeginAllocations();
    for (int i = 0; i < 20000; ++i) {
        HFQueryCAPILevel(&level);
        HFGetFeatureLength(&length);
        HFImageBitmapGetData(handle, &view);
        HFImageStreamSetRotation(streamHandle, HF_CAMERA_ROTATION_0);
    }
    uint64_t nativeAllocations = IFTestEndAllocations();
    double nativeTime = (double)(clock() - start) / CLOCKS_PER_SEC;
    start = clock();
    IFTestBeginAllocations();
    for (int i = 0; i < 20000; ++i) {
        [IFDiagnostics getCAPILevel:&level error:NULL];
        [IFFeatureBuffer getLength:&length error:NULL];
        [bitmap getBorrowedData:&view error:NULL];
        [stream setRotation:HF_CAMERA_ROTATION_0 error:NULL];
    }
    uint64_t adapterAllocations = IFTestEndAllocations();
    double adapterTime = (double)(clock() - start) / CLOCKS_PER_SEC;
    CHECK(!IFTestAllocationObservationSupported() || (nativeAllocations == 0 && adapterAllocations == 0));
    if (!IFTestAllocationObservationSupported()) printf("Allocation observer disabled under ASan; allocation counts below are not measurements.\n");
    printf("Hot paths: 80000 C calls %.6fs, Objective-C %.6fs; allocations %llu/%llu\n", nativeTime, adapterTime,
           (unsigned long long)nativeAllocations, (unsigned long long)adapterAllocations);
    HInt32 before = 0, after = 0;
    OK([IFDiagnostics getLiveStreamCount:&before error:&error]);
    dispatch_apply(8, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t worker) {
      for (int i = 0; i < 256; ++i) {
          @autoreleasepool {
              NSError *localError = nil;
              IFImageStream *temporary = [[IFImageStream alloc] initEmptyWithError:&localError];
              CHECK(temporary && !localError);
              HFUInt32 localLevel = 0;
              CHECK([IFDiagnostics getCAPILevel:&localLevel error:&localError]);
              CHECK(localLevel == HF_C_API_LEVEL);
          }
      }
    });
    OK([IFDiagnostics getLiveStreamCount:&after error:&error]);
    CHECK(before == after);
}

static void FeatureHubContract(void) {
    NSError *error = nil;
    HFFeatureHubConfiguration config = {HF_PK_MANUAL_INPUT, 0, NULL, -1, HF_SEARCH_MODE_EXHAUSTIVE};
    OK([IFFeatureHub enableWithConfiguration:config error:&error]);
    OK([IFFeatureHub setSearchThreshold:0.4 error:&error]);
    IFFeatureBuffer *owned = [[IFFeatureBuffer alloc] initWithError:&error];
    CHECK(owned);
    HFFaceFeature feature = owned.borrowedFeature;
    CHECK(feature.size > 0);
    memset(feature.data, 0, feature.size * sizeof(float));
    feature.data[0] = 1;
    HFFaceFeatureIdentity identity = {42, &feature};
    HFaceId allocated = -1;
    OK([IFFeatureHub insert:identity allocatedID:&allocated error:&error]);
    CHECK(allocated == 42);
    HInt32 count = 0;
    OK([IFFeatureHub getIdentityCount:&count error:&error]);
    CHECK(count == 1);
    HFFeatureHubSearchResultV2 result = {0};
    OK([IFFeatureHub search:feature borrowedResult:&result error:&error]);
    CHECK(result.found && result.id == 42);
    HFFaceFeatureIdentity found = {0};
    float score = 0;
    OK([IFFeatureHub search:feature confidence:&score borrowedIdentity:&found error:&error]);
    CHECK(found.id == 42 && score > 0.99);
    HFSearchTopKResults top = {0};
    OK([IFFeatureHub search:feature topK:1 borrowedResults:&top error:&error]);
    CHECK(top.size == 1 && top.ids[0] == 42);
    OK([IFFeatureHub getIdentityWithID:42 borrowedResult:&found error:&error]);
    CHECK(found.id == 42);
    HFFeatureHubExistingIds ids = {0};
    OK([IFFeatureHub getBorrowedExistingIDs:&ids error:&error]);
    OK([IFFeatureHub updateIdentity:identity error:&error]);
    OK([IFFeatureHub printTableWithError:&error]);
    OK([IFFeatureBuffer compare:feature with:feature similarity:&score error:&error]);
    CHECK(fabsf(score - 1) < 1e-5);
    HInt32 length = 0;
    OK([IFFeatureBuffer getLength:&length error:&error]);
    CHECK(length == feature.size);
    OK([IFFeatureBuffer getRecommendedThreshold:&score error:&error]);
    OK([IFFeatureBuffer convertSimilarity:score percentage:&score error:&error]);
    CHECK(isfinite(score));
    HFSimilarityConverterConfig converter = {0};
    OK([IFFeatureBuffer getSimilarityConverter:&converter error:&error]);
    OK([IFFeatureBuffer setSimilarityConverter:converter error:&error]);
    OK([IFFeatureHub removeIdentityWithID:42 error:&error]);
    OK([IFFeatureHub disableWithError:&error]);
    OK([owned closeWithError:&error]);
    BAD(HERR_INVALID_FACE_FEATURE, [owned closeWithError:&error]);
}

static void ModelContract(NSString *model, NSString *image) {
    NSError *error = nil;
    HFResourcePackInfo pack = {0};
    HResult validation = HFValidateResourcePack(model.fileSystemRepresentation, &pack);
    if (validation == HSUCCEED) {
        OK([IFRuntime validateResourcePackAtPath:model info:&pack error:&error]);
    } else {
        BAD(validation, [IFRuntime validateResourcePackAtPath:model info:&pack error:&error]);
    }
    OK([IFRuntime launchAtPath:model error:&error]);
    HInt32 launched = 0;
    OK([IFRuntime getLaunchStatus:&launched error:&error]);
    CHECK(launched);
    OK([IFRuntime reloadAtPath:model error:&error]);
    OK([IFRuntime setLandmarkEngine:HF_LANDMARK_HYPLMV2_0_25 error:&error]);
    HInt32 before = 0;
    OK([IFDiagnostics getLiveSessionCount:&before error:&error]);
    @autoreleasepool {
        HOption options = HF_ENABLE_FACE_RECOGNITION | HF_ENABLE_LIVENESS | HF_ENABLE_MASK_DETECT | HF_ENABLE_QUALITY |
                          HF_ENABLE_FACE_ATTRIBUTE | HF_ENABLE_INTERACTION | HF_ENABLE_FACE_EMOTION;
        HFSessionConfigV2 config = {0};
        config.structSize = sizeof(config);
        config.structVersion = HF_SESSION_CONFIG_V2_VERSION;
        config.featureMask = options;
        config.detectMode = HF_DETECT_MODE_ALWAYS_DETECT;
        config.maxDetectFaceNum = 3;
        config.detectPixelLevel = -1;
        config.trackByDetectModeFPS = -1;
        IFSession *session = [[IFSession alloc] initWithConfiguration:config error:&error];
        CHECK(session);
        HFSessionCustomParameter parameters = {0};
        parameters.enable_recognition = 1;
        IFSession *legacy = [[IFSession alloc] initWithParameters:parameters
                                                             mode:HF_DETECT_MODE_ALWAYS_DETECT
                                                     maximumFaces:1
                                                       pixelLevel:-1
                                                  framesPerSecond:-1
                                                            error:&error];
        CHECK(legacy);
        OK([legacy closeWithError:&error]);
        IFSession *optional = [[IFSession alloc] initWithOptions:options
                                                            mode:HF_DETECT_MODE_ALWAYS_DETECT
                                                    maximumFaces:3
                                                      pixelLevel:-1
                                                 framesPerSecond:-1
                                                           error:&error];
        CHECK(optional);
        OK([optional closeWithError:&error]);
        OK([session setTrackLostRecoveryEnabled:YES error:&error]);
        OK([session setLightTrackConfidenceThreshold:0.2 error:&error]);
        OK([session setTrackPreviewSize:192 error:&error]);
        HInt32 preview = 0;
        OK([session getTrackPreviewSize:&preview error:&error]);
        CHECK(preview == 192);
        OK([session setMinimumFacePixelSize:8 error:&error]);
        OK([session setDetectionThreshold:0.5 error:&error]);
        OK([session setTrackingSmoothRatio:0.5 error:&error]);
        OK([session setTrackingSmoothCacheFrames:3 error:&error]);
        OK([session setDetectionInterval:1 error:&error]);
        OK([session setLandmarkAugmentationCount:1 error:&error]);
        OK([session setTrackingTimingEnabled:YES error:&error]);
        IFImageBitmap *bitmap = [[IFImageBitmap alloc] initWithContentsOfFile:image channels:3 error:&error];
        CHECK(bitmap);
        HFImageBitmapData pixels = {0};
        OK([bitmap getBorrowedData:&pixels error:&error]);
        HFImageData input = {pixels.data, pixels.width, pixels.height, HF_STREAM_BGR, HF_CAMERA_ROTATION_0};
        IFImageStream *stream = [[IFImageStream alloc] initWithBorrowedData:input error:&error];
        CHECK(stream);
        HFMultipleFaceData faces = {0};
        OK([session trackStream:stream borrowedResult:&faces error:&error]);
        CHECK(faces.detectedNum > 0);
        HFFaceBasicToken token = faces.tokens[0];
        HInt32 tokenSize = 0;
        OK([IFFaceToken getTokenSize:&tokenSize error:&error]);
        char *tokenStorage = malloc(tokenSize);
        OK([IFFaceToken copyToken:token into:tokenStorage capacity:tokenSize error:&error]);
        HFFaceBasicToken copiedToken = {tokenSize, tokenStorage};
        HInt32 landmarkCount = 0;
        OK([IFFaceToken getDenseLandmarkCount:&landmarkCount error:&error]);
        HPoint2f *landmarks = calloc(landmarkCount, sizeof(HPoint2f));
        OK([IFFaceToken getDenseLandmarks:token into:landmarks capacity:landmarkCount error:&error]);
        CHECK(isfinite(landmarks[0].x));
        HPoint2f five[5];
        OK([IFFaceToken getFiveKeyPoints:token into:five capacity:5 error:&error]);
        float quality = 0;
        OK([session getQualityForToken:token result:&quality error:&error]);
        CHECK(isfinite(quality));
        HFFaceFeature feature = {0};
        OK([session extractFeatureFromStream:stream token:token borrowedResult:&feature error:&error]);
        CHECK(feature.size > 0);
        float *copy = calloc(feature.size, sizeof(float));
        OK([session copyFeatureFromStream:stream token:token into:copy capacity:feature.size error:&error]);
        CHECK(isfinite(copy[0]));
        BAD(HERR_INVALID_FACE_FEATURE, [session copyFeatureFromStream:stream
                                                                token:token
                                                                 into:copy
                                                             capacity:1
                                                                error:&error]);
        HFFaceFeature destination = {feature.size, copy};
        OK([session extractFeatureFromStream:stream token:token into:destination error:&error]);
        CHECK(destination.data == copy);
        IFImageBitmap *aligned = [session alignmentBitmapFromStream:stream token:token error:&error];
        CHECK(aligned);
        IFImageStream *alignedStream = [aligned snapshotStreamWithRotation:HF_CAMERA_ROTATION_0 error:&error];
        CHECK(alignedStream);
        OK([session extractAlignedFeatureFromStream:alignedStream into:destination error:&error]);
        OK([session processStream:stream faces:&faces options:options error:&error]);
        HFRGBLivenessConfidence live = {0};
        HFFaceMaskConfidence mask = {0};
        HFFaceQualityConfidence fq = {0};
        HFFaceInteractionState state = {0};
        HFFaceInteractionsActions actions = {0};
        HFFaceAttributeResult attributes = {0};
        HFFaceEmotionResult emotions = {0};
        OK([session getBorrowedRGBLiveness:&live error:&error]);
        CHECK(live.num == faces.detectedNum);
        OK([session getBorrowedMaskConfidence:&mask error:&error]);
        CHECK(mask.num == faces.detectedNum);
        OK([session getBorrowedQualityConfidence:&fq error:&error]);
        CHECK(fq.num == faces.detectedNum);
        OK([session getBorrowedInteractionState:&state error:&error]);
        CHECK(state.num == faces.detectedNum);
        OK([session getBorrowedInteractionActions:&actions error:&error]);
        CHECK(actions.num == faces.detectedNum);
        OK([session getBorrowedAttributes:&attributes error:&error]);
        CHECK(attributes.num == faces.detectedNum);
        OK([session getBorrowedEmotions:&emotions error:&error]);
        CHECK(emotions.num == faces.detectedNum);
        parameters.enable_liveness = 1;
        OK([session processStream:stream faces:&faces parameters:parameters error:&error]);
        OK([session getDebugPreviewImageSize:&preview error:&error]);
        OK([session printTrackingTimingWithError:&error]);
        NSError *scopeError = nil;
        CHECK([session
            withBorrowedFacesFromStream:stream
                                   body:^(HFMultipleFaceData scopedFaces) {
                                     CHECK(scopedFaces.detectedNum > 0);
                                     NSError *nestedError = nil;
                                     CHECK(![session clearTrackingWithError:&nestedError]);
                                     CHECK(nestedError.code == HERR_INVALID_PARAM);
                                     CHECK([session
                                         withBorrowedFeatureFromStream:stream
                                                                 token:scopedFaces.tokens[0]
                                                                  body:^(HFFaceFeature borrowed) {
                                                                    CHECK(borrowed.size > 0 && borrowed.data);
                                                                    NSError *closeError = nil;
                                                                    CHECK(![session closeWithError:&closeError]);
                                                                    CHECK(closeError.code == HERR_INVALID_PARAM);
                                                                  }
                                                                 error:&nestedError]);
                                   }
                                  error:&scopeError]);
        IFFaceSnapshot *snapshot = [session snapshotFromStream:stream error:&error];
        CHECK(snapshot);
        HFMultipleFaceData frozen = {0};
        OK([snapshot getBorrowedFaces:&frozen error:&error]);
        CHECK(frozen.detectedNum > 0);
        void *frozenAddress = frozen.tokens[0].data;
        OK([session trackStream:stream borrowedResult:&faces error:&error]);
        CHECK(frozen.tokens[0].data == frozenAddress);
        OK([IFFaceToken getFiveKeyPoints:copiedToken into:five capacity:5 error:&error]);
        HFFaceCaptureConfig captureConfig = {0};
        OK([IFCaptureSession getDefaultConfiguration:&captureConfig error:&error]);
        captureConfig.filterMask = 0;
        IFCaptureSession *capture = [[IFCaptureSession alloc] initWithSession:session
                                                                configuration:captureConfig
                                                                        error:&error];
        CHECK(capture);
        HFFaceCaptureProgress progress = {0};
        OK([capture updateStream:stream frameID:1 timestampMilliseconds:1 progress:&progress error:&error]);
        OK([capture updateStream:stream
                         snapshot:snapshot
                          frameID:2
            timestampMilliseconds:2
                         progress:&progress
                            error:&error]);
        HFFaceCaptureResult results[HF_FACE_CAPTURE_MAX_RESULTS];
        HFUInt32 count = 0;
        OK([capture getResults:results capacity:HF_FACE_CAPTURE_MAX_RESULTS count:&count error:&error]);
        CHECK(count <= HF_FACE_CAPTURE_MAX_RESULTS);
        OK([capture finishWithProgress:&progress error:&error]);
        OK([capture resetWithError:&error]);
        OK([capture closeWithError:&error]);
        OK([session clearTrackingWithError:&error]);
        HInt32 n = 0;
        OK([IFDiagnostics getLiveSessionCount:&n error:&error]);
        HFSession *handles = calloc(n, sizeof(HFSession));
        OK([IFDiagnostics getBorrowedLiveSessions:handles capacity:n error:&error]);
        free(handles);
        OK([IFDiagnostics getLiveStreamCount:&n error:&error]);
        HFImageStream *streams = calloc(n, sizeof(HFImageStream));
        OK([IFDiagnostics getBorrowedLiveStreams:streams capacity:n error:&error]);
        free(streams);
        OK([session closeWithError:&error]);
        OK([snapshot getBorrowedFaces:&frozen error:&error]);
        CHECK(frozen.tokens[0].data == frozenAddress);
        OK([snapshot closeWithError:&error]);
        BAD(HERR_INVALID_CONTEXT_HANDLE, [session clearTrackingWithError:&error]);
        free(copy);
        free(landmarks);
        free(tokenStorage);
    }
    HInt32 after = 0;
    OK([IFDiagnostics getLiveSessionCount:&after error:&error]);
    CHECK(before == after);
    OK([IFRuntime terminateWithError:&error]);
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        CHECK(argc == 2 || argc == 4);
        NSString *output = @(argv[1]);
        Metadata();
        Images(output);
        PixelBuffers();
        HotPaths();
        FeatureHubContract();
        if (argc == 4) ModelContract(@(argv[2]), @(argv[3]));
        printf("Objective-C contract: %u checks passed (%s)\n", checks, argc == 4 ? "with model" : "without model");
    }
    return 0;
}
