package com.insightface.sdk.inspireface.jni;

import java.lang.ref.WeakReference;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import static com.insightface.sdk.inspireface.jni.Native.*;
import static com.insightface.sdk.inspireface.jni.NativeConstants.*;
import com.insightface.sdk.inspireface.jni.NativeTypes.*;

/** Runs against the real native SDK with -Xcheck:jni, without Android or test doubles. */
public final class ContractTest {
    private static int assertions;
    private static void expect(boolean condition) {
        assertions++;
        if (!condition) throw new AssertionError("Assertion " + assertions);
    }
    private static void ok(long status) {
        assertions++;
        InspireFaceException.check(status);
    }
    private static void bad(Runnable action) {
        assertions++;
        try { action.run(); } catch (IllegalArgumentException expected) { return; }
        throw new AssertionError("Expected argument validation failure");
    }
    private static ByteBuffer bytes(int count) { return ByteBuffer.allocateDirect(count).order(ByteOrder.nativeOrder()); }
    private static void cpuEngine() throws Exception {
        // Exercise loading/configuration before any C API call or model launch.
        expect(CPUEngine.getGlobalPowerMode() == CPUEngine.PowerMode.NORMAL);
        java.lang.reflect.Method setter = CPUEngine.class.getDeclaredMethod("nativeSetGlobalPowerMode", int.class);
        setter.setAccessible(true);
        for (CPUEngine.PowerMode mode : CPUEngine.PowerMode.values()) {
            CPUEngine.setGlobalPowerMode(mode);
            expect(CPUEngine.getGlobalPowerMode() == mode);
            bad(() -> CPUEngine.setGlobalPowerMode(null));
            for (int invalid : new int[] {-1, 3, Integer.MAX_VALUE}) {
                expect(((Integer) setter.invoke(null, invalid)).longValue() == HERR_INVALID_PARAM);
                expect(CPUEngine.getGlobalPowerMode() == mode);
            }
        }
        CPUEngine.setGlobalPowerMode(CPUEngine.PowerMode.NORMAL);
    }
    private static String text(ByteBuffer buffer) {
        ByteBuffer view = buffer.duplicate();
        byte[] value = new byte[view.remaining()];
        view.get(value);
        int count = 0;
        while (count < value.length && value[count] != 0) count++;
        return new String(value, 0, count, StandardCharsets.UTF_8);
    }
    private static void metadata() {
        HFInspireFaceVersion version = new HFInspireFaceVersion();
        ok(HFQueryInspireFaceVersion(version)); expect(version.major >= 1);
        int[] value = new int[1];
        ok(HFQueryCAPILevel(value)); expect(value[0] == HF_C_API_LEVEL);
        for (int component = 0; component < HF_COMPONENT_COUNT; component++) {
            HFComponentVersion info = new HFComponentVersion();
            ok(HFQueryInspireFaceComponentVersion(component, info));
            expect(info.state >= HF_COMPONENT_VERSION_DISABLED && info.state <= HF_COMPONENT_VERSION_UNKNOWN);
        }
        HFInspireFaceExtendedInformation extra = new HFInspireFaceExtendedInformation();
        ok(HFQueryInspireFaceExtendedInformation(extra)); expect(extra.information.length == 256);
        ok(HFGetErrorMessage(HERR_INVALID_PARAM, null, 0, value));
        ByteBuffer message = bytes(value[0]);
        ok(HFGetErrorMessage(HERR_INVALID_PARAM, message, value[0], value)); expect(!text(message).isEmpty());
        expect(HFGetErrorMessage(HERR_INVALID_PARAM, bytes(1), 1, value) == HERR_INVALID_BUFFER_SIZE);
        ok(HFQueryInspireFaceComponentVersions(null, 0, value));
        ByteBuffer components = bytes(value[0]);
        ok(HFQueryInspireFaceComponentVersions(components, value[0], value)); expect(text(components).toLowerCase(java.util.Locale.ROOT).contains("mnn"));
        ok(HFQueryInspireFaceDiagnosticInformation(null, 0, value));
        ByteBuffer diagnostic = bytes(value[0]);
        ok(HFQueryInspireFaceDiagnosticInformation(diagnostic, value[0], value)); expect(value[0] > 1);
        ok(HFSetLogLevel(HF_LOG_NONE));
        ok(HFLogPrint(HF_LOG_INFO, "literal %s %n and emoji \uD83D\uDE42"));
        ok(HFLogDisable());
        ok(HFDeBugShowResourceStatistics());
        ok(HFQueryExpansiveHardwareRGACompileOption(value)); expect(value[0] == 0);
        ByteBuffer dma = bytes(256);
        ok(HFQueryExpansiveHardwareRockchipDmaHeapPath(dma));
        String originalPath = text(dma);
        ok(HFSetExpansiveHardwareRockchipDmaHeapPath("/tmp/\u6D4B\u8BD5-\uD83D\uDE42"));
        ok(HFQueryExpansiveHardwareRockchipDmaHeapPathWithSize(dma, 256));
        expect(text(dma).equals("/tmp/\u6D4B\u8BD5-\uD83D\uDE42"));
        ok(HFSetExpansiveHardwareRockchipDmaHeapPath(originalPath));
        ok(HFSwitchImageProcessingBackend(HF_IMAGE_PROCESSING_CPU));
        ok(HFSetImageProcessAlignedWidth(16));
        ok(HFSetAppleCoreMLInferenceMode(HF_APPLE_COREML_INFERENCE_MODE_CPU));
        ok(HFSetCudaDeviceId(0));
        ok(HFGetCudaDeviceId(value)); expect(value[0] == 0);
        expect(HFPrintCudaDeviceInfo() != 0);
        expect(HFGetNumCudaDevices(value) != 0);
        // This capability query can succeed with a false result on CPU builds.
        long supported = HFCheckCudaDeviceSupport(value);
        expect(supported != 0 || value[0] == 0);
        HFFaceDetectPixelList levels = new HFFaceDetectPixelList();
        ok(HFQuerySupportedPixelLevelsForFaceDetection(levels)); expect(levels.size > 0 && levels.size <= 20);
        bad(() -> HFQueryCAPILevel(new int[0]));
        bad(() -> HFQueryInspireFaceVersion(null));
        bad(() -> HFGetErrorMessage(2, ByteBuffer.allocate(20), 20, value));
        bad(() -> HFGetErrorMessage(2, bytes(20).asReadOnlyBuffer(), 20, value));
        bad(() -> HFGetErrorMessage(2, bytes(1), 100, value));
        bad(() -> HFLaunchInspireFace("embedded\u0000NUL"));
        bad(() -> HFLogPrint(HF_LOG_INFO, null));
        try { InspireFaceException.check(HERR_INVALID_PARAM); throw new AssertionError(); }
        catch (InspireFaceException error) { expect(error.getCode() == HERR_INVALID_PARAM); }
    }
    private static void images(Path output) {
        long[] stream = new long[1], bitmap = new long[1], copy = new long[1];
        ByteBuffer pixels = bytes(16 + 8 * 8 * 3);
        pixels.position(16);
        for (int i = 16; i < pixels.capacity(); i++) pixels.put(i, (byte) 80);
        HFImageData input = new HFImageData();
        input.data = pixels; input.width = 8; input.height = 8;
        input.format = HF_STREAM_BGR; input.rotation = HF_CAMERA_ROTATION_0;
        ok(HFCreateImageStream(input, stream));
        pixels.put(16, (byte) 123); // Changes must reach the borrowed stream without a copy.
        ok(HFCreateImageBitmapFromImageStreamProcess(stream[0], bitmap, 0, 1));
        HFImageBitmapData decoded = new HFImageBitmapData();
        ok(HFImageBitmapGetData(bitmap[0], decoded));
        expect(decoded.width == 8 && decoded.data.get(0) == (byte) 123);
        ok(HFReleaseImageBitmap(bitmap[0]));
        bad(() -> HFImageStreamSetFormat(stream[0], HF_STREAM_BGRA));
        WeakReference<ByteBuffer> reference = new WeakReference<>(pixels);
        input.data = null; pixels = null;
        System.gc(); expect(reference.get() != null);
        ok(HFCreateImageBitmapFromImageStreamProcess(stream[0], bitmap, 0, 1));
        ok(HFReleaseImageBitmap(bitmap[0]));
        ok(HFReleaseImageStream(stream[0]));
        expect(HFReleaseImageStream(stream[0]) == HERR_INVALID_IMAGE_STREAM_HANDLE);
        ok(HFCreateImageStreamEmpty(stream));
        ok(HFImageStreamSetFormat(stream[0], HF_STREAM_BGR));
        ByteBuffer replacement = bytes(8 * 8 * 3);
        replacement.put(0, (byte) 66);
        bad(() -> HFImageStreamSetBuffer(stream[0], bytes(3), 8, 8));
        ok(HFImageStreamSetBuffer(stream[0], replacement, 8, 8));
        ok(HFImageStreamSetRotation(stream[0], HF_CAMERA_ROTATION_90));
        ok(HFImageStreamSetRotation(stream[0], HF_CAMERA_ROTATION_0));
        HFImageBitmapData source = new HFImageBitmapData();
        source.data = replacement; source.width = 8; source.height = 8; source.channels = 3;
        ok(HFCreateImageBitmap(source, bitmap));
        replacement.put(0, (byte) 99); // Bitmap creation intentionally copies.
        ok(HFImageBitmapGetData(bitmap[0], decoded)); expect(decoded.data.get(0) == (byte) 66);
        ok(HFImageBitmapCopy(bitmap[0], copy));
        HFaceRect rect = new HFaceRect(); rect.x = 1; rect.y = 1; rect.width = 3; rect.height = 3;
        HColor color = new HColor(); color.r = 1;
        HPoint2f point = new HPoint2f(); point.x = 3; point.y = 3;
        HPoint2i pointI = new HPoint2i(); pointI.x = 4; pointI.y = 4;
        ok(HFImageBitmapDrawRect(bitmap[0], rect, color, 1));
        ok(HFImageBitmapDrawCircleF(bitmap[0], point, 1, color, 1));
        ok(HFImageBitmapDrawCircle(bitmap[0], pointI, 1, color, 1));
        // Keep Unicode conversion in our JNI path, rather than the JDK's macOS path normalizer.
        String unicode = output.toString() + "/\u56FE\u50CF-\uD83D\uDE42.png";
        ok(HFImageBitmapWriteToFile(bitmap[0], unicode));
        long[] loaded = new long[1];
        ok(HFCreateImageBitmapFromFilePath(unicode, 3, loaded));
        ok(HFReleaseImageBitmap(loaded[0]));
        ok(HFDeBugImageStreamDecodeSave(stream[0], output.resolve("stream.png").toString()));
        // Resolve GUI entry points without opening windows.
        expect(HFImageBitmapShow(0, "JNI test", 1) != 0);
        HFDeBugImageStreamImShow(0);
        ok(HFReleaseImageStream(stream[0]));
        ok(HFReleaseImageBitmap(copy[0]));
        ok(HFReleaseImageBitmap(bitmap[0]));
        for (int format = 0; format < 8; format++) {
            for (int rotation = 0; rotation < 4; rotation++) {
                input.data = bytes(8 * 8 * 4); input.format = format; input.rotation = rotation;
                ok(HFCreateImageStream(input, stream)); ok(HFReleaseImageStream(stream[0]));
            }
        }
        input.data = bytes(3); input.width = 8; input.height = 8;
        bad(() -> HFCreateImageStream(input, stream));
        input.data = bytes(256); input.format = HF_STREAM_YUV_NV21; input.width = 7;
        bad(() -> HFCreateImageStream(input, stream));
        bad(() -> HFDeBugGetUnreleasedStreams(new long[0], 1));
    }
    private static void featureHub(HFFaceFeature feature) {
        HFFeatureHubConfiguration config = new HFFeatureHubConfiguration();
        config.primaryKeyMode = HF_PK_MANUAL_INPUT;
        config.persistenceDbPath = "";
        config.searchThreshold = 0.1f;
        config.searchMode = HF_SEARCH_MODE_EXHAUSTIVE;
        ok(HFFeatureHubDataEnable(config));
        try {
            ok(HFFeatureHubFaceSearchThresholdSetting(0.1f));
            HFFaceFeatureIdentity identity = new HFFaceFeatureIdentity();
            identity.id = 0x123456789abcdefL; identity.feature = feature;
            long[] id = new long[1]; int[] count = new int[1]; float[] confidence = new float[1];
            ok(HFFeatureHubInsertFeature(identity, id)); expect(id[0] == identity.id);
            ok(HFFeatureHubGetFaceCount(count)); expect(count[0] == 1);
            HFFaceFeatureIdentity found = new HFFaceFeatureIdentity();
            ok(HFFeatureHubFaceSearch(feature, confidence, found));
            expect(found.id == identity.id && confidence[0] > 0.99f && found.feature.size == feature.size);
            HFFeatureHubSearchResultV2 result = new HFFeatureHubSearchResultV2();
            ok(HFFeatureHubFaceSearchV2(feature, result)); expect(result.found != 0 && result.id == identity.id);
            HFSearchTopKResults top = new HFSearchTopKResults();
            ok(HFFeatureHubFaceSearchTopK(feature, 1, top)); expect(top.size == 1 && top.ids.getLong(0) == identity.id);
            HFFeatureHubExistingIds ids = new HFFeatureHubExistingIds();
            ok(HFFeatureHubGetExistingIds(ids)); expect(ids.size == 1 && ids.ids.getLong(0) == identity.id);
            ok(HFFeatureHubGetFaceIdentity(identity.id, found)); expect(found.feature.size == feature.size);
            ok(HFFeatureHubFaceUpdate(identity));
            ok(HFFeatureHubViewDBTable());
            ok(HFFeatureHubFaceRemove(identity.id));
            ok(HFFeatureHubGetFaceCount(count)); expect(count[0] == 0);
        } finally { ok(HFFeatureHubDataDisable()); }
    }
    private static void model(String model, String image) {
        CPUEngine.PowerMode powerMode = CPUEngine.getGlobalPowerMode();
        HFResourcePackInfo pack = new HFResourcePackInfo();
        // Older encrypted packs can load while the optional validator reports unsupported metadata.
        expect(HFValidateResourcePack(model, pack) >= 0);
        pack.structVersion = 99;
        expect(HFValidateResourcePack(model, pack) == HERR_INVALID_PARAM);
        ok(HFLaunchInspireFace(model));
        expect(CPUEngine.getGlobalPowerMode() == powerMode);
        try {
            int[] value = new int[1]; float[] score = new float[1];
            ok(HFQueryInspireFaceLaunchStatus(value)); expect(value[0] != 0);
            ok(HFReloadInspireFace(model));
            expect(CPUEngine.getGlobalPowerMode() == powerMode);
            ok(HFSwitchLandmarkEngine(HF_LANDMARK_HYPLMV2_0_25));
            HFSessionCustomParameter parameters = new HFSessionCustomParameter();
            parameters.enable_recognition = 1; parameters.enable_liveness = 1;
            parameters.enable_mask_detect = 1; parameters.enable_face_quality = 1;
            parameters.enable_face_attribute = 1; parameters.enable_interaction_liveness = 1;
            parameters.enable_face_emotion = 1;
            int features = HF_ENABLE_FACE_RECOGNITION | HF_ENABLE_LIVENESS | HF_ENABLE_MASK_DETECT |
                    HF_ENABLE_QUALITY | HF_ENABLE_FACE_ATTRIBUTE | HF_ENABLE_INTERACTION | HF_ENABLE_FACE_EMOTION;
            long[] session = new long[1], bitmap = new long[1], stream = new long[1], snapshot = new long[1];
            ok(HFCreateInspireFaceSession(parameters, HF_DETECT_MODE_ALWAYS_DETECT, 3, -1, -1, session));
            ok(HFReleaseInspireFaceSession(session[0]));
            ok(HFCreateInspireFaceSessionOptional(features, HF_DETECT_MODE_ALWAYS_DETECT, 3, -1, -1, session));
            ok(HFReleaseInspireFaceSession(session[0]));
            HFSessionConfigV2 config = new HFSessionConfigV2();
            config.featureMask = features; config.detectMode = HF_DETECT_MODE_ALWAYS_DETECT;
            config.maxDetectFaceNum = 3; config.detectPixelLevel = -1; config.trackByDetectModeFPS = -1;
            ok(HFCreateInspireFaceSessionV2(config, session));
            try {
                ok(HFSessionSetTrackLostRecoveryMode(session[0], 1));
                ok(HFSessionSetLightTrackConfidenceThreshold(session[0], 0.2f));
                ok(HFSessionSetTrackPreviewSize(session[0], 192));
                ok(HFSessionGetTrackPreviewSize(session[0], value)); expect(value[0] == 192);
                ok(HFSessionSetFilterMinimumFacePixelSize(session[0], 8));
                ok(HFSessionSetFaceDetectThreshold(session[0], 0.5f));
                ok(HFSessionSetTrackModeSmoothRatio(session[0], 0.5f));
                ok(HFSessionSetTrackModeNumSmoothCacheFrame(session[0], 3));
                ok(HFSessionSetTrackModeDetectInterval(session[0], 1));
                ok(HFSessionSetLandmarkAugmentationNum(session[0], 1));
                ok(HFSessionSetEnableTrackCostSpend(session[0], 1));
                ok(HFCreateImageBitmapFromFilePath(image, 3, bitmap));
                HFImageBitmapData pixels = new HFImageBitmapData();
                ok(HFImageBitmapGetData(bitmap[0], pixels));
                HFImageData input = new HFImageData();
                input.data = pixels.data; input.width = pixels.width; input.height = pixels.height; input.format = HF_STREAM_BGR;
                ok(HFCreateImageStream(input, stream));
                try {
                    HFMultipleFaceData faces = new HFMultipleFaceData();
                    ok(HFExecuteFaceTrack(session[0], stream[0], faces));
                    expect(faces.detectedNum > 0 && faces.rects.length == faces.detectedNum);
                    expect(faces.rects[0].width > 0 && faces.detConfidence.getFloat(0) > 0);
                    expect(Float.isFinite(faces.angles.roll.getFloat(0)));
                    HFFaceBasicToken token = faces.tokens[0];
                    ok(HFGetFaceBasicTokenSize(value));
                    HFFaceBasicToken ownedToken = new HFFaceBasicToken();
                    ownedToken.size = value[0]; ownedToken.data = bytes(value[0]);
                    ok(HFCopyFaceBasicToken(token, ownedToken.data, value[0]));
                    ok(HFGetNumOfFaceDenseLandmark(value));
                    HPoint2f[] landmarks = new HPoint2f[value[0]];
                    ok(HFGetFaceDenseLandmarkFromFaceToken(token, landmarks, value[0])); expect(Float.isFinite(landmarks[0].x));
                    HPoint2f[] five = new HPoint2f[5];
                    ok(HFGetFaceFiveKeyPointsFromFaceToken(token, five, 5)); expect(Float.isFinite(five[0].y));
                    bad(() -> HFGetFaceFiveKeyPointsFromFaceToken(token, new HPoint2f[1], 5));
                    ok(HFFaceQualityDetect(session[0], token, score)); expect(Float.isFinite(score[0]));
                    HFFaceFeature borrowed = new HFFaceFeature(), owned = new HFFaceFeature();
                    ok(HFFaceFeatureExtract(session[0], stream[0], token, borrowed));
                    expect(HFReleaseFaceFeature(borrowed) == HERR_INVALID_FACE_FEATURE);
                    ok(HFGetFeatureLength(value)); expect(borrowed.size == value[0]);
                    ok(HFCreateFaceFeature(owned)); expect(owned.size == borrowed.size);
                    try {
                        ok(HFFaceFeatureExtractTo(session[0], stream[0], token, owned));
                        ok(HFFaceComparison(owned, borrowed, score)); expect(score[0] > 0.99f);
                        ByteBuffer copied = bytes(owned.size * 4);
                        ok(HFFaceFeatureExtractCpy(session[0], stream[0], token, copied));
                        expect(Math.abs(copied.getFloat(0) - owned.data.getFloat(0)) < 0.0001f);
                        bad(() -> HFFaceFeatureExtractCpy(session[0], stream[0], token, bytes(4)));
                        HFFaceFeature shortFeature = new HFFaceFeature();
                        shortFeature.size = owned.size; shortFeature.data = bytes(4);
                        bad(() -> HFFaceFeatureExtractTo(session[0], stream[0], token, shortFeature));
                        long[] aligned = new long[1], alignedStream = new long[1];
                        ok(HFFaceGetFaceAlignmentImage(session[0], stream[0], token, aligned));
                        ok(HFCreateImageStreamFromImageBitmap(aligned[0], 0, alignedStream));
                        ok(HFFaceFeatureExtractWithAlignmentImage(session[0], alignedStream[0], owned));
                        ok(HFReleaseImageStream(alignedStream[0])); ok(HFReleaseImageBitmap(aligned[0]));
                        featureHub(owned);
                    } finally { ok(HFReleaseFaceFeature(owned)); }
                    expect(owned.size == 0 && owned.data == null);
                    expect(HFReleaseFaceFeature(owned) == HERR_INVALID_FACE_FEATURE);
                    ok(HFGetRecommendedCosineThreshold(score));
                    ok(HFCosineSimilarityConvertToPercentage(0.8f, score)); expect(Float.isFinite(score[0]));
                    HFSimilarityConverterConfig similarity = new HFSimilarityConverterConfig();
                    ok(HFGetCosineSimilarityConverter(similarity)); ok(HFUpdateCosineSimilarityConverter(similarity));
                    ok(HFMultipleFacePipelineProcessOptional(session[0], stream[0], faces, features));
                    HFRGBLivenessConfidence live = new HFRGBLivenessConfidence();
                    HFFaceMaskConfidence mask = new HFFaceMaskConfidence();
                    HFFaceQualityConfidence quality = new HFFaceQualityConfidence();
                    HFFaceInteractionState state = new HFFaceInteractionState();
                    HFFaceInteractionsActions actions = new HFFaceInteractionsActions();
                    HFFaceAttributeResult attributes = new HFFaceAttributeResult();
                    HFFaceEmotionResult emotion = new HFFaceEmotionResult();
                    ok(HFGetRGBLivenessConfidence(session[0], live)); expect(live.num == faces.detectedNum);
                    ok(HFGetFaceMaskConfidence(session[0], mask)); expect(mask.confidence.isDirect());
                    ok(HFGetFaceQualityConfidence(session[0], quality)); expect(Float.isFinite(quality.confidence.getFloat(0)));
                    ok(HFGetFaceInteractionStateResult(session[0], state)); expect(state.num == faces.detectedNum);
                    ok(HFGetFaceInteractionActionsResult(session[0], actions)); expect(actions.num == faces.detectedNum);
                    ok(HFGetFaceAttributeResult(session[0], attributes)); expect(attributes.ageBracket.remaining() >= 4);
                    ok(HFGetFaceEmotionResult(session[0], emotion)); expect(emotion.emotion.remaining() >= 4);
                    ok(HFMultipleFacePipelineProcess(session[0], stream[0], faces, parameters));
                    ok(HFSessionLastFaceDetectionGetDebugPreviewImageSize(session[0], value));
                    ok(HFSessionPrintTrackCostSpend(session[0]));
                    ok(HFExecuteFaceTrackSnapshot(session[0], stream[0], snapshot));
                    try {
                        HFMultipleFaceData frozen = new HFMultipleFaceData();
                        ok(HFGetFaceResultSnapshotData(snapshot[0], frozen));
                        ok(HFExecuteFaceTrack(session[0], stream[0], faces));
                        ok(HFGetFaceFiveKeyPointsFromFaceToken(frozen.tokens[0], five, 5));
                        ok(HFGetFaceFiveKeyPointsFromFaceToken(ownedToken, five, 5));
                        HFFaceCaptureConfig captureConfig = new HFFaceCaptureConfig();
                        ok(HFGetDefaultFaceCaptureConfig(captureConfig)); captureConfig.filterMask = 0;
                        long[] capture = new long[1];
                        ok(HFCreateFaceCaptureSession(session[0], captureConfig, capture));
                        try {
                            long frame = (1L << 53) + 1;
                            HFFaceCaptureProgress progress = new HFFaceCaptureProgress();
                            ok(HFUpdateFaceCaptureSession(capture[0], stream[0], frame, frame, progress));
                            expect(progress.frameId == frame && progress.timestampMs == frame);
                            ok(HFUpdateFaceCaptureSessionWithSnapshot(capture[0], stream[0], snapshot[0], frame + 1, frame + 1, progress));
                            ok(HFFinishFaceCaptureSession(capture[0], progress));
                            HFFaceCaptureResult[] results = new HFFaceCaptureResult[HF_FACE_CAPTURE_MAX_RESULTS];
                            ok(HFGetFaceCaptureResults(capture[0], null, 0, value));
                            expect(value[0] > 0);
                            ok(HFGetFaceCaptureResults(capture[0], results, results.length, value));
                            expect(value[0] > 0 && results[0].token.size > 0 && results[0].frameId >= frame);
                            bad(() -> HFGetFaceCaptureResults(capture[0], new HFFaceCaptureResult[0], 1, value));
                            ok(HFResetFaceCaptureSession(capture[0]));
                        } finally { ok(HFReleaseFaceCaptureSession(capture[0])); }
                    } finally { ok(HFReleaseFaceResultSnapshot(snapshot[0])); }
                    ok(HFDeBugGetUnreleasedSessionsCount(value)); expect(value[0] == 1);
                    long[] active = new long[value[0]];
                    ok(HFDeBugGetUnreleasedSessions(active, active.length)); expect(active[0] == session[0]);
                    ok(HFDeBugGetUnreleasedStreamsCount(value)); expect(value[0] == 1);
                    active = new long[value[0]];
                    ok(HFDeBugGetUnreleasedStreams(active, active.length)); expect(active[0] == stream[0]);
                    ok(HFSessionClearTrackingFace(session[0]));
                } finally { ok(HFReleaseImageStream(stream[0])); ok(HFReleaseImageBitmap(bitmap[0])); }
            } finally { ok(HFReleaseInspireFaceSession(session[0])); }
        } finally {
            ok(HFTerminateInspireFace());
            expect(CPUEngine.getGlobalPowerMode() == powerMode);
        }
    }
    public static void main(String[] args) throws Exception {
        Path output = Files.createTempDirectory("inspireface-jni-");
        try {
            cpuEngine();
            metadata();
            images(output);
            try {
                for (CPUEngine.PowerMode mode : CPUEngine.PowerMode.values()) {
                    CPUEngine.setGlobalPowerMode(mode);
                    model(args[0], args[1]);
                }
            } finally { CPUEngine.setGlobalPowerMode(CPUEngine.PowerMode.NORMAL); }
            int[] count = new int[1];
            ok(HFDeBugGetUnreleasedSessionsCount(count)); expect(count[0] == 0);
            ok(HFDeBugGetUnreleasedStreamsCount(count)); expect(count[0] == 0);
            System.out.println("JNI contract passed: " + assertions + " assertions");
        } finally {
            try (java.util.stream.Stream<Path> paths = Files.walk(output)) {
                paths.sorted(Comparator.reverseOrder()).forEach(path -> {
                    try { Files.delete(path); } catch (Exception failure) { throw new RuntimeException(failure); }
                });
            }
        }
    }
}
