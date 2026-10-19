#import "IFInspireFace.h"

NSErrorDomain const IFErrorDomain = @"com.inspireface.error";
BOOL IFCheck(HResult status, NSError **error) {
    if (status == HSUCCEED) return YES;
    if (error) {
        char message[512] = {0};
        HFGetErrorMessage(status, message, sizeof(message), NULL);
        NSString *description = [NSString stringWithUTF8String:message] ?: @"InspireFace error";
        *error = [NSError errorWithDomain:IFErrorDomain
                                     code:status
                                 userInfo:@{NSLocalizedDescriptionKey : description}];
    }
    return NO;
}
@interface IFImageStream () {
    HFImageStream _handle;
    CVPixelBufferRef _pixelBuffer;
    HFImageFormat _pixelFormat;
}
- (void)releasePixelBuffer;
- (instancetype)initWithOwnedHandle:(HFImageStream)handle;
@end

@interface IFImageBitmap () {
    HFImageBitmap _handle;
}
- (instancetype)initWithOwnedHandle:(HFImageBitmap)handle;
@end

@interface IFSession () {
    HFSession _handle;
    BOOL _borrowingFaces;
    BOOL _borrowingFeature;
}
- (BOOL)isBorrowing;
- (instancetype)initWithOwnedHandle:(HFSession)handle;
@end

@interface IFFaceSnapshot () {
    HFFaceResultSnapshot _handle;
}
- (instancetype)initWithOwnedHandle:(HFFaceResultSnapshot)handle;
@end

@interface IFCaptureSession () {
    HFFaceCaptureSession _handle;
    IFSession *_session;
}
- (instancetype)initWithOwnedHandle:(HFFaceCaptureSession)handle;
@end

@implementation IFRuntime
+ (BOOL)launchAtPath:(NSString *)path error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFLaunchInspireFace(path.fileSystemRepresentation), error);
}
+ (BOOL)reloadAtPath:(NSString *)path error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFReloadInspireFace(path.fileSystemRepresentation), error);
}
+ (BOOL)terminateWithError:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFTerminateInspireFace(), error);
}
+ (BOOL)getLaunchStatus:(HInt32 *)status error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFQueryInspireFaceLaunchStatus(status), error);
}
+ (BOOL)setImageProcessingBackend:(HFImageProcessingBackend)backend error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFSwitchImageProcessingBackend(backend), error);
}
+ (BOOL)setImageProcessingAlignedWidth:(HInt32)width error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFSetImageProcessAlignedWidth(width), error);
}
+ (BOOL)setCoreMLInferenceMode:(HFAppleCoreMLInferenceMode)mode error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFSetAppleCoreMLInferenceMode(mode), error);
}
+ (BOOL)setLandmarkEngine:(HFSessionLandmarkEngine)engine error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFSwitchLandmarkEngine(engine), error);
}
+ (BOOL)getSupportedDetectionPixelLevels:(HFFaceDetectPixelList *)levels error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFQuerySupportedPixelLevelsForFaceDetection(levels), error);
}
+ (BOOL)validateResourcePackAtPath:(NSString *)path
                              info:(HFResourcePackInfo *_Nullable)info
                             error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFValidateResourcePack(path.fileSystemRepresentation, info), error);
}
@end

@implementation IFImageStream
- (instancetype)initWithOwnedHandle:(HFImageStream)handle {
    self = [super init];
    if (self) _handle = handle;
    return self;
}
- (HFImageStream)nativeHandle {
    return _handle;
}
- (BOOL)closeWithError:(NSError **)error {
    HResult status = HFReleaseImageStream(_handle);
    if (status == HSUCCEED) {
        _handle = NULL;
        [self releasePixelBuffer];
    }
    return IFCheck(status, error);
}
- (void)dealloc {
    if (_handle) HFReleaseImageStream(_handle);
    [self releasePixelBuffer];
}
- (void)releasePixelBuffer {
    if (_pixelBuffer) {
        CVPixelBufferUnlockBaseAddress(_pixelBuffer, kCVPixelBufferLock_ReadOnly);
        CVPixelBufferRelease(_pixelBuffer);
        _pixelBuffer = NULL;
    }
}
- (nullable instancetype)initWithPixelBuffer:(CVPixelBufferRef)pixelBuffer
                                    rotation:(HFRotation)rotation
                                       error:(NSError **)error {
    if (!pixelBuffer || CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly) != kCVReturnSuccess) {
        IFCheck(HERR_INVALID_IMAGE_STREAM_PARAM, error);
        return nil;
    }
    size_t width = CVPixelBufferGetWidth(pixelBuffer), height = CVPixelBufferGetHeight(pixelBuffer);
    OSType format = CVPixelBufferGetPixelFormatType(pixelBuffer);
    HFImageFormat hfFormat = HF_STREAM_BGRA;
    size_t channels = 4;
    BOOL valid = width > 0 && height > 0 && width <= INT32_MAX && height <= INT32_MAX;
    uint8_t *base = CVPixelBufferGetBaseAddress(pixelBuffer);
    if (format == kCVPixelFormatType_32BGRA)
        hfFormat = HF_STREAM_BGRA;
    else if (format == kCVPixelFormatType_32RGBA)
        hfFormat = HF_STREAM_RGBA;
    else if (format == kCVPixelFormatType_OneComponent8) {
        hfFormat = HF_STREAM_GRAY;
        channels = 1;
    } else if (format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange ||
               format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange) {
        hfFormat = HF_STREAM_YUV_NV12;
        channels = 0;
        valid = valid && !(width & 1) && !(height & 1) && CVPixelBufferGetPlaneCount(pixelBuffer) == 2;
        if (valid) {
            base = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0);
            valid = base && CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0) == width &&
                    CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1) == width &&
                    (uintptr_t)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1) == (uintptr_t)base + width * height;
        }
    } else
        valid = NO;
    if (channels)
        valid = valid && !CVPixelBufferIsPlanar(pixelBuffer) &&
                CVPixelBufferGetBytesPerRow(pixelBuffer) == width * channels;
    if (!valid || !base) {
        CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
        IFCheck(HERR_INVALID_IMAGE_STREAM_PARAM, error);
        return nil;
    }
    HFImageData data = {base, (HInt32)width, (HInt32)height, hfFormat, rotation};
    self = [self initWithBorrowedData:data error:error];
    if (!self) {
        CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
        return nil;
    }
    _pixelBuffer = CVPixelBufferRetain(pixelBuffer);
    _pixelFormat = hfFormat;
    return self;
}
- (nullable instancetype)initWithBorrowedData:(HFImageData)data error:(NSError *_Nullable *_Nullable)error {
    HFImageStream handle = NULL;
    if (!IFCheck(HFCreateImageStream(&data, &handle), error)) return nil;
    return [self initWithOwnedHandle:handle];
}
- (nullable instancetype)initEmptyWithError:(NSError *_Nullable *_Nullable)error {
    HFImageStream handle = NULL;
    if (!IFCheck(HFCreateImageStreamEmpty(&handle), error)) return nil;
    return [self initWithOwnedHandle:handle];
}
- (BOOL)setBorrowedBuffer:(uint8_t *)buffer
                    width:(HInt32)width
                   height:(HInt32)height
                    error:(NSError *_Nullable *_Nullable)error {
    HResult status = HFImageStreamSetBuffer(_handle, buffer, width, height);
    if (status == HSUCCEED && _pixelBuffer) {
        void *base = CVPixelBufferIsPlanar(_pixelBuffer) ? CVPixelBufferGetBaseAddressOfPlane(_pixelBuffer, 0)
                                                         : CVPixelBufferGetBaseAddress(_pixelBuffer);
        if (buffer != base) [self releasePixelBuffer];
    }
    return IFCheck(status, error);
}
- (BOOL)setRotation:(HFRotation)value error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFImageStreamSetRotation(_handle, value), error);
}
- (BOOL)setFormat:(HFImageFormat)value error:(NSError *_Nullable *_Nullable)error {
    if (_pixelBuffer && value != _pixelFormat) return IFCheck(HERR_INVALID_IMAGE_STREAM_PARAM, error);
    return IFCheck(HFImageStreamSetFormat(_handle, value), error);
}
- (nullable IFImageBitmap *)processedBitmapWithRotation:(BOOL)rotate
                                                  scale:(float)scale
                                                  error:(NSError *_Nullable *_Nullable)error {
    HFImageBitmap handle = NULL;
    if (!IFCheck(HFCreateImageBitmapFromImageStreamProcess(_handle, &handle, rotate, scale), error)) return nil;
    return [[IFImageBitmap alloc] initWithOwnedHandle:handle];
}
- (BOOL)saveDebugImageAtPath:(NSString *)path error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFDeBugImageStreamDecodeSave(_handle, path.fileSystemRepresentation), error);
}
- (void)showDebugImage {
    HFDeBugImageStreamImShow(_handle);
}
@end

@implementation IFImageBitmap
- (instancetype)initWithOwnedHandle:(HFImageBitmap)handle {
    self = [super init];
    if (self) _handle = handle;
    return self;
}
- (HFImageBitmap)nativeHandle {
    return _handle;
}
- (BOOL)closeWithError:(NSError **)error {
    HResult status = HFReleaseImageBitmap(_handle);
    if (status == HSUCCEED) _handle = NULL;
    return IFCheck(status, error);
}
- (void)dealloc {
    if (_handle) HFReleaseImageBitmap(_handle);
}
- (nullable instancetype)initCopyingData:(HFImageBitmapData)data error:(NSError *_Nullable *_Nullable)error {
    HFImageBitmap handle = NULL;
    if (!IFCheck(HFCreateImageBitmap(&data, &handle), error)) return nil;
    return [self initWithOwnedHandle:handle];
}
- (nullable instancetype)initWithContentsOfFile:(NSString *)path
                                       channels:(HInt32)channels
                                          error:(NSError *_Nullable *_Nullable)error {
    HFImageBitmap handle = NULL;
    if (!IFCheck(HFCreateImageBitmapFromFilePath(path.fileSystemRepresentation, channels, &handle), error)) return nil;
    return [self initWithOwnedHandle:handle];
}
- (nullable IFImageBitmap *)copyBitmapWithError:(NSError *_Nullable *_Nullable)error {
    HFImageBitmap handle = NULL;
    if (!IFCheck(HFImageBitmapCopy(_handle, &handle), error)) return nil;
    return [[IFImageBitmap alloc] initWithOwnedHandle:handle];
}
- (nullable IFImageStream *)snapshotStreamWithRotation:(HFRotation)rotation error:(NSError *_Nullable *_Nullable)error {
    HFImageStream handle = NULL;
    if (!IFCheck(HFCreateImageStreamFromImageBitmap(_handle, rotation, &handle), error)) return nil;
    return [[IFImageStream alloc] initWithOwnedHandle:handle];
}
- (BOOL)getBorrowedData:(HFImageBitmapData *)data error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFImageBitmapGetData(_handle, data), error);
}
- (BOOL)writeToFile:(NSString *)path error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFImageBitmapWriteToFile(_handle, path.fileSystemRepresentation), error);
}
- (BOOL)drawRect:(HFaceRect)rect
           color:(HColor)color
       thickness:(HInt32)thickness
           error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFImageBitmapDrawRect(_handle, rect, color, thickness), error);
}
- (BOOL)drawCircle:(HPoint2f)point
            radius:(HInt32)radius
             color:(HColor)color
         thickness:(HInt32)thickness
             error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFImageBitmapDrawCircleF(_handle, point, radius, color, thickness), error);
}
- (BOOL)drawIntegerCircle:(HPoint2i)point
                   radius:(HInt32)radius
                    color:(HColor)color
                thickness:(HInt32)thickness
                    error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFImageBitmapDrawCircle(_handle, point, radius, color, thickness), error);
}
- (BOOL)showWithTitle:(NSString *)title delay:(HInt32)delay error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFImageBitmapShow(_handle, (char *)title.UTF8String, delay), error);
}
@end

@implementation IFSession
- (BOOL)isBorrowing {
    return _borrowingFaces || _borrowingFeature;
}
- (BOOL)beginBorrowingFacesFromStream:(IFImageStream *)stream
                               result:(HFMultipleFaceData *)result
                                error:(NSError **)error {
    if (![self trackStream:stream borrowedResult:result error:error]) return NO;
    _borrowingFaces = YES;
    return YES;
}
- (void)endBorrowingFaces {
    _borrowingFaces = NO;
}
- (BOOL)beginBorrowingFeatureFromStream:(IFImageStream *)stream
                                  token:(HFFaceBasicToken)token
                                 result:(HFFaceFeature *)result
                                  error:(NSError **)error {
    if (![self extractFeatureFromStream:stream token:token borrowedResult:result error:error]) return NO;
    _borrowingFeature = YES;
    return YES;
}
- (void)endBorrowingFeature {
    _borrowingFeature = NO;
}
- (BOOL)withBorrowedFacesFromStream:(IFImageStream *)stream
                               body:(void(NS_NOESCAPE ^)(HFMultipleFaceData))body
                              error:(NSError **)error {
    HFMultipleFaceData result = {0};
    if (![self beginBorrowingFacesFromStream:stream result:&result error:error]) return NO;
    @try {
        body(result);
    } @finally {
        [self endBorrowingFaces];
    }
    return YES;
}
- (BOOL)withBorrowedFeatureFromStream:(IFImageStream *)stream
                                token:(HFFaceBasicToken)token
                                 body:(void(NS_NOESCAPE ^)(HFFaceFeature))body
                                error:(NSError **)error {
    HFFaceFeature result = {0};
    if (![self beginBorrowingFeatureFromStream:stream token:token result:&result error:error]) return NO;
    @try {
        body(result);
    } @finally {
        [self endBorrowingFeature];
    }
    return YES;
}
- (instancetype)initWithOwnedHandle:(HFSession)handle {
    self = [super init];
    if (self) _handle = handle;
    return self;
}
- (HFSession)nativeHandle {
    return _handle;
}
- (BOOL)closeWithError:(NSError **)error {
    if ([self isBorrowing]) return IFCheck(HERR_INVALID_PARAM, error);
    HResult status = HFReleaseInspireFaceSession(_handle);
    if (status == HSUCCEED) _handle = NULL;
    return IFCheck(status, error);
}
- (void)dealloc {
    if (_handle) HFReleaseInspireFaceSession(_handle);
}
- (nullable instancetype)initWithConfiguration:(HFSessionConfigV2)configuration
                                         error:(NSError *_Nullable *_Nullable)error {
    HFSession handle = NULL;
    if (!IFCheck(HFCreateInspireFaceSessionV2(&configuration, &handle), error)) return nil;
    return [self initWithOwnedHandle:handle];
}
- (nullable instancetype)initWithParameters:(HFSessionCustomParameter)parameters
                                       mode:(HFDetectMode)mode
                               maximumFaces:(HInt32)maximumFaces
                                 pixelLevel:(HInt32)pixelLevel
                            framesPerSecond:(HInt32)framesPerSecond
                                      error:(NSError *_Nullable *_Nullable)error {
    HFSession handle = NULL;
    if (!IFCheck(HFCreateInspireFaceSession(parameters, mode, maximumFaces, pixelLevel, framesPerSecond, &handle),
                 error))
        return nil;
    return [self initWithOwnedHandle:handle];
}
- (nullable instancetype)initWithOptions:(HOption)options
                                    mode:(HFDetectMode)mode
                            maximumFaces:(HInt32)maximumFaces
                              pixelLevel:(HInt32)pixelLevel
                         framesPerSecond:(HInt32)framesPerSecond
                                   error:(NSError *_Nullable *_Nullable)error {
    HFSession handle = NULL;
    if (!IFCheck(HFCreateInspireFaceSessionOptional(options, mode, maximumFaces, pixelLevel, framesPerSecond, &handle),
                 error))
        return nil;
    return [self initWithOwnedHandle:handle];
}
- (BOOL)setTrackLostRecoveryEnabled:(BOOL)value error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFSessionSetTrackLostRecoveryMode(_handle, value), error);
}
- (BOOL)setLightTrackConfidenceThreshold:(float)value error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFSessionSetLightTrackConfidenceThreshold(_handle, value), error);
}
- (BOOL)setTrackPreviewSize:(HInt32)value error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFSessionSetTrackPreviewSize(_handle, value), error);
}
- (BOOL)setMinimumFacePixelSize:(HInt32)value error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFSessionSetFilterMinimumFacePixelSize(_handle, value), error);
}
- (BOOL)setDetectionThreshold:(float)value error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFSessionSetFaceDetectThreshold(_handle, value), error);
}
- (BOOL)setTrackingSmoothRatio:(float)value error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFSessionSetTrackModeSmoothRatio(_handle, value), error);
}
- (BOOL)setTrackingSmoothCacheFrames:(HInt32)value error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFSessionSetTrackModeNumSmoothCacheFrame(_handle, value), error);
}
- (BOOL)setDetectionInterval:(HInt32)value error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFSessionSetTrackModeDetectInterval(_handle, value), error);
}
- (BOOL)setLandmarkAugmentationCount:(HInt32)value error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFSessionSetLandmarkAugmentationNum(_handle, value), error);
}
- (BOOL)setTrackingTimingEnabled:(BOOL)value error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFSessionSetEnableTrackCostSpend(_handle, value), error);
}
- (BOOL)getTrackPreviewSize:(HInt32 *)result error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFSessionGetTrackPreviewSize(_handle, result), error);
}
- (BOOL)getDebugPreviewImageSize:(HInt32 *)result error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFSessionLastFaceDetectionGetDebugPreviewImageSize(_handle, result), error);
}
- (BOOL)getBorrowedRGBLiveness:(HFRGBLivenessConfidence *)result error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFGetRGBLivenessConfidence(_handle, result), error);
}
- (BOOL)getBorrowedMaskConfidence:(HFFaceMaskConfidence *)result error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFGetFaceMaskConfidence(_handle, result), error);
}
- (BOOL)getBorrowedQualityConfidence:(HFFaceQualityConfidence *)result error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFGetFaceQualityConfidence(_handle, result), error);
}
- (BOOL)getBorrowedInteractionState:(HFFaceInteractionState *)result error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFGetFaceInteractionStateResult(_handle, result), error);
}
- (BOOL)getBorrowedInteractionActions:(HFFaceInteractionsActions *)result error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFGetFaceInteractionActionsResult(_handle, result), error);
}
- (BOOL)getBorrowedAttributes:(HFFaceAttributeResult *)result error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFGetFaceAttributeResult(_handle, result), error);
}
- (BOOL)getBorrowedEmotions:(HFFaceEmotionResult *)result error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFGetFaceEmotionResult(_handle, result), error);
}
- (BOOL)clearTrackingWithError:(NSError *_Nullable *_Nullable)error {
    if ([self isBorrowing]) return IFCheck(HERR_INVALID_PARAM, error);
    return IFCheck(HFSessionClearTrackingFace(_handle), error);
}
- (BOOL)printTrackingTimingWithError:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFSessionPrintTrackCostSpend(_handle), error);
}
- (BOOL)trackStream:(IFImageStream *)stream
     borrowedResult:(HFMultipleFaceData *)result
              error:(NSError *_Nullable *_Nullable)error {
    if ([self isBorrowing]) return IFCheck(HERR_INVALID_PARAM, error);
    return IFCheck(HFExecuteFaceTrack(_handle, stream.nativeHandle, result), error);
}
- (nullable IFFaceSnapshot *)snapshotFromStream:(IFImageStream *)stream error:(NSError *_Nullable *_Nullable)error {
    if ([self isBorrowing]) {
        IFCheck(HERR_INVALID_PARAM, error);
        return nil;
    }
    HFFaceResultSnapshot handle = NULL;
    if (!IFCheck(HFExecuteFaceTrackSnapshot(_handle, stream.nativeHandle, &handle), error)) return nil;
    return [[IFFaceSnapshot alloc] initWithOwnedHandle:handle];
}
- (BOOL)extractFeatureFromStream:(IFImageStream *)stream
                           token:(HFFaceBasicToken)token
                  borrowedResult:(HFFaceFeature *)feature
                           error:(NSError *_Nullable *_Nullable)error {
    if (_borrowingFeature) return IFCheck(HERR_INVALID_PARAM, error);
    return IFCheck(HFFaceFeatureExtract(_handle, stream.nativeHandle, token, feature), error);
}
- (BOOL)extractFeatureFromStream:(IFImageStream *)stream
                           token:(HFFaceBasicToken)token
                            into:(HFFaceFeature)feature
                           error:(NSError *_Nullable *_Nullable)error {
    if (_borrowingFeature) return IFCheck(HERR_INVALID_PARAM, error);
    return IFCheck(HFFaceFeatureExtractTo(_handle, stream.nativeHandle, token, feature), error);
}
- (BOOL)copyFeatureFromStream:(IFImageStream *)stream
                        token:(HFFaceBasicToken)token
                         into:(float *)buffer
                     capacity:(HInt32)capacity
                        error:(NSError *_Nullable *_Nullable)error {
    HInt32 count = 0;
    if (!IFCheck(HFGetFeatureLength(&count), error)) return NO;
    if (!buffer || capacity < count) return IFCheck(HERR_INVALID_FACE_FEATURE, error);
    if (_borrowingFeature) return IFCheck(HERR_INVALID_PARAM, error);
    return IFCheck(HFFaceFeatureExtractCpy(_handle, stream.nativeHandle, token, buffer), error);
}
- (nullable IFImageBitmap *)alignmentBitmapFromStream:(IFImageStream *)stream
                                                token:(HFFaceBasicToken)token
                                                error:(NSError *_Nullable *_Nullable)error {
    HFImageBitmap handle = NULL;
    if (!IFCheck(HFFaceGetFaceAlignmentImage(_handle, stream.nativeHandle, token, &handle), error)) return nil;
    return [[IFImageBitmap alloc] initWithOwnedHandle:handle];
}
- (BOOL)extractAlignedFeatureFromStream:(IFImageStream *)stream
                                   into:(HFFaceFeature)feature
                                  error:(NSError *_Nullable *_Nullable)error {
    if (_borrowingFeature) return IFCheck(HERR_INVALID_PARAM, error);
    return IFCheck(HFFaceFeatureExtractWithAlignmentImage(_handle, stream.nativeHandle, feature), error);
}
- (BOOL)processStream:(IFImageStream *)stream
                faces:(HFMultipleFaceData *)faces
           parameters:(HFSessionCustomParameter)parameters
                error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFMultipleFacePipelineProcess(_handle, stream.nativeHandle, faces, parameters), error);
}
- (BOOL)processStream:(IFImageStream *)stream
                faces:(HFMultipleFaceData *)faces
              options:(HInt32)options
                error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFMultipleFacePipelineProcessOptional(_handle, stream.nativeHandle, faces, options), error);
}
- (BOOL)getQualityForToken:(HFFaceBasicToken)token result:(float *)result error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFFaceQualityDetect(_handle, token, result), error);
}
@end

@implementation IFFaceSnapshot
- (instancetype)initWithOwnedHandle:(HFFaceResultSnapshot)handle {
    self = [super init];
    if (self) _handle = handle;
    return self;
}
- (HFFaceResultSnapshot)nativeHandle {
    return _handle;
}
- (BOOL)closeWithError:(NSError **)error {
    HResult status = HFReleaseFaceResultSnapshot(_handle);
    if (status == HSUCCEED) _handle = NULL;
    return IFCheck(status, error);
}
- (void)dealloc {
    if (_handle) HFReleaseFaceResultSnapshot(_handle);
}
- (BOOL)getBorrowedFaces:(HFMultipleFaceData *)result error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFGetFaceResultSnapshotData(_handle, result), error);
}
@end

@implementation IFCaptureSession
- (instancetype)initWithOwnedHandle:(HFFaceCaptureSession)handle {
    self = [super init];
    if (self) _handle = handle;
    return self;
}
- (HFFaceCaptureSession)nativeHandle {
    return _handle;
}
- (BOOL)closeWithError:(NSError **)error {
    HResult status = HFReleaseFaceCaptureSession(_handle);
    if (status == HSUCCEED) {
        _handle = NULL;
        _session = nil;
    }
    return IFCheck(status, error);
}
- (void)dealloc {
    if (_handle) HFReleaseFaceCaptureSession(_handle);
}
+ (BOOL)getDefaultConfiguration:(HFFaceCaptureConfig *)configuration error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFGetDefaultFaceCaptureConfig(configuration), error);
}
- (nullable instancetype)initWithSession:(IFSession *)session
                           configuration:(HFFaceCaptureConfig)configuration
                                   error:(NSError *_Nullable *_Nullable)error {
    HFFaceCaptureSession handle = NULL;
    if (!IFCheck(HFCreateFaceCaptureSession(session.nativeHandle, &configuration, &handle), error)) return nil;
    self = [self initWithOwnedHandle:handle];
    if (self) _session = session;
    return self;
}
- (BOOL)updateStream:(IFImageStream *)stream
                  frameID:(uint64_t)frameID
    timestampMilliseconds:(uint64_t)timestamp
                 progress:(HFFaceCaptureProgress *)progress
                    error:(NSError *_Nullable *_Nullable)error {
    if ([_session isBorrowing]) return IFCheck(HERR_INVALID_PARAM, error);
    return IFCheck(HFUpdateFaceCaptureSession(_handle, stream.nativeHandle, frameID, timestamp, progress), error);
}
- (BOOL)updateStream:(IFImageStream *)stream
                 snapshot:(IFFaceSnapshot *)snapshot
                  frameID:(uint64_t)frameID
    timestampMilliseconds:(uint64_t)timestamp
                 progress:(HFFaceCaptureProgress *)progress
                    error:(NSError *_Nullable *_Nullable)error {
    if ([_session isBorrowing]) return IFCheck(HERR_INVALID_PARAM, error);
    return IFCheck(HFUpdateFaceCaptureSessionWithSnapshot(_handle, stream.nativeHandle, snapshot.nativeHandle, frameID,
                                                          timestamp, progress),
                   error);
}
- (BOOL)getResults:(HFFaceCaptureResult *_Nullable)results
          capacity:(uint32_t)capacity
             count:(uint32_t *)count
             error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFGetFaceCaptureResults(_handle, results, capacity, count), error);
}
- (BOOL)finishWithProgress:(HFFaceCaptureProgress *)progress error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFFinishFaceCaptureSession(_handle, progress), error);
}
- (BOOL)resetWithError:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFResetFaceCaptureSession(_handle), error);
}
@end

@implementation IFFeatureBuffer {
    HFFaceFeature _feature;
}
- (HFFaceFeature)borrowedFeature {
    return _feature;
}
- (void)dealloc {
    if (_feature.data) HFReleaseFaceFeature(&_feature);
}
- (nullable instancetype)initWithError:(NSError *_Nullable *_Nullable)error {
    self = [super init];
    if (self && !IFCheck(HFCreateFaceFeature(&_feature), error)) return nil;
    return self;
}
- (BOOL)closeWithError:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFReleaseFaceFeature(&_feature), error);
}
+ (BOOL)getLength:(HInt32 *)length error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFGetFeatureLength(length), error);
}
+ (BOOL)compare:(HFFaceFeature)first
           with:(HFFaceFeature)second
     similarity:(float *)similarity
          error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFFaceComparison(first, second, similarity), error);
}
+ (BOOL)getRecommendedThreshold:(float *)threshold error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFGetRecommendedCosineThreshold(threshold), error);
}
+ (BOOL)convertSimilarity:(float)similarity percentage:(float *)percentage error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFCosineSimilarityConvertToPercentage(similarity, percentage), error);
}
+ (BOOL)setSimilarityConverter:(HFSimilarityConverterConfig)configuration error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFUpdateCosineSimilarityConverter(configuration), error);
}
+ (BOOL)getSimilarityConverter:(HFSimilarityConverterConfig *)configuration error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFGetCosineSimilarityConverter(configuration), error);
}
@end

@implementation IFFeatureHub
+ (BOOL)enableWithConfiguration:(HFFeatureHubConfiguration)configuration error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFFeatureHubDataEnable(configuration), error);
}
+ (BOOL)disableWithError:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFFeatureHubDataDisable(), error);
}
+ (BOOL)setSearchThreshold:(float)threshold error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFFeatureHubFaceSearchThresholdSetting(threshold), error);
}
+ (BOOL)insert:(HFFaceFeatureIdentity)identity
    allocatedID:(HFaceId *)allocatedID
          error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFFeatureHubInsertFeature(identity, allocatedID), error);
}
+ (BOOL)search:(HFFaceFeature)feature
          confidence:(float *)confidence
    borrowedIdentity:(HFFaceFeatureIdentity *)identity
               error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFFeatureHubFaceSearch(feature, confidence, identity), error);
}
+ (BOOL)search:(HFFaceFeature)feature
    borrowedResult:(HFFeatureHubSearchResultV2 *)result
             error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFFeatureHubFaceSearchV2(feature, result), error);
}
+ (BOOL)search:(HFFaceFeature)feature
               topK:(HInt32)topK
    borrowedResults:(HFSearchTopKResults *)results
              error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFFeatureHubFaceSearchTopK(feature, topK, results), error);
}
+ (BOOL)removeIdentityWithID:(HFaceId)identityID error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFFeatureHubFaceRemove(identityID), error);
}
+ (BOOL)updateIdentity:(HFFaceFeatureIdentity)identity error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFFeatureHubFaceUpdate(identity), error);
}
+ (BOOL)getIdentityWithID:(HFaceId)identityID
           borrowedResult:(HFFaceFeatureIdentity *)identity
                    error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFFeatureHubGetFaceIdentity(identityID, identity), error);
}
+ (BOOL)getIdentityCount:(HInt32 *)count error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFFeatureHubGetFaceCount(count), error);
}
+ (BOOL)printTableWithError:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFFeatureHubViewDBTable(), error);
}
+ (BOOL)getBorrowedExistingIDs:(HFFeatureHubExistingIds *)ids error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFFeatureHubGetExistingIds(ids), error);
}
@end

@implementation IFFaceToken
+ (BOOL)copyToken:(HFFaceBasicToken)token
             into:(char *)buffer
         capacity:(HInt32)capacity
            error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFCopyFaceBasicToken(token, buffer, capacity), error);
}
+ (BOOL)getTokenSize:(HInt32 *)count error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFGetFaceBasicTokenSize(count), error);
}
+ (BOOL)getDenseLandmarkCount:(HInt32 *)count error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFGetNumOfFaceDenseLandmark(count), error);
}
+ (BOOL)getDenseLandmarks:(HFFaceBasicToken)token
                     into:(HPoint2f *)points
                 capacity:(HInt32)capacity
                    error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFGetFaceDenseLandmarkFromFaceToken(token, points, capacity), error);
}
+ (BOOL)getFiveKeyPoints:(HFFaceBasicToken)token
                    into:(HPoint2f *)points
                capacity:(HInt32)capacity
                   error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFGetFaceFiveKeyPointsFromFaceToken(token, points, capacity), error);
}
@end

@implementation IFDiagnostics
+ (BOOL)getVersion:(HFInspireFaceVersion *)value error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFQueryInspireFaceVersion(value), error);
}
+ (BOOL)getCAPILevel:(HFUInt32 *)value error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFQueryCAPILevel(value), error);
}
+ (BOOL)getExtendedInformation:(HFInspireFaceExtendedInformation *)value error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFQueryInspireFaceExtendedInformation(value), error);
}
+ (BOOL)getLiveSessionCount:(HInt32 *)value error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFDeBugGetUnreleasedSessionsCount(value), error);
}
+ (BOOL)getLiveStreamCount:(HInt32 *)value error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFDeBugGetUnreleasedStreamsCount(value), error);
}
+ (BOOL)getComponentVersion:(HFComponentType)component
                     result:(HFComponentVersion *)result
                      error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFQueryInspireFaceComponentVersion(component, result), error);
}
+ (BOOL)getComponentVersions:(char *_Nullable)buffer
                    capacity:(HInt32)capacity
                requiredSize:(HInt32 *_Nullable)requiredSize
                       error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFQueryInspireFaceComponentVersions(buffer, capacity, requiredSize), error);
}
+ (BOOL)getDiagnosticInformation:(char *_Nullable)buffer
                        capacity:(HInt32)capacity
                    requiredSize:(HInt32 *_Nullable)requiredSize
                           error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFQueryInspireFaceDiagnosticInformation(buffer, capacity, requiredSize), error);
}
+ (BOOL)getErrorMessage:(HResult)code
                   into:(char *_Nullable)buffer
               capacity:(HInt32)capacity
           requiredSize:(HInt32 *_Nullable)requiredSize
                  error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFGetErrorMessage(code, buffer, capacity, requiredSize), error);
}
+ (BOOL)setLogLevel:(HFLogLevel)level error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFSetLogLevel(level), error);
}
+ (BOOL)disableLoggingWithError:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFLogDisable(), error);
}
+ (BOOL)printResourceStatisticsWithError:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFDeBugShowResourceStatistics(), error);
}
+ (BOOL)logMessage:(NSString *)message level:(HFLogLevel)level error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFLogPrint(level, "%s", message.UTF8String), error);
}
+ (BOOL)getBorrowedLiveSessions:(HFSession _Nullable *_Nonnull)buffer
                       capacity:(HInt32)capacity
                          error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFDeBugGetUnreleasedSessions(buffer, capacity), error);
}
+ (BOOL)getBorrowedLiveStreams:(HFImageStream _Nullable *_Nonnull)buffer
                      capacity:(HInt32)capacity
                         error:(NSError *_Nullable *_Nullable)error {
    return IFCheck(HFDeBugGetUnreleasedStreams(buffer, capacity), error);
}
@end
