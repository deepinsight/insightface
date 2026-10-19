#pragma once
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#include "inspireface.h"

NS_ASSUME_NONNULL_BEGIN
#define IF_APPLE_EXPORT __attribute__((visibility("default")))
FOUNDATION_EXPORT NSErrorDomain const IFErrorDomain;
/** Returns immediately on success. Only constructs an NSError when requested on failure. */
FOUNDATION_EXPORT BOOL IFCheck(HResult status, NSError *_Nullable *_Nullable error);
@class IFImageStream, IFImageBitmap, IFSession, IFFaceSnapshot, IFCaptureSession;

/**
 All calls are synchronous. Serialize operations on a session and changes to global
 runtime/FeatureHub state. Borrowed pointers follow C API lifetimes even if their
 Objective-C owner remains alive. Direct C calls can also invalidate these views.
 No implicit copying or per-element boxing occurs in descriptor/buffer methods.
 */

NS_SWIFT_NAME(InspireFaceRuntime)
__attribute__((objc_subclassing_restricted))
IF_APPLE_EXPORT @interface IFRuntime : NSObject
- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;
// C API: HFLaunchInspireFace
+ (BOOL)launchAtPath:(NSString *)path error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(launch(path:));
// C API: HFReloadInspireFace
+ (BOOL)reloadAtPath:(NSString *)path error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(reload(path:));
// C API: HFTerminateInspireFace
+ (BOOL)terminateWithError:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(terminate());
// C API: HFQueryInspireFaceLaunchStatus
+ (BOOL)getLaunchStatus:(HInt32 *)status error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getLaunchStatus(_:));
// C API: HFSwitchImageProcessingBackend
+ (BOOL)setImageProcessingBackend:(HFImageProcessingBackend)backend
                            error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(setImageProcessingBackend(_:));
// C API: HFSetImageProcessAlignedWidth
+ (BOOL)setImageProcessingAlignedWidth:(HInt32)width
                                 error:(NSError *_Nullable *_Nullable)error
    NS_SWIFT_NAME(setImageProcessingAlignedWidth(_:));
// C API: HFSetAppleCoreMLInferenceMode
+ (BOOL)setCoreMLInferenceMode:(HFAppleCoreMLInferenceMode)mode
                         error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(setCoreMLInferenceMode(_:));
// C API: HFSwitchLandmarkEngine
+ (BOOL)setLandmarkEngine:(HFSessionLandmarkEngine)engine
                    error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(setLandmarkEngine(_:));
// C API: HFQuerySupportedPixelLevelsForFaceDetection
+ (BOOL)getSupportedDetectionPixelLevels:(HFFaceDetectPixelList *)levels
                                   error:(NSError *_Nullable *_Nullable)error
    NS_SWIFT_NAME(getSupportedDetectionPixelLevels(_:));
// C API: HFValidateResourcePack
+ (BOOL)validateResourcePackAtPath:(NSString *)path
                              info:(HFResourcePackInfo *_Nullable)info
                             error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(validateResourcePack(path:info:));
@end

NS_SWIFT_NAME(ImageStream)
__attribute__((objc_subclassing_restricted))
IF_APPLE_EXPORT @interface IFImageStream : NSObject
- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;
/** Borrowed C handle. Use close for release; never release this handle directly through C. */
@property(nonatomic, readonly, nullable) HFImageStream nativeHandle;
/** Releases once; further calls report the C invalid-handle error. All borrowed views expire. */
// C API: HFReleaseImageStream
- (BOOL)closeWithError:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(close());
/** Does not copy pixels. Keep the source allocation alive and unchanged until close or replacement. Never retain a
 * pointer obtained from a temporary Swift buffer scope. */
/** Locks and retains tightly packed BGRA/RGBA/gray/NV12 storage until close or buffer replacement.
 Rejects padded rows or noncontiguous planes with HERR_INVALID_IMAGE_STREAM_PARAM; never copies pixels. */
- (nullable instancetype)initWithPixelBuffer:(CVPixelBufferRef)pixelBuffer
                                    rotation:(HFRotation)rotation
                                       error:(NSError *_Nullable *_Nullable)error;
// C API: HFCreateImageStream
- (nullable instancetype)initWithBorrowedData:(HFImageData)data
                                        error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(init(borrowing:));
// C API: HFCreateImageStreamEmpty
- (nullable instancetype)initEmptyWithError:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(init());
/** No copy or ownership transfer. The caller provides sufficient storage for the configured format. */
// C API: HFImageStreamSetBuffer
- (BOOL)setBorrowedBuffer:(uint8_t *)buffer
                    width:(HInt32)width
                   height:(HInt32)height
                    error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(setBorrowedBuffer(_:width:height:));
// C API: HFImageStreamSetRotation
- (BOOL)setRotation:(HFRotation)value error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(setRotation(_:));
// C API: HFImageStreamSetFormat
- (BOOL)setFormat:(HFImageFormat)value error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(setFormat(_:));
/** Allocates the processed image; conversion is explicit. */
// C API: HFCreateImageBitmapFromImageStreamProcess
- (nullable IFImageBitmap *)processedBitmapWithRotation:(BOOL)rotate
                                                  scale:(float)scale
                                                  error:(NSError *_Nullable *_Nullable)error
    NS_SWIFT_NAME(processedBitmap(rotate:scale:));
// C API: HFDeBugImageStreamDecodeSave
- (BOOL)saveDebugImageAtPath:(NSString *)path
                       error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(saveDebugImage(path:));
/** Uses the C build's GUI/file fallback. No additional display subsystem is installed. */
// C API: HFDeBugImageStreamImShow
- (void)showDebugImage;
@end

NS_SWIFT_NAME(ImageBitmap)
__attribute__((objc_subclassing_restricted))
IF_APPLE_EXPORT @interface IFImageBitmap : NSObject
- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;
/** Borrowed C handle. Use close for release; never release this handle directly through C. */
@property(nonatomic, readonly, nullable) HFImageBitmap nativeHandle;
/** Releases once; further calls report the C invalid-handle error. All borrowed views expire. */
// C API: HFReleaseImageBitmap
- (BOOL)closeWithError:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(close());
/** Copies pixels once, as required by HFCreateImageBitmap. */
// C API: HFCreateImageBitmap
- (nullable instancetype)initCopyingData:(HFImageBitmapData)data
                                   error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(init(copying:));
// C API: HFCreateImageBitmapFromFilePath
- (nullable instancetype)initWithContentsOfFile:(NSString *)path
                                       channels:(HInt32)channels
                                          error:(NSError *_Nullable *_Nullable)error
    NS_SWIFT_NAME(init(contentsOfFile:channels:));
// C API: HFImageBitmapCopy
- (nullable IFImageBitmap *)copyBitmapWithError:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(copyBitmap());
/** The C API copies a pixel snapshot. Use a borrowed stream over getBorrowedData for a no-copy path with caller-managed
 * lifetime. */
// C API: HFCreateImageStreamFromImageBitmap
- (nullable IFImageStream *)snapshotStreamWithRotation:(HFRotation)rotation
                                                 error:(NSError *_Nullable *_Nullable)error
    NS_SWIFT_NAME(snapshotStream(rotation:));
/** Borrowed pixels expire when the bitmap closes; drawing mutates the same storage. */
// C API: HFImageBitmapGetData
- (BOOL)getBorrowedData:(HFImageBitmapData *)data
                  error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getBorrowedData(_:) );
// C API: HFImageBitmapWriteToFile
- (BOOL)writeToFile:(NSString *)path error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(write(toFile:));
// C API: HFImageBitmapDrawRect
- (BOOL)drawRect:(HFaceRect)rect
           color:(HColor)color
       thickness:(HInt32)thickness
           error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(draw(rect:color:thickness:));
// C API: HFImageBitmapDrawCircleF
- (BOOL)drawCircle:(HPoint2f)point
            radius:(HInt32)radius
             color:(HColor)color
         thickness:(HInt32)thickness
             error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(drawCircle(at:radius:color:thickness:));
// C API: HFImageBitmapDrawCircle
- (BOOL)drawIntegerCircle:(HPoint2i)point
                   radius:(HInt32)radius
                    color:(HColor)color
                thickness:(HInt32)thickness
                    error:(NSError *_Nullable *_Nullable)error
    NS_SWIFT_NAME(drawIntegerCircle(at:radius:color:thickness:));
// C API: HFImageBitmapShow
- (BOOL)showWithTitle:(NSString *)title
                delay:(HInt32)delay
                error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(show(title:delay:));
@end

NS_SWIFT_NAME(FaceSession)
__attribute__((objc_subclassing_restricted))
IF_APPLE_EXPORT @interface IFSession : NSObject
- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;
/** Borrowed C handle. Use close for release; never release this handle directly through C. */
@property(nonatomic, readonly, nullable) HFSession nativeHandle;
/** Releases once; further calls report the C invalid-handle error. All borrowed views expire. */
// C API: HFReleaseInspireFaceSession
- (BOOL)closeWithError:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(close());
// C API: HFCreateInspireFaceSessionV2
- (nullable instancetype)initWithConfiguration:(HFSessionConfigV2)configuration
                                         error:(NSError *_Nullable *_Nullable)error;
// C API: HFCreateInspireFaceSession
- (nullable instancetype)initWithParameters:(HFSessionCustomParameter)parameters
                                       mode:(HFDetectMode)mode
                               maximumFaces:(HInt32)maximumFaces
                                 pixelLevel:(HInt32)pixelLevel
                            framesPerSecond:(HInt32)framesPerSecond
                                      error:(NSError *_Nullable *_Nullable)error;
// C API: HFCreateInspireFaceSessionOptional
- (nullable instancetype)initWithOptions:(HOption)options
                                    mode:(HFDetectMode)mode
                            maximumFaces:(HInt32)maximumFaces
                              pixelLevel:(HInt32)pixelLevel
                         framesPerSecond:(HInt32)framesPerSecond
                                   error:(NSError *_Nullable *_Nullable)error;
// C API: HFSessionSetTrackLostRecoveryMode
- (BOOL)setTrackLostRecoveryEnabled:(BOOL)value
                              error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(setTrackLostRecoveryEnabled(_:));
// C API: HFSessionSetLightTrackConfidenceThreshold
- (BOOL)setLightTrackConfidenceThreshold:(float)value
                                   error:(NSError *_Nullable *_Nullable)error
    NS_SWIFT_NAME(setLightTrackConfidenceThreshold(_:));
// C API: HFSessionSetTrackPreviewSize
- (BOOL)setTrackPreviewSize:(HInt32)value
                      error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(setTrackPreviewSize(_:));
// C API: HFSessionSetFilterMinimumFacePixelSize
- (BOOL)setMinimumFacePixelSize:(HInt32)value
                          error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(setMinimumFacePixelSize(_:));
// C API: HFSessionSetFaceDetectThreshold
- (BOOL)setDetectionThreshold:(float)value
                        error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(setDetectionThreshold(_:));
// C API: HFSessionSetTrackModeSmoothRatio
- (BOOL)setTrackingSmoothRatio:(float)value
                         error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(setTrackingSmoothRatio(_:));
// C API: HFSessionSetTrackModeNumSmoothCacheFrame
- (BOOL)setTrackingSmoothCacheFrames:(HInt32)value
                               error:(NSError *_Nullable *_Nullable)error
    NS_SWIFT_NAME(setTrackingSmoothCacheFrames(_:));
// C API: HFSessionSetTrackModeDetectInterval
- (BOOL)setDetectionInterval:(HInt32)value
                       error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(setDetectionInterval(_:));
// C API: HFSessionSetLandmarkAugmentationNum
- (BOOL)setLandmarkAugmentationCount:(HInt32)value
                               error:(NSError *_Nullable *_Nullable)error
    NS_SWIFT_NAME(setLandmarkAugmentationCount(_:));
// C API: HFSessionSetEnableTrackCostSpend
- (BOOL)setTrackingTimingEnabled:(BOOL)value
                           error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(setTrackingTimingEnabled(_:));
/** Writes only the result descriptor. Any referenced arrays belong to the session and may be overwritten by later
 * processing; serialize access. */
// C API: HFSessionGetTrackPreviewSize
- (BOOL)getTrackPreviewSize:(HInt32 *)result
                      error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getTrackPreviewSize(_:));
/** Writes only the result descriptor. Any referenced arrays belong to the session and may be overwritten by later
 * processing; serialize access. */
// C API: HFSessionLastFaceDetectionGetDebugPreviewImageSize
- (BOOL)getDebugPreviewImageSize:(HInt32 *)result
                           error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getDebugPreviewImageSize(_:));
/** Writes only the result descriptor. Any referenced arrays belong to the session and may be overwritten by later
 * processing; serialize access. */
// C API: HFGetRGBLivenessConfidence
- (BOOL)getBorrowedRGBLiveness:(HFRGBLivenessConfidence *)result
                         error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getBorrowedRGBLiveness(_:));
/** Writes only the result descriptor. Any referenced arrays belong to the session and may be overwritten by later
 * processing; serialize access. */
// C API: HFGetFaceMaskConfidence
- (BOOL)getBorrowedMaskConfidence:(HFFaceMaskConfidence *)result
                            error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getBorrowedMaskConfidence(_:));
/** Writes only the result descriptor. Any referenced arrays belong to the session and may be overwritten by later
 * processing; serialize access. */
// C API: HFGetFaceQualityConfidence
- (BOOL)getBorrowedQualityConfidence:(HFFaceQualityConfidence *)result
                               error:(NSError *_Nullable *_Nullable)error
    NS_SWIFT_NAME(getBorrowedQualityConfidence(_:));
/** Writes only the result descriptor. Any referenced arrays belong to the session and may be overwritten by later
 * processing; serialize access. */
// C API: HFGetFaceInteractionStateResult
- (BOOL)getBorrowedInteractionState:(HFFaceInteractionState *)result
                              error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getBorrowedInteractionState(_:));
/** Writes only the result descriptor. Any referenced arrays belong to the session and may be overwritten by later
 * processing; serialize access. */
// C API: HFGetFaceInteractionActionsResult
- (BOOL)getBorrowedInteractionActions:(HFFaceInteractionsActions *)result
                                error:(NSError *_Nullable *_Nullable)error
    NS_SWIFT_NAME(getBorrowedInteractionActions(_:));
/** Writes only the result descriptor. Any referenced arrays belong to the session and may be overwritten by later
 * processing; serialize access. */
// C API: HFGetFaceAttributeResult
- (BOOL)getBorrowedAttributes:(HFFaceAttributeResult *)result
                        error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getBorrowedAttributes(_:));
/** Writes only the result descriptor. Any referenced arrays belong to the session and may be overwritten by later
 * processing; serialize access. */
// C API: HFGetFaceEmotionResult
- (BOOL)getBorrowedEmotions:(HFFaceEmotionResult *)result
                      error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getBorrowedEmotions(_:));
// C API: HFSessionClearTrackingFace
- (BOOL)clearTrackingWithError:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(clearTracking());
// C API: HFSessionPrintTrackCostSpend
- (BOOL)printTrackingTimingWithError:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(printTrackingTiming());
/** No result payload copy. Faces/tokens expire on the next tracking/reset/release operation. Do not mutate the session
 * while consuming borrowed faces, including through the C API. */
// C API: HFExecuteFaceTrack
- (BOOL)trackStream:(IFImageStream *)stream
     borrowedResult:(HFMultipleFaceData *)result
              error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(track(_:borrowedResult:));
/** Explicitly creates an independent, owned snapshot using the C snapshot API. */
// C API: HFExecuteFaceTrackSnapshot
- (nullable IFFaceSnapshot *)snapshotFromStream:(IFImageStream *)stream
                                          error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(snapshot(from:));
/** Borrows the session feature cache. Later feature extraction/release invalidates the view; no payload copy is added.
 */
// C API: HFFaceFeatureExtract
- (BOOL)extractFeatureFromStream:(IFImageStream *)stream
                           token:(HFFaceBasicToken)token
                  borrowedResult:(HFFaceFeature *)feature
                           error:(NSError *_Nullable *_Nullable)error
    NS_SWIFT_NAME(extractFeature(from:token:borrowedResult:));
/** Writes directly to caller-provided storage. Capacity must match the C feature length contract. */
// C API: HFFaceFeatureExtractTo
- (BOOL)extractFeatureFromStream:(IFImageStream *)stream
                           token:(HFFaceBasicToken)token
                            into:(HFFaceFeature)feature
                           error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(extractFeature(from:token:into:));
// C API: HFFaceFeatureExtractCpy
- (BOOL)copyFeatureFromStream:(IFImageStream *)stream
                        token:(HFFaceBasicToken)token
                         into:(float *)buffer
                     capacity:(HInt32)capacity
                        error:(NSError *_Nullable *_Nullable)error
    NS_SWIFT_NAME(copyFeature(from:token:into:capacity:));
// C API: HFFaceGetFaceAlignmentImage
- (nullable IFImageBitmap *)alignmentBitmapFromStream:(IFImageStream *)stream
                                                token:(HFFaceBasicToken)token
                                                error:(NSError *_Nullable *_Nullable)error
    NS_SWIFT_NAME(alignmentBitmap(from:token:));
// C API: HFFaceFeatureExtractWithAlignmentImage
- (BOOL)extractAlignedFeatureFromStream:(IFImageStream *)stream
                                   into:(HFFaceFeature)feature
                                  error:(NSError *_Nullable *_Nullable)error
    NS_SWIFT_NAME(extractAlignedFeature(from:into:));
// C API: HFMultipleFacePipelineProcess
- (BOOL)processStream:(IFImageStream *)stream
                faces:(HFMultipleFaceData *)faces
           parameters:(HFSessionCustomParameter)parameters
                error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(process(_:faces:parameters:));
// C API: HFMultipleFacePipelineProcessOptional
- (BOOL)processStream:(IFImageStream *)stream
                faces:(HFMultipleFaceData *)faces
              options:(HInt32)options
                error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(process(_:faces:options:));
// C API: HFFaceQualityDetect
- (BOOL)getQualityForToken:(HFFaceBasicToken)token
                    result:(float *)result
                     error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getQuality(for:result:));
/** Scoped borrowed descriptors. Reentrant tracking/reset/close through this adapter fails.
 Calls through the original C API bypass this guard and remain the caller's responsibility. */
- (BOOL)withBorrowedFacesFromStream:(IFImageStream *)stream
                               body:(void(NS_NOESCAPE ^)(HFMultipleFaceData faces))body
                              error:(NSError *_Nullable *_Nullable)error;
- (BOOL)withBorrowedFeatureFromStream:(IFImageStream *)stream
                                token:(HFFaceBasicToken)token
                                 body:(void(NS_NOESCAPE ^)(HFFaceFeature feature))body
                                error:(NSError *_Nullable *_Nullable)error;
/** Low-level scope pairing used by the Swift overlay to avoid a heap box for throwing generic callbacks.
 Each successful begin MUST have one matching end before another invalidating operation. */
- (BOOL)beginBorrowingFacesFromStream:(IFImageStream *)stream
                               result:(HFMultipleFaceData *)result
                                error:(NSError *_Nullable *_Nullable)error
    NS_SWIFT_NAME(beginBorrowingFaces(from:result:));
- (void)endBorrowingFaces;
- (BOOL)beginBorrowingFeatureFromStream:(IFImageStream *)stream
                                  token:(HFFaceBasicToken)token
                                 result:(HFFaceFeature *)result
                                  error:(NSError *_Nullable *_Nullable)error
    NS_SWIFT_NAME(beginBorrowingFeature(from:token:result:));
- (void)endBorrowingFeature;
@end

NS_SWIFT_NAME(FaceSnapshot)
__attribute__((objc_subclassing_restricted))
IF_APPLE_EXPORT @interface IFFaceSnapshot : NSObject
- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;
/** Borrowed C handle. Use close for release; never release this handle directly through C. */
@property(nonatomic, readonly, nullable) HFFaceResultSnapshot nativeHandle;
/** Releases once; further calls report the C invalid-handle error. All borrowed views expire. */
// C API: HFReleaseFaceResultSnapshot
- (BOOL)closeWithError:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(close());
/** The descriptor borrows immutable snapshot storage, valid until close. Subsequent session tracking does not
 * invalidate it. */
// C API: HFGetFaceResultSnapshotData
- (BOOL)getBorrowedFaces:(HFMultipleFaceData *)result
                   error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getBorrowedFaces(_:) );
@end

NS_SWIFT_NAME(FaceCaptureSession)
__attribute__((objc_subclassing_restricted))
IF_APPLE_EXPORT @interface IFCaptureSession : NSObject
- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;
/** Borrowed C handle. Use close for release; never release this handle directly through C. */
@property(nonatomic, readonly, nullable) HFFaceCaptureSession nativeHandle;
/** Releases once; further calls report the C invalid-handle error. All borrowed views expire. */
// C API: HFReleaseFaceCaptureSession
- (BOOL)closeWithError:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(close());
// C API: HFGetDefaultFaceCaptureConfig
+ (BOOL)getDefaultConfiguration:(HFFaceCaptureConfig *)configuration
                          error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getDefaultConfiguration(_:) );
// C API: HFCreateFaceCaptureSession
- (nullable instancetype)initWithSession:(IFSession *)session
                           configuration:(HFFaceCaptureConfig)configuration
                                   error:(NSError *_Nullable *_Nullable)error;
// C API: HFUpdateFaceCaptureSession
- (BOOL)updateStream:(IFImageStream *)stream
                  frameID:(uint64_t)frameID
    timestampMilliseconds:(uint64_t)timestamp
                 progress:(HFFaceCaptureProgress *)progress
                    error:(NSError *_Nullable *_Nullable)error
    NS_SWIFT_NAME(update(_:frameID:timestampMilliseconds:progress:));
// C API: HFUpdateFaceCaptureSessionWithSnapshot
- (BOOL)updateStream:(IFImageStream *)stream
                 snapshot:(IFFaceSnapshot *)snapshot
                  frameID:(uint64_t)frameID
    timestampMilliseconds:(uint64_t)timestamp
                 progress:(HFFaceCaptureProgress *)progress
                    error:(NSError *_Nullable *_Nullable)error
    NS_SWIFT_NAME(update(_:snapshot:frameID:timestampMilliseconds:progress:));
/** Copies descriptors into caller storage, not token payloads. Tokens expire on next update/reset/finish/close. A nil
 * buffer and zero capacity query count. */
// C API: HFGetFaceCaptureResults
- (BOOL)getResults:(HFFaceCaptureResult *_Nullable)results
          capacity:(uint32_t)capacity
             count:(uint32_t *)count
             error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getResults(_:capacity:count:));
// C API: HFFinishFaceCaptureSession
- (BOOL)finishWithProgress:(HFFaceCaptureProgress *)progress
                     error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(finish(progress:));
// C API: HFResetFaceCaptureSession
- (BOOL)resetWithError:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(reset());
@end

NS_SWIFT_NAME(FaceFeatureBuffer)
__attribute__((objc_subclassing_restricted))
IF_APPLE_EXPORT @interface IFFeatureBuffer : NSObject
- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;
/** A descriptor borrowing this owned allocation. Writes are visible through every view. Invalid after close. */
@property(nonatomic, readonly) HFFaceFeature borrowedFeature;
// C API: HFCreateFaceFeature
- (nullable instancetype)initWithError:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(init());
// C API: HFReleaseFaceFeature
- (BOOL)closeWithError:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(close());
// C API: HFGetFeatureLength
+ (BOOL)getLength:(HInt32 *)length error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getLength(_:) );
// C API: HFFaceComparison
+ (BOOL)compare:(HFFaceFeature)first
           with:(HFFaceFeature)second
     similarity:(float *)similarity
          error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(compare(_:with:similarity:));
// C API: HFGetRecommendedCosineThreshold
+ (BOOL)getRecommendedThreshold:(float *)threshold
                          error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getRecommendedThreshold(_:) );
// C API: HFCosineSimilarityConvertToPercentage
+ (BOOL)convertSimilarity:(float)similarity
               percentage:(float *)percentage
                    error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(convert(similarity:percentage:));
// C API: HFUpdateCosineSimilarityConverter
+ (BOOL)setSimilarityConverter:(HFSimilarityConverterConfig)configuration
                         error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(setSimilarityConverter(_:) );
// C API: HFGetCosineSimilarityConverter
+ (BOOL)getSimilarityConverter:(HFSimilarityConverterConfig *)configuration
                         error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getSimilarityConverter(_:) );
@end

NS_SWIFT_NAME(FeatureHub)
__attribute__((objc_subclassing_restricted))
IF_APPLE_EXPORT @interface IFFeatureHub : NSObject
- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;
/** Process-wide database; borrowed configuration strings are consumed synchronously. */
// C API: HFFeatureHubDataEnable
+ (BOOL)enableWithConfiguration:(HFFeatureHubConfiguration)configuration
                          error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(enable(configuration:));
// C API: HFFeatureHubDataDisable
+ (BOOL)disableWithError:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(disable());
// C API: HFFeatureHubFaceSearchThresholdSetting
+ (BOOL)setSearchThreshold:(float)threshold
                     error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(setSearchThreshold(_:));
// C API: HFFeatureHubInsertFeature
+ (BOOL)insert:(HFFaceFeatureIdentity)identity
    allocatedID:(HFaceId *)allocatedID
          error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(insert(_:allocatedID:));
/** Borrowed result; consume before the next search on this thread. */
// C API: HFFeatureHubFaceSearch
+ (BOOL)search:(HFFaceFeature)feature
          confidence:(float *)confidence
    borrowedIdentity:(HFFaceFeatureIdentity *)identity
               error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(search(_:confidence:borrowedIdentity:));
/** Borrowed feature valid until the next single search on the same thread. */
// C API: HFFeatureHubFaceSearchV2
+ (BOOL)search:(HFFaceFeature)feature
    borrowedResult:(HFFeatureHubSearchResultV2 *)result
             error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(search(_:borrowedResult:));
/** Borrowed arrays valid until the next top-K search on this thread. */
// C API: HFFeatureHubFaceSearchTopK
+ (BOOL)search:(HFFaceFeature)feature
               topK:(HInt32)topK
    borrowedResults:(HFSearchTopKResults *)results
              error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(search(_:topK:borrowedResults:));
// C API: HFFeatureHubFaceRemove
+ (BOOL)removeIdentityWithID:(HFaceId)identityID
                       error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(removeIdentity(id:));
// C API: HFFeatureHubFaceUpdate
+ (BOOL)updateIdentity:(HFFaceFeatureIdentity)identity
                 error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(updateIdentity(_:));
/** Borrowed result; consume before another identity lookup on this thread. */
// C API: HFFeatureHubGetFaceIdentity
+ (BOOL)getIdentityWithID:(HFaceId)identityID
           borrowedResult:(HFFaceFeatureIdentity *)identity
                    error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getIdentity(id:borrowedResult:));
// C API: HFFeatureHubGetFaceCount
+ (BOOL)getIdentityCount:(HInt32 *)count error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getIdentityCount(_:));
// C API: HFFeatureHubViewDBTable
+ (BOOL)printTableWithError:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(printTable());
/** Borrowed IDs; consume before the next IDs query on this thread. */
// C API: HFFeatureHubGetExistingIds
+ (BOOL)getBorrowedExistingIDs:(HFFeatureHubExistingIds *)ids
                         error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getBorrowedExistingIDs(_:));
@end

NS_SWIFT_NAME(FaceTokenUtilities)
__attribute__((objc_subclassing_restricted))
IF_APPLE_EXPORT @interface IFFaceToken : NSObject
- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;
// C API: HFCopyFaceBasicToken
+ (BOOL)copyToken:(HFFaceBasicToken)token
             into:(char *)buffer
         capacity:(HInt32)capacity
            error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(copy(_:into:capacity:));
// C API: HFGetFaceBasicTokenSize
+ (BOOL)getTokenSize:(HInt32 *)count error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getTokenSize(_:) );
// C API: HFGetNumOfFaceDenseLandmark
+ (BOOL)getDenseLandmarkCount:(HInt32 *)count
                        error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getDenseLandmarkCount(_:) );
// C API: HFGetFaceDenseLandmarkFromFaceToken
+ (BOOL)getDenseLandmarks:(HFFaceBasicToken)token
                     into:(HPoint2f *)points
                 capacity:(HInt32)capacity
                    error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getDenseLandmarks(_:into:capacity:));
// C API: HFGetFaceFiveKeyPointsFromFaceToken
+ (BOOL)getFiveKeyPoints:(HFFaceBasicToken)token
                    into:(HPoint2f *)points
                capacity:(HInt32)capacity
                   error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getFiveKeyPoints(_:into:capacity:));
@end

NS_SWIFT_NAME(InspireFaceDiagnostics)
__attribute__((objc_subclassing_restricted))
IF_APPLE_EXPORT @interface IFDiagnostics : NSObject
- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;
// C API: HFQueryInspireFaceVersion
+ (BOOL)getVersion:(HFInspireFaceVersion *)value
             error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getVersion(_:) );
// C API: HFQueryCAPILevel
+ (BOOL)getCAPILevel:(HFUInt32 *)value error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getCAPILevel(_:) );
// C API: HFQueryInspireFaceExtendedInformation
+ (BOOL)getExtendedInformation:(HFInspireFaceExtendedInformation *)value
                         error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getExtendedInformation(_:) );
// C API: HFDeBugGetUnreleasedSessionsCount
+ (BOOL)getLiveSessionCount:(HInt32 *)value
                      error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getLiveSessionCount(_:) );
// C API: HFDeBugGetUnreleasedStreamsCount
+ (BOOL)getLiveStreamCount:(HInt32 *)value
                     error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getLiveStreamCount(_:) );
// C API: HFQueryInspireFaceComponentVersion
+ (BOOL)getComponentVersion:(HFComponentType)component
                     result:(HFComponentVersion *)result
                      error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getComponentVersion(_:result:));
/** No temporary string allocation. A nil buffer and zero capacity query the required size, including the terminator. */
// C API: HFQueryInspireFaceComponentVersions
+ (BOOL)getComponentVersions:(char *_Nullable)buffer
                    capacity:(HInt32)capacity
                requiredSize:(HInt32 *_Nullable)requiredSize
                       error:(NSError *_Nullable *_Nullable)error
    NS_SWIFT_NAME(getComponentVersions(_:capacity:requiredSize:));
/** No temporary string allocation. A nil buffer and zero capacity query the required size, including the terminator. */
// C API: HFQueryInspireFaceDiagnosticInformation
+ (BOOL)getDiagnosticInformation:(char *_Nullable)buffer
                        capacity:(HInt32)capacity
                    requiredSize:(HInt32 *_Nullable)requiredSize
                           error:(NSError *_Nullable *_Nullable)error
    NS_SWIFT_NAME(getDiagnosticInformation(_:capacity:requiredSize:));
// C API: HFGetErrorMessage
+ (BOOL)getErrorMessage:(HResult)code
                   into:(char *_Nullable)buffer
               capacity:(HInt32)capacity
           requiredSize:(HInt32 *_Nullable)requiredSize
                  error:(NSError *_Nullable *_Nullable)error
    NS_SWIFT_NAME(getErrorMessage(_:into:capacity:requiredSize:));
// C API: HFSetLogLevel
+ (BOOL)setLogLevel:(HFLogLevel)level error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(setLogLevel(_:) );
// C API: HFLogDisable
+ (BOOL)disableLoggingWithError:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(disableLogging());
// C API: HFDeBugShowResourceStatistics
+ (BOOL)printResourceStatisticsWithError:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(printResourceStatistics());
/** A literal message, never a printf format string. Swift interpolation and NSString formatting can be performed
 * explicitly by the caller. Preserves C filtering and truncation. */
// C API: HFLogPrint
+ (BOOL)logMessage:(NSString *)message
             level:(HFLogLevel)level
             error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(log(_:level:));
/** Writes borrowed handles into caller storage. Does not transfer ownership or create resource objects. */
// C API: HFDeBugGetUnreleasedSessions
+ (BOOL)getBorrowedLiveSessions:(HFSession _Nullable *_Nonnull)buffer
                       capacity:(HInt32)capacity
                          error:(NSError *_Nullable *_Nullable)error
    NS_SWIFT_NAME(getBorrowedLiveSessions(_:capacity:));
/** Writes borrowed handles into caller storage. Does not transfer ownership or create resource objects. */
// C API: HFDeBugGetUnreleasedStreams
+ (BOOL)getBorrowedLiveStreams:(HFImageStream _Nullable *_Nonnull)buffer
                      capacity:(HInt32)capacity
                         error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(getBorrowedLiveStreams(_:capacity:));
@end

NS_ASSUME_NONNULL_END
