import Foundation
@_exported import InspireFace

// Native spellings retain the exact C representation: passing descriptors never boxes
// an element or copies the buffers they refer to.
public typealias PixelFormat = HFImageFormat
public typealias ImageRotation = HFRotation
public typealias DetectionMode = HFDetectMode
public typealias CoreMLInferenceMode = HFAppleCoreMLInferenceMode
public typealias LandmarkEngine = HFSessionLandmarkEngine
public typealias LogLevel = HFLogLevel
public typealias FaceToken = HFFaceBasicToken
public typealias FaceFeatureView = HFFaceFeature
public typealias CaptureConfiguration = HFFaceCaptureConfig
public typealias CaptureProgress = HFFaceCaptureProgress
public typealias CaptureResult = HFFaceCaptureResult
public typealias FeatureHubConfiguration = HFFeatureHubConfiguration
public typealias FeatureSearchResult = HFFeatureHubSearchResultV2
extension HFImageFormat {
  public static let rgb = HF_STREAM_RGB
  public static let bgr = HF_STREAM_BGR
  public static let rgba = HF_STREAM_RGBA
  public static let bgra = HF_STREAM_BGRA
  public static let nv12 = HF_STREAM_YUV_NV12
  public static let nv21 = HF_STREAM_YUV_NV21
  public static let i420 = HF_STREAM_I420
  public static let gray = HF_STREAM_GRAY
}
extension HFRotation {
  public static let degrees0 = HF_CAMERA_ROTATION_0
  public static let degrees90 = HF_CAMERA_ROTATION_90
  public static let degrees180 = HF_CAMERA_ROTATION_180
  public static let degrees270 = HF_CAMERA_ROTATION_270
}
extension HFDetectMode {
  public static let alwaysDetect = HF_DETECT_MODE_ALWAYS_DETECT
  public static let lightTracking = HF_DETECT_MODE_LIGHT_TRACK
  public static let trackingByDetection = HF_DETECT_MODE_TRACK_BY_DETECTION
}
extension HFAppleCoreMLInferenceMode {
  public static let cpu = HF_APPLE_COREML_INFERENCE_MODE_CPU
  public static let gpu = HF_APPLE_COREML_INFERENCE_MODE_GPU
  public static let neuralEngine = HF_APPLE_COREML_INFERENCE_MODE_ANE
}
extension HFSessionLandmarkEngine {
  public static let hyperLandmark025 = HF_LANDMARK_HYPLMV2_0_25
  public static let hyperLandmark050 = HF_LANDMARK_HYPLMV2_0_50
  public static let insightFace106 = HF_LANDMARK_INSIGHTFACE_2D106_TRACK
}
extension HFLogLevel {
  public static let none = HF_LOG_NONE
  public static let debug = HF_LOG_DEBUG
  public static let info = HF_LOG_INFO
  public static let warning = HF_LOG_WARN
  public static let error = HF_LOG_ERROR
  public static let fatal = HF_LOG_FATAL
}

/// Options map exactly to the public C API feature bits, including future combinations.
public struct FaceFeatures: OptionSet {
  public let rawValue: UInt64
  public init(rawValue: UInt64) { self.rawValue = rawValue }
  public static let recognition = Self(rawValue: UInt64(HF_ENABLE_FACE_RECOGNITION))
  public static let rgbLiveness = Self(rawValue: UInt64(HF_ENABLE_LIVENESS))
  public static let irLiveness = Self(rawValue: UInt64(HF_ENABLE_IR_LIVENESS))
  public static let mask = Self(rawValue: UInt64(HF_ENABLE_MASK_DETECT))
  public static let attributes = Self(rawValue: UInt64(HF_ENABLE_FACE_ATTRIBUTE))
  public static let quality = Self(rawValue: UInt64(HF_ENABLE_QUALITY))
  public static let interaction = Self(rawValue: UInt64(HF_ENABLE_INTERACTION))
  public static let pose = Self(rawValue: UInt64(HF_ENABLE_FACE_POSE))
  public static let emotion = Self(rawValue: UInt64(HF_ENABLE_FACE_EMOTION))
}

public struct SessionConfiguration {
  public var features: FaceFeatures
  public var detectionMode: HFDetectMode
  public var maximumFaces: Int32
  public var pixelLevel: Int32
  public var framesPerSecond: Int32
  public init(
    features: FaceFeatures = [], detectionMode: HFDetectMode = HF_DETECT_MODE_ALWAYS_DETECT,
    maximumFaces: Int32 = 1, pixelLevel: Int32 = -1, framesPerSecond: Int32 = -1
  ) {
    self.features = features
    self.detectionMode = detectionMode
    self.maximumFaces = maximumFaces
    self.pixelLevel = pixelLevel
    self.framesPerSecond = framesPerSecond
  }
  public var cValue: HFSessionConfigV2 {
    var value = HFSessionConfigV2()
    value.structSize = UInt32(MemoryLayout<HFSessionConfigV2>.size)
    value.structVersion = UInt32(HF_SESSION_CONFIG_V2_VERSION)
    value.featureMask = features.rawValue
    value.detectMode = Int32(detectionMode.rawValue)
    value.maxDetectFaceNum = maximumFaces
    value.detectPixelLevel = pixelLevel
    value.trackByDetectModeFPS = framesPerSecond
    return value
  }
}

/// A descriptor-only view. Never store it beyond the owner's documented validity interval.
/// Direct C calls, close, and later tracking can invalidate its pointers even if the owner lives.
public struct BorrowedFaces {
  public let cValue: HFMultipleFaceData
  public var count: Int { Int(cValue.detectedNum) }
  public var rectangles: UnsafeBufferPointer<HFaceRect> {
    UnsafeBufferPointer(start: cValue.rects, count: count)
  }
  public var trackIDs: UnsafeBufferPointer<Int32> {
    UnsafeBufferPointer(start: cValue.trackIds, count: count)
  }
  public var trackCounts: UnsafeBufferPointer<Int32> {
    UnsafeBufferPointer(start: cValue.trackCounts, count: count)
  }
  public var confidences: UnsafeBufferPointer<Float> {
    UnsafeBufferPointer(start: cValue.detConfidence, count: count)
  }
  public var tokens: UnsafeBufferPointer<HFFaceBasicToken> {
    UnsafeBufferPointer(start: cValue.tokens, count: count)
  }
  public var roll: UnsafeBufferPointer<Float> {
    UnsafeBufferPointer(start: cValue.angles.roll, count: count)
  }
  public var yaw: UnsafeBufferPointer<Float> {
    UnsafeBufferPointer(start: cValue.angles.yaw, count: count)
  }
  public var pitch: UnsafeBufferPointer<Float> {
    UnsafeBufferPointer(start: cValue.angles.pitch, count: count)
  }
}

private func checkedCount(_ count: Int) throws -> Int32 {
  guard let result = Int32(exactly: count) else {
    throw NSError(domain: IFErrorDomain, code: Int(HERR_INVALID_PARAM))
  }
  return result
}

/// Validates tightly packed image storage without allocating. Strided/planar Apple images
/// must be checked or explicitly converted before using this buffer entry point.
private func requiredImageBytes(width: Int32, height: Int32, format: HFImageFormat) throws -> Int {
  guard width > 0, height > 0 else {
    throw NSError(domain: IFErrorDomain, code: Int(HERR_INVALID_IMAGE_STREAM_PARAM))
  }
  let (pixels, overflow) = Int(width).multipliedReportingOverflow(by: Int(height))
  var channels = 0
  switch format {
  case HF_STREAM_RGB, HF_STREAM_BGR: channels = 3
  case HF_STREAM_RGBA, HF_STREAM_BGRA: channels = 4
  case HF_STREAM_GRAY: channels = 1
  case HF_STREAM_YUV_NV12, HF_STREAM_YUV_NV21, HF_STREAM_I420:
    guard width % 2 == 0, height % 2 == 0, !overflow, pixels <= Int.max / 3 else {
      throw NSError(domain: IFErrorDomain, code: Int(HERR_INVALID_IMAGE_STREAM_PARAM))
    }
    let bytes = pixels / 2 * 3
    guard bytes <= Int(Int32.max) else {
      throw NSError(domain: IFErrorDomain, code: Int(HERR_INVALID_IMAGE_STREAM_PARAM))
    }
    return bytes
  default: throw NSError(domain: IFErrorDomain, code: Int(HERR_INVALID_IMAGE_STREAM_PARAM))
  }
  let (bytes, byteOverflow) = pixels.multipliedReportingOverflow(by: channels)
  guard !overflow && !byteOverflow && bytes <= Int(Int32.max) else {
    throw NSError(domain: IFErrorDomain, code: Int(HERR_INVALID_IMAGE_STREAM_PARAM))
  }
  return bytes
}

extension ImageStream {
  /// Creates only a stream handle. Pixels are borrowed during body and are never copied.
  /// Do not return the stream or its native handle, or mutate its source storage in body.
  public static func withBorrowedBytes<R>(
    _ bytes: UnsafeMutableRawBufferPointer, width: Int32, height: Int32,
    format: HFImageFormat, rotation: HFRotation = HF_CAMERA_ROTATION_0,
    _ body: (ImageStream) throws -> R
  ) throws -> R {
    let count = try requiredImageBytes(width: width, height: height, format: format)
    guard bytes.count >= count, let base = bytes.baseAddress else {
      throw NSError(domain: IFErrorDomain, code: Int(HERR_INVALID_IMAGE_STREAM_PARAM))
    }
    let data = HFImageData(
      data: base.assumingMemoryBound(to: UInt8.self), width: width, height: height,
      format: format, rotation: rotation)
    let stream = try ImageStream(borrowing: data)
    defer { try? stream.close() }
    return try body(stream)
  }

  /// Reuses this stream's C handle; no adapter buffer or object allocation.
  public func setBorrowedBytes(
    _ bytes: UnsafeMutableRawBufferPointer, width: Int32, height: Int32,
    format: HFImageFormat
  ) throws {
    let count = try requiredImageBytes(width: width, height: height, format: format)
    guard bytes.count >= count, let base = bytes.baseAddress else {
      throw NSError(domain: IFErrorDomain, code: Int(HERR_INVALID_IMAGE_STREAM_PARAM))
    }
    try setFormat(format)
    try setBorrowedBuffer(base.assumingMemoryBound(to: UInt8.self), width: width, height: height)
  }
}

extension FaceSession {
  public convenience init(configuration: SessionConfiguration) throws {
    try self.init(configuration: configuration.cValue)
  }
  /// No array construction. Do not track/reset/close this session again inside body or
  /// let the view escape. Extraction and pipeline methods may consume the current tokens.
  public func withUnsafeFaces<R>(in stream: ImageStream, _ body: (BorrowedFaces) throws -> R) throws
    -> R
  {
    var result = HFMultipleFaceData()
    try beginBorrowingFaces(from: stream, result: &result)
    defer { endBorrowingFaces() }
    return try withExtendedLifetime(self) { try body(BorrowedFaces(cValue: result)) }
  }
  /// Borrows the feature cache. Do not extract another feature or close in body.
  public func withUnsafeFeature<R>(
    in stream: ImageStream, token: HFFaceBasicToken,
    _ body: (UnsafeBufferPointer<Float>) throws -> R
  ) throws -> R {
    var result = HFFaceFeature()
    try beginBorrowingFeature(from: stream, token: token, result: &result)
    defer { endBorrowingFeature() }
    return try withExtendedLifetime(self) {
      try body(UnsafeBufferPointer(start: result.data, count: Int(result.size)))
    }
  }
  /// Caller storage is reused; the C implementation writes into it directly.
  public func extractFeature(
    from stream: ImageStream, token: HFFaceBasicToken,
    into buffer: UnsafeMutableBufferPointer<Float>
  ) throws {
    try extractFeature(
      from: stream, token: token,
      into: HFFaceFeature(size: try checkedCount(buffer.count), data: buffer.baseAddress))
  }
  public func copyFeature(
    from stream: ImageStream, token: HFFaceBasicToken,
    into buffer: UnsafeMutableBufferPointer<Float>
  ) throws {
    guard let base = buffer.baseAddress else {
      throw NSError(domain: IFErrorDomain, code: Int(HERR_INVALID_FACE_FEATURE))
    }
    try copyFeature(from: stream, token: token, into: base, capacity: checkedCount(buffer.count))
  }
}

extension FaceSnapshot {
  /// The snapshot owns its storage independently of the session; no payload copy here.
  /// Do not close it in body or let borrowed pointers outlive it.
  public func withUnsafeFaces<R>(_ body: (BorrowedFaces) throws -> R) throws -> R {
    var result = HFMultipleFaceData()
    try getBorrowedFaces(&result)
    return try withExtendedLifetime(self) { try body(BorrowedFaces(cValue: result)) }
  }
}

extension ImageBitmap {
  /// Pixels are mutable and borrowed, valid until close. No Data/Array bridge is created.
  public func withUnsafeMutablePixels<R>(
    _ body: (UnsafeMutableRawBufferPointer, HFImageBitmapData) throws -> R
  ) throws -> R {
    var result = HFImageBitmapData()
    try getBorrowedData(&result)
    let count = Int(result.width) * Int(result.height) * Int(result.channels)
    return try withExtendedLifetime(self) {
      try body(UnsafeMutableRawBufferPointer(start: result.data, count: count), result)
    }
  }
}

extension FaceFeatureBuffer {
  public func withUnsafeMutableBufferPointer<R>(
    _ body: (UnsafeMutableBufferPointer<Float>) throws -> R
  ) rethrows -> R {
    let feature = borrowedFeature
    return try withExtendedLifetime(self) {
      try body(UnsafeMutableBufferPointer(start: feature.data, count: Int(feature.size)))
    }
  }
  public static func similarity(
    _ first: UnsafeBufferPointer<Float>, _ second: UnsafeBufferPointer<Float>
  ) throws -> Float {
    var result: Float = 0
    let a = HFFaceFeature(
      size: try checkedCount(first.count), data: UnsafeMutablePointer(mutating: first.baseAddress))
    let b = HFFaceFeature(
      size: try checkedCount(second.count), data: UnsafeMutablePointer(mutating: second.baseAddress)
    )
    try compare(a, with: b, similarity: &result)
    return result
  }
}

extension FaceCaptureSession {
  public static func defaultConfiguration() throws -> HFFaceCaptureConfig {
    var configuration = HFFaceCaptureConfig()
    try getDefaultConfiguration(&configuration)
    return configuration
  }
  /// Copies result descriptors, while token payloads remain borrowed until update/reset/finish/close.
  public func results(into buffer: UnsafeMutableBufferPointer<HFFaceCaptureResult>) throws -> Int {
    guard let capacity = UInt32(exactly: buffer.count) else {
      throw NSError(domain: IFErrorDomain, code: Int(HERR_INVALID_PARAM))
    }
    var count: UInt32 = 0
    try getResults(buffer.baseAddress, capacity: capacity, count: &count)
    return Int(count)
  }
}

extension FaceTokenUtilities {
  public static func copy(_ token: HFFaceBasicToken, into bytes: UnsafeMutableRawBufferPointer)
    throws
  {
    guard let base = bytes.baseAddress else {
      throw NSError(domain: IFErrorDomain, code: Int(HERR_INVALID_PARAM))
    }
    try copy(
      token, into: base.assumingMemoryBound(to: CChar.self), capacity: checkedCount(bytes.count))
  }
  public static func getDenseLandmarks(
    _ token: HFFaceBasicToken, into points: UnsafeMutableBufferPointer<HPoint2f>
  ) throws {
    guard let base = points.baseAddress else {
      throw NSError(domain: IFErrorDomain, code: Int(HERR_INVALID_PARAM))
    }
    try getDenseLandmarks(token, into: base, capacity: checkedCount(points.count))
  }
  public static func getFiveKeyPoints(
    _ token: HFFaceBasicToken, into points: UnsafeMutableBufferPointer<HPoint2f>
  ) throws {
    guard let base = points.baseAddress else {
      throw NSError(domain: IFErrorDomain, code: Int(HERR_INVALID_PARAM))
    }
    try getFiveKeyPoints(token, into: base, capacity: checkedCount(points.count))
  }
}
