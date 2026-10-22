import CoreVideo
import Foundation
import InspireFaceSwift

@_silgen_name("IFTestAllocationObservationSupported") func allocationObservationSupported() -> Int32
@_silgen_name("IFTestConsumeAllocation") func consumeAllocation(_ memory: UnsafeMutableRawPointer?)
@_silgen_name("IFTestBeginAllocations") func beginAllocations()
@_silgen_name("IFTestEndAllocations") func endAllocations() -> UInt64

var checks = 0
func expect(
  _ condition: @autoclosure () -> Bool, _ message: String = "contract", file: StaticString = #file,
  line: UInt = #line
) {
  checks += 1
  if !condition() { fatalError("\(file):\(line): \(message)") }
}
func expectError(_ code: Int, _ body: () throws -> Void) {
  do {
    try body()
    fatalError("expected error \(code)")
  } catch {
    let error = error as NSError
    expect(error.domain == IFErrorDomain)
    expect(error.code == code, "error \(error.code), expected \(code)")
  }
}

func metadata() throws {
  var version = HFInspireFaceVersion()
  var baseline = HFInspireFaceVersion()
  try InspireFaceDiagnostics.getVersion(&version)
  expect(HFQueryInspireFaceVersion(&baseline) == HSUCCEED)
  expect(
    version.major == baseline.major && version.minor == baseline.minor
      && version.patch == baseline.patch)
  var level: UInt32 = 0
  try InspireFaceDiagnostics.getCAPILevel(&level)
  expect(level == HF_C_API_LEVEL)
  var extended = HFInspireFaceExtendedInformation()
  try InspireFaceDiagnostics.getExtendedInformation(&extended)
  for index in 0..<HF_COMPONENT_COUNT.rawValue {
    var component = HFComponentVersion()
    try InspireFaceDiagnostics.getComponentVersion(
      HFComponentType(rawValue: index), result: &component)
  }
  var coreml = HFComponentVersion()
  try InspireFaceDiagnostics.getComponentVersion(HF_COMPONENT_COREML, result: &coreml)
  #if IF_TEST_COREML
    expect(coreml.state != HF_COMPONENT_VERSION_DISABLED)
  #else
    expect(coreml.state == HF_COMPONENT_VERSION_DISABLED)
  #endif
  var size: Int32 = 0
  try InspireFaceDiagnostics.getComponentVersions(nil, capacity: 0, requiredSize: &size)
  expect(size > 0)
  let buffer = UnsafeMutablePointer<CChar>.allocate(capacity: 8192)
  defer { buffer.deallocate() }
  try InspireFaceDiagnostics.getComponentVersions(buffer, capacity: 8192, requiredSize: &size)
  try InspireFaceDiagnostics.getDiagnosticInformation(buffer, capacity: 8192, requiredSize: &size)
  try InspireFaceDiagnostics.getErrorMessage(
    Int(HERR_INVALID_PARAM), into: buffer, capacity: 8192, requiredSize: &size)
  expect(size > 1)
  expectError(Int(HERR_INVALID_BUFFER_SIZE)) {
    try InspireFaceDiagnostics.getErrorMessage(
      Int(HERR_INVALID_PARAM), into: buffer, capacity: 1, requiredSize: &size)
  }
  try InspireFaceDiagnostics.setLogLevel(HF_LOG_NONE)
  try InspireFaceDiagnostics.log("literal %s %n %@", level: HF_LOG_INFO)
  try InspireFaceDiagnostics.disableLogging()
  try InspireFaceDiagnostics.printResourceStatistics()
  try InspireFaceRuntime.setImageProcessingBackend(HF_IMAGE_PROCESSING_CPU)
  try InspireFaceRuntime.setImageProcessingAlignedWidth(16)
  expectError(Int(HERR_INVALID_PARAM)) { try InspireFaceRuntime.setImageProcessingAlignedWidth(0) }
  try InspireFaceRuntime.setCoreMLInferenceMode(.cpu)
  expectError(Int(HERR_INVALID_PARAM)) {
    try InspireFaceRuntime.setCoreMLInferenceMode(HFAppleCoreMLInferenceMode(rawValue: 99))
  }
  var levels = HFFaceDetectPixelList()
  try InspireFaceRuntime.getSupportedDetectionPixelLevels(&levels)
}

func images(_ output: String) throws {
  var before: Int32 = 0
  try InspireFaceDiagnostics.getLiveStreamCount(&before)
  try autoreleasepool {
    let pixels = UnsafeMutablePointer<UInt8>.allocate(capacity: 256)
    pixels.initialize(repeating: 127, count: 256)
    defer { pixels.deallocate() }
    for format in 0..<8 {
      for rotation in 0..<4 {
        let input = HFImageData(
          data: pixels, width: 8, height: 8, format: HFImageFormat(rawValue: UInt32(format)),
          rotation: HFRotation(rawValue: UInt32(rotation)))
        let stream = try ImageStream(borrowing: input)
        try stream.close()
        expectError(Int(HERR_INVALID_IMAGE_STREAM_HANDLE)) { try stream.close() }
      }
    }
    let stream = try ImageStream()
    try stream.setFormat(HF_STREAM_BGR)
    try stream.setBorrowedBuffer(pixels, width: 8, height: 8)
    try stream.setRotation(.degrees0)
    try stream.setBorrowedBytes(
      UnsafeMutableRawBufferPointer(start: pixels, count: 256), width: 8, height: 8,
      format: HF_STREAM_BGR)
    let processed = try stream.processedBitmap(rotate: false, scale: 1)
    var data = HFImageBitmapData()
    try processed.getBorrowedData(&data)
    expect(data.width == 8)
    let bitmap = try ImageBitmap(
      copying: HFImageBitmapData(data: pixels, width: 8, height: 8, channels: 3))
    try bitmap.getBorrowedData(&data)
    expect(data.data != pixels && memcmp(data.data, pixels, 192) == 0)
    var cdata = HFImageBitmapData()
    expect(HFImageBitmapGetData(bitmap.nativeHandle, &cdata) == HSUCCEED)
    expect(cdata.data == data.data)
    try bitmap.withUnsafeMutablePixels { bytes, description in
      expect(bytes.baseAddress == UnsafeMutableRawPointer(data.data))
      expect(description.width == 8)
    }
    let copy = try bitmap.copyBitmap()
    let snapshot = try bitmap.snapshotStream(rotation: HF_CAMERA_ROTATION_0)
    try bitmap.draw(
      rect: HFaceRect(x: 1, y: 1, width: 3, height: 3), color: HColor(r: 1, g: 0, b: 0),
      thickness: 1)
    try bitmap.drawCircle(
      at: HPoint2f(x: 3, y: 3), radius: 1, color: HColor(r: 1, g: 0, b: 0), thickness: 1)
    try bitmap.drawIntegerCircle(
      at: HPoint2i(x: 4, y: 4), radius: 1, color: HColor(r: 1, g: 0, b: 0), thickness: 1)
    try bitmap.write(toFile: output + "/swift-bitmap.bmp")
    let loaded = try ImageBitmap(contentsOfFile: output + "/swift-bitmap.bmp", channels: 3)
    try stream.saveDebugImage(path: output + "/swift-debug.bmp")
    try bitmap.close()
    expectError(Int(HERR_INVALID_IMAGE_BITMAP_HANDLE)) {
      try bitmap.show(title: "closed", delay: 1)
    }
    try stream.close()
    stream.showDebugImage()
    let retained = try snapshot.processedBitmap(rotate: false, scale: 1)
    try retained.getBorrowedData(&data)
    expect(data.width == 8)
    try copy.close()
    try loaded.close()
    try snapshot.close()
    try processed.close()
    try retained.close()
    enum Marker: Error { case stop }
    do {
      try ImageStream.withBorrowedBytes(
        UnsafeMutableRawBufferPointer(start: pixels, count: 256), width: 8, height: 8,
        format: HF_STREAM_BGR
      ) { _ in throw Marker.stop }
    } catch Marker.stop { expect(true) }
    expectError(Int(HERR_INVALID_IMAGE_STREAM_PARAM)) {
      try ImageStream.withBorrowedBytes(
        UnsafeMutableRawBufferPointer(start: pixels, count: 1), width: 8, height: 8,
        format: HF_STREAM_BGR
      ) { _ -> Void in fatalError("insufficient buffer accepted") }
    }
  }
  var after: Int32 = 0
  try InspireFaceDiagnostics.getLiveStreamCount(&after)
  expect(before == after)
}

func pixelBuffers() throws {
  var buffer: CVPixelBuffer?
  let status = CVPixelBufferCreate(
    nil, 16, 8, kCVPixelFormatType_32BGRA,
    [kCVPixelBufferBytesPerRowAlignmentKey: 64] as CFDictionary, &buffer)
  expect(status == kCVReturnSuccess)
  let stream = try ImageStream(pixelBuffer: buffer!, rotation: HF_CAMERA_ROTATION_0)
  buffer = nil
  let bitmap = try stream.processedBitmap(rotate: false, scale: 1)
  var description = HFImageBitmapData()
  try bitmap.getBorrowedData(&description)
  expect(description.width == 16)
  try stream.close()
  try bitmap.close()
}

func hotPaths() throws {
  let pixels = UnsafeMutablePointer<UInt8>.allocate(capacity: 192)
  pixels.initialize(repeating: 0, count: 192)
  defer { pixels.deallocate() }
  let bitmap = try ImageBitmap(
    copying: HFImageBitmapData(data: pixels, width: 8, height: 8, channels: 3))
  let stream = try ImageStream(
    borrowing: HFImageData(
      data: pixels, width: 8, height: 8, format: HF_STREAM_BGR, rotation: HF_CAMERA_ROTATION_0))
  var level: UInt32 = 0
  var length: Int32 = 0
  var view = HFImageBitmapData()
  for _ in 0..<100 {
    try InspireFaceDiagnostics.getCAPILevel(&level)
    try FaceFeatureBuffer.getLength(&length)
    try bitmap.getBorrowedData(&view)
    try stream.setRotation(.degrees0)
  }
  beginAllocations()
  let allocation = malloc(777)
  consumeAllocation(allocation)
  let control = endAllocations()
  expect(allocation != nil && (allocationObservationSupported() == 0 || control > 0))
  free(allocation)
  let start = Date.timeIntervalSinceReferenceDate
  beginAllocations()
  for _ in 0..<20000 {
    try InspireFaceDiagnostics.getCAPILevel(&level)
    try FaceFeatureBuffer.getLength(&length)
    try bitmap.getBorrowedData(&view)
    try stream.setRotation(.degrees0)
  }
  let count = endAllocations()
  let duration = Date.timeIntervalSinceReferenceDate - start
  expect(allocationObservationSupported() == 0 || count == 0, "Swift successful hot calls allocated \(count) times")
  if allocationObservationSupported() == 0 { print("Allocation observer disabled under ASan; counts below are not measurements.") }
  print("Hot paths: 80000 Swift calls \(duration)s; allocations \(count)")
  try stream.close()
  try bitmap.close()
  var before: Int32 = 0
  try InspireFaceDiagnostics.getLiveStreamCount(&before)
  DispatchQueue.concurrentPerform(iterations: 8) { _ in
    for _ in 0..<256 {
      autoreleasepool {
        do {
          let stream = try ImageStream()
          var level: UInt32 = 0
          try InspireFaceDiagnostics.getCAPILevel(&level)
          if level != HF_C_API_LEVEL || stream.nativeHandle == nil {
            fatalError("parallel resource test")
          }
        } catch { fatalError("parallel failure: \(error)") }
      }
    }
  }
  var after: Int32 = 0
  try InspireFaceDiagnostics.getLiveStreamCount(&after)
  expect(after == before)
}

func featureHub() throws {
  var config = HFFeatureHubConfiguration()
  config.primaryKeyMode = HF_PK_MANUAL_INPUT
  config.searchMode = HF_SEARCH_MODE_EXHAUSTIVE
  config.searchThreshold = -1
  try FeatureHub.enable(configuration: config)
  defer { try? FeatureHub.disable() }
  try FeatureHub.setSearchThreshold(0.4)
  let owned = try FaceFeatureBuffer()
  var feature = owned.borrowedFeature
  owned.withUnsafeMutableBufferPointer { buffer in
    buffer.update(repeating: 0)
    buffer[0] = 1
    expect(buffer.baseAddress == feature.data)
  }
  try withUnsafeMutablePointer(to: &feature) { pointer in
    let identity = HFFaceFeatureIdentity(id: 42, feature: pointer)
    var allocated: Int64 = -1
    try FeatureHub.insert(identity, allocatedID: &allocated)
    expect(allocated == 42)
    var count: Int32 = 0
    try FeatureHub.getIdentityCount(&count)
    expect(count == 1)
    var result = HFFeatureHubSearchResultV2()
    try FeatureHub.search(pointer.pointee, borrowedResult: &result)
    expect(result.found != 0 && result.id == 42)
    var found = HFFaceFeatureIdentity()
    var confidence: Float = 0
    try FeatureHub.search(pointer.pointee, confidence: &confidence, borrowedIdentity: &found)
    expect(found.id == 42 && confidence > 0.99)
    var top = HFSearchTopKResults()
    try FeatureHub.search(pointer.pointee, topK: 1, borrowedResults: &top)
    expect(top.size == 1 && top.ids[0] == 42)
    try FeatureHub.getIdentity(id: 42, borrowedResult: &found)
    expect(found.id == 42)
    var ids = HFFeatureHubExistingIds()
    try FeatureHub.getBorrowedExistingIDs(&ids)
    try FeatureHub.updateIdentity(identity)
    try FeatureHub.printTable()
    try FaceFeatureBuffer.compare(pointer.pointee, with: pointer.pointee, similarity: &confidence)
    expect(abs(confidence - 1) < 1e-5)
  }
  try owned.withUnsafeMutableBufferPointer { values in
    let result = try FaceFeatureBuffer.similarity(
      UnsafeBufferPointer(values), UnsafeBufferPointer(values))
    expect(abs(result - 1) < 1e-5)
  }
  var length: Int32 = 0
  try FaceFeatureBuffer.getLength(&length)
  expect(length == feature.size)
  var threshold: Float = 0
  try FaceFeatureBuffer.getRecommendedThreshold(&threshold)
  var percentage: Float = 0
  try FaceFeatureBuffer.convert(similarity: threshold, percentage: &percentage)
  expect(percentage.isFinite)
  var converter = HFSimilarityConverterConfig()
  try FaceFeatureBuffer.getSimilarityConverter(&converter)
  try FaceFeatureBuffer.setSimilarityConverter(converter)
  try FeatureHub.removeIdentity(id: 42)
  try owned.close()
  expectError(Int(HERR_INVALID_FACE_FEATURE)) { try owned.close() }
}

func modelContract(_ model: String, _ image: String) throws {
  var pack = HFResourcePackInfo()
  let validation = model.withCString { HFValidateResourcePack($0, &pack) }
  if validation == HSUCCEED {
    try InspireFaceRuntime.validateResourcePack(path: model, info: &pack)
  } else {
    expectError(Int(validation)) {
      try InspireFaceRuntime.validateResourcePack(path: model, info: &pack)
    }
  }
  try InspireFaceRuntime.launch(path: model)
  defer { try? InspireFaceRuntime.terminate() }
  var launched: Int32 = 0
  try InspireFaceRuntime.getLaunchStatus(&launched)
  expect(launched != 0)
  try InspireFaceRuntime.reload(path: model)
  try InspireFaceRuntime.setLandmarkEngine(HF_LANDMARK_HYPLMV2_0_25)
  var before: Int32 = 0
  try InspireFaceDiagnostics.getLiveSessionCount(&before)
  try autoreleasepool {
    let features: FaceFeatures = [
      .recognition, .rgbLiveness, .mask, .quality, .attributes, .interaction, .emotion,
    ]
    let config = SessionConfiguration(features: features, maximumFaces: 3)
    let session = try FaceSession(configuration: config)
    let cSession = try FaceSession(configuration: config.cValue)
    try cSession.close()
    var parameters = HFSessionCustomParameter()
    parameters.enable_recognition = 1
    let legacy = try FaceSession(
      parameters: parameters, mode: HF_DETECT_MODE_ALWAYS_DETECT, maximumFaces: 1, pixelLevel: -1,
      framesPerSecond: -1)
    try legacy.close()
    let optional = try FaceSession(
      options: Int32(features.rawValue), mode: HF_DETECT_MODE_ALWAYS_DETECT, maximumFaces: 3,
      pixelLevel: -1, framesPerSecond: -1)
    try optional.close()
    try session.setTrackLostRecoveryEnabled(true)
    try session.setLightTrackConfidenceThreshold(0.2)
    try session.setTrackPreviewSize(192)
    var preview: Int32 = 0
    try session.getTrackPreviewSize(&preview)
    expect(preview == 192)
    try session.setMinimumFacePixelSize(8)
    try session.setDetectionThreshold(0.5)
    try session.setTrackingSmoothRatio(0.5)
    try session.setTrackingSmoothCacheFrames(3)
    try session.setDetectionInterval(1)
    try session.setLandmarkAugmentationCount(1)
    try session.setTrackingTimingEnabled(true)
    let bitmap = try ImageBitmap(contentsOfFile: image, channels: 3)
    var pixels = HFImageBitmapData()
    try bitmap.getBorrowedData(&pixels)
    let stream = try ImageStream(
      borrowing: HFImageData(
        data: pixels.data, width: pixels.width, height: pixels.height, format: HF_STREAM_BGR,
        rotation: HF_CAMERA_ROTATION_0))
    var faces = HFMultipleFaceData()
    try session.track(stream, borrowedResult: &faces)
    expect(faces.detectedNum > 0)
    let token = faces.tokens[0]
    var tokenSize: Int32 = 0
    try FaceTokenUtilities.getTokenSize(&tokenSize)
    let tokenBytes = UnsafeMutableRawBufferPointer.allocate(byteCount: Int(tokenSize), alignment: 8)
    defer { tokenBytes.deallocate() }
    try FaceTokenUtilities.copy(token, into: tokenBytes)
    let copiedToken = HFFaceBasicToken(size: tokenSize, data: tokenBytes.baseAddress)
    var landmarksCount: Int32 = 0
    try FaceTokenUtilities.getDenseLandmarkCount(&landmarksCount)
    let landmarks = UnsafeMutableBufferPointer<HPoint2f>.allocate(capacity: Int(landmarksCount))
    defer { landmarks.deallocate() }
    try FaceTokenUtilities.getDenseLandmarks(token, into: landmarks)
    expect(landmarks[0].x.isFinite)
    let five = UnsafeMutableBufferPointer<HPoint2f>.allocate(capacity: 5)
    defer { five.deallocate() }
    try FaceTokenUtilities.getFiveKeyPoints(token, into: five)
    var quality: Float = 0
    try session.getQuality(for: token, result: &quality)
    expect(quality.isFinite)
    var feature = HFFaceFeature()
    try session.extractFeature(from: stream, token: token, borrowedResult: &feature)
    expect(feature.size > 0)
    let output = UnsafeMutableBufferPointer<Float>.allocate(capacity: Int(feature.size))
    defer { output.deallocate() }
    try session.copyFeature(from: stream, token: token, into: output)
    expect(output[0].isFinite)
    try session.extractFeature(from: stream, token: token, into: output)
    expectError(Int(HERR_INVALID_FACE_FEATURE)) {
      try session.copyFeature(
        from: stream, token: token,
        into: UnsafeMutableBufferPointer(start: output.baseAddress, count: 1))
    }
    try session.withUnsafeFeature(in: stream, token: token) { borrowed in
      expect(borrowed.count == output.count)
      expect(
        memcmp(borrowed.baseAddress, output.baseAddress, output.count * MemoryLayout<Float>.size)
          == 0)
      expectError(Int(HERR_INVALID_PARAM)) { try session.close() }
      expectError(Int(HERR_INVALID_PARAM)) {
        try session.extractFeature(from: stream, token: token, into: output)
      }
    }
    let aligned = try session.alignmentBitmap(from: stream, token: token)
    let alignedStream = try aligned.snapshotStream(rotation: HF_CAMERA_ROTATION_0)
    try session.extractAlignedFeature(
      from: alignedStream, into: HFFaceFeature(size: Int32(output.count), data: output.baseAddress))
    try session.process(stream, faces: &faces, options: Int32(features.rawValue))
    var live = HFRGBLivenessConfidence()
    var mask = HFFaceMaskConfidence()
    var fq = HFFaceQualityConfidence()
    var state = HFFaceInteractionState()
    var actions = HFFaceInteractionsActions()
    var attributes = HFFaceAttributeResult()
    var emotions = HFFaceEmotionResult()
    try session.getBorrowedRGBLiveness(&live)
    expect(live.num == faces.detectedNum)
    try session.getBorrowedMaskConfidence(&mask)
    expect(mask.num == faces.detectedNum)
    try session.getBorrowedQualityConfidence(&fq)
    expect(fq.num == faces.detectedNum)
    try session.getBorrowedInteractionState(&state)
    expect(state.num == faces.detectedNum)
    try session.getBorrowedInteractionActions(&actions)
    expect(actions.num == faces.detectedNum)
    try session.getBorrowedAttributes(&attributes)
    expect(attributes.num == faces.detectedNum)
    try session.getBorrowedEmotions(&emotions)
    expect(emotions.num == faces.detectedNum)
    parameters.enable_liveness = 1
    try session.process(stream, faces: &faces, parameters: parameters)
    try session.getDebugPreviewImageSize(&preview)
    try session.printTrackingTiming()
    let snapshot = try session.snapshot(from: stream)
    var frozen = HFMultipleFaceData()
    try snapshot.getBorrowedFaces(&frozen)
    expect(frozen.detectedNum > 0)
    let address = frozen.tokens[0].data
    try session.withUnsafeFaces(in: stream) { borrowed in
      expect(borrowed.count > 0)
      expect(borrowed.tokens.baseAddress == UnsafePointer(borrowed.cValue.tokens))
      expectError(Int(HERR_INVALID_PARAM)) { try session.clearTracking() }
      expectError(Int(HERR_INVALID_PARAM)) { _ = try session.snapshot(from: stream) }
    }
    try snapshot.withUnsafeFaces { borrowed in expect(borrowed.tokens[0].data == address) }
    try FaceTokenUtilities.getFiveKeyPoints(copiedToken, into: five)
    var captureConfig = try FaceCaptureSession.defaultConfiguration()
    captureConfig.filterMask = 0
    let capture = try FaceCaptureSession(session: session, configuration: captureConfig)
    var progress = HFFaceCaptureProgress()
    try capture.update(stream, frameID: 1, timestampMilliseconds: 1, progress: &progress)
    try capture.update(
      stream, snapshot: snapshot, frameID: 2, timestampMilliseconds: 2, progress: &progress)
    let results = UnsafeMutableBufferPointer<HFFaceCaptureResult>.allocate(
      capacity: Int(HF_FACE_CAPTURE_MAX_RESULTS))
    defer { results.deallocate() }
    let count = try capture.results(into: results)
    expect(count <= results.count)
    try capture.finish(progress: &progress)
    try capture.reset()
    try capture.close()
    try session.clearTracking()
    var n: Int32 = 0
    try InspireFaceDiagnostics.getLiveSessionCount(&n)
    let handles = UnsafeMutablePointer<HFSession?>.allocate(capacity: Int(n))
    defer { handles.deallocate() }
    try InspireFaceDiagnostics.getBorrowedLiveSessions(handles, capacity: n)
    try InspireFaceDiagnostics.getLiveStreamCount(&n)
    let streams = UnsafeMutablePointer<HFImageStream?>.allocate(capacity: Int(n))
    defer { streams.deallocate() }
    try InspireFaceDiagnostics.getBorrowedLiveStreams(streams, capacity: n)
    try session.close()
    try snapshot.getBorrowedFaces(&frozen)
    expect(frozen.tokens[0].data == address)
    try snapshot.close()
    expectError(Int(HERR_INVALID_CONTEXT_HANDLE)) { try session.clearTracking() }
    try stream.close()
    try bitmap.close()
    try alignedStream.close()
    try aligned.close()
  }
  var after: Int32 = 0
  try InspireFaceDiagnostics.getLiveSessionCount(&after)
  expect(after == before)
}

let args = CommandLine.arguments
expect(args.count == 2 || args.count == 4)
do {
  try metadata()
  try images(args[1])
  try pixelBuffers()
  try hotPaths()
  try featureHub()
  if args.count == 4 { try modelContract(args[2], args[3]) }
  print(
    "Swift contract: \(checks) assertions and all throwing operations passed (\(args.count == 4 ? "with model" : "without model"))"
  )
} catch { fatalError("unexpected error: \(error)") }
