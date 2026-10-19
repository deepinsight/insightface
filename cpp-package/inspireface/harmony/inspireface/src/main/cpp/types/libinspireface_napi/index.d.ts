export interface NativeSessionHandle {
  readonly __inspireFaceSessionBrand: string;
}

export interface NativeImageStreamHandle {
  readonly __inspireFaceImageStreamBrand: string;
}

export interface NativeFaceResultHandle {
  readonly __inspireFaceResultBrand: string;
}

export interface NativeImageBitmapHandle {
  readonly __inspireFaceImageBitmapBrand: string;
}

export interface NativeFaceCaptureHandle {
  readonly __inspireFaceCaptureBrand: string;
}

export interface NativeVersion {
  major: number;
  minor: number;
  patch: number;
}

export interface NativeSessionOptions {
  featureMask?: number;
  detectMode?: number;
  maxFaces?: number;
  detectPixelLevel?: number;
  trackFps?: number;
}

export interface NativeSessionRuntimeOptions {
  previewSize?: number;
  minimumFaceSize?: number;
  smoothCacheFrames?: number;
  detectInterval?: number;
  landmarkAugmentation?: number;
  detectThreshold?: number;
  smoothRatio?: number;
  lightTrackThreshold?: number;
  trackLostRecovery?: boolean;
}

export interface NativeRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface NativeTrackedFace {
  rect: NativeRect;
  trackId: number;
  trackCount: number;
  confidence: number;
  roll: number;
  yaw: number;
  pitch: number;
  token: Uint8Array;
}

export interface NativeTrackResult {
  detectedNum: number;
  faces: NativeTrackedFace[];
  handle: NativeFaceResultHandle;
}

export interface NativeFaceCaptureConfig {
  filterMask?: number;
  outputCount?: number;
  minTrackCount?: number;
  stableDurationMs?: number;
  collectDurationMs?: number;
  maxCollectDurationMs?: number;
  trackLostGraceMs?: number;
  minCandidateIntervalMs?: number;
  minFaceWidthRatio?: number;
  maxFaceWidthRatio?: number;
  maxCenterOffsetX?: number;
  maxCenterOffsetY?: number;
  boundaryMarginRatio?: number;
  maxCenterMotionRatio?: number;
  maxSizeChangeRatio?: number;
  maxAbsYaw?: number;
  maxAbsPitch?: number;
  maxAbsRoll?: number;
  minQualityScore?: number;
  minSharpnessScore?: number;
  minBrightnessScore?: number;
  maxBrightnessScore?: number;
}

export interface NativeFaceCaptureMetrics {
  availableMetrics: number;
  faceWidthRatio: number;
  centerOffsetX: number;
  centerOffsetY: number;
  stabilityScore: number;
  poseScore: number;
  qualityScore: number;
  sharpnessScore: number;
  brightnessScore: number;
}

export interface NativeFaceCaptureProgress {
  state: number;
  candidateCount: number;
  frameId: number;
  timestampMs: number;
  trackId: number;
  trackCount: number;
  evaluatedFilters: number;
  rejectReasons: number;
  progress: number;
  currentScore: number;
  metrics: NativeFaceCaptureMetrics;
}

export interface NativeFaceCaptureResult {
  frameId: number;
  timestampMs: number;
  trackId: number;
  trackCount: number;
  score: number;
  rect: NativeRect;
  roll: number;
  yaw: number;
  pitch: number;
  token: Uint8Array;
  metrics: NativeFaceCaptureMetrics;
}

export interface NativeResourcePackInfo {
  archiveFileCount: number;
  modelCount: number;
  tag: string;
  version: string;
  major: string;
  releaseDate: string;
}

export interface NativePipelineResult {
  detectedNum: number;
  rgbLiveness: Float32Array;
  maskConfidence: Float32Array;
  qualityConfidence: Float32Array;
  leftEyeStatusConfidence: Float32Array;
  rightEyeStatusConfidence: Float32Array;
  normal: Int32Array;
  shake: Int32Array;
  jawOpen: Int32Array;
  headRaise: Int32Array;
  blink: Int32Array;
  race: Int32Array;
  gender: Int32Array;
  ageBracket: Int32Array;
  emotion: Int32Array;
}

export interface NativeSimilarityConverterConfig {
  threshold: number;
  middleScore: number;
  steepness: number;
  outputMin: number;
  outputMax: number;
}

export interface NativeFeatureHubConfiguration {
  primaryKeyMode?: number;
  enablePersistence?: boolean;
  persistenceDbPath?: string;
  searchThreshold?: number;
  searchMode?: number;
}

export interface NativeFeatureHubSearchResult {
  found: boolean;
  id: bigint;
  confidence: number;
  feature: Float32Array;
}

export interface NativeFeatureHubTopKResult {
  ids: bigint[];
  confidence: Float32Array;
}

export interface NativeComponentVersion {
  major: number;
  minor: number;
  patch: number;
  state: number;
}

export interface NativeBitmapData {
  data: Uint8Array;
  width: number;
  height: number;
  channels: number;
}

export interface NativePoint {
  x: number;
  y: number;
}

export interface NativeDebugResourceCounts {
  sessions: number;
  streams: number;
}

export interface NativeColor {
  r: number;
  g: number;
  b: number;
}

interface InspireFaceNativeModule {
  getErrorMessage(code: number): string;
  validateResourcePack(resourcePath: string): NativeResourcePackInfo;
  launch(resourcePath: string): void;
  reload(resourcePath: string): void;
  terminate(): void;
  isLaunched(): boolean;
  getVersion(): NativeVersion;
  getCapiLevel(): number;
  createSession(options: NativeSessionOptions): NativeSessionHandle;
  releaseSession(handle: NativeSessionHandle): void;
  configureSession(handle: NativeSessionHandle, options: NativeSessionRuntimeOptions): void;
  clearTracking(handle: NativeSessionHandle): void;
  getSessionPreviewSize(handle: NativeSessionHandle): number;
  getLastDetectionDebugPreviewSize(handle: NativeSessionHandle): number;
  getSupportedPixelLevels(): Int32Array;
  switchLandmarkEngine(engine: number): void;
  createImageStream(data: Uint8Array, width: number, height: number, format: number,
    rotation: number): NativeImageStreamHandle;
  updateImageStream(handle: NativeImageStreamHandle, data: Uint8Array, width: number, height: number,
    format: number, rotation: number): void;
  releaseImageStream(handle: NativeImageStreamHandle): void;
  createImageBitmap(data: Uint8Array, width: number, height: number, channels: number): NativeImageBitmapHandle;
  createImageBitmapFromFile(filePath: string, channels: number): NativeImageBitmapHandle;
  copyImageBitmap(handle: NativeImageBitmapHandle): NativeImageBitmapHandle;
  releaseImageBitmap(handle: NativeImageBitmapHandle): void;
  createImageStreamFromBitmap(handle: NativeImageBitmapHandle, rotation: number): NativeImageStreamHandle;
  createImageBitmapFromStream(handle: NativeImageStreamHandle, rotate: number, scale: number): NativeImageBitmapHandle;
  getImageBitmapData(handle: NativeImageBitmapHandle): NativeBitmapData;
  writeImageBitmap(handle: NativeImageBitmapHandle, filePath: string): void;
  drawImageBitmapRect(handle: NativeImageBitmapHandle, rect: NativeRect, color: NativeColor, thickness: number): void;
  drawImageBitmapCircle(handle: NativeImageBitmapHandle, point: NativePoint, radius: number,
    color: NativeColor, thickness: number): void;
  showImageBitmap(handle: NativeImageBitmapHandle, title: string, delay: number): void;
  track(session: NativeSessionHandle, image: NativeImageStreamHandle): NativeTrackResult;
  releaseFaceResult(handle: NativeFaceResultHandle): void;
  getDefaultFaceCaptureConfig(): NativeFaceCaptureConfig;
  createFaceCaptureSession(session: NativeSessionHandle, config: NativeFaceCaptureConfig): NativeFaceCaptureHandle;
  updateFaceCaptureSession(capture: NativeFaceCaptureHandle, image: NativeImageStreamHandle,
    faces: NativeFaceResultHandle | null, frameId: number, timestampMs: number): NativeFaceCaptureProgress;
  getFaceCaptureResults(capture: NativeFaceCaptureHandle): NativeFaceCaptureResult[];
  finishFaceCaptureSession(capture: NativeFaceCaptureHandle): NativeFaceCaptureProgress;
  resetFaceCaptureSession(capture: NativeFaceCaptureHandle): void;
  releaseFaceCaptureSession(capture: NativeFaceCaptureHandle): void;
  processPipeline(session: NativeSessionHandle, image: NativeImageStreamHandle,
    faces: NativeFaceResultHandle, featureMask: number): NativePipelineResult;
  detectFaceQuality(session: NativeSessionHandle, token: Uint8Array): number;
  getDenseLandmarks(token: Uint8Array): Float32Array;
  getFiveKeyPoints(token: Uint8Array): Float32Array;
  extractFeature(session: NativeSessionHandle, image: NativeImageStreamHandle,
    token: Uint8Array): Float32Array;
  getFaceAlignmentImage(session: NativeSessionHandle, image: NativeImageStreamHandle,
    token: Uint8Array): NativeImageBitmapHandle;
  extractFeatureFromAlignmentImage(session: NativeSessionHandle, image: NativeImageStreamHandle): Float32Array;
  getFeatureLength(): number;
  compareFeatures(first: Float32Array, second: Float32Array): number;
  getRecommendedThreshold(): number;
  similarityToPercentage(similarity: number): number;
  getSimilarityConverter(): NativeSimilarityConverterConfig;
  updateSimilarityConverter(config: NativeSimilarityConverterConfig): void;
  featureHubEnable(config: NativeFeatureHubConfiguration): void;
  featureHubDisable(): void;
  featureHubViewTable(): void;
  featureHubSetThreshold(threshold: number): void;
  featureHubInsert(id: bigint, feature: Float32Array): bigint;
  featureHubUpdate(id: bigint, feature: Float32Array): void;
  featureHubRemove(id: bigint): void;
  featureHubGet(id: bigint): Float32Array;
  featureHubGetCount(): number;
  featureHubGetIds(): bigint[];
  featureHubSearch(feature: Float32Array): NativeFeatureHubSearchResult;
  featureHubSearchTopK(feature: Float32Array, topK: number): NativeFeatureHubTopKResult;
  getComponentVersion(component: number): NativeComponentVersion;
  getComponentVersions(): string;
  getDiagnosticInformation(): string;
  getExtendedInformation(): string;
  queryRgaEnabled(): boolean;
  setRgaDmaHeapPath(path: string): void;
  getRgaDmaHeapPath(): string;
  switchImageProcessingBackend(backend: number): void;
  setImageProcessAlignedWidth(width: number): void;
  setCoreMlInferenceMode(mode: number): void;
  setCudaDeviceId(deviceId: number): void;
  getCudaDeviceId(): number;
  printCudaDeviceInfo(): void;
  getCudaDeviceCount(): number;
  isCudaSupported(): boolean;
  configureTrackCost(session: NativeSessionHandle, enabled: boolean): void;
  printTrackCost(session: NativeSessionHandle): void;
  getDebugResourceCounts(): NativeDebugResourceCounts;
  saveDebugImageStream(handle: NativeImageStreamHandle, filePath: string): void;
  showDebugImageStream(handle: NativeImageStreamHandle): void;
  showDebugResourceStatistics(): void;
  setLogLevel(level: number): void;
  disableLog(): void;
  logPrint(level: number, message: string): void;
}

declare const inspireFaceNative: InspireFaceNativeModule;
export default inspireFaceNative;
