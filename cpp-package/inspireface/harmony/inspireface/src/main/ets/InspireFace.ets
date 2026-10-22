import native, { NativeFaceCaptureHandle, NativeFaceResultHandle, NativeImageBitmapHandle,
  NativeImageStreamHandle, NativeSessionHandle } from 'libinspireface_napi.so';

export enum ErrorCode {
  SUCCESS = 0,
  UNKNOWN = 1,
  INVALID_PARAM = 2,
  INVALID_IMAGE_STREAM_HANDLE = 3,
  INVALID_CONTEXT_HANDLE = 4,
  INVALID_FACE_TOKEN = 5,
  INVALID_FACE_FEATURE = 6,
  INVALID_FACE_LIST = 7,
  INVALID_BUFFER_SIZE = 8,
  INVALID_IMAGE_STREAM_PARAM = 9,
  INVALID_SERIALIZATION_FAILED = 10,
  INVALID_DETECTION_INPUT = 11,
  INVALID_IMAGE_BITMAP_HANDLE = 12,
  IMAGE_STREAM_DECODE_FAILED = 13,
  UNSUPPORTED = 14,
  SESSION_FUNCTION_UNUSABLE = 101,
  SESSION_TRACKER_FAILURE = 102,
  SESSION_PIPELINE_FAILURE = 103,
  SESSION_INVALID_RESOURCE = 104,
  SESSION_LANDMARK_NUM_NOT_MATCH = 105,
  SESSION_LANDMARK_NOT_ENABLE = 106,
  SESSION_KEY_POINT_NUM_NOT_MATCH = 107,
  SESSION_RECOGNITION_EXTRACT_FAILURE = 108,
  SESSION_RECOGNITION_FEATURE_ERROR = 109,
  SESSION_FACE_DATA_ERROR = 110,
  SESSION_FACE_RECOGNITION_OPTION_ERROR = 111,
  FEATURE_HUB_DISABLED = 201,
  FEATURE_HUB_INSERT_FAILURE = 202,
  FEATURE_HUB_FEATURE_NOT_FOUND = 203,
  FEATURE_HUB_INVALID_FEATURE = 204,
  FEATURE_HUB_DATABASE_FAILURE = 205,
  ARCHIVE_LOAD_FAILURE = 251,
  ARCHIVE_MODEL_LOAD_FAILURE = 252,
  ARCHIVE_FILE_FORMAT_ERROR = 253,
  ARCHIVE_REPEATED_LOAD = 254,
  ARCHIVE_NOT_LOADED = 255,
  CUDA_NOT_SUPPORTED = 301,
  CUDA_TENSOR_RT_NOT_SUPPORTED = 302,
  CUDA_UNKNOWN_ERROR = 303,
  CUDA_DISABLED = 304,
  IMAGE_PROCESS_FAILURE = 305,
  EXTENSION_ERROR = 351,
  EXTENSION_ML_MODEL_LOAD_FAILED = 352,
  EXTENSION_HETERO_MODEL_TAG_ERROR = 353,
  EXTENSION_RECOGNITION_HEAD_CONFIG_ERROR = 354,
  EXTENSION_HETERO_MODEL_NOT_MATCH = 355,
  EXTENSION_HETERO_MODEL_NOT_LOADED = 356,
  CAPTURE_INVALID_CONFIG = 401,
  CAPTURE_REQUIRED_FEATURE_OFF = 402,
  CAPTURE_FRAME_OUT_OF_ORDER = 403,
  CAPTURE_INVALID_HANDLE = 404
}

export enum FaceCaptureFilter {
  NONE = 0,
  FACE_COUNT = 1 << 0,
  FACE_SIZE = 1 << 1,
  FACE_POSITION = 1 << 2,
  FACE_BOUNDARY = 1 << 3,
  STABILITY = 1 << 4,
  POSE = 1 << 5,
  QUALITY = 1 << 6,
  SHARPNESS = 1 << 7,
  BRIGHTNESS = 1 << 8,
  TRACK_COUNT = 1 << 9
}

export enum FaceCaptureRejectReason {
  NONE = 0,
  NO_FACE = 1 << 0,
  MULTIPLE_FACES = 1 << 1,
  FACE_TOO_SMALL = 1 << 2,
  FACE_TOO_LARGE = 1 << 3,
  FACE_OFF_CENTER = 1 << 4,
  FACE_OUT_OF_BOUNDS = 1 << 5,
  UNSTABLE = 1 << 6,
  POSE = 1 << 7,
  QUALITY = 1 << 8,
  SHARPNESS = 1 << 9,
  BRIGHTNESS = 1 << 10,
  TRACK_COUNT_TOO_LOW = 1 << 11
}

export enum FaceCaptureState {
  IDLE = 0,
  STABILIZING = 1,
  COLLECTING = 2,
  READY = 3,
  FINISHED = 4,
  TRACK_LOST = 5
}

export enum Feature {
  NONE = 0x00000000,
  FACE_RECOGNITION = 0x00000002,
  LIVENESS = 0x00000004,
  IR_LIVENESS = 0x00000008,
  MASK_DETECT = 0x00000010,
  FACE_ATTRIBUTE = 0x00000020,
  QUALITY = 0x00000080,
  INTERACTION = 0x00000100,
  FACE_POSE = 0x00000200,
  FACE_EMOTION = 0x00000400
}

export enum DetectMode {
  ALWAYS_DETECT = 0,
  LIGHT_TRACK = 1,
  TRACK_BY_DETECTION = 2
}

export enum ImageFormat {
  RGB = 0,
  BGR = 1,
  RGBA = 2,
  BGRA = 3,
  YUV_NV12 = 4,
  YUV_NV21 = 5,
  I420 = 6,
  GRAY = 7
}

export enum Rotation {
  DEGREE_0 = 0,
  DEGREE_90 = 1,
  DEGREE_180 = 2,
  DEGREE_270 = 3
}

export enum LogLevel {
  NONE = 0,
  DEBUG = 1,
  INFO = 2,
  WARN = 3,
  ERROR = 4,
  FATAL = 5
}

export enum LandmarkEngine {
  HYPLMV2_0_25 = 0,
  HYPLMV2_0_50 = 1,
  INSIGHTFACE_2D106_TRACK = 2
}

export enum PrimaryKeyMode {
  AUTO_INCREMENT = 0,
  MANUAL_INPUT = 1
}

export enum SearchMode {
  EAGER = 0,
  EXHAUSTIVE = 1
}

export enum ComponentType {
  MNN = 0,
  INSPIRE_CV = 1,
  EIGEN = 2,
  SQLITE = 3,
  SQLITE_VEC = 4,
  NLOHMANN_JSON = 5,
  OPENCV = 6,
  TENSOR_RT = 7,
  CUDA = 8,
  RKNN = 9,
  RGA = 10,
  CORE_ML = 11
}

export enum ComponentVersionState {
  DISABLED = 0,
  KNOWN = 1,
  UNKNOWN = 2
}

export enum Race {
  BLACK = 0,
  ASIAN = 1,
  LATINO_HISPANIC = 2,
  MIDDLE_EASTERN = 3,
  WHITE = 4
}

export enum Gender {
  FEMALE = 0,
  MALE = 1
}

export enum AgeBracket {
  AGE_0_2 = 0,
  AGE_3_9 = 1,
  AGE_10_19 = 2,
  AGE_20_29 = 3,
  AGE_30_39 = 4,
  AGE_40_49 = 5,
  AGE_50_59 = 6,
  AGE_60_69 = 7,
  AGE_70_PLUS = 8
}

export enum Emotion {
  NEUTRAL = 0,
  HAPPY = 1,
  SAD = 2,
  SURPRISE = 3,
  FEAR = 4,
  DISGUST = 5,
  ANGER = 6
}

export enum ImageProcessingBackend {
  CPU = 0,
  RGA = 1
}

export enum CoreMlInferenceMode {
  CPU = 0,
  GPU = 1,
  ANE = 2
}

export interface Version {
  major: number;
  minor: number;
  patch: number;
}

export interface SessionOptions {
  featureMask?: number;
  detectMode?: DetectMode;
  maxFaces?: number;
  detectPixelLevel?: number;
  trackFps?: number;
}

export interface SessionRuntimeOptions {
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

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Point {
  x: number;
  y: number;
}

export interface Color {
  r: number;
  g: number;
  b: number;
}

export interface BitmapData {
  data: Uint8Array;
  width: number;
  height: number;
  channels: number;
}

export interface TrackedFace {
  rect: Rect;
  trackId: number;
  trackCount: number;
  confidence: number;
  roll: number;
  yaw: number;
  pitch: number;
  token: Uint8Array;
}

export interface FaceTrackResult {
  detectedNum: number;
  faces: TrackedFace[];
}

export interface FaceCaptureConfig {
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

export interface FaceCaptureMetrics {
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

export interface FaceCaptureProgress {
  state: FaceCaptureState;
  candidateCount: number;
  frameId: number;
  timestampMs: number;
  trackId: number;
  trackCount: number;
  evaluatedFilters: number;
  rejectReasons: number;
  progress: number;
  currentScore: number;
  metrics: FaceCaptureMetrics;
}

export interface FaceCaptureResult {
  frameId: number;
  timestampMs: number;
  trackId: number;
  trackCount: number;
  score: number;
  rect: Rect;
  roll: number;
  yaw: number;
  pitch: number;
  token: Uint8Array;
  metrics: FaceCaptureMetrics;
}

export interface ResourcePackInfo {
  archiveFileCount: number;
  modelCount: number;
  tag: string;
  version: string;
  major: string;
  releaseDate: string;
}

export interface PipelineResult {
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

export interface SimilarityConverterConfig {
  threshold: number;
  middleScore: number;
  steepness: number;
  outputMin: number;
  outputMax: number;
}

export interface FeatureHubConfiguration {
  primaryKeyMode?: PrimaryKeyMode;
  enablePersistence?: boolean;
  persistenceDbPath?: string;
  searchThreshold?: number;
  searchMode?: SearchMode;
}

export interface FeatureHubSearchResult {
  found: boolean;
  id: bigint;
  confidence: number;
  feature: Float32Array;
}

export interface FeatureHubTopKResult {
  ids: bigint[];
  confidence: Float32Array;
}

export interface ComponentVersion {
  major: number;
  minor: number;
  patch: number;
  state: ComponentVersionState;
}

export interface DebugResourceCounts {
  sessions: number;
  streams: number;
}

interface SessionState {
  handle: NativeSessionHandle;
  closed: boolean;
  featureMask: number;
}

interface ImageStreamState {
  handle: NativeImageStreamHandle;
  closed: boolean;
}

interface FaceResultState {
  handle: NativeFaceResultHandle;
  closed: boolean;
}

interface ImageBitmapState {
  handle: NativeImageBitmapHandle;
  closed: boolean;
}

interface FaceCaptureSessionState {
  handle: NativeFaceCaptureHandle;
  closed: boolean;
}

const sessionStates: WeakMap<Session, SessionState> = new WeakMap<Session, SessionState>();
const imageStreamStates: WeakMap<ImageStream, ImageStreamState> = new WeakMap<ImageStream, ImageStreamState>();
const faceResultStates: WeakMap<FaceTrackResult, FaceResultState> = new WeakMap<FaceTrackResult, FaceResultState>();
const imageBitmapStates: WeakMap<ImageBitmap, ImageBitmapState> = new WeakMap<ImageBitmap, ImageBitmapState>();
const faceCaptureStates: WeakMap<FaceCaptureSession, FaceCaptureSessionState> =
  new WeakMap<FaceCaptureSession, FaceCaptureSessionState>();

function requireSessionState(session: Session): SessionState {
  const state: SessionState | undefined = sessionStates.get(session);
  if (state === undefined || state.closed) {
    throw new Error('InspireFace session has already been closed');
  }
  return state;
}

function requireImageStreamState(image: ImageStream): ImageStreamState {
  const state: ImageStreamState | undefined = imageStreamStates.get(image);
  if (state === undefined || state.closed) {
    throw new Error('InspireFace image stream has already been closed');
  }
  return state;
}

function requireFaceResultState(result: FaceTrackResult): FaceResultState {
  const state: FaceResultState | undefined = faceResultStates.get(result);
  if (state === undefined || state.closed) {
    throw new Error('InspireFace face result has already been closed or was not created by this SDK');
  }
  return state;
}

function requireImageBitmapState(bitmap: ImageBitmap): ImageBitmapState {
  const state: ImageBitmapState | undefined = imageBitmapStates.get(bitmap);
  if (state === undefined || state.closed) {
    throw new Error('InspireFace image bitmap has already been closed');
  }
  return state;
}

function requireFaceCaptureState(capture: FaceCaptureSession): FaceCaptureSessionState {
  const state: FaceCaptureSessionState | undefined = faceCaptureStates.get(capture);
  if (state === undefined || state.closed) {
    throw new Error('InspireFace face capture session has already been closed');
  }
  return state;
}

/**
 * Owns an immutable native copy of one image. Call close() when processing is complete.
 */
export class ImageStream {
  public constructor(data: Uint8Array, width: number, height: number, format: ImageFormat,
    rotation: Rotation = Rotation.DEGREE_0, internalHandle?: NativeImageStreamHandle) {
    const handle: NativeImageStreamHandle = internalHandle ??
      native.createImageStream(data, width, height, format, rotation);
    imageStreamStates.set(this, { handle: handle, closed: false });
  }

  public static fromBitmap(bitmap: ImageBitmap, rotation: Rotation = Rotation.DEGREE_0): ImageStream {
    const handle: NativeImageStreamHandle = native.createImageStreamFromBitmap(
      requireImageBitmapState(bitmap).handle, rotation);
    return new ImageStream(new Uint8Array(0), 0, 0, ImageFormat.RGB, Rotation.DEGREE_0, handle);
  }

  public update(data: Uint8Array, width: number, height: number, format: ImageFormat,
    rotation: Rotation = Rotation.DEGREE_0): void {
    native.updateImageStream(requireImageStreamState(this).handle, data, width, height, format, rotation);
  }

  public saveDebugImage(filePath: string): void {
    native.saveDebugImageStream(requireImageStreamState(this).handle, filePath);
  }

  /** Desktop-style display is unavailable in the standard HarmonyOS build and reports UNSUPPORTED. */
  public showDebugImage(): void {
    native.showDebugImageStream(requireImageStreamState(this).handle);
  }

  public close(): void {
    const state: ImageStreamState | undefined = imageStreamStates.get(this);
    if (state === undefined || state.closed) {
      return;
    }
    native.releaseImageStream(state.handle);
    state.closed = true;
  }

  public isClosed(): boolean {
    const state: ImageStreamState | undefined = imageStreamStates.get(this);
    return state === undefined || state.closed;
  }
}

/** Owns a native BGR or grayscale bitmap. */
export class ImageBitmap {
  public constructor(data: Uint8Array, width: number, height: number, channels: number,
    internalHandle?: NativeImageBitmapHandle) {
    const handle: NativeImageBitmapHandle = internalHandle ?? native.createImageBitmap(data, width, height, channels);
    imageBitmapStates.set(this, { handle: handle, closed: false });
  }

  public static fromFile(filePath: string, channels: number = 3): ImageBitmap {
    const handle: NativeImageBitmapHandle = native.createImageBitmapFromFile(filePath, channels);
    return new ImageBitmap(new Uint8Array(0), 0, 0, channels, handle);
  }

  public static fromImageStream(image: ImageStream, rotate: boolean = true, scale: number = 1.0): ImageBitmap {
    const handle: NativeImageBitmapHandle = native.createImageBitmapFromStream(
      requireImageStreamState(image).handle, rotate ? 1 : 0, scale);
    return new ImageBitmap(new Uint8Array(0), 0, 0, 3, handle);
  }

  public copy(): ImageBitmap {
    const handle: NativeImageBitmapHandle = native.copyImageBitmap(requireImageBitmapState(this).handle);
    return new ImageBitmap(new Uint8Array(0), 0, 0, 3, handle);
  }

  public getData(): BitmapData {
    return native.getImageBitmapData(requireImageBitmapState(this).handle);
  }

  public writeToFile(filePath: string): void {
    native.writeImageBitmap(requireImageBitmapState(this).handle, filePath);
  }

  public drawRect(rect: Rect, color: Color, thickness: number = 1): void {
    native.drawImageBitmapRect(requireImageBitmapState(this).handle, rect, color, thickness);
  }

  public drawCircle(point: Point, radius: number, color: Color, thickness: number = 1): void {
    native.drawImageBitmapCircle(requireImageBitmapState(this).handle, point, radius, color, thickness);
  }

  /** Desktop-style display is unavailable in the standard HarmonyOS build and reports UNSUPPORTED. */
  public show(title: string, delay: number = 0): void {
    native.showImageBitmap(requireImageBitmapState(this).handle, title, delay);
  }

  public close(): void {
    const state: ImageBitmapState | undefined = imageBitmapStates.get(this);
    if (state === undefined || state.closed) {
      return;
    }
    native.releaseImageBitmap(state.handle);
    state.closed = true;
  }

  public isClosed(): boolean {
    const state: ImageBitmapState | undefined = imageBitmapStates.get(this);
    return state === undefined || state.closed;
  }
}

/**
 * Owns one face-processing session. Calls on the same session are serialized by the native bridge.
 */
export class Session {
  public constructor(options: SessionOptions = {}) {
    const handle: NativeSessionHandle = native.createSession(options);
    sessionStates.set(this, { handle: handle, closed: false, featureMask: options.featureMask ?? Feature.NONE });
  }

  public configure(options: SessionRuntimeOptions): void {
    native.configureSession(requireSessionState(this).handle, options);
  }

  public clearTracking(): void {
    native.clearTracking(requireSessionState(this).handle);
  }

  public getPreviewSize(): number {
    return native.getSessionPreviewSize(requireSessionState(this).handle);
  }

  public getLastDetectionDebugPreviewSize(): number {
    return native.getLastDetectionDebugPreviewSize(requireSessionState(this).handle);
  }

  public setTrackCostMeasurement(enabled: boolean): void {
    native.configureTrackCost(requireSessionState(this).handle, enabled);
  }

  public printTrackCost(): void {
    native.printTrackCost(requireSessionState(this).handle);
  }

  public track(image: ImageStream): FaceTrackResult {
    const session: SessionState = requireSessionState(this);
    const stream: ImageStreamState = requireImageStreamState(image);
    const result = native.track(session.handle, stream.handle);
    const publicResult: FaceTrackResult = { detectedNum: result.detectedNum, faces: result.faces };
    faceResultStates.set(publicResult, { handle: result.handle, closed: false });
    return publicResult;
  }

  public processPipeline(image: ImageStream, faces: FaceTrackResult, featureMask?: number): PipelineResult {
    const session: SessionState = requireSessionState(this);
    const stream: ImageStreamState = requireImageStreamState(image);
    const result: FaceResultState = requireFaceResultState(faces);
    return native.processPipeline(session.handle, stream.handle, result.handle, featureMask ?? session.featureMask);
  }

  public detectFaceQuality(face: TrackedFace): number {
    return native.detectFaceQuality(requireSessionState(this).handle, face.token);
  }

  public releaseFaceResult(result: FaceTrackResult): void {
    const state: FaceResultState | undefined = faceResultStates.get(result);
    if (state === undefined || state.closed) {
      return;
    }
    native.releaseFaceResult(state.handle);
    state.closed = true;
  }

  public extractFeature(image: ImageStream, face: TrackedFace): Float32Array {
    const session: SessionState = requireSessionState(this);
    const stream: ImageStreamState = requireImageStreamState(image);
    return native.extractFeature(session.handle, stream.handle, face.token);
  }

  public getFaceAlignmentImage(image: ImageStream, face: TrackedFace): ImageBitmap {
    const session: SessionState = requireSessionState(this);
    const stream: ImageStreamState = requireImageStreamState(image);
    const handle: NativeImageBitmapHandle = native.getFaceAlignmentImage(session.handle, stream.handle, face.token);
    return new ImageBitmap(new Uint8Array(0), 0, 0, 3, handle);
  }

  public extractFeatureFromAlignmentImage(image: ImageStream): Float32Array {
    const session: SessionState = requireSessionState(this);
    const stream: ImageStreamState = requireImageStreamState(image);
    return native.extractFeatureFromAlignmentImage(session.handle, stream.handle);
  }

  public close(): void {
    const state: SessionState | undefined = sessionStates.get(this);
    if (state === undefined || state.closed) {
      return;
    }
    native.releaseSession(state.handle);
    state.closed = true;
  }

  public isClosed(): boolean {
    const state: SessionState | undefined = sessionStates.get(this);
    return state === undefined || state.closed;
  }
}

/**
 * Synchronous capture policy attached to a Session. It owns no camera or worker
 * thread. Pass an existing FaceTrackResult to update() to avoid tracking twice.
 */
export class FaceCaptureSession {
  public constructor(session: Session, config: FaceCaptureConfig = {}) {
    const handle: NativeFaceCaptureHandle = native.createFaceCaptureSession(
      requireSessionState(session).handle, config);
    faceCaptureStates.set(this, { handle: handle, closed: false });
  }

  public static getDefaultConfig(): FaceCaptureConfig {
    return native.getDefaultFaceCaptureConfig();
  }

  /** A supplied FaceTrackResult must describe this frame; its trackCount is immutable. */
  public update(image: ImageStream, frameId: number, timestampMs: number,
    faces?: FaceTrackResult): FaceCaptureProgress {
    const capture: FaceCaptureSessionState = requireFaceCaptureState(this);
    const stream: ImageStreamState = requireImageStreamState(image);
    const faceHandle: NativeFaceResultHandle | null = faces === undefined
      ? null
      : requireFaceResultState(faces).handle;
    return native.updateFaceCaptureSession(capture.handle, stream.handle, faceHandle, frameId, timestampMs);
  }

  public getResults(): FaceCaptureResult[] {
    return native.getFaceCaptureResults(requireFaceCaptureState(this).handle);
  }

  public finish(): FaceCaptureProgress {
    return native.finishFaceCaptureSession(requireFaceCaptureState(this).handle);
  }

  public reset(): void {
    native.resetFaceCaptureSession(requireFaceCaptureState(this).handle);
  }

  public close(): void {
    const state: FaceCaptureSessionState | undefined = faceCaptureStates.get(this);
    if (state === undefined || state.closed) return;
    native.releaseFaceCaptureSession(state.handle);
    state.closed = true;
  }

  public isClosed(): boolean {
    const state: FaceCaptureSessionState | undefined = faceCaptureStates.get(this);
    return state === undefined || state.closed;
  }
}

export class InspireFace {
  public static getErrorMessage(code: number): string {
    return native.getErrorMessage(code);
  }

  public static validateResourcePack(resourcePath: string): ResourcePackInfo {
    return native.validateResourcePack(resourcePath);
  }

  public static launch(resourcePath: string): void {
    native.launch(resourcePath);
  }

  public static reload(resourcePath: string): void {
    native.reload(resourcePath);
  }

  public static terminate(): void {
    native.terminate();
  }

  public static isLaunched(): boolean {
    return native.isLaunched();
  }

  public static getVersion(): Version {
    return native.getVersion();
  }

  public static getCapiLevel(): number {
    return native.getCapiLevel();
  }

  public static getSupportedPixelLevels(): Int32Array {
    return native.getSupportedPixelLevels();
  }

  public static switchLandmarkEngine(engine: LandmarkEngine): void {
    native.switchLandmarkEngine(engine);
  }

  public static createSession(options: SessionOptions = {}): Session {
    return new Session(options);
  }

  public static createImageStream(data: Uint8Array, width: number, height: number, format: ImageFormat,
    rotation: Rotation = Rotation.DEGREE_0): ImageStream {
    return new ImageStream(data, width, height, format, rotation);
  }

  public static getDenseLandmarks(face: TrackedFace): Float32Array {
    return native.getDenseLandmarks(face.token);
  }

  public static getFiveKeyPoints(face: TrackedFace): Float32Array {
    return native.getFiveKeyPoints(face.token);
  }

  public static getFeatureLength(): number {
    return native.getFeatureLength();
  }

  public static compareFeatures(first: Float32Array, second: Float32Array): number {
    return native.compareFeatures(first, second);
  }

  public static getRecommendedThreshold(): number {
    return native.getRecommendedThreshold();
  }

  public static similarityToPercentage(similarity: number): number {
    return native.similarityToPercentage(similarity);
  }

  public static getSimilarityConverter(): SimilarityConverterConfig {
    return native.getSimilarityConverter();
  }

  public static updateSimilarityConverter(config: SimilarityConverterConfig): void {
    native.updateSimilarityConverter(config);
  }

  public static getComponentVersion(component: ComponentType): ComponentVersion {
    return native.getComponentVersion(component);
  }

  public static getComponentVersions(): string {
    return native.getComponentVersions();
  }

  public static getDiagnosticInformation(): string {
    return native.getDiagnosticInformation();
  }

  public static getExtendedInformation(): string {
    return native.getExtendedInformation();
  }

  public static isRgaEnabled(): boolean {
    return native.queryRgaEnabled();
  }

  public static setRgaDmaHeapPath(path: string): void {
    native.setRgaDmaHeapPath(path);
  }

  public static getRgaDmaHeapPath(): string {
    return native.getRgaDmaHeapPath();
  }

  public static switchImageProcessingBackend(backend: ImageProcessingBackend): void {
    native.switchImageProcessingBackend(backend);
  }

  public static setImageProcessAlignedWidth(width: number): void {
    native.setImageProcessAlignedWidth(width);
  }

  /** Returns UNSUPPORTED on HarmonyOS builds because CoreML is an Apple-only backend. */
  public static setCoreMlInferenceMode(mode: CoreMlInferenceMode): void {
    native.setCoreMlInferenceMode(mode);
  }

  /** CUDA calls are retained for API parity and return UNSUPPORTED on the standard HarmonyOS package. */
  public static setCudaDeviceId(deviceId: number): void {
    native.setCudaDeviceId(deviceId);
  }

  public static getCudaDeviceId(): number {
    return native.getCudaDeviceId();
  }

  public static printCudaDeviceInfo(): void {
    native.printCudaDeviceInfo();
  }

  public static getCudaDeviceCount(): number {
    return native.getCudaDeviceCount();
  }

  public static isCudaSupported(): boolean {
    return native.isCudaSupported();
  }

  public static getDebugResourceCounts(): DebugResourceCounts {
    return native.getDebugResourceCounts();
  }

  public static showDebugResourceStatistics(): void {
    native.showDebugResourceStatistics();
  }

  public static setLogLevel(level: LogLevel): void {
    native.setLogLevel(level);
  }

  public static disableLog(): void {
    native.disableLog();
  }

  public static log(level: LogLevel, message: string): void {
    native.logPrint(level, message);
  }
}

export class FeatureHub {
  public static enable(config: FeatureHubConfiguration = {}): void {
    native.featureHubEnable(config);
  }

  public static disable(): void {
    native.featureHubDisable();
  }

  public static viewTable(): void {
    native.featureHubViewTable();
  }

  public static setSearchThreshold(threshold: number): void {
    native.featureHubSetThreshold(threshold);
  }

  public static insert(feature: Float32Array, id: bigint = -1n): bigint {
    return native.featureHubInsert(id, feature);
  }

  public static update(id: bigint, feature: Float32Array): void {
    native.featureHubUpdate(id, feature);
  }

  public static remove(id: bigint): void {
    native.featureHubRemove(id);
  }

  public static get(id: bigint): Float32Array {
    return native.featureHubGet(id);
  }

  public static getCount(): number {
    return native.featureHubGetCount();
  }

  public static getIds(): bigint[] {
    return native.featureHubGetIds();
  }

  public static search(feature: Float32Array): FeatureHubSearchResult {
    return native.featureHubSearch(feature);
  }

  public static searchTopK(feature: Float32Array, topK: number): FeatureHubTopKResult {
    return native.featureHubSearchTopK(feature, topK);
  }
}
