import ctypes

import numpy as np
from .core import *
from typing import Tuple, List
from dataclasses import dataclass
from loguru import logger
from .utils import ResourceManager
from .utils.resource import set_use_oss_download
from . import herror as errcode
# Exception system
from .exception import (
    check_error, validate_image_format, validate_feature_data, 
    validate_session_initialized, handle_c_api_errors,
    InspireFaceError, InvalidInputError, SystemNotReadyError, 
    ProcessingError, ResourceError, HardwareError, FeatureHubError
)

# If True, the latest model will not be verified
IGNORE_VERIFICATION_OF_THE_LATEST_MODEL = False

def ignore_check_latest_model(ignore: bool):
    global IGNORE_VERIFICATION_OF_THE_LATEST_MODEL
    IGNORE_VERIFICATION_OF_THE_LATEST_MODEL = ignore

def use_oss_download(use_oss: bool = True):
    """Enable OSS download instead of ModelScope (for backward compatibility)
    
    Args:
        use_oss (bool): If True, use OSS download; if False, use ModelScope (default)
    """
    set_use_oss_download(use_oss)

class ImageStream(object):
    """
    ImageStream class handles the conversion of image data from various sources into a format compatible with the InspireFace library.
    It allows loading image data from numpy arrays, buffer objects, and directly from OpenCV images.
    """

    @staticmethod
    def load_from_cv_image(image: np.ndarray, stream_format=HF_STREAM_BGR, rotation=HF_CAMERA_ROTATION_0):
        """
        Load image data from an OpenCV image (numpy ndarray).

        Args:
            image (np.ndarray): The image data as a numpy array.
            stream_format (int): The format of the image data (e.g., BGR, RGB).
            rotation (int): The rotation angle to be applied to the image data.

        Returns:
            ImageStream: An instance of the ImageStream class initialized with the provided image data.

        Raises:
            InvalidInputError: If the image does not have 3 or 4 channels.
        """
        validate_image_format(image, "Load from CV image")
        h, w, c = image.shape
        return ImageStream(image, w, h, stream_format, rotation)

    @staticmethod
    def load_from_ndarray(data: np.ndarray, width: int, height: int, stream_format: int, rotation: int):
        """
        Load image data from a numpy array specifying width and height explicitly.

        Args:
            data (np.ndarray): The raw image data.
            width (int): The width of the image.
            height (int): The height of the image.
            stream_format (int): The format of the image data.
            rotation (int): The rotation angle to be applied to the image data.

        Returns:
            ImageStream: An instance of the ImageStream class.
        """
        return ImageStream(data, width, height, stream_format, rotation)

    @staticmethod
    def load_from_buffer(data, width: int, height: int, stream_format: int, rotation: int):
        """
        Load image data from a buffer (like bytes or bytearray).

        Args:
            data: The buffer containing the image data.
            width (int): The width of the image.
            height (int): The height of the image.
            stream_format (int): The format of the image data.
            rotation (int): The rotation angle to be applied to the image data.

        Returns:
            ImageStream: An instance of the ImageStream class.
        """
        return ImageStream(data, width, height, stream_format, rotation)

    def __init__(self, data, width: int, height: int, stream_format: int, rotation: int):
        """
        Initialize the ImageStream object with provided data and configuration.

        Args:
            data: The image data (numpy array or buffer).
            width (int): The width of the image.
            height (int): The height of the image.
            stream_format (int): The format of the image data.
            rotation (int): The rotation applied to the image.

        Raises:
            ResourceError: If there is an error in creating the image stream.
        """
        self.rotate = rotation
        self.data_format = stream_format
        if isinstance(data, np.ndarray):
            data_ptr = ctypes.cast(data.ctypes.data, ctypes.POINTER(ctypes.c_uint8))
        else:
            data_ptr = ctypes.cast(data, ctypes.POINTER(ctypes.c_uint8))
        image_struct = HFImageData()
        image_struct.data = data_ptr
        image_struct.width = width
        image_struct.height = height
        image_struct.format = self.data_format
        image_struct.rotation = self.rotate
        self._handle = HFImageStream()
        ret = HFCreateImageStream(PHFImageData(image_struct), self._handle)
        check_error(ret, "Create ImageStream", width=width, height=height, format=stream_format)

    def write_to_file(self, file_path: str):
        """
        Write the image stream to a file. Like PATH/image.jpg
        """
        ret = HFDeBugImageStreamDecodeSave(self._handle, file_path)
        check_error(ret, "Write ImageStream to file", file_path=file_path)

    def release(self):
        """
        Release the resources associated with the ImageStream.

        Logs an error if the release fails.
        """
        if self._handle is not None:
            ret = HFReleaseImageStream(self._handle)
            if ret != 0:
                logger.warning(f"Failed to release ImageStream: error code {ret}")

    def __del__(self):
        """
        Ensure that resources are released when the ImageStream object is garbage collected.
        """
        self.release()

    def debug_show(self):
        """
        Display the image using a debug function provided by the library.
        """
        HFDeBugImageStreamImShow(self._handle)

    @property
    def handle(self):
        """
        Return the internal handle of the image stream.
        Returns:
        The handle to the internal image stream, used for interfacing with the underlying C/C++ library.
        """
        return self._handle


# == Session API ==

@dataclass
class FaceExtended:
    """
    A data class to hold extended face information with confidence levels for various attributes.

    Attributes:
        rgb_liveness_confidence (float): Confidence level of RGB-based liveness detection.
        mask_confidence (float): Confidence level of mask detection on the face.
        quality_confidence (float): Confidence level of the overall quality of the face capture.
    """
    rgb_liveness_confidence: float
    mask_confidence: float
    quality_confidence: float
    left_eye_status_confidence: float
    right_eye_status_confidence: float
    action_normal: int
    action_jaw_open: int
    action_shake: int
    action_blink: int
    action_head_raise: int
    race: int
    gender: int
    age_bracket: int
    emotion: int

    def __repr__(self) -> str:
        return f"FaceExtended(rgb_liveness_confidence={self.rgb_liveness_confidence}, mask_confidence={self.mask_confidence}, quality_confidence={self.quality_confidence}, left_eye_status_confidence={self.left_eye_status_confidence}, right_eye_status_confidence={self.right_eye_status_confidence}, action_normal={self.action_normal}, action_jaw_open={self.action_jaw_open}, action_shake={self.action_shake}, action_blink={self.action_blink}, action_head_raise={self.action_head_raise}, race={self.race}, gender={self.gender}, age_bracket={self.age_bracket}, emotion={self.emotion})"


class FaceInformation:
    """
    Holds detailed information about a detected face including location and orientation.

    Attributes:
        track_id (int): Unique identifier for tracking the face across frames.
        location (Tuple): Coordinates of the face in the form (x, y, width, height).
        roll (float): Roll angle of the face.
        yaw (float): Yaw angle of the face.
        pitch (float): Pitch angle of the face.
        _token (HFFaceBasicToken): A token containing low-level details about the face.
        _feature (np.array, optional): An optional numpy array holding the facial feature data.

    Methods:
        __init__: Initializes a new instance of FaceInformation.
    """

    def __init__(self,
                 track_id: int,
                 track_count: int,
                 detection_confidence: float,
                 location: Tuple,
                 roll: float,
                 yaw: float,
                 pitch: float,
                 _token: HFFaceBasicToken,
                 _feature: np.array = None):
        self.track_id = track_id
        self.track_count = track_count
        self.detection_confidence = detection_confidence
        self.location = location
        self.roll = roll
        self.yaw = yaw
        self.pitch = pitch

        # Calculate the required buffer size for the face token and copy it.
        token_size = HInt32()
        HFGetFaceBasicTokenSize(HPInt32(token_size))
        buffer_size = token_size.value
        self.buffer = create_string_buffer(buffer_size)
        ret = HFCopyFaceBasicToken(_token, self.buffer, token_size)
        check_error(ret, "Copy face basic token", track_id=track_id)

        # Store the copied token.
        self._token = HFFaceBasicToken()
        self._token.size = buffer_size
        self._token.data = cast(addressof(self.buffer), c_void_p)

    def __repr__(self) -> str:
        return f"FaceInformation(track_id={self.track_id}, track_count={self.track_count}, detection_confidence={self.detection_confidence}, location={self.location}, roll={self.roll}, yaw={self.yaw}, pitch={self.pitch})"


@dataclass
class SessionCustomParameter:
    """
    A data class for configuring the optional parameters in a face recognition session.

    Attributes are set to False by default and can be enabled as needed.

    Methods:
        _c_struct: Converts the Python attributes to a C-compatible structure for session configuration.
    """
    enable_recognition: bool = False
    enable_liveness: bool = False
    enable_ir_liveness: bool = False
    enable_mask_detect: bool = False
    enable_face_attribute: bool = False
    enable_face_quality: bool = False
    enable_interaction_liveness: bool = False
    enable_face_emotion: bool = False

    def _c_struct(self):
        """
        Creates a C structure from the current state of the instance.

        Returns:
            HFSessionCustomParameter: The corresponding C structure with proper type conversions.
        """
        custom_param = HFSessionCustomParameter(
            enable_recognition=int(self.enable_recognition),
            enable_liveness=int(self.enable_liveness),
            enable_ir_liveness=int(self.enable_ir_liveness),
            enable_mask_detect=int(self.enable_mask_detect),
            enable_face_attribute=int(self.enable_face_attribute),
            enable_face_quality=int(self.enable_face_quality),
            enable_interaction_liveness=int(self.enable_interaction_liveness),
            enable_face_emotion=int(self.enable_face_emotion)
        )

        return custom_param

    def __repr__(self) -> str:
        return f"SessionCustomParameter(enable_recognition={self.enable_recognition}, enable_liveness={self.enable_liveness}, enable_ir_liveness={self.enable_ir_liveness}, enable_mask_detect={self.enable_mask_detect}, enable_face_attribute={self.enable_face_attribute}, enable_face_quality={self.enable_face_quality}, enable_interaction_liveness={self.enable_interaction_liveness}, enable_face_emotion={self.enable_face_emotion})"


class InspireFaceSession(object):
    """
    Manages a session for face detection and recognition processes using the InspireFace library.

    Attributes:
        multiple_faces (HFMultipleFaceData): Stores data about multiple detected faces during the session.
        _sess (HFSession): The handle to the underlying library session.
        param (int or SessionCustomParameter): Configuration parameters or flags for the session.

    """

    def __init__(self, param, detect_mode: int = HF_DETECT_MODE_ALWAYS_DETECT,
                 max_detect_num: int = 10, detect_pixel_level=-1, track_by_detect_mode_fps=-1):
        """
        Initializes a new session with the provided configuration parameters.
        
        Args:
            param (int or SessionCustomParameter): Configuration parameters or flags.
            detect_mode (int): Detection mode to be used (e.g., image-based detection).
            max_detect_num (int): Maximum number of faces to detect.
            
        Raises:
            SystemNotReadyError: If InspireFace is not launched.
            ProcessingError: If session creation fails.
        """
        # Initialize _sess to None first to prevent AttributeError in __del__
        self._sess = None
        self.multiple_faces = None
        self.param = param
        
        # If InspireFace is not initialized, run launch() use Pikachu model
        if not query_launch_status():
            ret = launch()
            if not ret:
                raise SystemNotReadyError("Failed to launch InspireFace automatically")

        self._sess = HFSession()
        
        if isinstance(self.param, SessionCustomParameter):
            ret = HFCreateInspireFaceSession(self.param._c_struct(), detect_mode, max_detect_num, detect_pixel_level,
                                             track_by_detect_mode_fps, self._sess)
        elif isinstance(self.param, int):
            ret = HFCreateInspireFaceSessionOptional(self.param, detect_mode, max_detect_num, detect_pixel_level,
                                                     track_by_detect_mode_fps, self._sess)
        else:
            raise InvalidInputError("Session parameter must be SessionCustomParameter or int", 
                                   context={'param_type': type(self.param).__name__})
        
        check_error(ret, "Create InspireFace session", 
                   detect_mode=detect_mode, max_detect_num=max_detect_num)

    @handle_c_api_errors("Face detection")
    def face_detection(self, image) -> List[FaceInformation]:
        """
        Detects faces in the given image and returns a list of FaceInformation objects containing detailed face data.
        
        Args:
            image (np.ndarray or ImageStream): The image in which to detect faces.
            
        Returns:
            List[FaceInformation]: A list of detected face information.
            
        Raises:
            ResourceError: If session is not initialized.
            ProcessingError: If face detection fails.
        """
        validate_session_initialized(self, "Face detection")
        stream = self._get_image_stream(image)
        self.multiple_faces = HFMultipleFaceData()
        ret = HFExecuteFaceTrack(self._sess, stream.handle,
                                PHFMultipleFaceData(self.multiple_faces))
        check_error(ret, "Execute face tracking")

        if self.multiple_faces.detectedNum > 0:
            boxes = self._get_faces_boundary_boxes()
            track_ids = self._get_faces_track_ids()
            euler_angle = self._get_faces_euler_angle()
            tokens = self._get_faces_tokens()
            track_counts = self._get_faces_track_counts()

            infos = list()
            for idx in range(self.multiple_faces.detectedNum):
                top_left = (boxes[idx][0], boxes[idx][1])
                bottom_right = (boxes[idx][0] + boxes[idx][2], boxes[idx][1] + boxes[idx][3])
                roll = euler_angle[idx][0]
                yaw = euler_angle[idx][1]
                pitch = euler_angle[idx][2]
                track_id = track_ids[idx]
                _token = tokens[idx]
                detection_confidence = self.multiple_faces.detConfidence[idx]
                track_count = track_counts[idx]

                info = FaceInformation(
                    location=(top_left[0], top_left[1], bottom_right[0], bottom_right[1]),
                    roll=roll,
                    yaw=yaw,
                    pitch=pitch,
                    track_id=track_id,
                    track_count=track_count,
                    _token=_token,
                    detection_confidence=detection_confidence,
                )
                infos.append(info)

            return infos
        else:
            return []
        
    def get_face_five_key_points(self, single_face: FaceInformation):
        """Get five key points for a detected face"""
        validate_session_initialized(self, "Get face five key points")
        num_landmarks = 5
        landmarks_array = (HPoint2f * num_landmarks)()
        ret = HFGetFaceFiveKeyPointsFromFaceToken(single_face._token, landmarks_array, num_landmarks)
        check_error(ret, "Get face five key points", track_id=single_face.track_id)

        landmark = []
        for point in landmarks_array:
            landmark.append(point.x)
            landmark.append(point.y)

        return np.asarray(landmark).reshape(-1, 2)

    def get_face_dense_landmark(self, single_face: FaceInformation):
        """Get dense landmarks for a detected face"""
        validate_session_initialized(self, "Get face dense landmark")
        num_landmarks = HInt32()
        HFGetNumOfFaceDenseLandmark(byref(num_landmarks))
        landmarks_array = (HPoint2f * num_landmarks.value)()
        ret = HFGetFaceDenseLandmarkFromFaceToken(single_face._token, landmarks_array, num_landmarks)
        check_error(ret, "Get face dense landmark", track_id=single_face.track_id)

        landmark = []
        for point in landmarks_array:
            landmark.append(point.x)
            landmark.append(point.y)

        return np.asarray(landmark).reshape(-1, 2)
    
    def print_track_cost_spend(self):
        """Print tracking cost statistics"""
        validate_session_initialized(self, "Print track cost spend")
        ret = HFSessionPrintTrackCostSpend(self._sess)
        check_error(ret, "Print track cost spend")

    def set_enable_track_cost_spend(self, enable: bool):
        """Enable or disable track cost spend monitoring"""
        validate_session_initialized(self, "Set enable track cost spend")
        ret = HFSessionSetEnableTrackCostSpend(self._sess, enable)
        check_error(ret, "Set enable track cost spend", enable=enable)
    
    def set_detection_confidence_threshold(self, threshold: float):
        """
        Sets the detection confidence threshold for the face detection session.

        Args:
            threshold (float): The confidence threshold for face detection.
        """
        validate_session_initialized(self, "Set detection confidence threshold")
        ret = HFSessionSetFaceDetectThreshold(self._sess, threshold)
        check_error(ret, "Set detection confidence threshold", threshold=threshold)

    def set_track_preview_size(self, size=192):
        """
        Sets the preview size for the face tracking session.

        Args:
            size (int, optional): The size of the preview area for face tracking. Default is 192.
        """
        validate_session_initialized(self, "Set track preview size")
        ret = HFSessionSetTrackPreviewSize(self._sess, size)
        check_error(ret, "Set track preview size", size=size)

    def set_filter_minimum_face_pixel_size(self, min_size=32):
        """Set minimum face pixel size filter"""
        validate_session_initialized(self, "Set filter minimum face pixel size")
        ret = HFSessionSetFilterMinimumFacePixelSize(self._sess, min_size)
        check_error(ret, "Set filter minimum face pixel size", min_size=min_size)

    def set_track_mode_smooth_ratio(self, ratio=0.025):
        """Set track mode smooth ratio"""
        validate_session_initialized(self, "Set track mode smooth ratio")
        ret = HFSessionSetTrackModeSmoothRatio(self._sess, ratio)
        check_error(ret, "Set track mode smooth ratio", ratio=ratio)

    def set_track_mode_num_smooth_cache_frame(self, num=15):
        """Set track mode number of smooth cache frames"""
        validate_session_initialized(self, "Set track mode num smooth cache frame")
        ret = HFSessionSetTrackModeNumSmoothCacheFrame(self._sess, num)
        check_error(ret, "Set track mode num smooth cache frame", num=num)

    def set_track_model_detect_interval(self, num=20):
        """Set track model detect interval"""
        validate_session_initialized(self, "Set track model detect interval")
        ret = HFSessionSetTrackModeDetectInterval(self._sess, num)
        check_error(ret, "Set track model detect interval", num=num)

    def set_landmark_augmentation_num(self, num=1):
        """Set landmark augmentation number"""
        validate_session_initialized(self, "Set landmark augmentation num")
        ret = HFSessionSetLandmarkAugmentationNum(self._sess, num)
        check_error(ret, "Set landmark augmentation num", num=num)

    def set_track_lost_recovery_mode(self, value=False):
        """Set track lost recovery mode"""
        validate_session_initialized(self, "Set track lost recovery mode")
        ret = HFSessionSetTrackLostRecoveryMode(self._sess, value)
        check_error(ret, "Set track lost recovery mode", value=value)

    @handle_c_api_errors("Face pipeline processing")
    def face_pipeline(self, image, faces: List[FaceInformation], exec_param) -> List[FaceExtended]:
        """
        Processes detected faces to extract additional attributes based on the provided execution parameters.

        Args:
            image (np.ndarray or ImageStream): The image from which faces are detected.
            faces (List[FaceInformation]): A list of FaceInformation objects containing detected face data.
            exec_param (SessionCustomParameter or int): Custom parameters for processing faces.

        Returns:
            List[FaceExtended]: A list of FaceExtended objects with updated attributes like mask confidence, liveness, etc.
        """
        validate_session_initialized(self, "Face pipeline processing")
        stream = self._get_image_stream(image)
        fn, pm, flag = self._get_processing_function_and_param(exec_param)
        tokens = [face._token for face in faces]
        tokens_array = (HFFaceBasicToken * len(tokens))(*tokens)
        tokens_ptr = cast(tokens_array, PHFFaceBasicToken)

        multi_faces = HFMultipleFaceData()
        multi_faces.detectedNum = len(tokens)
        multi_faces.tokens = tokens_ptr
        ret = fn(self._sess, stream.handle, PHFMultipleFaceData(multi_faces), pm)

        check_error(ret, "Face pipeline processing", num_faces=len(faces))

        extends = [FaceExtended(-1.0, -1.0, -1.0, -1.0, -1.0, 0, 0, 0, 0, 0, -1, -1, -1, -1) for _ in range(len(faces))]
        self._update_mask_confidence(exec_param, flag, extends)
        self._update_rgb_liveness_confidence(exec_param, flag, extends)
        self._update_face_quality_confidence(exec_param, flag, extends)
        self._update_face_attribute_confidence(exec_param, flag, extends)
        self._update_face_interact_confidence(exec_param, flag, extends)
        self._update_face_emotion_confidence(exec_param, flag, extends)

        return extends

    @handle_c_api_errors("Face feature extraction")
    def face_feature_extract(self, image, face_information: FaceInformation):
        """
        Extracts facial features from a specified face within an image for recognition or comparison purposes.

        Args:
            image (np.ndarray or ImageStream): The image from which the face features are to be extracted.
            face_information (FaceInformation): The FaceInformation object containing the details of the face.

        Returns:
            np.ndarray: A numpy array containing the extracted facial features, or None if the extraction fails.
        """
        validate_session_initialized(self, "Face feature extraction")
        stream = self._get_image_stream(image)
        feature_length = HInt32()
        HFGetFeatureLength(byref(feature_length))

        feature = np.zeros((feature_length.value,), dtype=np.float32)
        ret = HFFaceFeatureExtractCpy(self._sess, stream.handle, face_information._token,
                                      feature.ctypes.data_as(ctypes.POINTER(HFloat)))

        check_error(ret, "Face feature extraction", track_id=face_information.track_id)
        return feature

    @staticmethod
    def _get_image_stream(image):
        """Convert image to ImageStream if needed"""
        if isinstance(image, np.ndarray):
            return ImageStream.load_from_cv_image(image)
        elif isinstance(image, ImageStream):
            return image
        else:
            raise InvalidInputError("Image must be numpy.ndarray or ImageStream", 
                                   context={'input_type': type(image).__name__})

    @staticmethod
    def _get_processing_function_and_param(exec_param):
        """Get processing function and parameters"""
        if isinstance(exec_param, SessionCustomParameter):
            return HFMultipleFacePipelineProcess, exec_param._c_struct(), "object"
        elif isinstance(exec_param, int):
            return HFMultipleFacePipelineProcessOptional, exec_param, "bitmask"
        else:
            raise InvalidInputError("exec_param must be SessionCustomParameter or int",
                                   context={'param_type': type(exec_param).__name__})

    def _update_mask_confidence(self, exec_param, flag, extends):
        """Update mask confidence in extends list"""
        if (flag == "object" and exec_param.enable_mask_detect) or (
                flag == "bitmask" and exec_param & HF_ENABLE_MASK_DETECT):
            mask_results = HFFaceMaskConfidence()
            ret = HFGetFaceMaskConfidence(self._sess, PHFFaceMaskConfidence(mask_results))
            if ret == errcode.HSUCCEED:
                for i in range(mask_results.num):
                    extends[i].mask_confidence = mask_results.confidence[i]
            else:
                logger.warning(f"Failed to get mask confidence: error code {ret}")

    def _update_face_interact_confidence(self, exec_param, flag, extends):
        """Update face interaction confidence in extends list"""
        if (flag == "object" and exec_param.enable_interaction_liveness) or (
                flag == "bitmask" and exec_param & HF_ENABLE_INTERACTION):
            results = HFFaceInteractionState()
            ret = HFGetFaceInteractionStateResult(self._sess, PHFFaceInteractionState(results))
            if ret == errcode.HSUCCEED:
                for i in range(results.num):
                    extends[i].left_eye_status_confidence = results.leftEyeStatusConfidence[i]
                    extends[i].right_eye_status_confidence = results.rightEyeStatusConfidence[i]
            else:
                logger.warning(f"Failed to get face interaction state: error code {ret}")
                
            actions = HFFaceInteractionsActions()
            ret = HFGetFaceInteractionActionsResult(self._sess, PHFFaceInteractionsActions(actions))
            if ret == errcode.HSUCCEED:
                for i in range(results.num):
                    extends[i].action_normal = actions.normal[i]
                    extends[i].action_shake = actions.shake[i]
                    extends[i].action_jaw_open = actions.jawOpen[i]
                    extends[i].action_head_raise = actions.headRaise[i]
                    extends[i].action_blink = actions.blink[i]
            else:
                logger.warning(f"Failed to get face interaction actions: error code {ret}")

    def _update_face_emotion_confidence(self, exec_param, flag, extends):
        """Update face emotion confidence in extends list"""
        if (flag == "object" and exec_param.enable_face_emotion) or (
                flag == "bitmask" and exec_param & HF_ENABLE_FACE_EMOTION):
            emotion_results = HFFaceEmotionResult()
            ret = HFGetFaceEmotionResult(self._sess, PHFFaceEmotionResult(emotion_results))
            if ret == errcode.HSUCCEED:
                for i in range(emotion_results.num):
                    extends[i].emotion = emotion_results.emotion[i]
            else:
                logger.warning(f"Failed to get face emotion result: error code {ret}")

    def _update_rgb_liveness_confidence(self, exec_param, flag, extends: List[FaceExtended]):
        """Update RGB liveness confidence in extends list"""
        if (flag == "object" and exec_param.enable_liveness) or (
                flag == "bitmask" and exec_param & HF_ENABLE_LIVENESS):
            liveness_results = HFRGBLivenessConfidence()
            ret = HFGetRGBLivenessConfidence(self._sess, PHFRGBLivenessConfidence(liveness_results))
            if ret == errcode.HSUCCEED:
                for i in range(liveness_results.num):
                    extends[i].rgb_liveness_confidence = liveness_results.confidence[i]
            else:
                logger.warning(f"Failed to get RGB liveness confidence: error code {ret}")

    def _update_face_attribute_confidence(self, exec_param, flag, extends: List[FaceExtended]):
        """Update face attribute confidence in extends list"""
        if (flag == "object" and exec_param.enable_face_attribute) or (
                flag == "bitmask" and exec_param & HF_ENABLE_FACE_ATTRIBUTE):
            attribute_results = HFFaceAttributeResult()
            ret = HFGetFaceAttributeResult(self._sess, PHFFaceAttributeResult(attribute_results))
            if ret == errcode.HSUCCEED:
                for i in range(attribute_results.num):
                    extends[i].gender = attribute_results.gender[i]
                    extends[i].age_bracket = attribute_results.ageBracket[i]
                    extends[i].race = attribute_results.race[i]
            else:
                logger.warning(f"Failed to get face attribute result: error code {ret}")

    def _update_face_quality_confidence(self, exec_param, flag, extends: List[FaceExtended]):
        """Update face quality confidence in extends list"""
        if (flag == "object" and exec_param.enable_face_quality) or (
                flag == "bitmask" and exec_param & HF_ENABLE_QUALITY):
            quality_results = HFFaceQualityConfidence()
            ret = HFGetFaceQualityConfidence(self._sess, PHFFaceQualityConfidence(quality_results))
            if ret == errcode.HSUCCEED:
                for i in range(quality_results.num):
                    extends[i].quality_confidence = quality_results.confidence[i]
            else:
                logger.warning(f"Failed to get face quality confidence: error code {ret}")

    def _get_faces_boundary_boxes(self) -> List:
        """Get face boundary boxes from detection results"""
        num_of_faces = self.multiple_faces.detectedNum
        rects_ptr = self.multiple_faces.rects
        rects = [(rects_ptr[i].x, rects_ptr[i].y, rects_ptr[i].width, rects_ptr[i].height) for i in range(num_of_faces)]
        return rects

    def _get_faces_track_ids(self) -> List:
        """Get face track IDs from detection results"""
        num_of_faces = self.multiple_faces.detectedNum
        track_ids_ptr = self.multiple_faces.trackIds
        track_ids = [track_ids_ptr[i] for i in range(num_of_faces)]
        return track_ids

    def _get_faces_euler_angle(self) -> List:
        """Get face euler angles from detection results"""
        num_of_faces = self.multiple_faces.detectedNum
        euler_angle = self.multiple_faces.angles
        angles = [(euler_angle.roll[i], euler_angle.yaw[i], euler_angle.pitch[i]) for i in range(num_of_faces)]
        return angles
     
    def _get_faces_track_counts(self) -> List:
        """Get face track counts from detection results"""
        num_of_faces = self.multiple_faces.detectedNum
        track_counts_ptr = self.multiple_faces.trackCounts
        track_counts = [track_counts_ptr[i] for i in range(num_of_faces)]
        return track_counts

    def _get_faces_tokens(self) -> List[HFFaceBasicToken]:
        """Get face tokens from detection results"""
        num_of_faces = self.multiple_faces.detectedNum
        tokens_ptr = self.multiple_faces.tokens
        tokens = [tokens_ptr[i] for i in range(num_of_faces)]
        return tokens

    def release(self):
        """Release session resources"""
        if self._sess is not None:
            HFReleaseInspireFaceSession(self._sess)
            self._sess = None

    def __del__(self):
        self.release()


# == Global API ==

def _check_modelscope_availability():
    """
    Check if ModelScope is available when needed and provide helpful error message if not.
    
    Exits the program if ModelScope is needed but not available and OSS is not enabled.
    """
    import sys
    from .utils.resource import USE_OSS_DOWNLOAD
    
    # Dynamic check for ModelScope availability (don't rely on cached MODELSCOPE_AVAILABLE)
    modelscope_available = True
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        print("ModelScope import successful")
    except Exception as e:
        modelscope_available = False
        print(f"ModelScope import failed: {e}")
    
    if not USE_OSS_DOWNLOAD and not modelscope_available:
        print("ModelScope is not available, cannot download models!")
        print("\nPlease choose one of the following solutions:")
        print("1. Reinstall ModelScope with all dependencies:")
        print("   pip install --upgrade modelscope")
        print("\n2. Install missing dependencies manually:")
        print("   pip install filelock")
        print("\n3. Switch to OSS download mode:")
        print("   import inspireface as isf")
        print("   isf.use_oss_download(True)  # Execute before calling launch()")
        print("\nNote: OSS download requires stable international network connection")
        sys.exit(1)

def launch(model_name: str = "Pikachu", resource_path: str = None) -> bool:
    """
    Launches the InspireFace system with the specified resource directory.

    Args:
        model_name (str): the name of the model to use.
        resource_path (str): if None, use the default model path.

    Returns:
        bool: True if the system was successfully launched, False otherwise.

    Raises:
        SystemNotReadyError: If launch fails due to resource issues.
    """
    if resource_path is None:
        from .utils.resource import USE_OSS_DOWNLOAD
        
        # Check if ModelScope is available when needed
        _check_modelscope_availability()
        
        # Use ModelScope by default unless OSS is forced
        sm = ResourceManager(use_modelscope=not USE_OSS_DOWNLOAD)
        resource_path = sm.get_model(model_name, ignore_verification=IGNORE_VERIFICATION_OF_THE_LATEST_MODEL)
    path_c = String(bytes(resource_path, encoding="utf8"))
    ret = HFLaunchInspireFace(path_c)
    if ret != 0:
        if ret == errcode.HERR_ARCHIVE_REPETITION_LOAD:
            logger.warning("Duplicate loading was found")
            return True
        else:
            check_error(ret, "Launch InspireFace", model_name=model_name, resource_path=resource_path)
    return True

def pull_latest_model(model_name: str = "Pikachu") -> str:
    """
    Pulls the latest model from the resource manager.

    Args:
        model_name (str): the name of the model to use.

    Returns:
        str: Path to the downloaded model.
    """
    from .utils.resource import USE_OSS_DOWNLOAD
    
    # Check if ModelScope is available when needed
    _check_modelscope_availability()
    
    sm = ResourceManager(use_modelscope=not USE_OSS_DOWNLOAD)
    resource_path = sm.get_model(model_name, re_download=True)
    return resource_path

def reload(model_name: str = "Pikachu", resource_path: str = None) -> bool:
    """
    Reloads the InspireFace system with the specified resource directory.

    Args:
        model_name (str): the name of the model to use.
        resource_path (str): if None, use the default model path.

    Returns:
        bool: True if reload was successful.
    """
    if resource_path is None:
        from .utils.resource import USE_OSS_DOWNLOAD
        
        # Check if ModelScope is available when needed
        _check_modelscope_availability()
        
        sm = ResourceManager(use_modelscope=not USE_OSS_DOWNLOAD)
        resource_path = sm.get_model(model_name, ignore_verification=IGNORE_VERIFICATION_OF_THE_LATEST_MODEL)
    path_c = String(bytes(resource_path, encoding="utf8"))
    ret = HFReloadInspireFace(path_c)
    if ret != 0:
        if ret == errcode.HERR_ARCHIVE_REPETITION_LOAD:
            logger.warning("Duplicate loading was found")
            return True
        else:
            check_error(ret, "Reload InspireFace", model_name=model_name, resource_path=resource_path)
    return True

def terminate() -> bool:
    """
    Terminates the InspireFace system.

    Returns:
        bool: True if the system was successfully terminated, False otherwise.
    """
    ret = HFTerminateInspireFace()
    check_error(ret, "Terminate InspireFace")
    return True

def query_launch_status() -> bool:
    """
    Queries the launch status of the InspireFace SDK.

    Returns:
        bool: True if InspireFace is launched, False otherwise.
    """
    status = HInt32()
    ret = HFQueryInspireFaceLaunchStatus(byref(status))
    check_error(ret, "Query launch status")
    return status.value == 1

def switch_landmark_engine(engine: int):
    """Switch landmark engine"""
    ret = HFSwitchLandmarkEngine(engine)
    check_error(ret, "Switch landmark engine", engine=engine)
    return True

def switch_image_processing_backend(backend: int):
    """Switch image processing backend"""
    ret = HFSwitchImageProcessingBackend(backend)
    check_error(ret, "Switch image processing backend", backend=backend)
    return True

def set_image_process_aligned_width(width: int):
    """Set the image process aligned width"""
    ret = HFSetImageProcessAlignedWidth(width)
    check_error(ret, "Set image process aligned width", width=width)
    return True

@dataclass
class FeatureHubConfiguration:
    """
    Configuration settings for managing the feature hub, including database and search settings.

    Attributes:
        primary_key_mode (int): Primary key mode for the database.
        enable_persistence (bool): Flag to indicate if the database should be used.
        persistence_db_path (str): Path to the database file.
        search_threshold (float): The threshold value for considering a match.
        search_mode (int): The mode of searching in the database.
    """
    primary_key_mode: int
    enable_persistence: bool
    persistence_db_path: str
    search_threshold: float
    search_mode: int

    def _c_struct(self):
        """
        Converts the data class attributes to a C-compatible structure for use in the InspireFace SDK.

        Returns:
            HFFeatureHubConfiguration: A C-structure for feature hub configuration.
        """
        return HFFeatureHubConfiguration(
            primaryKeyMode=self.primary_key_mode,
            enablePersistence=int(self.enable_persistence),
            persistenceDbPath=String(bytes(self.persistence_db_path, encoding="utf8")),
            searchThreshold=self.search_threshold,
            searchMode=self.search_mode
        )


def feature_hub_enable(config: FeatureHubConfiguration) -> bool:
    """
    Enables the feature hub with the specified configuration.

    Args:
        config (FeatureHubConfiguration): Configuration settings for the feature hub.

    Returns:
        bool: True if successfully enabled, False otherwise.
    """
    ret = HFFeatureHubDataEnable(config._c_struct())
    check_error(ret, "Enable FeatureHub")
    return True


def feature_hub_disable() -> bool:
    """
    Disables the feature hub.

    Returns:
        bool: True if successfully disabled, False otherwise.
    """
    ret = HFFeatureHubDataDisable()
    check_error(ret, "Disable FeatureHub")
    return True


def feature_comparison(feature1: np.ndarray, feature2: np.ndarray) -> float:
    """
    Compares two facial feature arrays to determine their similarity.

    Args:
        feature1 (np.ndarray): The first feature array.
        feature2 (np.ndarray): The second feature array.

    Returns:
        float: A similarity score, where -1.0 indicates an error during comparison.
    """
    validate_feature_data(feature1, "Feature comparison")
    validate_feature_data(feature2, "Feature comparison")
    faces = [feature1, feature2]
    feats = []
    for face in faces:
        feature = HFFaceFeature()
        data_ptr = face.ctypes.data_as(HPFloat)
        feature.size = HInt32(face.size)
        feature.data = data_ptr
        feats.append(feature)

    comparison_result = HFloat()
    ret = HFFaceComparison(feats[0], feats[1], HPFloat(comparison_result))
    check_error(ret, "Face feature comparison")
    return float(comparison_result.value)


class FaceIdentity(object):
    """
    Represents an identity based on facial features, associating the features with a custom ID and a tag.

    Attributes:
        feature (np.ndarray): The facial features as a numpy array.
        id (int): A custom identifier for the face identity.

    Methods:
        __init__: Initializes a new instance of FaceIdentity.
        from_ctypes: Converts a C structure to a FaceIdentity instance.
        _c_struct: Converts the instance back to a compatible C structure.
    """

    def __init__(self, data: np.ndarray, id: int):
        """
        Initializes a new FaceIdentity instance with facial feature data, a custom identifier, and a tag.

        Args:
            data (np.ndarray): The facial feature data.
            id (int): A custom identifier for tracking or referencing the face identity.
        """
        validate_feature_data(data, "FaceIdentity initialization")
        self.feature = data
        self.id = id

    def __repr__(self) -> str:
        return f"FaceIdentity(id={self.id}, feature={self.feature})"

    @staticmethod
    def from_ctypes(raw_identity: HFFaceFeatureIdentity):
        """
        Converts a ctypes structure representing a face identity into a FaceIdentity object.

        Args:
            raw_identity (HFFaceFeatureIdentity): The ctypes structure containing the face identity data.

        Returns:
            FaceIdentity: An instance of FaceIdentity with data extracted from the ctypes structure.
        """
        feature_size = raw_identity.feature.contents.size
        feature_data_ptr = raw_identity.feature.contents.data
        feature_data = np.ctypeslib.as_array(cast(feature_data_ptr, HPFloat), (feature_size,))
        id_ = raw_identity.id

        return FaceIdentity(data=feature_data, id=id_)

    def _c_struct(self):
        """
        Converts this FaceIdentity instance into a C-compatible structure for use with InspireFace APIs.

        Returns:
            HFFaceFeatureIdentity: A C structure representing this face identity.
        """
        feature = HFFaceFeature()
        data_ptr = self.feature.ctypes.data_as(HPFloat)
        feature.size = HInt32(self.feature.size)
        feature.data = data_ptr
        return HFFaceFeatureIdentity(
            id=HFaceId(self.id),
            feature=PHFFaceFeature(feature)
        )


def feature_hub_set_search_threshold(threshold: float):
    """
    Sets the search threshold for face matching in the FeatureHub.

    Args:
        threshold (float): The similarity threshold for determining a match.
    """
    HFFeatureHubFaceSearchThresholdSetting(threshold)


def feature_hub_face_insert(face_identity: FaceIdentity) -> Tuple[bool, int]:
    """
    Inserts a face identity into the FeatureHub database.

    Args:
        face_identity (FaceIdentity): The face identity to insert.

    Returns:
        Tuple[bool, int]: (True, allocated_id) if the face identity was successfully inserted.
    """
    alloc_id = HFaceId()
    ret = HFFeatureHubInsertFeature(face_identity._c_struct(), HPFaceId(alloc_id))
    check_error(ret, "Insert face feature into FeatureHub", identity_id=face_identity.id)
    return True, int(alloc_id.value)


@dataclass
class SearchResult:
    """
    Represents the result of a face search operation with confidence level and the most similar face identity found.

    Attributes:
        confidence (float): The confidence score of the search result, indicating the similarity.
        similar_identity (FaceIdentity): The face identity that most closely matches the search query.
    """
    confidence: float
    similar_identity: FaceIdentity

    def __repr__(self) -> str:
        return f"SearchResult(confidence={self.confidence}, similar_identity={self.similar_identity})"

def feature_hub_face_search(data: np.ndarray) -> SearchResult:
    """
    Searches for the most similar face identity in the feature hub based on provided facial features.

    Args:
        data (np.ndarray): The facial feature data to search for.

    Returns:
        SearchResult: The search result containing the confidence and the most similar identity found.
    """
    validate_feature_data(data, "FeatureHub face search")
    feature = HFFaceFeature(size=HInt32(data.size), data=data.ctypes.data_as(HPFloat))
    confidence = HFloat()
    most_similar = HFFaceFeatureIdentity()
    ret = HFFeatureHubFaceSearch(feature, HPFloat(confidence), PHFFaceFeatureIdentity(most_similar))
    check_error(ret, "Search face in FeatureHub")
    
    if most_similar.id != -1:
        search_identity = FaceIdentity.from_ctypes(most_similar)
        return SearchResult(confidence=confidence.value, similar_identity=search_identity)
    else:
        none = FaceIdentity(np.zeros(0, dtype=np.float32), most_similar.id)
        return SearchResult(confidence=confidence.value, similar_identity=none)


def feature_hub_face_search_top_k(data: np.ndarray, top_k: int) -> List[Tuple]:
    """
    Searches for the top 'k' most similar face identities in the feature hub based on provided facial features.

    Args:
        data (np.ndarray): The facial feature data to search for.
        top_k (int): The number of top results to retrieve.

    Returns:
        List[Tuple]: A list of tuples, each containing the confidence and custom ID of the top results.
    """
    validate_feature_data(data, "FeatureHub face search top k")
    feature = HFFaceFeature(size=HInt32(data.size), data=data.ctypes.data_as(HPFloat))
    results = HFSearchTopKResults()
    ret = HFFeatureHubFaceSearchTopK(feature, top_k, PHFSearchTopKResults(results))
    outputs = []
    if ret == errcode.HSUCCEED:
        for idx in range(results.size):
            confidence = results.confidence[idx]
            id_ = results.ids[idx]
            outputs.append((confidence, id_))
    return outputs


def feature_hub_face_update(face_identity: FaceIdentity) -> bool:
    """
    Updates an existing face identity in the feature hub.

    Args:
        face_identity (FaceIdentity): The face identity to update.

    Returns:
        bool: True if the update was successful, False otherwise.

    Notes:
        Logs an error if the update operation fails.
    """
    ret = HFFeatureHubFaceUpdate(face_identity._c_struct())
    if ret != 0:
        logger.error(f"Failed to update face feature data in FeatureHub: {ret}")
        return False
    return True


def feature_hub_face_remove(custom_id: int) -> bool:
    """
    Removes a face identity from the feature hub using its custom ID.

    Args:
        custom_id (int): The custom ID of the face identity to remove.

    Returns:
        bool: True if the face was successfully removed, False otherwise.

    Notes:
        Logs an error if the removal operation fails.
    """
    ret = HFFeatureHubFaceRemove(HFaceId(custom_id))
    if ret != 0:
        logger.error(f"Failed to remove face feature data from FeatureHub: {ret}")
        return False
    return True


def feature_hub_get_face_identity(custom_id: int):
    """
    Retrieves a face identity from the feature hub using its custom ID.

    Args:
        custom_id (int): The custom ID of the face identity to retrieve.

    Returns:
        FaceIdentity: The face identity retrieved, or None if the operation fails.

    Notes:
        Logs an error if retrieving the face identity fails.
    """
    identify = HFFaceFeatureIdentity()
    ret = HFFeatureHubGetFaceIdentity(HFaceId(custom_id), PHFFaceFeatureIdentity(identify))
    check_error(ret, "Get face identity from FeatureHub", custom_id=custom_id)

    return FaceIdentity.from_ctypes(identify)


def feature_hub_get_face_count() -> int:
    """
    Retrieves the total count of face identities stored in the feature hub.

    Returns:
        int: The count of face identities.

    Notes:
        Logs an error if the operation to retrieve the count fails.
    """
    count = HInt32()
    ret = HFFeatureHubGetFaceCount(HPInt32(count))
    check_error(ret, "Get face count")

    return int(count.value)


def feature_hub_get_face_id_list() -> List[int]:
    """
    Retrieves a list of face IDs from the feature hub.

    Returns:
        List[int]: A list of face IDs.
    """
    ids = HFFeatureHubExistingIds()
    ptr = PHFFeatureHubExistingIds(ids)
    ret = HFFeatureHubGetExistingIds(ptr)
    if ret != 0:
        logger.error(f"Failed to get face id list: {ret}")
    return [int(ids.ids[i]) for i in range(ids.size)]

def view_table_in_terminal():
    """
    Displays the database table of face identities in the terminal.

    Notes:
        Logs an error if the operation to view the table fails.
    """
    ret = HFFeatureHubViewDBTable()
    check_error(ret, "View DB table")

def get_recommended_cosine_threshold() -> float:
    """
    Retrieves the recommended cosine threshold.
    """
    threshold = HFloat()
    HFGetRecommendedCosineThreshold(threshold)
    return float(threshold.value)

def get_similarity_converter_config() -> dict:
    """
    Retrieves the similarity converter configuration.
    """
    config = HFSimilarityConverterConfig()
    ret = HFGetCosineSimilarityConverter(PHFSimilarityConverterConfig(config))
    if ret != 0:
        logger.error(f"Failed to get cosine similarity converter config: {ret}")
    cfg = { 
        "threshold": config.threshold,
        "middleScore": config.middleScore,
        "steepness": config.steepness,
        "outputMin": config.outputMin,
        "outputMax": config.outputMax
    }
    return cfg

def set_similarity_converter_config(cfg: dict):
    """
    Sets the similarity converter configuration.
    """
    config = HFSimilarityConverterConfig()
    config.threshold = cfg["threshold"]
    config.middleScore = cfg["middleScore"]
    config.steepness = cfg["steepness"]
    config.outputMin = cfg["outputMin"]
    config.outputMax = cfg["outputMax"]
    HFUpdateCosineSimilarityConverter(config)

def cosine_similarity_convert_to_percentage(similarity: float) -> float:
    """
    Converts a cosine similarity score to a percentage similarity score.
    """
    result = HFloat()
    ret = HFCosineSimilarityConvertToPercentage(HFloat(similarity), HPFloat(result))
    if ret != 0:
        logger.error(f"Failed to convert cosine similarity to percentage: {ret}")
    return float(result.value)

def version() -> str:
    """
    Retrieves the version of the InspireFace library.

    Returns:
        str: The version string of the library.
    """
    ver = HFInspireFaceVersion()
    HFQueryInspireFaceVersion(PHFInspireFaceVersion(ver))
    return f"{ver.major}.{ver.minor}.{ver.patch}"


def set_logging_level(level: int) -> None:
    """
    Sets the logging level of the InspireFace library.

    Args:
        level (int): The level to set the logging to.
    """
    HFSetLogLevel(level)

def disable_logging() -> None:
    """
    Disables all logging from the InspireFace library.
    """
    HFLogDisable()

def show_system_resource_statistics():
    """
    Displays the system resource information.
    """
    HFDeBugShowResourceStatistics()

def switch_apple_coreml_inference_mode(mode: int):
    """
    Switches the Apple CoreML inference mode.
    """
    ret = HFSetAppleCoreMLInferenceMode(mode)
    if ret != 0:
        logger.error(f"Failed to set Apple CoreML inference mode: {ret}")
        return False
    return True

def set_expansive_hardware_rockchip_dma_heap_path(path: str):
    """
    Sets the path to the expansive hardware Rockchip DMA heap.
    """
    ret = HFSetExpansiveHardwareRockchipDmaHeapPath(path)
    check_error(ret, "Set expansive hardware Rockchip DMA heap path", path=path)

def query_expansive_hardware_rockchip_dma_heap_path() -> str:
    """
    Queries the path to the expansive hardware Rockchip DMA heap.
    """
    path = HString()
    ret = HFQueryExpansiveHardwareRockchipDmaHeapPath(path)
    check_error(ret, "Query expansive hardware Rockchip DMA heap path")
    return str(path.value)


def set_cuda_device_id(device_id: int):
    """
    Sets the CUDA device ID.
    """
    ret = HFSetCudaDeviceId(device_id)
    check_error(ret, "Set CUDA device ID", device_id=device_id)

def get_cuda_device_id() -> int:
    """
    Gets the CUDA device ID.
    """
    id = HInt32()
    ret = HFGetCudaDeviceId(id)
    check_error(ret, "Get CUDA device ID")
    return int(id.value)

def print_cuda_device_info():
    """
    Prints the CUDA device information.
    """
    HFPrintCudaDeviceInfo()
    
def get_num_cuda_devices() -> int:
    """
    Gets the number of CUDA devices.
    """
    num = HInt32()
    ret = HFGetNumCudaDevices(num)
    check_error(ret, "Get number of CUDA devices")
    return int(num.value)

def check_cuda_device_support() -> bool:
    """
    Checks if the CUDA device is supported.
    """
    is_support = HInt32()
    ret = HFCheckCudaDeviceSupport(is_support)
    check_error(ret, "Check CUDA device support")
    return bool(is_support.value)
