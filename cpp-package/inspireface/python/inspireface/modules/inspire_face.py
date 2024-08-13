import ctypes

import cv2
import numpy as np
from .core import *
from typing import Tuple, List
from dataclasses import dataclass
from loguru import logger


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
            Exception: If the image does not have 3 or 4 channels.
        """
        h, w, c = image.shape
        if c != 3 and c != 4:
            raise Exception("The channel must be 3 or 4.")
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
            Exception: If there is an error in creating the image stream.
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
        if ret != 0:
            raise Exception("Error in creating ImageStream")

    def release(self):
        """
        Release the resources associated with the ImageStream.

        Logs an error if the release fails.
        """
        if self._handle is not None:
            ret = HFReleaseImageStream(self._handle)
            if ret != 0:
                logger.error(f"Release ImageStream error: {ret}")

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
                 location: Tuple,
                 roll: float,
                 yaw: float,
                 pitch: float,
                 _token: HFFaceBasicToken,
                 _feature: np.array = None):
        self.track_id = track_id
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
        if ret != 0:
            logger.error("Failed to copy face basic token")

        # Store the copied token.
        self._token = HFFaceBasicToken()
        self._token.size = buffer_size
        self._token.data = cast(addressof(self.buffer), c_void_p)


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
            enable_interaction_liveness=int(self.enable_interaction_liveness)
        )

        return custom_param


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
            Exception: If session creation fails.
        """
        self.multiple_faces = None
        self._sess = HFSession()
        self.param = param
        if isinstance(self.param, SessionCustomParameter):
            ret = HFCreateInspireFaceSession(self.param._c_struct(), detect_mode, max_detect_num, detect_pixel_level,
                                             track_by_detect_mode_fps, self._sess)
        elif isinstance(self.param, int):
            ret = HFCreateInspireFaceSessionOptional(self.param, detect_mode, max_detect_num, detect_pixel_level,
                                                     track_by_detect_mode_fps, self._sess)
        else:
            raise NotImplemented("")
        if ret != 0:
            st = f"Create session error: {ret}"
            raise Exception(st)

    def face_detection(self, image) -> List[FaceInformation]:
        """
        Detects faces in the given image and returns a list of FaceInformation objects containing detailed face data.
        Args:
            image (np.ndarray or ImageStream): The image in which to detect faces.
        Returns:
            List[FaceInformation]: A list of detected face information.
        """
        stream = self._get_image_stream(image)
        self.multiple_faces = HFMultipleFaceData()
        ret = HFExecuteFaceTrack(self._sess, stream.handle,
                                 PHFMultipleFaceData(self.multiple_faces))
        if ret != 0:
            logger.error(f"Face detection error: ", {ret})
            return []

        if self.multiple_faces.detectedNum > 0:
            boxes = self._get_faces_boundary_boxes()
            track_ids = self._get_faces_track_ids()
            euler_angle = self._get_faces_euler_angle()
            tokens = self._get_faces_tokens()

            infos = list()
            for idx in range(self.multiple_faces.detectedNum):
                top_left = (boxes[idx][0], boxes[idx][1])
                bottom_right = (boxes[idx][0] + boxes[idx][2], boxes[idx][1] + boxes[idx][3])
                roll = euler_angle[idx][0]
                yaw = euler_angle[idx][1]
                pitch = euler_angle[idx][2]
                track_id = track_ids[idx]
                _token = tokens[idx]

                info = FaceInformation(
                    location=(top_left[0], top_left[1], bottom_right[0], bottom_right[1]),
                    roll=roll,
                    yaw=yaw,
                    pitch=pitch,
                    track_id=track_id,
                    _token=_token,
                )
                infos.append(info)

            return infos
        else:
            return []

    def get_face_dense_landmark(self, single_face: FaceInformation):
        num_landmarks = HInt32()
        HFGetNumOfFaceDenseLandmark(byref(num_landmarks))
        landmarks_array = (HPoint2f * num_landmarks.value)()
        ret = HFGetFaceDenseLandmarkFromFaceToken(single_face._token, landmarks_array, num_landmarks)
        if ret != 0:
            logger.error(f"An error occurred obtaining a dense landmark for a single face: {ret}")

        landmark = []
        for point in landmarks_array:
            landmark.append(point.x)
            landmark.append(point.y)

        return np.asarray(landmark).reshape(-1, 2)

    def set_track_preview_size(self, size=192):
        """
        Sets the preview size for the face tracking session.

        Args:
            size (int, optional): The size of the preview area for face tracking. Default is 192.

        Notes:
            If setting the preview size fails, an error is logged with the returned status code.
        """
        ret = HFSessionSetTrackPreviewSize(self._sess, size)
        if ret != 0:
            logger.error(f"Set track preview size error: {ret}")

    def set_filter_minimum_face_pixel_size(self, min_size=32):
        ret = HFSessionSetFilterMinimumFacePixelSize(self._sess, min_size)
        if ret != 0:
            logger.error(f"Set filter minimum face pixel size error: {ret}")

    def face_pipeline(self, image, faces: List[FaceInformation], exec_param) -> List[FaceExtended]:
        """
        Processes detected faces to extract additional attributes based on the provided execution parameters.

        Args:
            image (np.ndarray or ImageStream): The image from which faces are detected.
            faces (List[FaceInformation]): A list of FaceInformation objects containing detected face data.
            exec_param (SessionCustomParameter or int): Custom parameters for processing faces.

        Returns:
            List[FaceExtended]: A list of FaceExtended objects with updated attributes like mask confidence, liveness, etc.

        Notes:
            If the face pipeline processing fails, an error is logged and an empty list is returned.
        """
        stream = self._get_image_stream(image)
        fn, pm, flag = self._get_processing_function_and_param(exec_param)
        tokens = [face._token for face in faces]
        tokens_array = (HFFaceBasicToken * len(tokens))(*tokens)
        tokens_ptr = cast(tokens_array, PHFFaceBasicToken)

        multi_faces = HFMultipleFaceData()
        multi_faces.detectedNum = len(tokens)
        multi_faces.tokens = tokens_ptr
        ret = fn(self._sess, stream.handle, PHFMultipleFaceData(multi_faces), pm)

        if ret != 0:
            logger.error(f"Face pipeline error: {ret}")
            return []

        extends = [FaceExtended(-1.0, -1.0, -1.0, -1.0, -1.0, 0, 0, 0, 0, 0, -1, -1, -1) for _ in range(len(faces))]
        self._update_mask_confidence(exec_param, flag, extends)
        self._update_rgb_liveness_confidence(exec_param, flag, extends)
        self._update_face_quality_confidence(exec_param, flag, extends)
        self._update_face_attribute_confidence(exec_param, flag, extends)
        self._update_face_interact_confidence(exec_param, flag, extends)

        return extends

    def face_feature_extract(self, image, face_information: FaceInformation):
        """
        Extracts facial features from a specified face within an image for recognition or comparison purposes.

        Args:
            image (np.ndarray or ImageStream): The image from which the face features are to be extracted.
            face_information (FaceInformation): The FaceInformation object containing the details of the face.

        Returns:
            np.ndarray: A numpy array containing the extracted facial features, or None if the extraction fails.

        Notes:
            If the feature extraction process fails, an error is logged and None is returned.
        """
        stream = self._get_image_stream(image)
        feature_length = HInt32()
        HFGetFeatureLength(byref(feature_length))

        feature = np.zeros((feature_length.value,), dtype=np.float32)
        ret = HFFaceFeatureExtractCpy(self._sess, stream.handle, face_information._token,
                                      feature.ctypes.data_as(ctypes.POINTER(HFloat)))

        if ret != 0:
            logger.error(f"Face feature extract error: {ret}")
            return None

        return feature

    @staticmethod
    def _get_image_stream(image):
        if isinstance(image, np.ndarray):
            return ImageStream.load_from_cv_image(image)
        elif isinstance(image, ImageStream):
            return image
        else:
            raise NotImplemented("Place check input type.")

    @staticmethod
    def _get_processing_function_and_param(exec_param):
        if isinstance(exec_param, SessionCustomParameter):
            return HFMultipleFacePipelineProcess, exec_param._c_struct(), "object"
        elif isinstance(exec_param, int):
            return HFMultipleFacePipelineProcessOptional, exec_param, "bitmask"
        else:
            raise NotImplemented("Unsupported parameter type")

    def _update_mask_confidence(self, exec_param, flag, extends):
        if (flag == "object" and exec_param.enable_mask_detect) or (
                flag == "bitmask" and exec_param & HF_ENABLE_MASK_DETECT):
            mask_results = HFFaceMaskConfidence()
            ret = HFGetFaceMaskConfidence(self._sess, PHFFaceMaskConfidence(mask_results))
            if ret == 0:
                for i in range(mask_results.num):
                    extends[i].mask_confidence = mask_results.confidence[i]
            else:
                logger.error(f"Get mask result error: {ret}")

    def _update_face_interact_confidence(self, exec_param, flag, extends):
        if (flag == "object" and exec_param.enable_interaction_liveness) or (
                flag == "bitmask" and exec_param & HF_ENABLE_INTERACTION):
            results = HFFaceIntereactionState()
            ret = HFGetFaceIntereactionStateResult(self._sess, PHFFaceIntereactionState(results))
            if ret == 0:
                for i in range(results.num):
                    extends[i].left_eye_status_confidence = results.leftEyeStatusConfidence[i]
                    extends[i].right_eye_status_confidence = results.rightEyeStatusConfidence[i]
            else:
                logger.error(f"Get face interact result error: {ret}")
            actions = HFFaceIntereactionsActions()
            ret = HFGetFaceIntereactionActionsResult(self._sess, PHFFaceIntereactionsActions(actions))
            if ret == 0:
                for i in range(results.num):
                    extends[i].action_normal = actions.normal[i]
                    extends[i].action_shake = actions.shake[i]
                    extends[i].action_jaw_open = actions.jawOpen[i]
                    extends[i].action_head_raise = actions.headRiase[i]
                    extends[i].action_blink = actions.blink[i]
            else:
                logger.error(f"Get face action result error: {ret}")

    def _update_rgb_liveness_confidence(self, exec_param, flag, extends: List[FaceExtended]):
        if (flag == "object" and exec_param.enable_liveness) or (
                flag == "bitmask" and exec_param & HF_ENABLE_LIVENESS):
            liveness_results = HFRGBLivenessConfidence()
            ret = HFGetRGBLivenessConfidence(self._sess, PHFRGBLivenessConfidence(liveness_results))
            if ret == 0:
                for i in range(liveness_results.num):
                    extends[i].rgb_liveness_confidence = liveness_results.confidence[i]
            else:
                logger.error(f"Get rgb liveness result error: {ret}")

    def _update_face_attribute_confidence(self, exec_param, flag, extends: List[FaceExtended]):
        if (flag == "object" and exec_param.enable_face_attribute) or (
                flag == "bitmask" and exec_param & HF_ENABLE_FACE_ATTRIBUTE):
            attribute_results = HFFaceAttributeResult()
            ret = HFGetFaceAttributeResult(self._sess, PHFFaceAttributeResult(attribute_results))
            if ret == 0:
                for i in range(attribute_results.num):
                    extends[i].gender = attribute_results.gender[i]
                    extends[i].age_bracket = attribute_results.ageBracket[i]
                    extends[i].race = attribute_results.race[i]
            else:
                logger.error(f"Get face attribute result error: {ret}")

    def _update_face_quality_confidence(self, exec_param, flag, extends: List[FaceExtended]):
        if (flag == "object" and exec_param.enable_face_quality) or (
                flag == "bitmask" and exec_param & HF_ENABLE_QUALITY):
            quality_results = HFFaceQualityConfidence()
            ret = HFGetFaceQualityConfidence(self._sess, PHFFaceQualityConfidence(quality_results))
            if ret == 0:
                for i in range(quality_results.num):
                    extends[i].quality_confidence = quality_results.confidence[i]
            else:
                logger.error(f"Get quality result error: {ret}")

    def _get_faces_boundary_boxes(self) -> List:
        num_of_faces = self.multiple_faces.detectedNum
        rects_ptr = self.multiple_faces.rects
        rects = [(rects_ptr[i].x, rects_ptr[i].y, rects_ptr[i].width, rects_ptr[i].height) for i in range(num_of_faces)]

        return rects

    def _get_faces_track_ids(self) -> List:
        num_of_faces = self.multiple_faces.detectedNum
        track_ids_ptr = self.multiple_faces.trackIds
        track_ids = [track_ids_ptr[i] for i in range(num_of_faces)]

        return track_ids

    def _get_faces_euler_angle(self) -> List:
        num_of_faces = self.multiple_faces.detectedNum
        euler_angle = self.multiple_faces.angles
        angles = [(euler_angle.roll[i], euler_angle.yaw[i], euler_angle.pitch[i]) for i in range(num_of_faces)]

        return angles

    def _get_faces_tokens(self) -> List[HFFaceBasicToken]:
        num_of_faces = self.multiple_faces.detectedNum
        tokens_ptr = self.multiple_faces.tokens
        tokens = [tokens_ptr[i] for i in range(num_of_faces)]

        return tokens

    def release(self):
        if self._sess is not None:
            HFReleaseInspireFaceSession(self._sess)
            self._sess = None

    def __del__(self):
        self.release()


# == Global API ==
def launch(resource_path: str) -> bool:
    """
    Launches the InspireFace system with the specified resource directory.

    Args:
        resource_path (str): The file path to the resource directory necessary for operation.

    Returns:
        bool: True if the system was successfully launched, False otherwise.

    Notes:
        A specific error is logged if duplicate loading is detected or if there is any other launch failure.
    """
    path_c = String(bytes(resource_path, encoding="utf8"))
    ret = HFLaunchInspireFace(path_c)
    if ret != 0:
        if ret == 1363:
            logger.warning("Duplicate loading was found")
            return True
        else:
            logger.error(f"Launch InspireFace failure: {ret}")
            return False
    return True


@dataclass
class FeatureHubConfiguration:
    """
    Configuration settings for managing the feature hub, including database and search settings.

    Attributes:
        feature_block_num (int): Number of features per block in the database.
        enable_use_db (bool): Flag to indicate if the database should be used.
        db_path (str): Path to the database file.
        search_threshold (float): The threshold value for considering a match.
        search_mode (int): The mode of searching in the database.
    """
    feature_block_num: int
    enable_use_db: bool
    db_path: str
    search_threshold: float
    search_mode: int

    def _c_struct(self):
        """
        Converts the data class attributes to a C-compatible structure for use in the InspireFace SDK.

        Returns:
            HFFeatureHubConfiguration: A C-structure for feature hub configuration.
        """
        return HFFeatureHubConfiguration(
            enableUseDb=int(self.enable_use_db),
            dbPath=String(bytes(self.db_path, encoding="utf8")),
            featureBlockNum=self.feature_block_num,
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

    Notes:
        Logs an error if enabling the feature hub fails.
    """
    ret = HFFeatureHubDataEnable(config._c_struct())
    if ret != 0:
        logger.error(f"FeatureHub enable failure: {ret}")
        return False
    return True


def feature_hub_disable() -> bool:
    """
    Disables the feature hub.

    Returns:
        bool: True if successfully disabled, False otherwise.

    Notes:
        Logs an error if disabling the feature hub fails.
    """
    ret = HFFeatureHubDataDisable()
    if ret != 0:
        logger.error(f"FeatureHub disable failure: {ret}")
        return False
    return True


def feature_comparison(feature1: np.ndarray, feature2: np.ndarray) -> float:
    """
    Compares two facial feature arrays to determine their similarity.

    Args:
        feature1 (np.ndarray): The first feature array.
        feature2 (np.ndarray): The second feature array.

    Returns:
        float: A similarity score, where -1.0 indicates an error during comparison.

    Notes:
        Logs an error if the comparison process fails.
    """
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
    if ret != 0:
        logger.error(f"Comparison error: {ret}")
        return -1.0

    return float(comparison_result.value)


class FaceIdentity(object):
    """
    Represents an identity based on facial features, associating the features with a custom ID and a tag.

    Attributes:
        feature (np.ndarray): The facial features as a numpy array.
        custom_id (int): A custom identifier for the face identity.
        tag (str): A tag or label associated with the face identity.

    Methods:
        __init__: Initializes a new instance of FaceIdentity.
        from_ctypes: Converts a C structure to a FaceIdentity instance.
        _c_struct: Converts the instance back to a compatible C structure.
    """

    def __init__(self, data: np.ndarray, custom_id: int, tag: str):
        """
        Initializes a new FaceIdentity instance with facial feature data, a custom identifier, and a tag.

        Args:
            data (np.ndarray): The facial feature data.
            custom_id (int): A custom identifier for tracking or referencing the face identity.
            tag (str): A descriptive tag or label for the face identity.
        """
        self.feature = data
        self.custom_id = custom_id
        self.tag = tag

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
        custom_id = raw_identity.customId
        tag = raw_identity.tag.data.decode('utf-8')

        return FaceIdentity(data=feature_data, custom_id=custom_id, tag=tag)

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
            customId=self.custom_id,
            tag=String(bytes(self.tag, encoding="utf8")),
            feature=PHFFaceFeature(feature)
        )


def feature_hub_set_search_threshold(threshold: float):
    """
    Sets the search threshold for face matching in the FeatureHub.

    Args:
        threshold (float): The similarity threshold for determining a match.
    """
    HFFeatureHubFaceSearchThresholdSetting(threshold)


def feature_hub_face_insert(face_identity: FaceIdentity) -> bool:
    """
    Inserts a face identity into the FeatureHub database.

    Args:
        face_identity (FaceIdentity): The face identity to insert.

    Returns:
        bool: True if the face identity was successfully inserted, False otherwise.

    Notes:
        Logs an error if the insertion process fails.
    """
    ret = HFFeatureHubInsertFeature(face_identity._c_struct())
    if ret != 0:
        logger.error(f"Failed to insert face feature data into FeatureHub: {ret}")
        return False
    return True


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


def feature_hub_face_search(data: np.ndarray) -> SearchResult:
    """
    Searches for the most similar face identity in the feature hub based on provided facial features.

    Args:
        data (np.ndarray): The facial feature data to search for.

    Returns:
        SearchResult: The search result containing the confidence and the most similar identity found.

    Notes:
        If the search operation fails, logs an error and returns a SearchResult with a confidence of -1.
    """
    feature = HFFaceFeature(size=HInt32(data.size), data=data.ctypes.data_as(HPFloat))
    confidence = HFloat()
    most_similar = HFFaceFeatureIdentity()
    ret = HFFeatureHubFaceSearch(feature, HPFloat(confidence), PHFFaceFeatureIdentity(most_similar))
    if ret != 0:
        logger.error(f"Failed to search face: {ret}")
        return SearchResult(confidence=-1, similar_identity=FaceIdentity(np.zeros(0), most_similar.customId, "None"))
    if most_similar.customId != -1:
        search_identity = FaceIdentity.from_ctypes(most_similar)
        return SearchResult(confidence=confidence.value, similar_identity=search_identity)
    else:
        none = FaceIdentity(np.zeros(0), most_similar.customId, "None")
        return SearchResult(confidence=confidence.value, similar_identity=none)


def feature_hub_face_search_top_k(data: np.ndarray, top_k: int) -> List[Tuple]:
    """
    Searches for the top 'k' most similar face identities in the feature hub based on provided facial features.

    Args:
        data (np.ndarray): The facial feature data to search for.
        top_k (int): The number of top results to retrieve.

    Returns:
        List[Tuple]: A list of tuples, each containing the confidence and custom ID of the top results.

    Notes:
        If the search operation fails, an empty list is returned.
    """
    feature = HFFaceFeature(size=HInt32(data.size), data=data.ctypes.data_as(HPFloat))
    results = HFSearchTopKResults()
    ret = HFFeatureHubFaceSearchTopK(feature, top_k, PHFSearchTopKResults(results))
    outputs = []
    if ret == 0:
        for idx in range(results.size):
            confidence = results.confidence[idx]
            customId = results.customIds[idx]
            outputs.append((confidence, customId))
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
    ret = HFFeatureHubFaceRemove(custom_id)
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
    ret = HFFeatureHubGetFaceIdentity(custom_id, PHFFaceFeatureIdentity(identify))
    if ret != 0:
        logger.error("Get face identity errors from FeatureHub")
        return None

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
    if ret != 0:
        logger.error(f"Failed to get count: {ret}")

    return int(count.value)


def view_table_in_terminal():
    """
    Displays the database table of face identities in the terminal.

    Notes:
        Logs an error if the operation to view the table fails.
    """
    ret = HFFeatureHubViewDBTable()
    if ret != 0:
        logger.error(f"Failed to view DB: {ret}")


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
