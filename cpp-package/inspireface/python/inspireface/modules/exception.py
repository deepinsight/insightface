from . import herror as errcode
from typing import Optional, Dict, Any


class InspireFaceError(Exception):
    """Base class for all InspireFace exceptions"""
    
    def __init__(self, message: str, error_code: Optional[int] = None, **context):
        super().__init__(message)
        self.error_code = error_code
        self.context = context
        self._error_name = self._get_error_name(error_code) if error_code else None
    
    def _get_error_name(self, error_code: int) -> str:
        """Get error name corresponding to error code"""
        for name, value in errcode.__dict__.items():
            if isinstance(value, int) and value == error_code:
                return name
        return f"UNKNOWN_ERROR"
    
    def __str__(self):
        base_msg = super().__str__()
        if self.error_code is not None:
            return f"[{self._error_name}({self.error_code})] {base_msg}"
        return base_msg
    
    @property
    def error_name(self) -> Optional[str]:
        """Get error name"""
        return self._error_name


class InvalidInputError(InspireFaceError, ValueError):
    """Input parameter/data format error"""
    pass


class SystemNotReadyError(InspireFaceError, RuntimeError):
    """System not initialized or resources not ready"""
    pass


class ProcessingError(InspireFaceError):
    """Business logic error during processing"""
    pass


class ResourceError(InspireFaceError, OSError):
    """Resource-related errors (handles, memory, files, etc.)"""
    pass


class HardwareError(InspireFaceError):
    """Hardware-related errors (CUDA, devices, etc.)"""
    pass


class FeatureHubError(InspireFaceError):
    """Feature hub related errors"""
    pass


# === Error code mapping table ===
ERROR_CODE_MAPPING = {
    # Input parameter errors
    'invalid_input': [
        errcode.HERR_INVALID_PARAM,
        errcode.HERR_INVALID_IMAGE_STREAM_PARAM,
        errcode.HERR_INVALID_BUFFER_SIZE,
        errcode.HERR_INVALID_DETECTION_INPUT,
    ],
    
    # System not ready
    'system_not_ready': [
        errcode.HERR_ARCHIVE_NOT_LOAD,
        errcode.HERR_SESS_INVALID_RESOURCE,
        errcode.HERR_ARCHIVE_LOAD_MODEL_FAILURE,
    ],
    
    # Processing errors
    'processing': [
        errcode.HERR_SESS_FUNCTION_UNUSABLE,
        errcode.HERR_SESS_TRACKER_FAILURE,
        errcode.HERR_SESS_PIPELINE_FAILURE,
        errcode.HERR_SESS_REC_EXTRACT_FAILURE,
        errcode.HERR_SESS_LANDMARK_NOT_ENABLE,
        errcode.HERR_IMAGE_STREAM_DECODE_FAILED,
    ],
    
    # Resource errors
    'resource': [
        errcode.HERR_INVALID_IMAGE_STREAM_HANDLE,
        errcode.HERR_INVALID_CONTEXT_HANDLE,
        errcode.HERR_INVALID_FACE_TOKEN,
        errcode.HERR_INVALID_FACE_FEATURE,
        errcode.HERR_INVALID_FACE_LIST,
        errcode.HERR_INVALID_IMAGE_BITMAP_HANDLE,
    ],
    
    # Hardware errors
    'hardware': [
        errcode.HERR_DEVICE_CUDA_NOT_SUPPORT,
        errcode.HERR_DEVICE_CUDA_UNKNOWN_ERROR,
    ],
    
    # Feature hub errors
    'feature_hub': [
        errcode.HERR_FT_HUB_DISABLE,
        errcode.HERR_FT_HUB_INSERT_FAILURE,
        errcode.HERR_FT_HUB_NOT_FOUND_FEATURE,
    ],
}


def check_error(error_code: int, operation: str = "", **context):
    """
    Check error code and raise corresponding exception
    
    Args:
        error_code: Error code returned by C library
        operation: Operation description for building error message
        **context: Additional context information
    
    Raises:
        Corresponding InspireFaceError subclass exception
    """
    if error_code == errcode.HSUCCEED:
        return
    
    # Get error name
    error_name = None
    for name, value in errcode.__dict__.items():
        if isinstance(value, int) and value == error_code:
            error_name = name
            break
    
    # Build basic error message
    if operation:
        message = f"{operation} failed"
        if error_name:
            message += f": {error_name}"
    else:
        message = error_name or f"Unknown error (code: {error_code})"
    
    # Select exception type based on error code
    exception_class = InspireFaceError  # Default exception type
    
    for category, codes in ERROR_CODE_MAPPING.items():
        if error_code in codes:
            if category == 'invalid_input':
                exception_class = InvalidInputError
            elif category == 'system_not_ready':
                exception_class = SystemNotReadyError
            elif category == 'processing':
                exception_class = ProcessingError
            elif category == 'resource':
                exception_class = ResourceError
            elif category == 'hardware':
                exception_class = HardwareError
            elif category == 'feature_hub':
                exception_class = FeatureHubError
            break
    
    # Raise corresponding exception
    raise exception_class(message, error_code, **context)


# === Convenient validation functions ===

def validate_image_format(image, operation: str = "Image validation"):
    """Validate image format"""
    import numpy as np
    
    if not isinstance(image, np.ndarray):
        raise InvalidInputError(
            f"{operation}: Input must be a numpy array",
            errcode.HERR_INVALID_PARAM,
            input_type=type(image).__name__
        )
    
    if len(image.shape) != 3:
        raise InvalidInputError(
            f"{operation}: Image must be 3-dimensional (H, W, C)",
            errcode.HERR_INVALID_IMAGE_STREAM_PARAM,
            actual_shape=image.shape
        )
    
    h, w, c = image.shape
    if c not in [3, 4]:
        raise InvalidInputError(
            f"{operation}: Image must have 3 or 4 channels",
            errcode.HERR_INVALID_IMAGE_STREAM_PARAM,
            actual_channels=c
        )


def validate_feature_data(data, operation: str = "Feature validation"):
    """Validate feature data format"""
    import numpy as np
    
    if not isinstance(data, np.ndarray):
        raise InvalidInputError(
            f"{operation}: Feature data must be a numpy array",
            errcode.HERR_INVALID_FACE_FEATURE,
            input_type=type(data).__name__
        )
    
    if data.dtype != np.float32:
        raise InvalidInputError(
            f"{operation}: Feature data must be in float32 format",
            errcode.HERR_INVALID_FACE_FEATURE,
            actual_dtype=str(data.dtype)
        )


def validate_session_initialized(session, operation: str = "Session operation"):
    """Validate if session is initialized"""
    if session is None or session._sess is None:
        raise ResourceError(
            f"{operation}: Session not initialized",
            errcode.HERR_INVALID_CONTEXT_HANDLE
        )


# === Exception handling decorators for special scenarios ===

def handle_c_api_errors(operation_name: str):
    """Decorator for wrapping C API calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if not isinstance(e, InspireFaceError):
                    # Wrap non-InspireFace exceptions as ProcessingError
                    raise ProcessingError(
                        f"{operation_name}: {str(e)}",
                        context={'original_exception': type(e).__name__}
                    ) from e
                raise
        return wrapper
    return decorator
