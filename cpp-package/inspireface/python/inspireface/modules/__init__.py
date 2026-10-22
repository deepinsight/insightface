"""Stable high-level InspireFace Python API."""

from types import ModuleType as _ModuleType

from .inspireface import (
    HF_IMAGE_PROCESSING_CPU,
    HF_IMAGE_PROCESSING_RGA,
    HF_INVALID_FACE_ID,
    HF_PK_AUTO_INCREMENT,
    HF_PK_MANUAL_INPUT,
    HF_SEARCH_MODE_EAGER,
    HF_SEARCH_MODE_EXHAUSTIVE,
    FaceExtended,
    FaceIdentity,
    FaceInformation,
    FeatureHubConfiguration,
    FeatureHubError,
    HardwareError,
    ImageStream,
    InspireFaceError,
    InspireFaceSession,
    InvalidInputError,
    ProcessingError,
    ResourceError,
    ResourcePackInfo,
    SearchResult,
    SessionCustomParameter,
    SystemNotReadyError,
    UnsupportedError,
    c_api_level,
    check_cuda_device_support,
    component_versions,
    cosine_similarity_convert_to_percentage,
    diagnostic_info,
    disable_logging,
    feature_comparison,
    feature_hub_disable,
    feature_hub_enable,
    feature_hub_face_insert,
    feature_hub_face_remove,
    feature_hub_face_search,
    feature_hub_face_search_top_k,
    feature_hub_face_update,
    feature_hub_get_face_count,
    feature_hub_get_face_id_list,
    feature_hub_get_face_identity,
    feature_hub_set_search_threshold,
    get_cuda_device_id,
    get_num_cuda_devices,
    get_recommended_cosine_threshold,
    get_similarity_converter_config,
    ignore_check_latest_model,
    launch,
    print_cuda_device_info,
    pull_latest_model,
    query_expansive_hardware_rockchip_dma_heap_path,
    query_launch_status,
    reload,
    set_cuda_device_id,
    set_expansive_hardware_rockchip_dma_heap_path,
    set_image_process_aligned_width,
    set_logging_level,
    set_similarity_converter_config,
    show_system_resource_statistics,
    switch_apple_coreml_inference_mode,
    switch_image_processing_backend,
    switch_landmark_engine,
    terminate,
    use_oss_download,
    validate_resource_pack,
    version,
    view_table_in_terminal,
)
from .capture import (
    FaceCaptureConfig,
    FaceCaptureFilter,
    FaceCaptureMetrics,
    FaceCaptureProgress,
    FaceCaptureRejectReason,
    FaceCaptureResult,
    FaceCaptureSession,
    FaceCaptureState,
    FaceDetectionSnapshot,
)


# Imported implementation modules are intentionally excluded. Direct imports
# of every long-standing API object above remain supported.
__all__ = tuple(
    name
    for name, value in globals().items()
    if not name.startswith("_") and not isinstance(value, _ModuleType)
)

del _ModuleType
