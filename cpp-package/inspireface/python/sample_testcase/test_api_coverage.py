"""Keep the Python public-API coverage manifest synchronized with the wrapper."""

import inspect
import unittest

from inspireface.modules import capture as capture_module
from inspireface.modules import inspireface as api_module

from .settings import TEST_RES_DIR


COVERED_PUBLIC_API = {
    "FaceExtended",
    "FaceCaptureConfig",
    "FaceCaptureFilter",
    "FaceCaptureMetrics",
    "FaceCaptureProgress",
    "FaceCaptureRejectReason",
    "FaceCaptureResult",
    "FaceCaptureSession",
    "FaceCaptureState",
    "FaceDetectionSnapshot",
    "FaceIdentity",
    "FaceInformation",
    "FeatureHubConfiguration",
    "ImageStream",
    "InspireFaceSession",
    "ResourcePackInfo",
    "SearchResult",
    "SessionCustomParameter",
    "c_api_level",
    "cosine_similarity_convert_to_percentage",
    "component_versions",
    "diagnostic_info",
    "disable_logging",
    "feature_comparison",
    "feature_hub_disable",
    "feature_hub_enable",
    "feature_hub_face_insert",
    "feature_hub_face_remove",
    "feature_hub_face_search",
    "feature_hub_face_search_top_k",
    "feature_hub_face_update",
    "feature_hub_get_face_count",
    "feature_hub_get_face_id_list",
    "feature_hub_get_face_identity",
    "feature_hub_set_search_threshold",
    "get_recommended_cosine_threshold",
    "get_similarity_converter_config",
    "launch",
    "query_launch_status",
    "reload",
    "set_logging_level",
    "set_similarity_converter_config",
    "terminate",
    "validate_resource_pack",
    "version",
}

EXCLUDED_PUBLIC_API = {
    # Network behavior has its own network-free sample_resource_download_guard.py.
    "ignore_check_latest_model",
    "pull_latest_model",
    "use_oss_download",
    # Interactive or diagnostic output is not suitable for an automated assertion.
    "show_system_resource_statistics",
    "view_table_in_terminal",
    # Backend and hardware switches require a matching target runner/device.
    "check_cuda_device_support",
    "get_cuda_device_id",
    "get_num_cuda_devices",
    "print_cuda_device_info",
    "query_expansive_hardware_rockchip_dma_heap_path",
    "set_cuda_device_id",
    "set_expansive_hardware_rockchip_dma_heap_path",
    "set_image_process_aligned_width",
    "switch_apple_coreml_inference_mode",
    "switch_image_processing_backend",
    "switch_landmark_engine",
}


class PublicApiCoverageCase(unittest.TestCase):
    def test_public_api_manifest_is_current(self):
        discovered = set()
        for module in (api_module, capture_module):
            discovered.update(
                name
                for name, value in vars(module).items()
                if not name.startswith("_")
                and (inspect.isfunction(value) or inspect.isclass(value))
                and getattr(value, "__module__", None) == module.__name__
            )
        accounted_for = COVERED_PUBLIC_API | EXCLUDED_PUBLIC_API
        self.assertEqual(
            discovered - accounted_for,
            set(),
            "new public Python APIs need a testcase or an explicit exclusion",
        )
        self.assertEqual(
            accounted_for - discovered,
            set(),
            "coverage manifest contains removed public APIs",
        )

    def test_cpp_fixture_tree_is_shared(self):
        required_directories = {
            "RD",
            "attribute",
            "bulk",
            "emotion",
            "pose",
            "reaction",
            "rotate",
            "search",
        }
        actual = {
            path.name
            for path in (TEST_RES_DIR / "data").iterdir()
            if path.is_dir()
        }
        self.assertTrue(required_directories.issubset(actual))
        self.assertTrue((TEST_RES_DIR / "pack" / "Pikachu").is_file())
