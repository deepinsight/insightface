"""Compatibility gates for the Python-facing package and ctypes boundary."""

import ctypes
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import inspireface as ifac
from inspireface.modules import inspireface as api_module
from inspireface.modules import herror as errcode
from inspireface.modules.core import native
from inspireface.modules.core._library_path import get_lib_path, platform_library_spec
from inspireface.modules.exception import (
    ProcessingError,
    SystemNotReadyError,
    UnsupportedError,
    handle_c_api_errors,
)


class PublicPackageContractCase(unittest.TestCase):
    def test_star_export_manifest_is_explicit_and_resolvable(self):
        self.assertIsInstance(ifac.__all__, tuple)
        self.assertEqual(len(ifac.__all__), len(set(ifac.__all__)))
        self.assertNotIn("core", ifac.__all__)
        self.assertNotIn("exception", ifac.__all__)
        self.assertNotIn("modules", ifac.__all__)
        self.assertNotIn("utils", ifac.__all__)
        for name in ifac.__all__:
            self.assertTrue(hasattr(ifac, name), name)

        # Accidental historical module exports remain addressable for callers
        # that used them, but new star imports do not receive them.
        for name in (
            "core",
            "exception",
            "herror",
            "inspireface",
            "modules",
            "param",
            "utils",
        ):
            self.assertTrue(hasattr(ifac, name), name)
            self.assertNotIn(name, ifac.__all__)

    def test_python_and_native_versions_have_separate_contracts(self):
        version_file = Path(ifac.__file__).resolve().parent.parent / "version.txt"
        if version_file.is_file():
            expected_version = version_file.read_text(encoding="utf-8").strip()
        else:
            try:
                from importlib.metadata import version as distribution_version
            except ImportError:  # Python 3.7
                from importlib_metadata import version as distribution_version
            expected_version = distribution_version("inspireface")
        self.assertEqual(ifac.__version__, expected_version)
        self.assertEqual(ifac.__native_version__, ifac.version())
        self.assertIs(ifac.native_version, ifac.version)

    def test_c_api_level_falls_back_for_legacy_native_libraries(self):
        self.assertEqual(ifac.c_api_level(), 2)
        with patch.dict(api_module.__dict__, {"HFQueryCAPILevel": None}):
            self.assertEqual(api_module.c_api_level(), 1)

    def test_face_identity_repr_is_bounded(self):
        identity = ifac.FaceIdentity(np.arange(512, dtype=np.float32), 42)
        representation = repr(identity)
        self.assertIn("id=42", representation)
        self.assertIn("shape=(512,)", representation)
        self.assertLess(len(representation), 120)
        self.assertEqual(identity.custom_id, identity.id)

    def test_launch_and_reload_preflight_new_native_but_preserve_legacy_loading(self):
        with patch.object(api_module, "validate_resource_pack") as validate, patch.object(
            api_module, "HFLaunchInspireFace", return_value=errcode.HSUCCEED
        ), patch.object(api_module, "HFReloadInspireFace", return_value=errcode.HSUCCEED):
            self.assertTrue(api_module.launch(resource_path="/model.pack"))
            self.assertTrue(api_module.reload(resource_path="/model.pack"))
        self.assertEqual(validate.call_count, 2)
        validate.assert_any_call("/model.pack")

        with patch.dict(api_module.__dict__, {"HFValidateResourcePack": None}), patch.object(
            api_module, "validate_resource_pack"
        ) as validate, patch.object(api_module, "HFLaunchInspireFace", return_value=errcode.HSUCCEED):
            self.assertTrue(api_module.launch(resource_path="/legacy-model.pack"))
        validate.assert_not_called()

        with patch.dict(api_module.__dict__, {"HFValidateResourcePack": None}):
            with self.assertRaises(UnsupportedError):
                api_module.validate_resource_pack("/legacy-model.pack")


class ExceptionBoundaryContractCase(unittest.TestCase):
    def test_programming_errors_keep_their_original_type(self):
        @handle_c_api_errors("test operation")
        def fail():
            raise TypeError("programmer error")

        with self.assertRaisesRegex(TypeError, "programmer error"):
            fail()

    def test_ctypes_errors_are_translated_with_flat_context(self):
        @handle_c_api_errors("test operation")
        def fail():
            raise ctypes.ArgumentError("bad native argument")

        with self.assertRaises(ProcessingError) as raised:
            fail()
        self.assertEqual(
            raised.exception.context,
            {"original_exception": "ArgumentError"},
        )
        self.assertIsInstance(raised.exception.__cause__, ctypes.ArgumentError)

    def test_explicit_session_lifecycle_does_not_auto_launch(self):
        with patch.object(api_module, "query_launch_status", return_value=False), patch.object(
            api_module,
            "launch",
        ) as launch:
            with self.assertRaises(SystemNotReadyError):
                ifac.InspireFaceSession(ifac.HF_ENABLE_NONE, auto_launch=False)
        launch.assert_not_called()


class NativePlatformContractCase(unittest.TestCase):
    def test_legacy_long_and_fixed_width_ids_follow_the_native_abi(self):
        # HResult is C long, while HFaceId is explicitly int64_t. They have
        # different widths on 64-bit Windows (LLP64), unlike Linux/macOS.
        self.assertEqual(ctypes.sizeof(native.HResult), ctypes.sizeof(ctypes.c_long))
        self.assertEqual(native.HResult(-1).value, -1)
        self.assertEqual(ctypes.sizeof(native.HFaceId), 8)
        self.assertEqual(native.HFaceId((1 << 40) + 17).value, (1 << 40) + 17)
        self.assertEqual(ctypes.sizeof(native.HFSession), ctypes.sizeof(ctypes.c_void_p))
        if sys.platform == "win32":
            self.assertEqual(ctypes.sizeof(native.HResult), 4)

    def test_generated_module_keeps_loader_and_native_symbol_compatibility(self):
        self.assertTrue(hasattr(native, "LibraryLoader"))
        self.assertTrue(hasattr(native, "add_library_search_dirs"))
        self.assertTrue(hasattr(native, "HFCreateInspireFaceSession"))
        self.assertTrue(hasattr(native, "HFFaceBasicToken"))
        self.assertTrue(hasattr(native, "HFCreateInspireFaceSessionV2"))
        self.assertTrue(hasattr(native, "HFQueryCAPILevel"))
        self.assertTrue(hasattr(native, "HFExecuteFaceTrackSnapshot"))
        self.assertTrue(hasattr(native, "HFGetFaceResultSnapshotData"))
        self.assertTrue(hasattr(native, "HFReleaseFaceResultSnapshot"))
        self.assertTrue(hasattr(native, "HFGetErrorMessage"))
        self.assertTrue(hasattr(native, "HFValidateResourcePack"))

    def test_native_error_message_uses_caller_owned_storage(self):
        required_size = native.HInt32()
        self.assertEqual(
            native.HFGetErrorMessage(
                native.HResult(errcode.HERR_INVALID_PARAM),
                None,
                0,
                ctypes.byref(required_size),
            ),
            errcode.HSUCCEED,
        )
        self.assertGreater(required_size.value, 1)

        buffer = ctypes.create_string_buffer(required_size.value)
        copied_size = native.HInt32()
        self.assertEqual(
            native.HFGetErrorMessage(
                native.HResult(errcode.HERR_INVALID_PARAM),
                buffer,
                len(buffer),
                ctypes.byref(copied_size),
            ),
            errcode.HSUCCEED,
        )
        self.assertEqual(copied_size.value, required_size.value)
        self.assertIn("parameter", buffer.value.decode("utf-8").lower())

    def test_level_two_ctypes_layout_is_fixed_width(self):
        self.assertEqual(ctypes.sizeof(native.HFStatus), 4)
        self.assertEqual(ctypes.sizeof(native.HFUInt32), 4)
        self.assertEqual(ctypes.sizeof(native.HFUInt64), 8)
        self.assertEqual(ctypes.sizeof(native.HFSessionConfigV2), 64)
        self.assertEqual(native.HFSessionConfigV2.structSize.offset, 0)
        self.assertEqual(native.HFSessionConfigV2.structVersion.offset, 4)
        self.assertEqual(native.HFSessionConfigV2.featureMask.offset, 8)
        self.assertEqual(native.HFSessionConfigV2.detectMode.offset, 16)
        self.assertEqual(native.HFSessionConfigV2.reserved.offset, 32)
        self.assertEqual(ctypes.sizeof(native.HFResourcePackInfo), 336)
        self.assertEqual(native.HFResourcePackInfo.structSize.offset, 0)
        self.assertEqual(native.HFResourcePackInfo.structVersion.offset, 4)
        self.assertEqual(native.HFResourcePackInfo.archiveFileCount.offset, 8)
        self.assertEqual(native.HFResourcePackInfo.tag.offset, 16)
        self.assertEqual(native.HFResourcePackInfo.reserved.offset, 272)

    def test_supported_platform_mappings(self):
        self.assertEqual(
            platform_library_spec("Darwin", "arm64"),
            ("darwin", "arm64", "libInspireFace.dylib"),
        )
        self.assertEqual(
            platform_library_spec("Linux", "x86_64"),
            ("linux", "x64", "libInspireFace.so"),
        )
        self.assertEqual(
            platform_library_spec("Windows", "AMD64"),
            ("windows", "x64", "libInspireFace.dll"),
        )

    def test_unsupported_architecture_is_not_silently_misclassified(self):
        with self.assertRaisesRegex(RuntimeError, "armv7l"):
            platform_library_spec("Linux", "armv7l")

    def test_explicit_library_override_is_validated(self):
        with self.assertRaisesRegex(RuntimeError, "not found"):
            get_lib_path(environ={"INSPIREFACE_LIBRARY_PATH": "/missing/inspireface.so"})
