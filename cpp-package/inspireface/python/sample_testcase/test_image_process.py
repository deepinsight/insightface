"""Image rotation and bitmap cases ported from C++ test_image_process.cpp."""

import ctypes
import tempfile
import unittest
from pathlib import Path

import numpy as np

import inspireface as ifac
from inspireface.modules.core import (
    HColor,
    HFaceRect,
    HF_CAMERA_ROTATION_0,
    HF_CAMERA_ROTATION_90,
    HF_CAMERA_ROTATION_180,
    HF_CAMERA_ROTATION_270,
    HF_STREAM_YUV_NV21,
    HFCreateImageBitmapFromFilePath,
    HFImageBitmap,
    HFImageBitmapCopy,
    HFImageBitmapData,
    HFImageBitmapDrawRect,
    HFImageBitmapGetData,
    HFImageBitmapWriteToFile,
    HFReleaseImageBitmap,
    String,
)
from inspireface.modules.exception import InvalidInputError, check_error

from .common import NativeResourceCaseMixin, decode_stream, image_mse, load_image
from .settings import data_path


class ImageProcessCase(NativeResourceCaseMixin, unittest.TestCase):
    def test_bgr_rotation_matches_cpp_fixtures(self):
        cases = (
            ("bulk/r0.jpg", HF_CAMERA_ROTATION_0),
            ("bulk/r90.jpg", HF_CAMERA_ROTATION_90),
            ("bulk/r180.jpg", HF_CAMERA_ROTATION_180),
            ("bulk/r270.jpg", HF_CAMERA_ROTATION_270),
        )
        decoded = []
        for relative_path, rotation in cases:
            image = load_image(relative_path)
            with ifac.ImageStream.load_from_cv_image(
                image,
                rotation=rotation,
            ) as stream:
                decoded.append(decode_stream(stream, apply_rotation=True))
        for index, image in enumerate(decoded[1:], start=1):
            self.assertEqual(image.shape, decoded[0].shape, cases[index][0])
            self.assertLessEqual(image_mse(image, decoded[0]), 0.001, cases[index][0])

    def test_nv21_dimension_and_rotation_contract_matches_cpp(self):
        cases = (
            ("bulk/r0_w330_h409_c3.nv21", 330, 409, HF_CAMERA_ROTATION_0),
            ("bulk/r90_w409_h330_c3.nv21", 409, 330, HF_CAMERA_ROTATION_90),
            ("bulk/r180_w330_h409_c3.nv21", 330, 409, HF_CAMERA_ROTATION_180),
            ("bulk/r270_w409_h330_c3.nv21", 409, 330, HF_CAMERA_ROTATION_270),
        )
        for relative_path, width, height, rotation in cases:
            payload = np.fromfile(str(data_path(relative_path)), dtype=np.uint8)
            with self.assertRaises(InvalidInputError, msg=relative_path):
                ifac.ImageStream.load_from_ndarray(
                    payload,
                    width,
                    height,
                    HF_STREAM_YUV_NV21,
                    rotation,
                )

        width = 8
        height = 8
        payload = np.full(width * height * 3 // 2, 128, dtype=np.uint8)
        for rotation in (
            HF_CAMERA_ROTATION_0,
            HF_CAMERA_ROTATION_90,
            HF_CAMERA_ROTATION_180,
            HF_CAMERA_ROTATION_270,
        ):
            with ifac.ImageStream.load_from_ndarray(
                payload,
                width,
                height,
                HF_STREAM_YUV_NV21,
                rotation,
            ) as stream:
                decoded = decode_stream(stream, apply_rotation=True)
            self.assertEqual(decoded.shape, (height, width, 3))
            self.assertTrue(np.isfinite(decoded).all())

    def test_bitmap_create_copy_draw_and_write(self):
        bitmap = HFImageBitmap()
        copied = HFImageBitmap()
        source = String(bytes(str(data_path("bulk/r90.jpg")), encoding="utf8"))
        check_error(
            HFCreateImageBitmapFromFilePath(source, 3, ctypes.byref(bitmap)),
            "Create bitmap from fixture",
        )
        try:
            check_error(HFImageBitmapCopy(bitmap, ctypes.byref(copied)), "Copy bitmap")
            try:
                original_data = HFImageBitmapData()
                copied_data = HFImageBitmapData()
                check_error(
                    HFImageBitmapGetData(bitmap, ctypes.byref(original_data)),
                    "Get original bitmap data",
                )
                check_error(
                    HFImageBitmapGetData(copied, ctypes.byref(copied_data)),
                    "Get copied bitmap data",
                )
                self.assertEqual(
                    (copied_data.width, copied_data.height, copied_data.channels),
                    (original_data.width, original_data.height, original_data.channels),
                )
                rectangle = HFaceRect(10, 10, 30, 30)
                color = HColor(0, 0, 255)
                check_error(
                    HFImageBitmapDrawRect(copied, rectangle, color, 2),
                    "Draw bitmap rectangle",
                )
                with tempfile.TemporaryDirectory(prefix="inspireface-bitmap-") as temp_dir:
                    output = Path(temp_dir) / "bitmap.png"
                    check_error(
                        HFImageBitmapWriteToFile(
                            copied,
                            String(bytes(str(output), encoding="utf8")),
                        ),
                        "Write bitmap",
                    )
                    self.assertTrue(output.is_file())
                    self.assertGreater(output.stat().st_size, 0)
            finally:
                check_error(HFReleaseImageBitmap(copied), "Release copied bitmap")
        finally:
            check_error(HFReleaseImageBitmap(bitmap), "Release original bitmap")
