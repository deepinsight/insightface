import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


def load_module(name):
    path = Path(__file__).resolve().parents[1] / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_module("prepare_wheels")
source = load_module("check_source_run")
VERSION = "1.2.4.post1"


class WheelCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifacts = self.root / "artifacts"
        self.output = self.root / "dist"
        for name in prepare.PLATFORMS:
            self.make_wheel(name)

    def make_wheel(self, artifact, *, suffix="", version=VERSION, metadata_version=VERSION,
                   purelib=False, native=True, tag=None, library_override=None, extra_libraries=()):
        platform, library = prepare.PLATFORMS[artifact]
        folder = self.artifacts / (artifact + suffix)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"inspireface-{version}-py3-none-{platform}.whl"
        info = f"inspireface-{version}.dist-info/"
        with zipfile.ZipFile(path, "w") as wheel:
            wheel.writestr(info + "METADATA", f"Name: inspireface\nVersion: {metadata_version}\n")
            wheel.writestr(info + "WHEEL", f"Root-Is-Purelib: false\nTag: {tag or ('py3-none-' + platform)}\n")
            if native:
                prefix = f"inspireface-{version}.data/purelib/" if purelib else ""
                wheel.writestr(prefix + "inspireface/modules/core/libs/" + (library_override or library), b"native-library")
            for extra in extra_libraries:
                wheel.writestr("inspireface/modules/core/libs/" + extra, b"other-native-library")
        return path

    def collect(self, legacy=None):
        return prepare.collect_wheels(self.artifacts, self.output, VERSION, legacy)

    def test_collects_five_distinct_platforms(self):
        self.make_wheel("macos-arm64-wheels", purelib=True)
        self.make_wheel("windows-x64-wheels", purelib=True)
        selected = self.collect()
        self.assertEqual(len(selected), 5)
        for path in selected:
            self.assertEqual(path.read_bytes(), (self.output / path.name).read_bytes())

    def test_manifest_and_job_output_use_validated_artifact_version(self):
        manifest = self.root / "metadata" / "release.json"
        outputs = self.root / "github-output"
        argv = ["prepare_wheels.py", "--artifacts", str(self.artifacts),
                "--output", str(self.output), "--version", VERSION,
                "--manifest", str(manifest)]
        with patch.object(sys, "argv", argv), patch.dict(os.environ, {
            "GITHUB_OUTPUT": str(outputs), "GITHUB_STEP_SUMMARY": "",
        }), patch.object(prepare, "project_version", side_effect=AssertionError("Do not use checkout version")):
            prepare.main()
        published = json.loads(manifest.read_text())
        self.assertEqual(published["version"], VERSION)
        self.assertEqual(len(published["wheels"]), 5)
        for path in self.output.glob("*.whl"):
            self.assertEqual(published["wheels"][path.name], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(outputs.read_text(), f"version={VERSION}\n")

    def test_invalid_release_emits_no_manifest_or_job_output(self):
        shutil.rmtree(self.artifacts / "windows-x64-wheels")
        manifest = self.root / "release.json"
        outputs = self.root / "github-output"
        argv = ["prepare_wheels.py", "--artifacts", str(self.artifacts),
                "--output", str(self.output), "--version", VERSION,
                "--manifest", str(manifest)]
        with patch.object(sys, "argv", argv), patch.dict(os.environ, {
            "GITHUB_OUTPUT": str(outputs), "GITHUB_STEP_SUMMARY": "",
        }), self.assertRaises(SystemExit):
            prepare.main()
        self.assertFalse(manifest.exists())
        self.assertFalse(outputs.exists())

    def test_legacy_matrix_selects_only_explicit_python_version(self):
        for artifact in ("macos-arm64-wheels", "macos-x86_64-wheels"):
            shutil.rmtree(self.artifacts / artifact)
            for version in ("3.8", "3.9", "3.10", "3.11", "3.12"):
                self.make_wheel(artifact, suffix=f"-py{version}", purelib=True)
        selected = self.collect("3.12")
        self.assertEqual(len(selected), 5)
        for path in selected:
            if path.parent.name.startswith("macos-"):
                self.assertTrue(path.parent.name.endswith("-py3.12"))

    def test_rejects_missing_platform_before_copying(self):
        shutil.rmtree(self.artifacts / "macos-x86_64-wheels")
        with self.assertRaisesRegex(ValueError, "exactly one artifact"):
            self.collect()
        self.assertFalse(self.output.exists())

    def test_rejects_legacy_four_platform_release_without_windows(self):
        shutil.rmtree(self.artifacts / "windows-x64-wheels")
        with self.assertRaisesRegex(ValueError, "exactly one artifact for win_amd64"):
            self.collect("3.12")
        self.assertFalse(self.output.exists())

    def test_rejects_windows_metadata_with_wrong_architecture_or_python_tag(self):
        for tag in ("py3-none-win32", "py3-none-win_arm64", "cp312-cp312-win_amd64"):
            with self.subTest(tag=tag):
                self.make_wheel("windows-x64-wheels", tag=tag)
                with self.assertRaisesRegex(ValueError, "expected py3-none platform tag"):
                    self.collect()
                self.assertFalse(self.output.exists())

    def test_rejects_windows_wheel_with_wrong_filename_tag(self):
        path = self.make_wheel("windows-x64-wheels")
        path.rename(path.with_name(path.name.replace("win_amd64", "win32")))
        with self.assertRaisesRegex(ValueError, "Expected inspireface-.*win_amd64"):
            self.collect()
        self.assertFalse(self.output.exists())

    def test_rejects_wrong_windows_dll_name_or_location(self):
        for library in ("windows/x86/libInspireFace.dll", "windows/x64/InspireFace.dll"):
            with self.subTest(library=library):
                self.make_wheel("windows-x64-wheels", library_override=library)
                with self.assertRaisesRegex(ValueError, "native library: windows/x64/libInspireFace.dll"):
                    self.collect()
                self.assertFalse(self.output.exists())

    def test_rejects_native_libraries_from_another_platform_or_dynamic_dependency(self):
        for extra in ("linux/x64/libInspireFace.so", "darwin/x64/libInspireFace.dylib",
                      "windows/x86/libInspireFace.dll", "windows/x64/MNN.dll"):
            with self.subTest(extra=extra):
                self.make_wheel("windows-x64-wheels", extra_libraries=[extra])
                with self.assertRaisesRegex(ValueError, "Unexpected native libraries"):
                    self.collect()
                self.assertFalse(self.output.exists())

    def test_rejects_windows_library_leaking_into_unix_wheel(self):
        self.make_wheel("manylinux2014-wheels", extra_libraries=["windows/x64/libInspireFace.dll"])
        with self.assertRaisesRegex(ValueError, "Unexpected native libraries"):
            self.collect()
        self.assertFalse(self.output.exists())

    def test_rejects_duplicate_windows_dll_in_root_and_purelib(self):
        self.make_wheel("windows-x64-wheels", purelib=True,
                        extra_libraries=["windows/x64/libInspireFace.dll"])
        with self.assertRaisesRegex(ValueError, "duplicated native library"):
            self.collect()
        self.assertFalse(self.output.exists())

    def test_rejects_duplicate_artifact_candidates(self):
        self.make_wheel("macos-arm64-wheels", suffix="-py3.12")
        with self.assertRaisesRegex(ValueError, "exactly one artifact"):
            self.collect("3.12")

    def test_rejects_multiple_wheels_in_one_artifact(self):
        self.make_wheel("macos-arm64-wheels", version="1.2.4.post2")
        with self.assertRaisesRegex(ValueError, "Expected one wheel"):
            self.collect()

    def test_rejects_wrong_release(self):
        with self.assertRaisesRegex(ValueError, "Expected inspireface-1.2.4.post2"):
            prepare.collect_wheels(self.artifacts, self.output, "1.2.4.post2")

    def test_rejects_metadata_version_mismatch(self):
        self.make_wheel("macos-arm64-wheels", metadata_version="1.2.3")
        with self.assertRaisesRegex(ValueError, "metadata does not match"):
            self.collect()

    def test_rejects_corrupt_zip_central_directory(self):
        path = self.make_wheel("macos-arm64-wheels")
        path.write_bytes(path.read_bytes().replace(b"PK\x01\x02", b"BAD!", 1))
        with self.assertRaisesRegex(ValueError, "Bad magic number for central directory"):
            self.collect()
        self.assertFalse(self.output.exists())

    def test_rejects_crc_corruption(self):
        path = self.make_wheel("macos-arm64-wheels")
        path.write_bytes(path.read_bytes().replace(b"native-library", b"broken-library"))
        with self.assertRaisesRegex(ValueError, "ZIP CRC check failed"):
            self.collect()

    def test_rejects_missing_native_library(self):
        self.make_wheel("macos-arm64-wheels", native=False)
        with self.assertRaisesRegex(ValueError, "native library"):
            self.collect()

    def test_does_not_overwrite_existing_release_files(self):
        self.output.mkdir()
        existing = self.output / "existing.whl"
        existing.write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "must be empty"):
            self.collect()
        self.assertEqual(existing.read_bytes(), b"keep")


class SourceRunTests(unittest.TestCase):
    def test_allows_failed_upload_from_completed_build(self):
        source.validate_source_run(self.run_data(), "HyperInspire/InspireFace")

    def run_data(self):
        return {"path": ".github/workflows/build_wheels.yaml", "status": "completed",
                "conclusion": "failure", "event": "push",
                "head_repository": {"full_name": "HyperInspire/InspireFace"}}

    def test_rejects_unrelated_or_unfinished_runs(self):
        for key, value in [("path", ".github/workflows/build.yaml"),
                           ("status", "in_progress"), ("event", "pull_request"),
                           ("head_repository", {"full_name": "other/InspireFace"})]:
            with self.subTest(key=key):
                run = self.run_data()
                run[key] = value
                with self.assertRaises(ValueError):
                    source.validate_source_run(run, "HyperInspire/InspireFace")


if __name__ == "__main__":
    unittest.main()
