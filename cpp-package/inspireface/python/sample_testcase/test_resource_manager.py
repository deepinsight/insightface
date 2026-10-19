"""Network-free resource download, verification, and concurrency gates."""

import hashlib
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from inspireface.modules.utils import resource as resource_module
from inspireface.modules.utils.resource import ResourceManager


class FakeResponse:
    def __init__(self, payload, delay=0.0):
        self.payload = payload
        self.delay = delay
        self.headers = {"content-length": str(len(payload))}
        self._read = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self, _size):
        if self._read:
            return b""
        self._read = True
        if self.delay:
            time.sleep(self.delay)
        return self.payload


class ResourceManagerCase(unittest.TestCase):
    def manager_for(self, root, payload):
        manager = ResourceManager(
            use_modelscope=False,
            base_dir=root,
        )
        manager._MODEL_LIST["Pikachu"] = {
            "url": "https://example.invalid/Pikachu",
            "filename": "Pikachu",
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        return manager

    def test_verified_download_is_cached_and_reports_progress(self):
        payload = b"verified model payload"
        progress = []
        with tempfile.TemporaryDirectory(prefix="inspireface-resource-") as temp_dir:
            manager = self.manager_for(temp_dir, payload)
            manager.progress_callback = lambda name, downloaded, total: progress.append(
                (name, downloaded, total)
            )
            with patch(
                "inspireface.modules.utils.resource.urllib.request.urlopen",
                return_value=FakeResponse(payload),
            ) as urlopen:
                first = manager.get_model("Pikachu")
                second = manager.get_model("Pikachu")

            self.assertEqual(first, second)
            self.assertEqual(Path(first).read_bytes(), payload)
            self.assertEqual(urlopen.call_count, 1)
            self.assertEqual(progress[-1], ("Pikachu", len(payload), len(payload)))
            self.assertFalse(Path(first).with_suffix(".downloading").exists())
            self.assertEqual(list(Path(temp_dir).rglob("*.part")), [])

    def test_failed_verification_preserves_existing_model(self):
        expected = b"expected payload"
        existing = b"existing model"
        with tempfile.TemporaryDirectory(prefix="inspireface-resource-") as temp_dir:
            manager = self.manager_for(temp_dir, expected)
            target = manager.models_dir / "Pikachu"
            target.write_bytes(existing)
            with patch(
                "inspireface.modules.utils.resource.urllib.request.urlopen",
                return_value=FakeResponse(b"corrupted payload"),
            ):
                with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                    manager.get_model("Pikachu", re_download=True)
            self.assertEqual(target.read_bytes(), existing)
            self.assertEqual(list(manager.models_dir.glob("*.part")), [])

    def test_concurrent_requests_perform_one_download(self):
        payload = b"concurrent model payload"
        barrier = threading.Barrier(2)
        with tempfile.TemporaryDirectory(prefix="inspireface-resource-") as temp_dir:
            manager = self.manager_for(temp_dir, payload)

            def fetch(_index):
                barrier.wait()
                return manager.get_model("Pikachu")

            with patch(
                "inspireface.modules.utils.resource.urllib.request.urlopen",
                side_effect=lambda *args, **kwargs: FakeResponse(payload, delay=0.05),
            ) as urlopen, ThreadPoolExecutor(max_workers=2) as executor:
                paths = list(executor.map(fetch, range(2)))

            self.assertEqual(paths[0], paths[1])
            self.assertEqual(urlopen.call_count, 1)
            self.assertEqual(Path(paths[0]).read_bytes(), payload)

    def test_unknown_model_fails_before_network_access(self):
        with tempfile.TemporaryDirectory(prefix="inspireface-resource-") as temp_dir:
            manager = ResourceManager(use_modelscope=False, base_dir=temp_dir)
            with patch(
                "inspireface.modules.utils.resource.urllib.request.urlopen"
            ) as urlopen:
                with self.assertRaisesRegex(ValueError, "Available models"):
                    manager.get_model("unknown")
            urlopen.assert_not_called()

    def test_modelscope_keeps_custom_repository_file_compatibility(self):
        with tempfile.TemporaryDirectory(prefix="inspireface-resource-") as temp_dir:
            repository = Path(temp_dir) / "repository"
            repository.mkdir()
            custom_model = repository / "custom_model"
            custom_model.write_bytes(b"custom")
            with patch.object(resource_module, "MODELSCOPE_AVAILABLE", True), patch.object(
                resource_module,
                "snapshot_download",
                return_value=str(repository),
            ) as snapshot:
                manager = ResourceManager(
                    use_modelscope=True,
                    base_dir=Path(temp_dir) / "cache",
                )
                self.assertEqual(manager.get_model("custom_model"), str(custom_model))
            self.assertEqual(
                snapshot.call_args[1]["allow_file_pattern"],
                ["custom_model"],
            )
