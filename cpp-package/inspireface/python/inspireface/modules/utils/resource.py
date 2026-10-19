"""Model resource discovery, download, verification, and local caching."""

import hashlib
import hmac
import logging
import os
import ssl
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Mapping, Optional, Union

from filelock import FileLock, Timeout

try:
    from modelscope.hub.snapshot_download import snapshot_download

    MODELSCOPE_AVAILABLE = True
except ImportError:
    snapshot_download = None
    MODELSCOPE_AVAILABLE = False


logger = logging.getLogger(__name__)

__all__ = (
    "MODELSCOPE_AVAILABLE",
    "ResourceManager",
    "USE_OSS_DOWNLOAD",
    "get_file_hash_sha256",
    "set_use_oss_download",
)

# Backward-compatible process-wide switch. New code should configure each
# ResourceManager through ``use_modelscope`` instead.
USE_OSS_DOWNLOAD = False


_MODEL_LIST = {
    "Pikachu": {
        "url": "https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/inspireface_modelzoo/t4/Pikachu",
        "filename": "Pikachu",
        "sha256": "5037ba1f49905b783a1c973d5d58b834a645922cc2814c8e3ca630a38dc24431",
    },
    "Megatron": {
        "url": "https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/inspireface_modelzoo/t4/Megatron",
        "filename": "Megatron",
        "sha256": "709fddf024d9f34ec034d8ef79a4779e1543b867b05e428c1d4b766f69287050",
    },
    "Megatron_TRT": {
        "url": "https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/inspireface_modelzoo/t4/Megatron_TRT",
        "filename": "Megatron_TRT",
        "sha256": "bc9123bdc510954b28d703b8ffe6023f469fb81123fd0b0b27fd452dfa369bab",
    },
    "Gundam_RK356X": {
        "url": "https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/inspireface_modelzoo/t4/Gundam_RK356X",
        "filename": "Gundam_RK356X",
        "sha256": "0fa12a425337ed98bd82610768a50de71cf93ef42a0929ba06cc94c86f4bd415",
    },
    "Gundam_RK3588": {
        "url": "https://inspireface-1259028827.cos.ap-singapore.myqcloud.com/inspireface_modelzoo/t4/Gundam_RK3588",
        "filename": "Gundam_RK3588",
        "sha256": "66070e8d654408b666a2210bd498a976bbad8b33aef138c623e652f8d956641e",
    },
}


def set_use_oss_download(use_oss: bool) -> None:
    """Select OSS for subsequently created and existing resource managers."""
    if not isinstance(use_oss, bool):
        raise TypeError("use_oss must be a bool")
    global USE_OSS_DOWNLOAD
    USE_OSS_DOWNLOAD = use_oss


def get_file_hash_sha256(file_path: Union[str, os.PathLike]) -> str:
    """Return a file's lowercase SHA-256 digest."""
    sha256 = hashlib.sha256()
    with Path(file_path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _remove_file_if_exists(file_path: Union[str, os.PathLike]) -> None:
    try:
        Path(file_path).unlink()
    except FileNotFoundError:
        pass


def _verify_file_sha256(
    file_path: Union[str, os.PathLike],
    expected_sha256: str,
) -> None:
    actual_sha256 = get_file_hash_sha256(file_path)
    if not hmac.compare_digest(actual_sha256.lower(), expected_sha256.lower()):
        raise RuntimeError(
            "SHA-256 mismatch: expected {}, got {}".format(
                expected_sha256,
                actual_sha256,
            )
        )


class ResourceManager:
    """Resolve model files from ModelScope or verified OSS downloads."""

    DOWNLOAD_TIMEOUT_SECONDS = 60
    LOCK_TIMEOUT_SECONDS = 120

    def __init__(
        self,
        use_modelscope: bool = True,
        modelscope_model_id: str = "tunmxy/InspireFace",
        base_dir: Optional[Union[str, os.PathLike]] = None,
        progress_callback: Optional[Callable[[str, int, Optional[int]], None]] = None,
    ) -> None:
        self.user_home = Path.home()
        self.base_dir = (
            Path(base_dir).expanduser()
            if base_dir is not None
            else self.user_home / ".inspireface"
        )
        self.models_dir = self.base_dir / "models"
        self.use_modelscope = bool(use_modelscope)
        self.modelscope_model_id = modelscope_model_id
        self.modelscope_cache_dir = self.base_dir / "ms"
        self.progress_callback = progress_callback
        # Preserve the private attribute used by older integrations while
        # preventing one instance from mutating another instance's registry.
        self._MODEL_LIST = {
            name: dict(model_info)
            for name, model_info in _MODEL_LIST.items()
        }

        self.models_dir.mkdir(parents=True, exist_ok=True)
        if self.use_modelscope:
            self.modelscope_cache_dir.mkdir(parents=True, exist_ok=True)
            if not MODELSCOPE_AVAILABLE:
                raise ImportError(
                    "ModelScope is unavailable. Install modelscope or call "
                    "inspireface.use_oss_download(True) before downloading models."
                )

    def _emit_progress(
        self,
        model_name: str,
        downloaded_bytes: int,
        total_bytes: Optional[int],
    ) -> None:
        if self.progress_callback is not None:
            self.progress_callback(model_name, downloaded_bytes, total_bytes)

    def _model_info(self, name: str) -> Mapping[str, str]:
        try:
            return self._MODEL_LIST[name]
        except KeyError as error:
            raise ValueError(
                "Model {!r} not found. Available models: {}".format(
                    name,
                    sorted(self._MODEL_LIST),
                )
            ) from error

    def _download_from_modelscope(self, model_name: str) -> str:
        if not MODELSCOPE_AVAILABLE or snapshot_download is None:
            raise ImportError("ModelScope is unavailable. Install it with: pip install modelscope")

        logger.info("Downloading model %s from ModelScope", model_name)
        try:
            cache_dir = snapshot_download(
                model_id=self.modelscope_model_id,
                cache_dir=str(self.modelscope_cache_dir),
                allow_file_pattern=[model_name],
            )
            model_file_path = Path(cache_dir) / model_name
            if not model_file_path.is_file():
                raise FileNotFoundError(
                    "Model file {!r} not found in downloaded repository".format(model_name)
                )
            return str(model_file_path)
        except Exception as error:
            raise RuntimeError(
                "Failed to download model from ModelScope: {}".format(error)
            ) from error

    def get_model(
        self,
        name: str,
        re_download: bool = False,
        ignore_verification: bool = False,
    ) -> str:
        """Return a local model path, downloading and verifying when needed."""
        if self.use_modelscope and not USE_OSS_DOWNLOAD:
            # ModelScope owns its cache layout and performs its own locking.
            # Calling snapshot_download again is the supported cache lookup.
            return self._download_from_modelscope(name)

        model_info = self._model_info(name)
        model_file = self.models_dir / model_info["filename"]
        lock_file = self.models_dir / ".{}.lock".format(model_info["filename"])
        try:
            with FileLock(str(lock_file), timeout=self.LOCK_TIMEOUT_SECONDS):
                return self._get_oss_model(
                    name,
                    model_info,
                    model_file,
                    re_download,
                    ignore_verification,
                )
        except Timeout as error:
            raise RuntimeError(
                "Timed out waiting for model download lock: {}".format(lock_file)
            ) from error

    def _get_oss_model(
        self,
        name: str,
        model_info: Mapping[str, str],
        model_file: Path,
        re_download: bool,
        ignore_verification: bool,
    ) -> str:
        downloading_flag = model_file.with_suffix(".downloading")
        _remove_file_if_exists(downloading_flag)

        if model_file.is_file() and not re_download:
            if ignore_verification:
                logger.warning("Model verification skipped for %s", name)
                return str(model_file)
            if hmac.compare_digest(
                get_file_hash_sha256(model_file).lower(),
                model_info["sha256"].lower(),
            ):
                return str(model_file)
            logger.warning("Model hash mismatch for %s; downloading again", name)

        temporary_file = None
        try:
            downloading_flag.touch()
            ssl_context = ssl.create_default_context()
            request = urllib.request.Request(
                model_info["url"],
                headers={"User-Agent": "InspireFace-Python"},
            )
            with urllib.request.urlopen(
                request,
                context=ssl_context,
                timeout=self.DOWNLOAD_TIMEOUT_SECONDS,
            ) as response:
                content_length = response.headers.get("content-length")
                total_size = int(content_length) if content_length else None
                downloaded_size = 0
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=str(self.models_dir),
                    prefix=".{}.".format(model_info["filename"]),
                    suffix=".part",
                    delete=False,
                ) as file_handle:
                    temporary_file = Path(file_handle.name)
                    while True:
                        buffer = response.read(8192)
                        if not buffer:
                            break
                        file_handle.write(buffer)
                        downloaded_size += len(buffer)
                        self._emit_progress(name, downloaded_size, total_size)

            if ignore_verification:
                logger.warning("Model verification skipped for %s", name)
            else:
                _verify_file_sha256(temporary_file, model_info["sha256"])

            os.replace(str(temporary_file), str(model_file))
            temporary_file = None
            logger.info("Model %s is ready at %s", name, model_file)
            return str(model_file)
        except Exception as error:
            if temporary_file is not None:
                _remove_file_if_exists(temporary_file)
            raise RuntimeError("Failed to download model {!r}: {}".format(name, error)) from error
        finally:
            _remove_file_if_exists(downloading_flag)

    def _download_from_modelscope_with_cache(
        self,
        name: str,
        re_download: bool = False,
    ) -> str:
        """Compatibility wrapper; ModelScope itself owns the cache lookup."""
        return self._download_from_modelscope(name)


if __name__ == "__main__":
    manager = ResourceManager()
    print(manager.get_model("Pikachu"))
