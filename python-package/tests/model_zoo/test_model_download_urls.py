import zipfile
from pathlib import Path

import pytest

from insightface.model_zoo import model_zoo
from insightface.utils import storage


def test_ensure_available_downloads_from_model_zoo_release(tmp_path, monkeypatch):
    downloads = []

    def download_file(url, path, overwrite):
        downloads.append((url, overwrite))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("detector.onnx", b"model")

    monkeypatch.setattr(storage, "download_file", download_file)

    package = storage.ensure_available(
        "models",
        "raccoon_s",
        root=str(tmp_path),
    )

    assert package == str(tmp_path / "models" / "raccoon_s")
    assert downloads == [
        (
            f"{storage.MODEL_ZOO_RELEASE_DOWNLOAD_URL}raccoon_s.zip",
            True,
        )
    ]
    assert (tmp_path / "models" / "raccoon_s" / "detector.onnx").is_file()


@pytest.mark.parametrize("download_zip", [False, True])
def test_download_onnx_uses_model_zoo_release(tmp_path, monkeypatch, download_zip):
    downloads = []
    model_path = tmp_path / "models" / "inswapper_128.onnx"
    filename = "inswapper_128.onnx.zip" if download_zip else "inswapper_128.onnx"

    def download_file(url, path, overwrite):
        downloads.append((url, path, overwrite))
        if download_zip:
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("inswapper_128.onnx", b"model")
        else:
            Path(path).write_bytes(b"model")

    monkeypatch.setattr(storage, "download_file", download_file)

    result = storage.download_onnx(
        "models",
        "inswapper_128.onnx",
        root=str(tmp_path),
        download_zip=download_zip,
    )

    assert result == str(model_path)
    assert model_path.read_bytes() == b"model"
    assert downloads == [
        (
            f"{storage.MODEL_ZOO_RELEASE_DOWNLOAD_URL}{filename}",
            str(tmp_path / "models" / filename),
            True,
        )
    ]


def test_get_model_downloads_then_loads_and_reuses_cached_onnx(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "cache"
    model_path = root / "models" / "inswapper_128.onnx"
    downloads = []
    loaded_paths = []
    model = object()

    def download_file(url, path, overwrite):
        downloads.append((url, path, overwrite))
        Path(path).write_bytes(b"model")

    class ModelRouter:
        def __init__(self, onnx_file):
            loaded_paths.append(onnx_file)
            assert Path(onnx_file).read_bytes() == b"model"

        def get_model(self, **kwargs):
            return model

    monkeypatch.setattr(storage, "download_file", download_file)
    monkeypatch.setattr(model_zoo, "ModelRouter", ModelRouter)

    for _ in range(2):
        assert model_zoo.get_model(
            "inswapper_128.onnx",
            download=True,
            root=str(root),
            providers=["CPUExecutionProvider"],
        ) is model

    assert loaded_paths == [str(model_path), str(model_path)]
    assert downloads == [
        (
            f"{storage.MODEL_ZOO_RELEASE_DOWNLOAD_URL}inswapper_128.onnx",
            str(model_path),
            True,
        )
    ]
