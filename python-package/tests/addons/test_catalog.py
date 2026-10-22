import hashlib

import pytest
import requests
from insightface.addons import catalog


@pytest.fixture
def artifact(monkeypatch):
    content = b"model contents"
    value = catalog.AddonArtifact(
        "liveness.onnx",
        "https://example.test/liveness.onnx",
        hashlib.sha256(content).hexdigest(),
        len(content),
    )
    monkeypatch.setattr(catalog, "ADDON_CATALOG", {"liveness": value})
    return value, content


def response(monkeypatch, chunks):
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            for chunk in chunks:
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr(catalog.requests, "get", get)
    return calls


def test_download_is_flat_verified_and_reused_offline(tmp_path, monkeypatch, artifact):
    spec, content = artifact
    calls = response(monkeypatch, [content[:3], b"", content[3:]])
    path = catalog.ensure_addon("liveness", root=tmp_path)
    assert path == tmp_path / "addons" / "liveness.onnx"
    assert path.read_bytes() == content
    assert list(path.parent.iterdir()) == [path]
    assert calls == [(spec.url, {"stream": True, "timeout": (10, 60)})]
    assert catalog.ensure_addon("liveness", root=tmp_path, download=False) == path
    assert len(calls) == 1


@pytest.mark.parametrize(
    "chunks,error",
    [
        ([b"truncated"], RuntimeError),
        ([b"x" * 100], RuntimeError),
        ([b"model", requests.ConnectionError("interrupted")], requests.ConnectionError),
    ],
)
def test_failed_download_does_not_leave_installed_or_partial_file(
    tmp_path, monkeypatch, artifact, chunks, error
):
    response(monkeypatch, chunks)
    with pytest.raises(error):
        catalog.ensure_addon("liveness", root=tmp_path)
    assert list((tmp_path / "addons").iterdir()) == []


def test_existing_corrupt_file_fails_without_replacement(
    tmp_path, monkeypatch, artifact
):
    directory = tmp_path / "addons"
    directory.mkdir()
    path = directory / "liveness.onnx"
    path.write_bytes(b"user file")
    calls = response(monkeypatch, [])
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        catalog.ensure_addon("liveness", root=tmp_path)
    assert calls == []
    assert path.read_bytes() == b"user file"


def test_unknown_addon_and_missing_offline_model_do_not_create_directories(tmp_path):
    with pytest.raises(ValueError, match="Unknown addon"):
        catalog.ensure_addon("../liveness", root=tmp_path)
    with pytest.raises(FileNotFoundError):
        catalog.ensure_addon("liveness", root=tmp_path, download=False)
    assert list(tmp_path.iterdir()) == []
