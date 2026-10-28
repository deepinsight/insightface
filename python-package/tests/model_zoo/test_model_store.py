import hashlib
import zipfile

from insightface.model_zoo import model_store


def _register(monkeypatch, name, content):
    sha1 = hashlib.sha1(content).hexdigest()
    monkeypatch.setitem(model_store._model_sha1, name, sha1)


def test_get_model_file_returns_cached_params(tmp_path, monkeypatch):
    content = b"cached-params"
    _register(monkeypatch, "dummy_v1", content)
    params = tmp_path / "dummy_v1" / "model-0000.params"
    params.parent.mkdir()
    params.write_bytes(content)

    def fail_download(*args, **kwargs):
        raise AssertionError("cached model must not be downloaded again")

    monkeypatch.setattr(model_store, "download", fail_download)

    assert model_store.get_model_file("dummy_v1", root=str(tmp_path)) == str(params)


def test_get_model_file_downloads_and_extracts_zip(tmp_path, monkeypatch):
    content = b"downloaded-params"
    _register(monkeypatch, "dummy_v1", content)
    calls = []

    def fake_download(url, path=None, overwrite=False):
        calls.append(url)
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("model-0000.params", content)
        return path

    monkeypatch.setattr(model_store, "download", fake_download)

    path = model_store.get_model_file("dummy_v1", root=str(tmp_path))

    assert calls == ["https://insightface.ai/files/models/dummy_v1.zip"]
    assert path == str(tmp_path / "dummy_v1" / "model-0000.params")
    assert not (tmp_path / "dummy_v1.zip").exists()
