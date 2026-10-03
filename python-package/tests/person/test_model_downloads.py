"""SDK first-use downloads validate complete packages before publishing them."""
import io
import json
import shutil
import stat
import urllib.error
import zipfile
from pathlib import Path

import pytest

from insightface.model_zoo import person_package


@pytest.fixture
def source_package(tmp_path, synthetic_person_manifest, synthetic_person_package):
    def create(name="cheetah_s"):
        document = synthetic_person_manifest(name, 320 if name == "cheetah_s" else 640)
        folder = synthetic_person_package(tmp_path / "source", document)
        (folder / "MODEL.LICENSE").write_text("synthetic test license")
        return folder
    return create


def stage_download(monkeypatch, source, wrapped=False, after_download=None):
    def download(url, destination):
        assert url == f"https://github.com/deepinsight/insightface/releases/download/model-zoo/{source.name}.zip"
        with zipfile.ZipFile(destination, "w") as archive:
            for path in source.iterdir():
                archive.write(path, f"{source.name}/{path.name}" if wrapped else path.name)
        if after_download:
            after_download()
    monkeypatch.setattr(person_package, "_download_person_archive", download)


def forbid_download(monkeypatch):
    monkeypatch.setattr(person_package, "_download_person_archive", lambda *args: pytest.fail("network"))


@pytest.mark.parametrize("name", ["cheetah_s", "cheetah_l"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_download_installs_complete_flat_or_wrapped_package(tmp_path, monkeypatch, source_package, name, wrapped):
    import onnxruntime
    source = source_package(name)
    stage_download(monkeypatch, source, wrapped)
    monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *args, **kwargs: pytest.fail("inference"))
    root = tmp_path / "installed"
    package = person_package.ensure_person_package(name, root)
    target = root / "models" / name
    assert Path(package["path"]) == target
    assert {path.name for path in target.iterdir()} == {path.name for path in source.iterdir()}
    assert all(Path(task["path"]).parent == target for task in package["tasks"].values())
    assert not list(target.parent.glob(".*-download-*"))
    forbid_download(monkeypatch)
    assert person_package.ensure_person_package(name, root) == package


def test_inspection_and_custom_missing_package_never_download(tmp_path, monkeypatch):
    forbid_download(monkeypatch)
    with pytest.raises(FileNotFoundError):
        person_package.load_person_package("cheetah_s", tmp_path)
    with pytest.raises(FileNotFoundError, match="automatic download supports"):
        person_package.ensure_person_package("custom_person", tmp_path)
    assert not (tmp_path / "models").exists()


@pytest.mark.parametrize("existing", ["empty", "partial", "file", "dangling_symlink"])
def test_existing_user_path_is_never_replaced(tmp_path, monkeypatch, existing):
    target = tmp_path / "models" / "cheetah_s"
    target.parent.mkdir()
    if existing == "file":
        target.write_text("keep")
    elif existing == "dangling_symlink":
        try:
            target.symlink_to(tmp_path / "absent", target_is_directory=True)
        except OSError as error:
            pytest.skip(f"symlinks unavailable: {error}")
    else:
        target.mkdir()
        if existing == "partial":
            (target / "user.txt").write_text("keep")
    forbid_download(monkeypatch)
    with pytest.raises(RuntimeError, match="will not be overwritten automatically"):
        person_package.ensure_person_package("cheetah_s", tmp_path)
    if existing == "file":
        assert target.read_text() == "keep"
    elif existing == "dangling_symlink":
        assert target.is_symlink()
    else:
        assert sorted(path.name for path in target.iterdir()) == (["user.txt"] if existing == "partial" else [])


@pytest.mark.parametrize("problem, message", [
    ("corrupt", "SHA-256 mismatch"),
    ("missing_model", "model file does not exist"),
    ("missing_license", "missing MODEL.LICENSE"),
    ("wrong_model_id", "model_id does not match package name"),
])
def test_invalid_download_never_becomes_installed(tmp_path, monkeypatch, source_package, problem, message):
    source = source_package()
    if problem == "corrupt":
        (source / "person_reid.onnx").write_bytes(b"invalid")
    elif problem == "wrong_model_id":
        manifest = source / "manifest.json"
        document = json.loads(manifest.read_text())
        document['model_id'] = 'another_model'
        manifest.write_text(json.dumps(document))
    else:
        (source / ("person_reid.onnx" if problem == "missing_model" else "MODEL.LICENSE")).unlink()
    stage_download(monkeypatch, source)
    root = tmp_path / "installed"
    with pytest.raises(RuntimeError, match=message):
        person_package.ensure_person_package("cheetah_s", root)
    assert list((root / "models").iterdir()) == []


def test_download_accepts_v2_defaults_and_optional_hashes(tmp_path, monkeypatch, source_package):
    source = source_package()
    path = source / 'manifest.json'
    document = json.loads(path.read_text())
    document['display_name'] = 'Cheetah S'
    document['future_metadata'] = {'ignored': True}
    for descriptor in document['tasks'].values():
        descriptor.pop('sha256')
        descriptor.pop('input_size', None)
        descriptor.pop('preprocessing', None)
    path.write_text(json.dumps(document))
    original = path.read_bytes()
    stage_download(monkeypatch, source)
    root = tmp_path / 'installed'
    result = person_package.ensure_person_package(source.name, root)
    assert result['display_name'] == 'Cheetah S'
    assert result['tasks']['detection']['input_size'] == [640, 640]
    assert result['tasks']['recognition']['preprocessing'] == {'mean': 127.5, 'std': 127.5}
    assert all(len(descriptor['sha256']) == 64 for descriptor in result['tasks'].values())
    assert (root / 'models' / source.name / 'manifest.json').read_bytes() == original


@pytest.mark.parametrize("complete", [False, True])
def test_local_package_created_during_download_is_preserved(tmp_path, monkeypatch, source_package, complete):
    source = source_package()
    root = tmp_path / "installed"
    target = root / "models" / source.name

    def create_local():
        if complete:
            shutil.copytree(source, target)
        else:
            target.mkdir()
        (target / "user.txt").write_text("keep")

    stage_download(monkeypatch, source, after_download=create_local)
    if complete:
        assert person_package.ensure_person_package(source.name, root)["model_id"] == source.name
    else:
        with pytest.raises(RuntimeError, match="will not be overwritten"):
            person_package.ensure_person_package(source.name, root)
    assert (target / "user.txt").read_text() == "keep"
    assert not list(target.parent.glob(".*-download-*"))


@pytest.mark.parametrize("entry", ["../outside", "/outside", "nested/../../outside", "nested\\outside", "C:/outside"])
def test_archive_paths_are_checked_before_any_extraction(tmp_path, entry):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("safe.txt", "must not be written")
        bundle.writestr(entry, "unsafe")
    destination = tmp_path / "unpacked"
    with pytest.raises(ValueError, match="unsafe model archive entry"):
        person_package._extract_person_archive(archive, destination)
    assert not destination.exists()


@pytest.mark.parametrize("kind", ["symlink", "duplicate"])
def test_archive_rejects_symlinks_and_duplicate_entries(tmp_path, kind):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        if kind == "symlink":
            info = zipfile.ZipInfo("link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            bundle.writestr(info, "../outside")
        else:
            bundle.writestr("same", "first")
            with pytest.warns(UserWarning, match="Duplicate name"):
                bundle.writestr("same", "second")
    with pytest.raises(ValueError, match="unsafe model archive entry"):
        person_package._extract_person_archive(archive, tmp_path / "unpacked")


def test_download_failure_reports_manual_install_path_and_cleans_staging(tmp_path, monkeypatch):
    requests = []

    def unavailable(request, **kwargs):
        requests.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(person_package.urllib.request, "urlopen", unavailable)
    with pytest.raises(RuntimeError, match="Install its complete archive manually") as error:
        person_package.ensure_person_package("cheetah_s", tmp_path)
    assert requests == ["https://github.com/deepinsight/insightface/releases/download/model-zoo/cheetah_s.zip"]
    assert str(tmp_path / "models" / "cheetah_s") in str(error.value)
    assert "404" in str(error.value)
    assert list((tmp_path / "models").iterdir()) == []


def test_download_uses_timeout_and_streams_response(tmp_path, monkeypatch):
    def response(request, *, timeout):
        assert request.full_url == "https://example.test/cheetah_s.zip"
        assert timeout == 120
        return io.BytesIO(b"archive bytes")
    monkeypatch.setattr(person_package.urllib.request, "urlopen", response)
    archive = tmp_path / "package.zip"
    person_package._download_person_archive("https://example.test/cheetah_s.zip", archive)
    assert archive.read_bytes() == b"archive bytes"


def test_prepare_models_ensures_package_before_opening_sessions(monkeypatch, tmp_path):
    def ensure(name, root):
        assert name == "cheetah_s" and root == tmp_path
        raise FileNotFoundError("download entry reached")
    monkeypatch.setattr(person_package, "ensure_person_package", ensure)
    with pytest.raises(FileNotFoundError, match="download entry reached"):
        person_package.prepare_models("cheetah_s", tmp_path)
