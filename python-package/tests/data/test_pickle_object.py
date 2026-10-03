import pickle
import sys

import pytest

from insightface.data import pickle_object


@pytest.fixture
def object_layout(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    package = bundle / "insightface" / "data"
    package.mkdir(parents=True)
    monkeypatch.setattr(pickle_object, "__file__", str(package / "pickle_object.py"))
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    return bundle, package


def _write_object(directory, value):
    objects = directory / "objects"
    objects.mkdir(parents=True)
    (objects / "meanshape_68.pkl").write_bytes(pickle.dumps(value))


@pytest.mark.parametrize("name", ["meanshape_68", "meanshape_68.pkl"])
@pytest.mark.parametrize("layout", ["normal", "package", "bundle", "both"])
def test_get_object_layouts(object_layout, monkeypatch, name, layout):
    bundle, package = object_layout
    monkeypatch.setattr(sys, "frozen", layout != "normal", raising=False)
    if layout in ("normal", "package", "both"):
        _write_object(package, {"source": "package", "points": [3, 7, 11]})
    if layout in ("bundle", "both"):
        _write_object(bundle, {"source": "bundle", "points": [2, 5, 13]})
    expected = (
        {"source": "bundle", "points": [2, 5, 13]}
        if layout in ("bundle", "both")
        else {"source": "package", "points": [3, 7, 11]}
    )

    assert pickle_object.get_object(name) == expected


@pytest.mark.parametrize("frozen", [False, True])
def test_get_object_missing(object_layout, monkeypatch, capsys, frozen):
    monkeypatch.setattr(sys, "frozen", frozen, raising=False)

    assert pickle_object.get_object("meanshape_68") is None
    assert "[Error] File not found:" in capsys.readouterr().out


def test_get_object_normal_install_ignores_bundle_root(object_layout, monkeypatch):
    bundle, _ = object_layout
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    _write_object(bundle, {"source": "bundle"})

    assert pickle_object.get_object("meanshape_68") is None
