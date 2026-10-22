from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _AlbumEngine:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.calls: list[str] = []

    def is_loaded(self) -> bool:
        return True

    def detect_faces(self, image, source_path=None):
        del image
        self.calls.append(str(source_path))
        return [
            SimpleNamespace(
                normed_embedding=np.array([1.0, 0.0], dtype=np.float32),
                bbox=np.array([0.0, 0.0, 100.0, 100.0], dtype=np.float32),
                crop=np.zeros((100, 100, 3), dtype=np.uint8),
                kps=None,
                det_score=0.99,
                quality_score=0.8,
            )
        ]


def test_album_background_job_keeps_engine_and_model_snapshot(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.storage import Storage
    from insightface.gui.pages import album_page as album_module
    from insightface.gui.pages.album_page import AlbumPage

    configure_qt_plugin_paths()
    app = QApplication.instance() or QApplication([])
    assert app is not None
    album_dir = tmp_path / "album"
    album_dir.mkdir()
    image_path = album_dir / "face.jpg"
    image_path.write_bytes(b"test image placeholder")

    storage = Storage(tmp_path / "album.db")
    original_engine = _AlbumEngine("raccoon_s")
    replacement_engine = _AlbumEngine("buffalo_l")
    context = SimpleNamespace(
        config=SimpleNamespace(ui_language="en"),
        storage=storage,
        engine=original_engine,
    )
    page = AlbumPage(context)
    page.folder_list.addItem(str(album_dir))

    monkeypatch.setattr(
        album_module,
        "read_image",
        lambda _path: np.zeros((160, 160, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        album_module,
        "encode_webp_thumbnail",
        lambda *_args, **_kwargs: b"thumbnail",
    )
    captured = {}

    def capture_task(_title, fn, on_result=None, **_kwargs):
        captured["fn"] = fn
        captured["on_result"] = on_result

    monkeypatch.setattr(page, "run_task", capture_task)

    page._run_import_refresh()
    context.engine = replacement_engine
    result = captured["fn"]()

    assert original_engine.calls == [str(image_path)]
    assert replacement_engine.calls == []
    assert result["model_name"] == "raccoon_s"
    faces = storage.list_media_faces(model_name="raccoon_s")
    assert len(faces) == 1
    assert faces[0]["model_name"] == "raccoon_s"
    assert storage.list_media_faces(model_name="buffalo_l") == []
    assert storage.load_album_results("raccoon_s")["clusters"]

    captured["on_result"](result)

    assert page.clusters == []
    assert "active model changed" in page.status_label.text()


def test_album_refresh_reuses_only_active_model_results(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.storage import Storage
    from insightface.gui.pages.album_page import AlbumPage

    configure_qt_plugin_paths()
    app = QApplication.instance() or QApplication([])
    assert app is not None
    storage = Storage(tmp_path / "album.db")
    image_path = str(tmp_path / "face.jpg")
    media_id = storage.add_media_item(image_path, "image")
    raccoon_face_id = storage.add_media_face(
        media_id,
        np.array([1.0, 0.0], dtype=np.float32),
        bbox=[0, 0, 100, 100],
        model_name="raccoon_s",
    )
    buffalo_face_id = storage.add_media_face(
        media_id,
        np.array([0.0, 1.0], dtype=np.float32),
        bbox=[0, 0, 100, 100],
        model_name="buffalo_l",
    )
    storage.save_album_results(
        [{"id": 1, "name": "Raccoon", "face_count": 1, "photo_count": 1}],
        {1: [{"id": raccoon_face_id}]},
        "DBSCAN",
        model_name="raccoon_s",
    )
    storage.save_album_results(
        [{"id": 1, "name": "Buffalo", "face_count": 1, "photo_count": 1}],
        {1: [{"id": buffalo_face_id}]},
        "DBSCAN",
        model_name="buffalo_l",
    )
    raccoon_engine = _AlbumEngine("raccoon_s")
    context = SimpleNamespace(
        config=SimpleNamespace(ui_language="en"),
        storage=storage,
        engine=raccoon_engine,
    )
    page = AlbumPage(context)

    page.refresh()
    assert [cluster["name"] for cluster in page.clusters] == ["Raccoon"]
    assert [face["model_name"] for face in page.cluster_items[1]] == [
        "raccoon_s"
    ]

    context.engine = _AlbumEngine("buffalo_l")
    page.refresh()
    assert [cluster["name"] for cluster in page.clusters] == ["Buffalo"]
    assert [face["model_name"] for face in page.cluster_items[1]] == [
        "buffalo_l"
    ]
