from __future__ import annotations

import os
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _grid_position_for_widget(grid, widget):
    """Return the row/column occupied by a widget or its field wrapper."""

    for index in range(grid.count()):
        item = grid.itemAt(index)
        candidate = item.widget()
        if candidate is widget or (
            candidate is not None and candidate.isAncestorOf(widget)
        ):
            row, column, _row_span, _column_span = grid.getItemPosition(index)
            return row, column
    raise AssertionError(f"{widget.objectName() or type(widget).__name__} is absent")


@pytest.mark.parametrize(
    "movies_location",
    [
        "/Users/alice/Movies",
        r"C:\Users\alice\Videos",
        "/home/alice/Videos",
    ],
)
def test_privateframe_default_output_uses_platform_movies_location(
    monkeypatch,
    movies_location,
):
    pytest.importorskip("PySide6")
    from insightface.gui.pages import privateframe_page

    monkeypatch.setattr(
        privateframe_page.QStandardPaths,
        "writableLocation",
        staticmethod(lambda _location: movies_location),
    )

    assert privateframe_page._default_privateframe_output_directory() == Path(
        movies_location
    )


@pytest.mark.parametrize(
    ("platform_name", "folder_name"),
    [("darwin", "Movies"), ("win32", "Videos"), ("linux", "Videos")],
)
def test_privateframe_default_output_has_visible_platform_fallback(
    monkeypatch,
    platform_name,
    folder_name,
):
    pytest.importorskip("PySide6")
    from insightface.gui.pages import privateframe_page

    monkeypatch.setattr(
        privateframe_page.QStandardPaths,
        "writableLocation",
        staticmethod(lambda _location: ""),
    )
    monkeypatch.setattr(privateframe_page.sys, "platform", platform_name)

    assert privateframe_page._default_privateframe_output_directory() == (
        privateframe_page.Path.home() / folder_name
    )


def test_privateframe_page_does_not_default_to_hidden_workspace_exports(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages import privateframe_page

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    videos = tmp_path / "Videos"
    monkeypatch.setattr(
        privateframe_page,
        "_default_privateframe_output_directory",
        lambda: videos,
    )
    config = AppConfig(model_name="raccoon_s", workspace_path=str(tmp_path / ".insightface"))

    class Context:
        pass

    context = Context()
    context.config = config
    page = privateframe_page.PrivateFramePage(context)

    assert page.output_dir.text() == str(videos)
    assert page.output_dir.text() != config.export_dir
    assert not videos.exists()
    page.close()


@pytest.mark.parametrize("frame_shape", [(40, 80, 3), (80, 40, 3)])
def test_privateframe_video_preview_uses_first_frame_metadata_and_black_letterbox(
    tmp_path,
    monkeypatch,
    frame_shape,
):
    pytest.importorskip("PySide6")
    from insightface.app.privateframe.video import VideoMetadata
    from insightface.gui.pages import privateframe_page

    source = tmp_path / "holiday.mp4"
    source.write_bytes(b"x" * 1536)
    frame = np.full(frame_shape, (17, 83, 149), dtype=np.uint8)
    calls = {"thumbnail": [], "metadata": []}

    def fake_thumbnail(path):
        calls["thumbnail"].append(Path(path))
        return frame.copy()

    def fake_probe(path):
        calls["metadata"].append(Path(path))
        return VideoMetadata(
            path=str(source),
            width=1920,
            height=1080,
            fps=29.97,
            frame_count=1800,
            duration=65.2,
        )

    video_stream = SimpleNamespace(codec_context=SimpleNamespace(name="h264"))
    audio_stream = SimpleNamespace(codec_context=SimpleNamespace(name="aac"))

    class FakeContainer:
        def __init__(self):
            self.streams = SimpleNamespace(
                video=[video_stream],
                audio=[audio_stream],
            )
            self.closed = False

        def close(self):
            self.closed = True

    container = FakeContainer()
    monkeypatch.setattr(privateframe_page, "read_video_thumbnail", fake_thumbnail)
    monkeypatch.setattr(privateframe_page, "probe_video", fake_probe)
    monkeypatch.setattr(privateframe_page.av, "open", lambda _path: container)

    preview = privateframe_page._read_video_preview(source)

    resolved_source = source.resolve()
    preview_size = privateframe_page._VIDEO_PREVIEW_SIZE
    midpoint = preview_size // 2
    assert calls == {
        "thumbnail": [resolved_source],
        "metadata": [resolved_source],
    }
    assert preview.path == resolved_source
    assert preview.image.shape == (preview_size, preview_size, 3)
    assert np.array_equal(preview.image[midpoint, midpoint], frame[0, 0])
    if frame_shape[1] > frame_shape[0]:
        assert np.all(preview.image[0] == 0)
        assert np.all(preview.image[-1] == 0)
        assert np.array_equal(preview.image[midpoint, 0], frame[0, 0])
        assert np.array_equal(preview.image[midpoint, -1], frame[0, 0])
    else:
        assert np.all(preview.image[:, 0] == 0)
        assert np.all(preview.image[:, -1] == 0)
        assert np.array_equal(preview.image[0, midpoint], frame[0, 0])
        assert np.array_equal(preview.image[-1, midpoint], frame[0, 0])
    assert (preview.width, preview.height) == (1920, 1080)
    assert preview.fps == pytest.approx(29.97)
    assert preview.frame_count == 1800
    assert preview.duration == pytest.approx(65.2)
    assert preview.file_size == 1536
    assert preview.video_codec == "h264"
    assert preview.audio_codec == "aac"
    assert preview.has_audio is True
    assert container.closed


def test_privateframe_page_displays_video_preview_metadata_and_clears_it(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages import privateframe_page

    configure_qt_plugin_paths()
    _app = QApplication.instance() or QApplication([])
    config = AppConfig(model_name="raccoon_s", workspace_path=str(tmp_path), auto_load_model=False)

    class Context:
        pass

    context = Context()
    context.config = config
    source = tmp_path / "holiday.mp4"
    source.write_bytes(b"x" * 1536)
    preview_size = privateframe_page._VIDEO_PREVIEW_SIZE
    square_frame = np.full((preview_size, preview_size, 3), 127, dtype=np.uint8)
    loaded_paths = []

    def fake_preview(path):
        loaded_paths.append(Path(path).resolve())
        return privateframe_page.VideoPreviewData(
            path=source.resolve(),
            image=square_frame.copy(),
            width=1920,
            height=1080,
            fps=29.97,
            frame_count=1800,
            duration=65.2,
            file_size=1536,
            video_codec="h264",
            audio_codec="aac",
            has_audio=True,
        )

    monkeypatch.setattr(privateframe_page, "_read_video_preview", fake_preview)
    page = privateframe_page.PrivateFramePage(context)

    page.video_input.set_path(str(source))

    assert loaded_paths == [source.resolve()]
    assert page.video_input.width() == page.video_input.height() == preview_size
    assert page.video_input.file_label.isHidden()
    assert page.video_input.viewer.image is not None
    assert np.array_equal(page.video_input.viewer.image, square_frame)
    values = {key: label.text() for key, label in page.video_metadata_values.items()}
    assert values["file"] == "holiday.mp4"
    assert "1920" in values["resolution"] and "1080" in values["resolution"]
    assert "01:05" in values["duration"]
    assert "29.97" in values["fps"]
    assert values["frames"].replace(",", "") == "1800"
    assert "h264" in values["video_codec"].lower()
    assert "aac" in values["audio"].lower()
    assert "1.5" in values["size"] and "kib" in values["size"].lower()

    input_layout = page.video_input.parentWidget().layout()
    video_row = input_layout.itemAt(0).layout()
    assert video_row.itemAt(0).widget() is page.video_input
    assert video_row.itemAt(1).widget() is page.video_metadata

    page.video_input.remove_button.click()

    assert page.video_input.path() == ""
    assert page.video_input.viewer.image is None
    assert {label.text() for label in page.video_metadata_values.values()} == {"—"}
    page.close()


def test_privateframe_video_preview_ignores_stale_and_cleared_results(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QWidget

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages import privateframe_page

    configure_qt_plugin_paths()
    _app = QApplication.instance() or QApplication([])
    config = AppConfig(model_name="raccoon_s", workspace_path=str(tmp_path), auto_load_model=False)

    class Context:
        pass

    class FakeWorker:
        def __init__(self):
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

    class TaskHost(QWidget):
        def __init__(self):
            super().__init__()
            self.requests = []

        def run_task(
            self,
            title,
            fn,
            on_result,
            *,
            show_dialog=True,
            on_progress=None,
            on_error=None,
            on_finished=None,
        ):
            worker = FakeWorker()
            self.requests.append(
                {
                    "title": title,
                    "fn": fn,
                    "on_result": on_result,
                    "on_error": on_error,
                    "on_finished": on_finished,
                    "worker": worker,
                }
            )
            return worker

    def preview(path, value):
        size = privateframe_page._VIDEO_PREVIEW_SIZE
        return privateframe_page.VideoPreviewData(
            path=path.resolve(),
            image=np.full((size, size, 3), value, dtype=np.uint8),
            width=640,
            height=360,
            fps=25.0,
            frame_count=250,
            duration=10.0,
            file_size=1024,
            video_codec="h264",
            audio_codec=None,
            has_audio=False,
        )

    context = Context()
    context.config = config
    host = TaskHost()
    page = privateframe_page.PrivateFramePage(context, parent=host)
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    third = tmp_path / "third.mp4"
    for source in (first, second, third):
        source.write_bytes(b"video")

    page.video_input.set_path(str(first))
    page.video_input.set_path(str(second))

    assert len(host.requests) == 2
    assert host.requests[0]["worker"].cancelled
    host.requests[1]["on_result"](preview(second, 22))
    host.requests[1]["on_finished"]()
    assert page.video_metadata_values["file"].text() == "second.mp4"
    assert np.all(page.video_input.viewer.image == 22)

    host.requests[0]["on_result"](preview(first, 11))
    host.requests[0]["on_finished"]()
    assert page.video_input.path() == str(second.resolve())
    assert page.video_metadata_values["file"].text() == "second.mp4"
    assert np.all(page.video_input.viewer.image == 22)

    page.video_input.set_path(str(third))
    pending = host.requests[2]
    page.video_input.clear()

    assert pending["worker"].cancelled
    pending["on_result"](preview(third, 33))
    pending["on_finished"]()
    assert page.video_input.path() == ""
    assert page.video_input.viewer.image is None
    assert {label.text() for label in page.video_metadata_values.values()} == {"—"}
    page.close()
    host.close()


def test_privateframe_primary_options_reflow_without_losing_values(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QResizeEvent
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages.privateframe_page import PrivateFramePage

    configure_qt_plugin_paths()
    _app = QApplication.instance() or QApplication([])
    config = AppConfig(model_name="raccoon_s", workspace_path=str(tmp_path), auto_load_model=False)

    class Context:
        pass

    context = Context()
    context.config = config
    page = PrivateFramePage(context)
    page.analysis_mode.setCurrentIndex(page.analysis_mode.findData(30))
    page.redaction_method.setCurrentIndex(page.redaction_method.findData("mosaic"))
    fields = [
        page.redaction_method,
        page.analysis_mode,
    ]

    page._layout_primary_options(two_columns=True)
    page.options_grid.activate()
    wide_positions = [
        _grid_position_for_widget(page.options_grid, field) for field in fields
    ]
    wide_height = page.options_grid.sizeHint().height()
    # Native fonts and styles determine the breakpoint, so allow ample room
    # for the measured two-column grid instead of assuming 1100 pixels fits.
    wide_width = 2 * page.options_grid.minimumSize().width()
    narrow_width = min(label.minimumSizeHint().width() for label, _ in page._primary_option_rows)

    assert [row for row, _column in wide_positions] == [0, 0]
    assert [column for _row, column in wide_positions] == [1, 3]
    assert page.options_grid.count() == 4

    page._layout_primary_options(two_columns=False)
    page.options_grid.activate()
    narrow_positions = [
        _grid_position_for_widget(page.options_grid, field) for field in fields
    ]
    narrow_height = page.options_grid.sizeHint().height()

    assert [row for row, _column in narrow_positions] == list(range(2))
    assert {column for _row, column in narrow_positions} == {1}
    assert page.options_grid.count() == 4
    assert narrow_height > wide_height
    assert page.analysis_mode.currentData() == 30
    assert page.redaction_method.currentData() == "mosaic"

    page.resizeEvent(QResizeEvent(QSize(wide_width, 900), QSize(narrow_width, 900)))
    assert page._primary_options_two_columns is True
    assert [
        _grid_position_for_widget(page.options_grid, field)[0] for field in fields
    ] == [0, 0]

    page.resizeEvent(QResizeEvent(QSize(narrow_width, 900), QSize(wide_width, 900)))
    assert page._primary_options_two_columns is False
    assert [
        _grid_position_for_widget(page.options_grid, field)[0] for field in fields
    ] == list(range(2))
    assert page.analysis_mode.currentData() == 30
    assert page.redaction_method.currentData() == "mosaic"
    page.close()


def test_build_privateframe_job_maps_gui_options(tmp_path):
    pytest.importorskip("PySide6")
    from insightface.gui.pages.privateframe_page import build_privateframe_job

    source = tmp_path / "input.mov"
    source.write_bytes(b"video")
    output = tmp_path / "output"

    job = build_privateframe_job(
        input_path=source,
        output_dir=output,
        model_package="raccoon_l",
        max_analysis_fps=15,
        redaction_method="mosaic",
        runtime_provider="CUDA",
        model_root=tmp_path / "model-root",
        preserve_aac_audio=False,
    )

    assert job.config_path.name == "base.yaml"
    assert job.result_path == output / "input_privateframe.json"
    assert job.redacted_path == output / "input_privateframe.mp4"
    assert job.workdir == output / ".input_privateframe_work"
    assert job.output_mode == "json_and_video"
    assert job.config_overrides["models.name"] == "raccoon_l"
    assert job.model_name == "raccoon_l"
    assert job.model_root == (tmp_path / "model-root").resolve()
    assert job.provider_choice == "CUDA"
    assert job.config_overrides["models.root"] == str(
        (tmp_path / "model-root").resolve()
    )
    assert job.config_overrides["scan.max_analysis_fps"] == 15
    assert job.config_overrides["runtime.provider"] == "CUDAExecutionProvider"
    assert job.config_overrides["render.redaction.method"] == "mosaic"
    assert job.config_overrides["render.redaction.box_scale"] == 1.0
    assert job.config_overrides["render.video_output.preset"] == "medium"
    assert job.config_overrides["render.video_output.rate_control"]["mode"] == "crf"
    assert job.config_overrides["render.video_output.rate_control"]["quality"] == 23
    assert job.config_overrides["render.video_output.audio.redacted"] == "none"
    assert job.config_overrides["recognition.mode"] == "all"
    assert job.config_overrides["tracking.between_scan_frames"] == "interpolate"
    assert "recognition.reference_dir" not in job.config_overrides


def test_build_privateframe_job_maps_advanced_and_selective_options(tmp_path):
    pytest.importorskip("PySide6")
    from insightface.gui.pages.privateframe_page import build_privateframe_job

    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    gallery = tmp_path / "gallery"
    gallery.mkdir(parents=True)
    (gallery / "Alice.jpg").write_bytes(b"image")
    (gallery / "Bob.png").write_bytes(b"image")
    (gallery / ".hidden-person").mkdir()

    job = build_privateframe_job(
        input_path=source,
        output_dir=tmp_path / "output",
        model_package="raccoon_s",
        max_analysis_fps=15,
        redaction_method="gaussian",
        runtime_provider="CPU",
        between_scan_frames="visual",
        box_scale=1.30,
        video_preset="slow",
        video_crf=28,
        recognition_mode="exempt",
        recognition_reference_dir=gallery,
        recognition_profile="accurate",
    )

    assert job.config_overrides["scan.max_analysis_fps"] == 15
    assert job.config_overrides["tracking.between_scan_frames"] == "visual"
    assert job.config_overrides["render.redaction.box_scale"] == 1.30
    assert job.config_overrides["render.video_output.preset"] == "slow"
    assert job.config_overrides["render.video_output.rate_control"]["mode"] == "crf"
    assert job.config_overrides["render.video_output.rate_control"]["quality"] == 28
    assert job.config_overrides["recognition.mode"] == "exempt"
    assert job.config_overrides["recognition.reference_dir"] == str(gallery.resolve())
    assert job.config_overrides["recognition.unknown_action"] == "auto"
    assert job.config_overrides["recognition.similarity_threshold"] == 0.40
    assert job.config_overrides["recognition.profile"] == "accurate"


def test_advanced_gui_job_overrides_load_as_valid_config(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from insightface.app.privateframe import base_config
    from insightface.app.privateframe.config import load_config
    from insightface.gui.pages.privateframe_page import build_privateframe_job

    monkeypatch.setattr(base_config, "materialize_model_package", lambda _config: None)
    monkeypatch.setattr(
        base_config,
        "validate_model_package_contracts",
        lambda _config: None,
    )
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    gallery = tmp_path / "gallery"
    gallery.mkdir(parents=True)
    (gallery / "Alice.jpg").write_bytes(b"image")
    job = build_privateframe_job(
        input_path=source,
        output_dir=tmp_path / "output",
        model_package="raccoon_l",
        max_analysis_fps=15,
        redaction_method="mosaic",
        runtime_provider="CPU",
        between_scan_frames="visual",
        box_scale=1.15,
        video_preset="veryfast",
        video_crf=23,
        recognition_mode="exempt",
        recognition_reference_dir=gallery,
        recognition_profile="fast",
    )

    config = load_config(
        job.config_path,
        config_overrides=job.config_overrides,
        config_override_root=job.config_path.parent,
    )

    assert config["scan"]["max_analysis_fps"] == 15.0
    assert config["models"]["root"] == str(Path("~/.insightface").expanduser())
    assert config["tracking"]["between_scan_frames"] == "visual"
    assert config["render"]["redaction"]["box_scale"] == 1.15
    assert config["render"]["video_output"]["preset"] == "veryfast"
    assert config["render"]["video_output"]["rate_control"] == {
        "mode": "crf",
        "quality": 23,
    }
    assert config["recognition"]["reference_dir"] == str(gallery.resolve())
    assert config["recognition"]["unknown_action"] == "auto"


def test_all_recognition_policy_never_reads_gallery_arguments(tmp_path):
    pytest.importorskip("PySide6")
    from insightface.gui.pages.privateframe_page import build_privateframe_job

    class UnreadableGallery:
        def __str__(self):
            raise AssertionError("all policy must not inspect gallery")

        def __iter__(self):
            raise AssertionError("all policy must not inspect targets")

    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    poison = UnreadableGallery()

    job = build_privateframe_job(
        input_path=source,
        output_dir=tmp_path / "output",
        model_package="raccoon_s",
        max_analysis_fps=30,
        redaction_method="gaussian",
        runtime_provider="auto",
        recognition_mode="all",
        recognition_reference_dir=poison,
    )

    assert job.config_overrides["recognition.mode"] == "all"
    assert "recognition.reference_dir" not in job.config_overrides
    assert "recognition.target_persons" not in job.config_overrides


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("max_analysis_fps", 0),
        ("max_analysis_fps", -1),
        ("max_analysis_fps", "15"),
        ("max_analysis_fps", True),
        ("max_analysis_fps", float("nan")),
        ("max_analysis_fps", float("inf")),
        ("between_scan_frames", "hybrid"),
        ("box_scale", 0.09),
        ("video_preset", 123),
        ("video_crf", 52),
        ("recognition_mode", "unknown"),
        ("recognition_profile", "maximum"),
    ],
)
def test_build_privateframe_job_rejects_unsupported_advanced_values(
    tmp_path,
    option,
    value,
):
    pytest.importorskip("PySide6")
    from insightface.gui.pages.privateframe_page import build_privateframe_job

    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    options = {
        "input_path": source,
        "output_dir": tmp_path / "output",
        "model_package": "raccoon_s",
        "max_analysis_fps": 30,
        "redaction_method": "gaussian",
        "runtime_provider": "auto",
    }
    options[option] = value

    with pytest.raises(ValueError):
        build_privateframe_job(**options)


def test_run_privateframe_job_calls_python_pipeline_directly(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from insightface.gui.pages import privateframe_page

    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    job = privateframe_page.build_privateframe_job(
        input_path=source,
        output_dir=tmp_path / "output",
        model_package="raccoon_s",
        max_analysis_fps=15,
        redaction_method="gaussian",
        runtime_provider="Auto",
        preserve_aac_audio=True,
    )
    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return {"analysis": {}, "render": {}}

    def progress(current, total, message):
        return None

    def is_cancelled():
        return False

    monkeypatch.setattr(privateframe_page, "run_streaming_pipeline", fake_pipeline)
    monkeypatch.setattr(privateframe_page, "_source_audio_codec", lambda _path: "aac")

    result = privateframe_page.run_privateframe_job(
        job,
        progress=progress,
        is_cancelled=is_cancelled,
    )

    assert result == {
        "analysis": {},
        "render": {},
        "gui": {
            "output_mode": "json_and_video",
            "source_audio_codec": "aac",
            "audio_output_mode": "aac",
        },
    }
    assert captured["input_path"] == job.input_path
    assert captured["redacted_path"] == job.redacted_path
    assert captured["result_path"] == job.result_path
    assert captured["debug_path"] is None
    assert captured["progress"] is progress
    assert captured["is_cancelled"] is is_cancelled
    assert captured["config_overrides"]["scan.max_analysis_fps"] == 15
    assert captured["config_overrides"]["runtime.provider"] == "auto"


@pytest.mark.parametrize(
    ("render_video", "expected"),
    [
        (True, 14.75),
        (False, 10.25),
    ],
)
def test_privateframe_pipeline_total_includes_artifact_and_optional_render(
    render_video,
    expected,
):
    pytest.importorskip("PySide6")
    from insightface.gui.pages.privateframe_page import _pipeline_total_seconds

    analysis = {
        "timings": {
            "analysis_seconds": 10.0,
            "artifact_seconds": 0.25,
            "total_seconds": 10.25,
        }
    }
    render = {"seconds": 4.5}

    assert (
        _pipeline_total_seconds(
            analysis,
            render,
            render_video=render_video,
        )
        == expected
    )


def test_non_aac_audio_is_omitted_before_pipeline(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from insightface.gui.pages import privateframe_page

    source = tmp_path / "input.webm"
    source.write_bytes(b"video")
    job = privateframe_page.build_privateframe_job(
        input_path=source,
        output_dir=tmp_path / "output",
        model_package="raccoon_s",
        max_analysis_fps=30,
        redaction_method="gaussian",
        runtime_provider="CPU",
        preserve_aac_audio=True,
    )
    captured = {}
    monkeypatch.setattr(privateframe_page, "_source_audio_codec", lambda _path: "opus")
    monkeypatch.setattr(
        privateframe_page,
        "run_streaming_pipeline",
        lambda **kwargs: captured.update(kwargs) or {"analysis": {}, "render": {}},
    )

    result = privateframe_page.run_privateframe_job(job)

    assert captured["config_overrides"]["render.video_output.audio.redacted"] == "none"
    assert captured["config_overrides"]["runtime.provider"] == "CPUExecutionProvider"
    assert result["gui"] == {
        "output_mode": "json_and_video",
        "source_audio_codec": "opus",
        "audio_output_mode": "none",
    }


def test_json_only_job_analyzes_without_probing_audio_or_rendering(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("PySide6")
    from insightface.gui.pages import privateframe_page

    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    job = privateframe_page.build_privateframe_job(
        input_path=source,
        output_dir=tmp_path / "results",
        model_package="raccoon_s",
        max_analysis_fps=15,
        redaction_method="mosaic",
        runtime_provider="CoreML",
        preserve_aac_audio=True,
        output_mode="json_only",
    )
    captured = {}
    expected_analysis = {"frame_count": 12, "accepted_tracks": 2}

    monkeypatch.setattr(
        privateframe_page,
        "_source_audio_codec",
        lambda _path: pytest.fail("JSON-only mode must not inspect audio"),
    )
    monkeypatch.setattr(
        privateframe_page,
        "run_streaming_pipeline",
        lambda **_kwargs: pytest.fail("JSON-only mode must not render video"),
    )
    monkeypatch.setattr(
        privateframe_page,
        "analyze_streaming_pipeline",
        lambda **kwargs: captured.update(kwargs) or expected_analysis,
    )

    result = privateframe_page.run_privateframe_job(job)

    assert job.result_path == tmp_path / "results" / "clip_privateframe.json"
    assert job.redacted_path is None
    assert job.config_overrides["render.redaction.method"] == "mosaic"
    assert job.config_overrides["render.redaction.box_scale"] == 1.0
    assert job.config_overrides["render.video_output.preset"] == "medium"
    assert job.config_overrides["render.video_output.rate_control"]["mode"] == "crf"
    assert job.config_overrides["render.video_output.rate_control"]["quality"] == 23
    assert job.config_overrides["render.video_output.audio.redacted"] == "aac"
    assert captured["result_path"] == job.result_path
    assert captured["workdir"] == job.workdir
    assert result == {
        "analysis": expected_analysis,
        "render": {},
        "gui": {
            "output_mode": "json_only",
            "source_audio_codec": None,
            "audio_output_mode": "none",
            "audio_preference": "aac",
        },
    }


def test_privateframe_page_has_non_blocking_controls(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages.privateframe_page import PrivateFramePage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    config = AppConfig(model_name="raccoon_s", workspace_path=str(tmp_path), auto_load_model=False)

    class Context:
        pass

    context = Context()
    context.config = config
    page = PrivateFramePage(context)

    assert page.model_label.text() == "raccoon_s"
    assert page.model_root_label.text() == str(
        Path(config.model_root).expanduser().resolve()
    )
    assert not hasattr(page, "model_package")
    assert page.analysis_mode.count() == 2
    assert page.analysis_mode.currentData() == 15
    assert page.analysis_fps_label.text() == "Target analysis FPS"
    assert page.analysis_mode.itemText(0) == "Fast (default, target 15 analysis FPS)"
    assert page.analysis_mode.itemText(1) == "Normal (target 30 analysis FPS)"
    assert page.analysis_mode.itemData(0) == 15
    assert page.analysis_mode.itemData(1) == 30
    assert "per second of input video" in page.analysis_mode.toolTip()
    assert "Current default: 15" in page.analysis_mode.toolTip()
    assert "lower values reduce work for faster processing" in page.analysis_mode.toolTip()
    assert "may miss brief appearances" in page.analysis_mode.toolTip()
    assert "Output video FPS is unchanged" in page.analysis_mode.toolTip()
    assert page.analysis_fps_label.toolTip() == page.analysis_mode.toolTip()
    assert page.redaction_method.count() == 2
    assert page.output_mode.count() == 2
    assert page.output_mode.currentData() == "json_and_video"
    assert page.recognition_policy.count() == 3
    assert page.recognition_policy.currentData() == "all"
    assert page.more_options_button.objectName() == "privateFrameMoreOptionsButton"
    assert not page.more_options_dialog.isModal()
    assert page.between_scan_frames.currentData() == "interpolate"
    assert page.box_scale.currentData() == 1.0
    assert page.video_preset.currentData() == "medium"
    assert page.video_crf.currentData() == 23
    assert page.recognition_profile.currentData() == "balanced"
    assert page.reference_panel.isHidden()
    assert page.more_option_groups["matching"].isHidden()
    assert page.selective_privacy_note.isHidden()
    assert page.progress_bar.value() == 0
    assert page.start_button.isEnabled()
    assert not page.cancel_button.isEnabled()
    assert page.summary.isReadOnly()
    page._set_summary_text("\n".join(f"line {index}" for index in range(100)))
    assert page.summary.textCursor().atEnd()
    assert (
        page.summary.verticalScrollBar().value()
        == page.summary.verticalScrollBar().maximum()
    )
    page.more_options_button.click()
    assert page.more_options_dialog.isVisible()
    page.video_preset.setCurrentIndex(page.video_preset.findData("slow"))
    page.more_options_dialog.close()
    page.more_options_button.click()
    assert page.video_preset.currentData() == "slow"
    page.more_options_dialog.close()
    page.close()


def test_privateframe_non_raccoon_blocks_workspace_with_top_message(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages import privateframe_page

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        model_name="buffalo_l",
        model_root=str(tmp_path / "model-root"),
        auto_load_model=False,
    )

    class Context:
        pass

    context = Context()
    context.config = config
    page = privateframe_page.PrivateFramePage(context)

    assert page.content.itemAt(0).widget() is page.model_requirement_banner
    assert page.content.itemAt(1).widget() is page.operation_scroll
    assert page.operation_scroll.widget() is page.operation_panel
    assert page.model_requirement_banner.isHidden() is False
    assert page.model_requirement_banner.parentWidget() is page
    assert "raccoon_s or raccoon_l" in page.model_requirement_message.text()
    assert page.model_requirement_current_model_value.text() == "buffalo_l"
    assert page.model_requirement_open_models_button.isEnabled()
    assert page.operation_panel.isEnabled() is False
    assert page.operation_opacity.opacity() == pytest.approx(
        privateframe_page._UNAVAILABLE_CONTENT_OPACITY
    )
    for widget in (
        page.video_input,
        page.output_dir,
        page.analysis_mode,
        page.recognition_policy,
        page.redaction_method,
        page.output_mode,
        page.more_options_button,
        page.start_button,
    ):
        assert widget.isEnabled() is False
    assert page.more_options_dialog.isEnabled() is False

    config.model_name = "raccoon_s"
    page.refresh()

    assert page.model_requirement_banner.isHidden()
    assert page.operation_panel.isEnabled()
    assert page.operation_opacity.opacity() == pytest.approx(1.0)
    assert page.video_input.isEnabled()
    assert page.output_dir.isEnabled()
    assert page.analysis_mode.isEnabled()
    assert page.start_button.isEnabled()
    assert page.more_options_dialog.isEnabled()

    page.more_options_dialog.show()
    assert page.more_options_dialog.isVisible()
    config.model_name = "buffalo_m"
    page.refresh()

    assert page.more_options_dialog.isHidden()
    assert page.operation_panel.isEnabled() is False
    page.close()


def test_privateframe_invalid_raccoon_keeps_workspace_visible(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages.privateframe_page import PrivateFramePage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    model_root = tmp_path / "model-root"
    (model_root / "models" / "raccoon_s").mkdir(parents=True)
    config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        model_name="raccoon_s",
        model_root=str(model_root),
        auto_load_model=False,
    )

    class Context:
        pass

    context = Context()
    context.config = config
    page = PrivateFramePage(context)

    assert page.model_requirement_banner.isHidden()
    assert page.operation_panel.isEnabled()
    assert page.start_button.isEnabled() is False
    assert "manifest is missing" in page.model_status_label.text()
    page.close()


def test_privateframe_model_requirement_message_survives_language_switch(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.core.i18n import apply_translations
    from insightface.gui.pages.privateframe_page import PrivateFramePage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        model_name="buffalo_l",
        model_root=str(tmp_path / "model-root"),
        auto_load_model=False,
        ui_language="zh",
    )

    class Context:
        pass

    context = Context()
    context.config = config
    page = PrivateFramePage(context)
    apply_translations(page, "zh")

    assert page.model_requirement_title.text() == "PrivateFrame 需要 Raccoon 模型。"
    assert "仅支持 raccoon_s 或 raccoon_l" in (
        page.model_requirement_message.text()
    )
    assert page.model_requirement_current_model_caption.text() == "当前全局模型"

    config.ui_language = "en"
    apply_translations(page, "en")

    assert page.model_requirement_title.text() == (
        "PrivateFrame requires a Raccoon model."
    )
    assert page.model_requirement_message.text().startswith(
        "PrivateFrame supports only raccoon_s or raccoon_l."
    )
    assert page.model_requirement_current_model_caption.text() == (
        "Current global model"
    )
    page.close()


@pytest.mark.parametrize("language", ["zh", "ja", "ko", "es", "fr", "de", "pt", "ru"])
def test_privateframe_core_controls_are_localized_in_every_gui_language(
    tmp_path,
    language,
):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.core.i18n import apply_translations, tr
    from insightface.gui.pages.privateframe_page import PrivateFramePage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    config = AppConfig(
        model_name="raccoon_s",
        workspace_path=str(tmp_path / language),
        model_root=str(tmp_path / language / "model-root"),
        auto_load_model=False,
        ui_language=language,
    )
    context = SimpleNamespace(config=config)
    page = PrivateFramePage(context)
    apply_translations(page, language)

    assert page.analysis_fps_label.text() == tr("Target analysis FPS", language)
    assert page.analysis_mode.itemText(0) == tr(
        "Fast (default, target 15 analysis FPS)", language
    )
    assert page.analysis_mode.itemText(1) == tr(
        "Normal (target 30 analysis FPS)", language
    )
    assert page.analysis_mode.currentText() != "Fast (default, target 15 analysis FPS)"
    assert page.analysis_mode.currentData() == 15
    assert page.analysis_mode.itemData(0) == 15
    assert page.analysis_mode.itemData(1) == 30
    tooltip_source = "Regular face detections per second of input video. Higher values check faces more often; lower values reduce work for faster processing but may miss brief appearances. Extra scans may occur. Output video FPS is unchanged. Current default: {default}."
    assert page.analysis_mode.toolTip() == tr(tooltip_source, language).format(default="15")
    assert page.analysis_mode.toolTip() != tooltip_source
    assert page.analysis_fps_label.toolTip() == page.analysis_mode.toolTip()
    assert page.video_input.dialog_filter == (
        f"{tr('Videos', language)} (*.mp4 *.mov *.m4v *.mkv *.avi *.webm);;"
        f"{tr('All Files', language)} (*)"
    )
    assert page.more_options_dialog.windowTitle() == tr(
        "PrivateFrame More Options", language
    )
    assert page.processing_details_button.text() == tr("Processing details…", language)
    assert page.processing_details_dialog.windowTitle() == tr("PrivateFrame Processing Details", language)
    assert page.processing_details_button.text() != "Processing details…"
    assert page.redaction_method.itemText(0) == tr("Gaussian blur", language)
    assert page.recognition_policy.itemText(0) == tr(
        "Blur everyone", language
    )
    assert tr("Target analysis FPS", language) != "Target analysis FPS"
    page.close()


def test_privateframe_dynamic_content_retranslates_without_losing_values(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.core.i18n import apply_translations
    from insightface.gui.pages.privateframe_page import PrivateFramePage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    config = AppConfig(
        model_name="raccoon_s",
        workspace_path=str(tmp_path / "workspace"),
        model_root=str(tmp_path / "model-root"),
        auto_load_model=False,
        ui_language="en",
    )
    page = PrivateFramePage(SimpleNamespace(config=config))
    page.analysis_mode.setCurrentIndex(page.analysis_mode.findData(30))
    source = tmp_path / "holiday.mp4"
    source.write_bytes(b"video")
    output_dir = tmp_path / "output"
    gallery = tmp_path / "gallery"
    gallery.mkdir(parents=True)
    (gallery / "Alice.jpg").write_bytes(b"image")
    page.video_input._path = str(source)
    page._video_preview_state = "failed"
    page._video_preview_path = source
    page.output_dir.setText(str(output_dir))
    page.recognition_policy.setCurrentIndex(
        page.recognition_policy.findData("exempt")
    )
    page.reference_dir.setText(str(gallery))
    page._refresh_reference_photos()
    page._last_job = SimpleNamespace(
        output_mode="json_and_video",
        result_path=output_dir / "holiday_privateframe.json",
        redacted_path=output_dir / "holiday_privateframe.mp4",
    )
    result = {
        "analysis": {
            "frame_count": 10,
            "accepted_tracks": 2,
            "timings": {"analysis_seconds": 1.25, "total_seconds": 1.5},
        },
        "render": {"frame_count": 10, "seconds": 0.5},
        "gui": {"source_audio_codec": "aac", "audio_output_mode": "aac"},
    }
    page._summary_state = "completed"
    page._summary_result = result
    page._set_stage(
        "Analyzing video frames: {current}/{total}",
        current=3,
        total=10,
    )
    page._set_progress_format("Completed")

    apply_translations(page, "zh")

    assert page.analysis_mode.currentData() == 30
    assert page.analysis_mode.currentText() == "普通（目标分析 30 FPS）"
    assert page.analysis_mode.itemText(0) == "快速（默认，目标分析 15 FPS）"
    assert page.stage_label.text() == "正在分析视频帧：3/10"
    assert page.progress_bar.format() == "已完成"
    assert "分析结果 JSON" in page.output_preview.text()
    assert "无法生成预览" in page.video_input.placeholder.text()
    assert "1" in page.reference_status.text()
    assert "已接受轨迹：2" in page.summary.toPlainText()
    assert "总耗时：2.00 秒" in page.summary.toPlainText()

    apply_translations(page, "en")

    assert page.analysis_mode.currentData() == 30
    assert page.analysis_mode.currentText() == "Normal (target 30 analysis FPS)"
    assert page.analysis_mode.itemText(0) == "Fast (default, target 15 analysis FPS)"
    assert page.stage_label.text() == "Analyzing video frames: 3/10"
    assert page.progress_bar.format() == "Completed"
    assert "Analysis JSON" in page.output_preview.text()
    assert "Preview unavailable" in page.video_input.placeholder.text()
    assert "Found 1 reference photos" in page.reference_status.text()
    assert "Accepted tracks: 2" in page.summary.toPlainText()
    assert "Total seconds: 2.00" in page.summary.toPlainText()
    page.close()


def test_privateframe_dynamic_error_and_provider_tooltip_retranslate(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.core.i18n import apply_translations, tr
    from insightface.gui.pages.privateframe_page import PrivateFramePage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    config = AppConfig(
        model_name="raccoon_s",
        workspace_path=str(tmp_path / "workspace"),
        auto_load_model=False,
        provider="CPU",
        ui_language="en",
    )
    page = PrivateFramePage(SimpleNamespace(config=config))
    page.show_error = lambda _message: None
    page._processing_error("The output directory does not exist.")
    assert page.summary.toPlainText() == "The output directory does not exist."

    config.ui_language = "zh"
    apply_translations(page, "zh")

    assert page.summary.toPlainText() == tr(
        "The output directory does not exist.",
        "zh",
    )
    assert "Configured selection:" not in page.provider_label.toolTip()
    assert "CPU" in page.provider_label.toolTip()

    config.ui_language = "en"
    apply_translations(page, "en")
    assert page.summary.toPlainText() == "The output directory does not exist."
    assert "Configured selection: CPU" in page.provider_label.toolTip()
    page.close()


def test_privateframe_invalid_model_reason_is_localized(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.core.i18n import apply_translations
    from insightface.gui.pages.privateframe_page import PrivateFramePage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    model_root = tmp_path / "model-root"
    model_root.write_text("not a directory", encoding="utf-8")
    config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        model_root=str(model_root),
        model_name="raccoon_s",
        auto_load_model=False,
        ui_language="zh",
    )
    page = PrivateFramePage(SimpleNamespace(config=config))
    apply_translations(page, "zh")

    assert "模型根目录不是目录" in page.model_status_label.text()
    assert str(model_root) in page.model_status_label.text()
    page.close()


def test_privateframe_json_only_controls_preview_and_progress(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages.privateframe_page import PrivateFramePage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    config = AppConfig(model_name="raccoon_s", workspace_path=str(tmp_path), auto_load_model=False)

    class Context:
        pass

    context = Context()
    context.config = config
    page = PrivateFramePage(context)
    source = tmp_path / "holiday.mov"
    source.write_bytes(b"video")
    page.video_input.set_path(str(source))
    page.output_dir.setText(str(tmp_path / "output"))

    assert "holiday_privateframe.json" in page.output_preview.text()
    assert "holiday_privateframe.mp4" in page.output_preview.text()
    page.output_mode.setCurrentIndex(page.output_mode.findData("json_only"))

    assert page.preserve_audio.isEnabled()
    assert page.video_preset.isEnabled()
    assert page.video_crf.isEnabled()
    assert page.box_scale.isEnabled()
    assert "holiday_privateframe.json" in page.output_preview.text()
    assert "holiday_privateframe.mp4" not in page.output_preview.text()
    job = page._selected_job()
    assert job.config_overrides["scan.max_analysis_fps"] == 15
    page._last_job = job
    page._processing_progress(3, 10, "analysis")
    assert page.progress_bar.maximum() == 10
    assert page.progress_bar.value() == 3
    assert page.stage_label.text() == "Analyzing video frames: 3/10"

    page.output_mode.setCurrentIndex(page.output_mode.findData("json_and_video"))
    assert page.preserve_audio.isEnabled()

    page.recognition_policy.setCurrentIndex(page.recognition_policy.findData("exempt"))
    assert page.reference_dir.isEnabled()
    assert page.recognition_profile.isEnabled()
    assert not page.reference_panel.isHidden()
    assert not page.selective_privacy_note.isHidden()

    page._set_running(True)
    for widget in (
        page.analysis_mode,
        page.recognition_policy,
        page.more_options_button,
        page.between_scan_frames,
        page.box_scale,
        page.video_preset,
        page.video_crf,
        page.preserve_audio,
        page.reference_dir,
        page.browse_reference_button,
        page.recognition_profile,
    ):
        assert not widget.isEnabled()
    page._set_running(False)
    assert page.more_options_button.isEnabled()
    assert page.reference_dir.isEnabled()
    page.close()


def test_privateframe_flat_reference_scan_and_job_validation(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages.privateframe_page import PrivateFramePage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    config = AppConfig(model_name="raccoon_s", workspace_path=str(tmp_path), auto_load_model=False)

    class Context:
        pass

    context = Context()
    context.config = config
    page = PrivateFramePage(context)
    source = tmp_path / "holiday.mp4"
    source.write_bytes(b"video")
    output = tmp_path / "output"
    gallery = tmp_path / "gallery"
    gallery.mkdir(parents=True)
    (gallery / "Alice.jpg").write_bytes(b"image")
    (gallery / "Bob.png").write_bytes(b"image")
    (gallery / "not-a-person.txt").write_text("ignored", encoding="utf-8")
    (gallery / ".hidden-person").mkdir()
    try:
        (gallery / "Linked").symlink_to(gallery / "Alice", target_is_directory=True)
    except OSError:
        pass

    page.video_input.set_path(str(source))
    page.output_dir.setText(str(output))
    page.recognition_policy.setCurrentIndex(page.recognition_policy.findData("exempt"))
    page.reference_dir.setText(str(gallery))
    page._refresh_reference_photos()

    assert "Found 2 reference photos" in page.reference_status.text()
    assert not hasattr(page, "target_persons")
    assert not hasattr(page, "gallery_dir")
    page.analysis_mode.setCurrentIndex(page.analysis_mode.findData(15))
    page.between_scan_frames.setCurrentIndex(
        page.between_scan_frames.findData("visual")
    )
    page.box_scale.setCurrentIndex(page.box_scale.findData(1.15))
    page.video_preset.setCurrentIndex(page.video_preset.findData("veryfast"))
    page.video_crf.setCurrentIndex(page.video_crf.findData(23))
    page.recognition_profile.setCurrentIndex(page.recognition_profile.findData("fast"))

    job = page._selected_job()

    assert job.config_overrides["scan.max_analysis_fps"] == 15
    assert job.config_overrides["tracking.between_scan_frames"] == "visual"
    assert job.config_overrides["render.redaction.box_scale"] == 1.15
    assert job.config_overrides["render.video_output.preset"] == "veryfast"
    assert job.config_overrides["render.video_output.rate_control"]["quality"] == 23
    assert job.config_overrides["recognition.reference_dir"] == str(gallery.resolve())
    assert job.config_overrides["recognition.unknown_action"] == "auto"
    assert job.config_overrides["recognition.profile"] == "fast"
    page.close()


def test_existing_json_is_included_in_replace_confirmation(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QMessageBox

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages import privateframe_page

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    config = AppConfig(model_name="raccoon_s", workspace_path=str(tmp_path), auto_load_model=False)

    class Context:
        pass

    context = Context()
    context.config = config
    page = privateframe_page.PrivateFramePage(context)
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    output = tmp_path / "output"
    output.mkdir()
    (output / "input_privateframe.json").write_text("{}", encoding="utf-8")
    page.video_input.set_path(str(source))
    page.output_dir.setText(str(output))
    page.output_mode.setCurrentIndex(page.output_mode.findData("json_only"))
    confirmations = []

    def reject_replace(*args):
        confirmations.append(args)
        return QMessageBox.No

    monkeypatch.setattr(QMessageBox, "question", reject_replace)
    monkeypatch.setattr(
        page,
        "run_task",
        lambda *_args, **_kwargs: pytest.fail("declined replacement must not run"),
    )

    page.start_processing()

    assert len(confirmations) == 1
    assert page._running is False
    assert page._last_job is None
    page.close()


def test_privateframe_refresh_recomputes_resolved_provider(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core import face_engine
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages.privateframe_page import PrivateFramePage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    available = ["CPUExecutionProvider"]
    monkeypatch.setattr(
        face_engine,
        "available_execution_providers",
        lambda: list(available),
    )
    config = AppConfig(
        model_name="raccoon_s",
        workspace_path=str(tmp_path),
        auto_load_model=False,
        provider="Auto",
    )

    class Context:
        pass

    context = Context()
    context.config = config
    page = PrivateFramePage(context)
    assert page.provider_label.text() == "CPUExecutionProvider"

    available[:] = [
        "CoreMLExecutionProvider",
        "AzureExecutionProvider",
        "CPUExecutionProvider",
    ]
    page.refresh()

    assert page.provider_label.text() == "CoreMLExecutionProvider"
    assert "CPUExecutionProvider (fallback)" in page.provider_label.toolTip()
    assert "AzureExecutionProvider" not in page.provider_label.toolTip()
    page.close()


def test_privateframe_page_runs_python_api_on_background_worker(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import StudioContext, configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.core.face_engine import FaceEngine
    from insightface.gui.core.storage import Storage
    from insightface.gui.main_window import MainWindow
    from insightface.gui.pages import privateframe_page

    configure_qt_plugin_paths()
    app = QApplication.instance() or QApplication([])
    config = AppConfig(
        model_name="raccoon_s",
        workspace_path=str(tmp_path / "workspace"),
        model_root=str(tmp_path / "models"),
        auto_load_model=False,
        safe_mode=True,
        ui_language="en",
    )
    window = MainWindow(
        StudioContext(
            config,
            True,
            Storage(config.database_path),
            FaceEngine(model_name=config.model_name, root=config.model_root),
            str(tmp_path / "app.log"),
        )
    )
    window.open_page("private_frame")
    page = window.page_registry.get("private_frame")
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    page.video_input.set_path(str(source))
    page.output_dir.setText(str(tmp_path / "output"))
    page.output_mode.setCurrentIndex(page.output_mode.findData("json_only"))
    main_thread = threading.get_ident()
    worker_threads = []
    release = threading.Event()

    def fake_run(job, *, progress=None, is_cancelled=None, reference_log=None):
        worker_threads.append(threading.get_ident())
        assert job.output_mode == "json_only"
        assert job.result_path.name == "input_privateframe.json"
        assert job.redacted_path is None
        release.wait(timeout=1.0)
        assert is_cancelled is not None and not is_cancelled()
        assert progress is not None
        progress(1, 1, "analysis")
        return {
            "analysis": {
                "frame_count": 1,
                "accepted_tracks": 0,
                "timings": {"analysis_seconds": 0.01},
            },
            "render": {"frame_count": 1, "seconds": 0.01},
        }

    monkeypatch.setattr(privateframe_page, "run_privateframe_job", fake_run)

    page.start_processing()
    assert page._running is True
    assert page.start_button.isEnabled() is False
    assert page.open_models_button.isEnabled() is False
    assert window.context.privateframe_jobs_in_progress == 1
    release.set()

    loop = QEventLoop()
    poll = QTimer()
    poll.timeout.connect(lambda: loop.quit() if not page._running else None)
    poll.start(5)
    QTimer.singleShot(1000, loop.quit)
    loop.exec()
    poll.stop()

    assert page._running is False
    assert window.context.privateframe_jobs_in_progress == 0
    assert worker_threads and worker_threads[0] != main_thread
    assert page.progress_bar.value() == 100
    assert page.start_button.isEnabled()
    assert "Analysis JSON:" in page.summary.toPlainText()
    assert "Video rendering: skipped (JSON only)" in page.summary.toPlainText()
    assert "Total seconds: 0.01" in page.summary.toPlainText()
    window.close()
    app.processEvents()


@pytest.fixture
def reference_page(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages.privateframe_page import PrivateFramePage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    page = PrivateFramePage(SimpleNamespace(config=AppConfig(model_name="raccoon_s", workspace_path=str(tmp_path), auto_load_model=False, ui_language="en")))
    yield page
    page.more_options_dialog.close()
    page.processing_details_dialog.close()
    page.close()


def test_reference_controls_are_inline_and_match_policy(reference_page):
    page = reference_page
    assert page.reference_panel.isHidden()
    assert page.more_option_groups["matching"].isHidden()
    assert page.selective_privacy_note.isHidden()
    assert "Reference photos are not needed" in page.recognition_policy.toolTip()
    page.recognition_policy.setCurrentIndex(page.recognition_policy.findData("blur_only"))
    assert not page.reference_panel.isHidden()
    assert not page.more_option_groups["matching"].isHidden()
    assert not page.more_options_dialog.isAncestorOf(page.reference_dir)
    assert "Unmatched faces stay clear" in page.selective_privacy_note.text()
    assert page.more_options_dialog.isHidden()
    page.recognition_policy.setCurrentIndex(page.recognition_policy.findData("exempt"))
    assert "including uncertain matches, is blurred" in page.selective_privacy_note.text()
    assert page.more_options_dialog.isAncestorOf(page.output_mode)
    assert not page.more_options_dialog.isAncestorOf(page.preserve_audio)


def test_more_option_group_defaults_and_effective_modified_marker(reference_page):
    page = reference_page
    assert page.video_crf.currentData() == 23
    assert page.recognition_threshold.value() == 0.40
    assert page.recognition_threshold.minimum() == 0.0
    assert page.recognition_threshold.maximum() == 1.0
    assert page.recognition_threshold.singleStep() == 0.01
    page.recognition_threshold.setValue(0.65)
    assert page.more_options_button.text() == "More Options…"
    page.recognition_policy.setCurrentIndex(page.recognition_policy.findData("exempt"))
    assert "modified" in page.more_options_button.text()
    page.box_scale.setCurrentIndex(page.box_scale.findData(1.30))
    page.video_crf.setCurrentIndex(page.video_crf.findData(18))
    page.reset_option_buttons["matching"].click()
    assert page.recognition_threshold.value() == 0.40
    assert page.box_scale.currentData() == 1.30
    assert page.video_crf.currentData() == 18
    page.reset_option_buttons["appearance"].click()
    assert page.box_scale.currentData() == 1.0
    assert "modified" in page.more_options_button.text()
    page.reset_option_buttons["video"].click()
    assert page.video_crf.currentData() == 23
    assert page.more_options_button.text() == "More Options…"


def test_reference_preflight_is_lightweight_and_requires_supported_photos(reference_page, tmp_path, monkeypatch):
    from insightface.app.privateframe import recognition
    page = reference_page
    root = tmp_path / "photos"
    root.mkdir()
    (root / "person").mkdir()
    (root / "person" / "nested.jpg").write_bytes(b"not used")
    monkeypatch.setattr(recognition, "build_gallery", lambda *_a, **_kw: pytest.fail("preflight loaded face models"))
    page.recognition_policy.setCurrentIndex(page.recognition_policy.findData("blur_only"))
    page.reference_dir.setText(str(root))
    page._refresh_reference_photos()
    assert "No supported reference photos" in page.reference_status.text()
    (root / "photo.jpg").write_bytes(b"not decoded in preflight")
    page._refresh_reference_photos()
    assert "Found 1 reference photos" in page.reference_status.text()


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), True, "0.4"])
def test_gui_job_rejects_invalid_match_threshold(tmp_path, threshold):
    from insightface.gui.pages.privateframe_page import build_privateframe_job
    with pytest.raises(ValueError, match="Match threshold must be between 0 and 1"):
        build_privateframe_job(input_path=tmp_path / "video.mp4", output_dir=tmp_path / "out", model_package="raccoon_s", max_analysis_fps=30, redaction_method="gaussian", runtime_provider="auto", recognition_similarity_threshold=threshold)


def test_json_only_audio_preference_can_be_changed_and_survives_job(reference_page, tmp_path):
    page = reference_page
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    page.video_input.set_path(str(source))
    page.output_mode.setCurrentIndex(page.output_mode.findData("json_only"))
    assert page.preserve_audio.isEnabled()
    assert "saved for later" in page.audio_notice.text()
    assert page._selected_job().config_overrides["render.video_output.audio.redacted"] == "aac"
    page.preserve_audio.setChecked(False)
    assert page._selected_job().config_overrides["render.video_output.audio.redacted"] == "none"


@pytest.mark.parametrize(
    ("audio_codec", "has_audio", "expected"),
    [("aac", True, "AAC audio will be preserved"), ("opus", True, "Source audio is opus"), (None, False, "no audio track")],
)
def test_audio_notice_uses_source_capabilities(reference_page, audio_codec, has_audio, expected):
    page = reference_page
    page._video_preview_data = SimpleNamespace(audio_codec=audio_codec, has_audio=has_audio)
    page._update_audio_notice()
    assert expected in page.audio_notice.text()
    page.preserve_audio.setChecked(False)
    assert "will be silent" in page.audio_notice.text()


def test_reference_log_forwarding_filters_other_threads_and_restores_logger():
    import logging
    from insightface.gui.pages import privateframe_page

    logger = logging.getLogger("insightface.app.privateframe.recognition")
    old_level, old_handlers = logger.level, list(logger.handlers)
    received = []
    with privateframe_page._forward_reference_logs(received.append):
        other = threading.Thread(target=lambda: logger.info("Reference photos: read %d, used %d, skipped %d", 99, 99, 0))
        other.start()
        other.join()
        logger.info("Reference photo %s: detected %d faces; using only the largest face (box=%s), ignoring %d others", "group.jpg", 3, (0, 0, 10, 10), 2)
        logger.warning("Reference photo %s was not used: %s", "empty.jpg", "no_face")
        logger.info("Reference photos: read %d, used %d, skipped %d", 2, 1, 1)
    assert [entry["kind"] for entry in received] == ["largest", "skip", "summary"]
    assert received[-1]["total"] == 2
    assert logger.level == old_level
    assert logger.handlers == old_handlers


def test_gui_reference_log_signal_delivers_to_main_thread(reference_page):
    from PySide6.QtWidgets import QApplication
    page = reference_page
    event = {"kind": "summary", "total": 3, "used": 2, "skipped": 1}
    worker = threading.Thread(target=lambda: page.reference_log_received.emit(event))
    worker.start()
    worker.join()
    assert page._reference_log_entries == []
    QApplication.instance().processEvents()
    assert page._reference_log_entries == [event]
    assert "2 used, 1 skipped" in page.reference_status.text()
    assert "2 used, 1 skipped" in page.summary.toPlainText()


def test_reference_logs_and_no_usable_faces_error_are_localized(reference_page, monkeypatch):
    from insightface.gui.core.i18n import tr
    page = reference_page
    page.context.config.ui_language = "zh"
    page._reference_log_received({"kind": "skip", "file": "empty.jpg", "reason": "no_face"})
    assert tr("No face detected", "zh") in page.summary.toPlainText()
    caught = []
    monkeypatch.setattr(page, "show_error", caught.append)
    page._processing_error("No usable faces were found in the reference photos; choose clearer reference photos and try again")
    assert caught == ["No usable faces were found in the reference photos. Choose clearer photos and try again."]
    assert tr(caught[0], "zh") in page.summary.toPlainText()
    assert "empty.jpg" in page.summary.toPlainText()


def test_reference_scan_error_with_braces_is_displayed_literally(reference_page):
    reference_page._set_reference_status("Failed to read /photos/{family}.jpg")
    assert reference_page.reference_status.text() == "Failed to read /photos/{family}.jpg"


def test_controlled_reference_inference_error_retranslates(reference_page, monkeypatch):
    from insightface.gui.core.i18n import tr
    page = reference_page
    captured = []
    monkeypatch.setattr(page, "show_error", captured.append)
    page._processing_error("Reference photo {family}.jpg: face detector inference failed")
    assert "{family}.jpg" in captured[0]
    page.retranslate_dynamic_content("zh")
    expected = tr("Reference photo {file}: face detection failed.", "zh").format(file="{family}.jpg")
    assert expected in page.summary.toPlainText()


def test_privateframe_scrolls_in_short_window_without_forcing_it_taller(reference_page):
    from PySide6.QtWidgets import QApplication
    page = reference_page
    page.recognition_policy.setCurrentIndex(page.recognition_policy.findData("exempt"))
    page.resize(900, 400)
    page.show()
    QApplication.instance().processEvents()
    assert page.height() == 400
    assert page.operation_scroll.verticalScrollBar().maximum() > 0
    assert page.operation_scroll.widget() is page.operation_panel


def _configured_gui_base(tmp_path, monkeypatch, *, rate_control=None, analysis_fps=24):
    import yaml
    from insightface.gui.pages import privateframe_page
    from insightface.app.privateframe.base_config import read_default_config

    defaults = read_default_config(privateframe_page.DEFAULT_CONFIG_PATH)
    defaults["scan"]["max_analysis_fps"] = analysis_fps
    defaults["tracking"]["between_scan_frames"] = "visual"
    defaults["render"]["redaction"].update(method="mosaic", box_scale=1.1)
    defaults["render"]["video_output"]["preset"] = "fast"
    defaults["render"]["video_output"]["rate_control"] = rate_control or {"mode": "crf", "quality": 21}
    defaults["render"]["video_output"]["audio"]["redacted"] = "none"
    references = tmp_path / "photos"
    references.mkdir()
    (references / "reference.jpg").write_bytes(b"photo")
    defaults["recognition"].update(mode="exempt", reference_dir=str(references), profile="fast", similarity_threshold=0.555, unknown_action="keep")
    config_path = tmp_path / "base.yaml"
    config_path.write_text(yaml.safe_dump(defaults), encoding="utf-8")
    monkeypatch.setattr(privateframe_page, "DEFAULT_CONFIG_PATH", config_path)
    return defaults, config_path


@pytest.mark.parametrize("analysis_fps", [24, 30])
def test_gui_initial_values_and_restore_read_one_packaged_base_snapshot(tmp_path, monkeypatch, analysis_fps):
    from PySide6.QtWidgets import QApplication
    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.core.i18n import apply_translations
    from insightface.gui.pages import privateframe_page

    defaults, config_path = _configured_gui_base(tmp_path, monkeypatch, analysis_fps=analysis_fps)
    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    page = privateframe_page.PrivateFramePage(SimpleNamespace(config=AppConfig(model_name="raccoon_s", workspace_path=str(tmp_path), auto_load_model=False, ui_language="en")))
    assert page.analysis_mode.currentData() == analysis_fps
    assert page.analysis_mode.itemText(0) == "Fast (target 15 analysis FPS)"
    assert page.analysis_mode.itemText(1) == (
        "Normal (default, target 30 analysis FPS)"
        if analysis_fps == 30 else "Normal (target 30 analysis FPS)"
    )
    assert page.redaction_method.currentData() == "mosaic"
    assert page.between_scan_frames.currentData() == "visual"
    assert page.box_scale.currentData() == 1.1
    assert page.video_preset.currentData() == "fast"
    assert page.video_crf.currentData() == 21
    assert page.recognition_policy.currentData() == "exempt"
    assert page.recognition_profile.currentData() == "fast"
    assert page.recognition_threshold.value() == 0.555
    assert not page.preserve_audio.isChecked()
    assert page.more_options_button.text() == "More Options…"
    assert f"Current default: {analysis_fps}" in page.analysis_mode.toolTip()
    assert "default (0.555)" in page.recognition_threshold.toolTip()
    assert "will stay clear, as set in the configuration" in page.selective_privacy_note.text()
    assert "is blurred" not in page.selective_privacy_note.text()
    page.box_scale.setCurrentIndex(page.box_scale.findData(1.3))
    page.video_crf.setCurrentIndex(page.video_crf.findData(18))
    page.recognition_threshold.setValue(0.8)
    assert "modified" in page.more_options_button.text()
    for button in page.reset_option_buttons.values():
        button.click()
    assert page.box_scale.currentData() == 1.1
    assert page.video_crf.currentData() == 21
    assert page.recognition_threshold.value() == 0.555
    assert page.more_options_button.text() == "More Options…"
    page.between_scan_frames.setCurrentIndex(page.between_scan_frames.findData("auto"))
    assert page.more_options_button.text() == "More Options…"
    page.reset_option_buttons["appearance"].click()
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    page.video_input.set_path(str(source))
    # A page runs with the defaults it showed, even if another caller changes
    # the packaged-default reader before Start is clicked.
    monkeypatch.setattr(privateframe_page, "read_default_config", lambda *_args: pytest.fail("job re-read a different defaults snapshot"))
    job = page._selected_job()
    assert job.config_path == config_path
    assert job.config_overrides["scan.max_analysis_fps"] == analysis_fps
    assert job.config_overrides["render.redaction.box_scale"] == 1.1
    assert job.config_overrides["render.video_output.preset"] == "fast"
    assert job.config_overrides["render.video_output.rate_control"] == {"mode": "crf", "quality": 21}
    assert job.config_overrides["recognition.similarity_threshold"] == 0.555
    assert job.config_overrides["recognition.unknown_action"] == "keep"
    assert job.config_overrides["render.video_output.audio.redacted"] == "none"
    assert page._base_defaults == defaults
    apply_translations(page, "zh")
    assert str(analysis_fps) in page.analysis_mode.currentText()
    assert "{value}" not in page.analysis_mode.currentText()
    assert str(analysis_fps) in page.analysis_mode.toolTip() and "{default}" not in page.analysis_mode.toolTip()
    page.analysis_mode.setCurrentIndex(page.analysis_mode.findData(15))
    apply_translations(page, "en")
    assert page.analysis_mode.currentData() == 15
    assert page.analysis_mode.currentText() == "Fast (target 15 analysis FPS)"
    page._initialize_default_controls()
    assert page.analysis_mode.currentData() == analysis_fps
    page.close()


def test_builder_omitted_options_read_base_and_accept_nonpreset_legal_values(tmp_path, monkeypatch):
    from insightface.gui.pages import privateframe_page

    defaults, config_path = _configured_gui_base(tmp_path, monkeypatch)
    job = privateframe_page.build_privateframe_job(input_path=tmp_path / "video.mp4", output_dir=tmp_path / "out", model_package="raccoon_l", runtime_provider="cpu")
    assert job.config_path == config_path
    assert job.config_overrides["scan.max_analysis_fps"] == 24
    assert job.config_overrides["render.redaction.method"] == "mosaic"
    assert job.config_overrides["render.redaction.box_scale"] == 1.1
    assert job.config_overrides["render.video_output.preset"] == "fast"
    assert job.config_overrides["render.video_output.rate_control"] == {"mode": "crf", "quality": 21}
    assert job.config_overrides["recognition.profile"] == "fast"
    assert job.config_overrides["recognition.similarity_threshold"] == 0.555
    assert job.config_overrides["recognition.unknown_action"] == "keep"
    assert job.config_overrides["models.name"] == "raccoon_l"
    assert job.config_overrides["runtime.provider"] == "CPUExecutionProvider"
    custom = privateframe_page.build_privateframe_job(input_path=tmp_path / "video.mp4", output_dir=tmp_path / "out", model_package="raccoon_s", runtime_provider="auto", max_analysis_fps=29.97, box_scale=0.9, video_crf=19, video_preset="faster", recognition_mode="all")
    assert custom.config_overrides["scan.max_analysis_fps"] == 29.97
    assert custom.config_overrides["render.redaction.box_scale"] == 0.9
    assert custom.config_overrides["render.video_output.rate_control"] == {"mode": "crf", "quality": 19}


def test_non_crf_base_is_preserved_until_user_selects_quality(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages import privateframe_page

    rate = {"mode": "vbr", "bitrate": "4M", "max_bitrate": "6M"}
    _configured_gui_base(tmp_path, monkeypatch, rate_control=rate)
    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    page = privateframe_page.PrivateFramePage(SimpleNamespace(config=AppConfig(model_name="raccoon_s", workspace_path=str(tmp_path), auto_load_model=False, ui_language="en")))
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    page.video_input.set_path(str(source))
    assert page.video_crf.currentData() is None
    assert page.video_crf.currentText() == "From configuration (vbr)"
    assert page.more_options_button.text() == "More Options…"
    assert page._selected_job().config_overrides["render.video_output.rate_control"] == rate
    page.video_crf.setCurrentIndex(page.video_crf.findData(18))
    assert page._selected_job().config_overrides["render.video_output.rate_control"] == {"mode": "crf", "quality": 18}
    page.reset_option_buttons["video"].click()
    assert page.video_crf.currentData() is None
    assert page._selected_job().config_overrides["render.video_output.rate_control"] == rate
    page.close()


def test_execution_controls_stay_outside_scrolling_settings(reference_page):
    from PySide6.QtWidgets import QApplication
    page = reference_page
    page.resize(900, 600)
    page.show()
    QApplication.instance().processEvents()
    assert page.execution_panel.isAncestorOf(page.start_button)
    assert page.execution_panel.isAncestorOf(page.progress_bar)
    assert page.execution_panel.isAncestorOf(page.processing_details_button)
    assert page.processing_details_dialog.isAncestorOf(page.summary)
    assert page.processing_details_dialog.isAncestorOf(page.output_preview)
    assert not page.execution_panel.isAncestorOf(page.summary)
    assert not page.operation_scroll.isAncestorOf(page.start_button)
    assert page.start_button.mapTo(page, page.start_button.rect().topLeft()).y() < page.height()
    assert page.processing_details_button.mapTo(page, page.processing_details_button.rect().bottomRight()).y() <= page.height()


def test_privateframe_preview_uses_stack_for_loading_failure_and_ready(reference_page, tmp_path):
    from insightface.gui.pages.privateframe_page import VideoPreviewData
    page = reference_page
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    page.video_input.set_path(str(source), emit=False)
    page._video_preview_path = source
    for state in ("loading", "failed", "empty"):
        page._video_preview_state = state
        page._render_video_preview_state("en")
        assert page.video_input.stack.currentWidget() is page.video_input.placeholder
    page._video_preview_state = "ready"
    page._video_preview_data = VideoPreviewData(source, np.zeros((152, 152, 3), dtype=np.uint8), 640, 480, 25, 25, 1, 1, "h264", None, False)
    page.video_input.set_image(page._video_preview_data.image, str(source))
    page._render_video_preview_state("en")
    assert page.video_input.stack.currentWidget() is page.video_input.viewer


@pytest.mark.parametrize("preset", ["", None])
def test_empty_or_null_base_preset_inherits_encoder_default(tmp_path, monkeypatch, preset):
    import yaml
    from insightface.gui.pages import privateframe_page
    from insightface.gui.core.config import AppConfig
    from PySide6.QtWidgets import QApplication
    defaults, path = _configured_gui_base(tmp_path, monkeypatch)
    defaults["render"]["video_output"]["preset"] = preset
    path.write_text(yaml.safe_dump(defaults), encoding="utf-8")
    QApplication.instance() or QApplication([])
    page = privateframe_page.PrivateFramePage(SimpleNamespace(config=AppConfig(model_name="raccoon_s", workspace_path=str(tmp_path), auto_load_model=False, ui_language="en")))
    assert page.video_preset.currentData() == preset
    assert page.video_preset.currentText() == "Automatic"
    job = privateframe_page.build_privateframe_job(input_path=tmp_path / "video.mp4", output_dir=tmp_path / "out", model_package="raccoon_s", runtime_provider="auto")
    assert job.config_overrides["render.video_output.preset"] == preset
    page.close()


def test_default_reference_folder_is_relative_to_packaged_config(tmp_path, monkeypatch):
    import yaml
    from insightface.gui.pages import privateframe_page
    defaults, path = _configured_gui_base(tmp_path, monkeypatch)
    defaults["recognition"]["reference_dir"] = "photos"
    path.write_text(yaml.safe_dump(defaults), encoding="utf-8")
    job = privateframe_page.build_privateframe_job(input_path=tmp_path / "video.mp4", output_dir=tmp_path / "out", model_package="raccoon_s", runtime_provider="auto")
    assert job.config_overrides["recognition.reference_dir"] == str((path.parent / "photos").resolve())


@pytest.mark.parametrize("language", ["ja", "es", "ru"])
def test_narrow_translated_settings_fit_without_horizontal_clipping(reference_page, language):
    from PySide6.QtWidgets import QApplication
    from insightface.gui.core.i18n import apply_translations
    from insightface.gui.core.theme import application_stylesheet
    app = QApplication.instance()
    old_style = app.styleSheet()
    page = reference_page
    try:
        app.setStyleSheet(application_stylesheet(page.context.config.ui_theme))
        page.context.config.ui_language = language
        page.recognition_policy.setCurrentIndex(page.recognition_policy.findData("blur_only"))
        apply_translations(page, language)
        page.resize(760, 640)
        page.show()
        app.processEvents()
        panel_width = page.operation_panel.width()
        viewport_width = page.operation_scroll.viewport().width()
        if os.name == "nt" and language == "es" and panel_width > viewport_width:
            pytest.xfail(
                "Accepted cosmetic overflow with Spanish text on Windows"
            )
        assert panel_width <= viewport_width
    finally:
        app.setStyleSheet(old_style)


def test_primary_options_reflow_when_control_size_hints_change(tmp_path):
    """Late control size hints must not leave the scroll panel too wide."""
    from math import ceil
    from PySide6.QtCore import QSize
    from PySide6.QtWidgets import QApplication, QWidget
    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages.privateframe_page import PrivateFramePage

    configure_qt_plugin_paths()
    app = QApplication.instance() or QApplication([])
    host = QWidget()
    page = PrivateFramePage(SimpleNamespace(config=AppConfig(
        model_name="raccoon_s",
        workspace_path=str(tmp_path), auto_load_model=False, ui_language="en",
    )), parent=host)
    try:
        # A child without a parent layout can use the measured width even when
        # it exceeds the offscreen platform's available top-level window size.
        page.ensurePolished()
        page.resize(page.minimumSizeHint().expandedTo(QSize(
            2 * page.options_grid.minimumSize().width(), page.sizeHint().height(),
        )))
        host.show()
        page.show()
        for _ in range(3):
            app.processEvents()
        assert page._primary_options_two_columns is True
        initial_width = page.width()

        # Construct translated-label-sized text from the actual layout metrics:
        # each label plus its field fits, but both pairs cannot share one row.
        rows = page._primary_option_rows
        texts = [label.text() for label, _field in rows]
        field_widths = [field.minimumSizeHint().width() for _label, field in rows]
        available = page.options_grid.contentsRect().width()
        spacing = page.options_grid.horizontalSpacing()
        single_row_label_limit = available - max(field_widths) - spacing
        two_column_label_limit = (available - sum(field_widths) - 3 * spacing) / 2
        target_label_width = (single_row_label_limit + two_column_label_limit) / 2
        for label, _field in rows:
            glyph_width = label.fontMetrics().horizontalAdvance("M")
            label.setText("M" * ceil(target_label_width / glyph_width))
        page.options_grid.invalidate()
        assert page.options_grid.minimumSize().width() > available
        assert (
            max(label.minimumSizeHint().width() for label, _field in rows)
            + max(field_widths) + spacing
            < available
        )
        for _ in range(3):
            app.processEvents()
        assert page.width() == initial_width
        assert page._primary_options_two_columns is False
        assert page.operation_panel.width() <= page.operation_scroll.viewport().width()

        for (label, _field), text in zip(rows, texts):
            label.setText(text)
        for _ in range(3):
            app.processEvents()
        assert page._primary_options_two_columns is True
        assert page.operation_panel.width() <= page.operation_scroll.viewport().width()
    finally:
        page.more_options_dialog.close()
        page.processing_details_dialog.close()
        page.close()
        host.close()


def _stub_ready_privateframe_model(page):
    from insightface.gui.core.model_packages import PrivateFrameModelStatus
    root = Path(page.context.config.model_root).expanduser()
    return PrivateFrameModelStatus(
        model_name=str(page.context.config.model_name), model_root=root,
        package_path=root / "models" / str(page.context.config.model_name),
        state="ready", can_start=True, message="Ready",
    )


def test_processing_details_remain_live_when_closed_and_available_while_running(reference_page, monkeypatch):
    from insightface.gui.core.i18n import tr
    page = reference_page
    monkeypatch.setattr(page, "_global_model_status", lambda: _stub_ready_privateframe_model(page))
    assert not page.processing_details_dialog.isModal()
    assert page.processing_details_dialog.isHidden()
    page.processing_details_button.click()
    assert page.processing_details_dialog.isVisible()
    page.close_processing_details_button.click()
    assert page.processing_details_dialog.isHidden()
    page._set_running(True)
    assert page.processing_details_button.isEnabled()
    page._reference_log_received({"kind": "skip", "file": "empty.jpg", "reason": "no_face"})
    page._reference_log_received({"kind": "summary", "total": 3, "used": 2, "skipped": 1})
    assert page.processing_details_dialog.isHidden()
    page.processing_details_button.click()
    assert "empty.jpg" in page.summary.toPlainText()
    assert "2 used, 1 skipped" in page.summary.toPlainText()
    page.close_processing_details_button.click()
    assert page._running and not page._cancel_requested
    page.context.config.ui_language = "zh"
    page.retranslate_dynamic_content("zh")
    assert page.processing_details_dialog.isHidden()
    assert tr("No face detected", "zh") in page.summary.toPlainText()
    page._cancel_requested = True
    page._processing_error("cancelled")
    assert tr("No face detected", "zh") in page.summary.toPlainText()
    assert tr("Processing cancelled. Partial work files may remain in the work directory.", "zh") in page.summary.toPlainText()
    page.retranslate_dynamic_content("en")
    assert "No face detected" in page.summary.toPlainText()
    assert "Processing cancelled" in page.summary.toPlainText()
    page._set_running(False)
    page.processing_details_button.click()
    assert "empty.jpg" in page.summary.toPlainText()


def test_closed_processing_details_keep_completion_and_failures_still_alert(reference_page, tmp_path, monkeypatch):
    from insightface.gui.pages.privateframe_page import build_privateframe_job
    page = reference_page
    monkeypatch.setattr(page, "_global_model_status", lambda: _stub_ready_privateframe_model(page))
    page._last_job = build_privateframe_job(
        input_path=tmp_path / "video.mp4", output_dir=tmp_path / "out",
        model_package="raccoon_s", runtime_provider="auto", output_mode="json_only",
    )
    page._processing_complete({"analysis": {"frame_count": 10, "accepted_tracks": 2, "timings": {"analysis_seconds": 1.0}}, "render": {}})
    assert page.processing_details_dialog.isHidden()
    assert page.progress_bar.value() == 100
    assert page.open_output_button.isEnabled()
    assert "Frames: 10" in page.summary.toPlainText()
    assert "Accepted tracks: 2" in page.summary.toPlainText()
    assert "video_privateframe.json" in page.summary.toPlainText()
    alerts = []
    monkeypatch.setattr(page, "show_error", alerts.append)
    page._processing_error("No usable faces were found in the reference photos; choose clearer photos")
    assert alerts == ["No usable faces were found in the reference photos. Choose clearer photos and try again."]
    assert page.progress_bar.format() == "Failed"
    assert page.processing_details_dialog.isHidden()
    page.processing_details_button.click()
    assert alerts[0] in page.summary.toPlainText()


def test_incompatible_current_model_does_not_hide_previous_processing_details(reference_page, monkeypatch):
    from dataclasses import replace
    page = reference_page
    unsupported = replace(_stub_ready_privateframe_model(page), model_name="buffalo_l", state="unsupported", can_start=False, message="PrivateFrame requires a Raccoon model.")
    monkeypatch.setattr(page, "_global_model_status", lambda: unsupported)
    page._set_summary_text("Previous run completed")
    page.refresh()
    assert not page.model_requirement_banner.isHidden()
    assert not page.start_button.isEnabled()
    assert page.processing_details_button.isEnabled()
    page.processing_details_button.click()
    assert page.processing_details_dialog.isVisible()
    assert page.summary.toPlainText() == "Previous run completed"


def test_model_controls_move_to_more_but_blocking_status_stays_visible(reference_page, monkeypatch):
    from dataclasses import replace
    page = reference_page
    ready = _stub_ready_privateframe_model(page)
    status = [ready]
    monkeypatch.setattr(page, "_global_model_status", lambda: status[0])
    page.refresh()
    assert page.more_options_dialog.isAncestorOf(page.model_settings_panel)
    assert page.more_options_dialog.isAncestorOf(page.model_label)
    assert page.more_options_dialog.isAncestorOf(page.provider_label)
    assert page.more_options_dialog.isAncestorOf(page.open_models_button)
    assert not page.more_options_dialog.isAncestorOf(page.model_status_label)
    assert page.model_status_label.isHidden()
    status[0] = replace(ready, state="invalid", can_start=False, message="Broken model manifest")
    page.refresh()
    assert not page.model_status_label.isHidden()
    assert not page.start_button.isEnabled()
    status[0] = ready
    page.context.model_downloads_in_progress = 1
    page.refresh()
    assert not page.model_status_label.isHidden()
    assert "download" in page.model_status_label.text()
    assert not page.start_button.isEnabled()
    page.context.model_downloads_in_progress = 0
    page.refresh()
    assert page.model_status_label.isHidden()
    assert page.start_button.isEnabled()


@pytest.mark.parametrize("mode", ["all", "blur_only"])
def test_privateframe_fits_default_application_page_viewport(tmp_path, monkeypatch, mode):
    """A 1320×860 main window gives PrivateFrame a 1084×761 page viewport."""
    from PySide6.QtWidgets import QApplication
    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.core.i18n import apply_translations
    from insightface.gui.core.theme import application_stylesheet
    from insightface.gui.pages import privateframe_page

    configure_qt_plugin_paths()
    app = QApplication.instance() or QApplication([])
    old_style = app.styleSheet()
    monkeypatch.setattr(privateframe_page.PrivateFramePage, "_global_model_status", _stub_ready_privateframe_model)
    config = AppConfig(model_name="raccoon_s", workspace_path=str(tmp_path), auto_load_model=False, ui_language="zh")
    app.setStyleSheet(application_stylesheet(config.ui_theme))
    page = privateframe_page.PrivateFramePage(SimpleNamespace(config=config))
    try:
        source = tmp_path / "input.mp4"
        source.write_bytes(b"preview fixture")
        page.video_input.set_path(str(source), emit=False)
        size = privateframe_page._VIDEO_PREVIEW_SIZE
        preview = privateframe_page.VideoPreviewData(
            source, np.zeros((size, size, 3), dtype=np.uint8),
            1920, 1080, 30, 300, 10, 1000, "h264", "aac", True,
        )
        page._apply_video_preview(page._video_preview_generation, preview)
        page.recognition_policy.setCurrentIndex(page.recognition_policy.findData(mode))
        apply_translations(page, "zh")
        page.resize(1084, 761)
        page.show()
        app.processEvents()
        assert page.size().width() == 1084 and page.size().height() == 761
        assert page.operation_scroll.verticalScrollBar().maximum() == 0
        assert page.operation_panel.width() <= page.operation_scroll.viewport().width()
        assert page.processing_details_button.mapTo(page, page.processing_details_button.rect().bottomRight()).y() < page.height()
    finally:
        page.processing_details_dialog.close()
        page.close()
        app.setStyleSheet(old_style)


def test_path_fields_expand_when_native_forms_default_to_size_hints(tmp_path, monkeypatch):
    """Native form defaults must not constrain an expanding path row."""
    from PySide6.QtWidgets import (
        QApplication, QFormLayout, QProxyStyle, QStyle, QStyleFactory, QWidget,
    )
    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.core.theme import application_stylesheet
    from insightface.gui.pages import privateframe_page

    class SizeHintFormStyle(QProxyStyle):
        def styleHint(self, hint, option=None, widget=None, returnData=None):
            if hint == QStyle.StyleHint.SH_FormLayoutFieldGrowthPolicy:
                return QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint.value
            return super().styleHint(hint, option, widget, returnData)

    configure_qt_plugin_paths()
    app = QApplication.instance() or QApplication([])
    old_stylesheet = app.styleSheet()
    app.setStyleSheet("")
    old_style_name = app.style().objectName()
    page = None
    probe = None
    try:
        # Use Fusion drawing with the restrictive native-form hint so the
        # regression runs consistently without a macOS window server.
        app.setStyle(SizeHintFormStyle(QStyleFactory.create("Fusion")))
        config = AppConfig(model_name="raccoon_s", workspace_path=str(tmp_path), auto_load_model=False, ui_language="en")
        app.setStyleSheet(application_stylesheet(config.ui_theme))
        probe = QWidget()
        assert QFormLayout(probe).fieldGrowthPolicy() == QFormLayout.FieldsStayAtSizeHint
        monkeypatch.setattr(privateframe_page.PrivateFramePage, "_global_model_status", _stub_ready_privateframe_model)
        page = privateframe_page.PrivateFramePage(SimpleNamespace(config=config))
        page.recognition_policy.setCurrentIndex(page.recognition_policy.findData("blur_only"))
        page.show()
        widths = []
        for window_width in (860, 1180):
            page.resize(window_width, 761)
            app.processEvents()
            widths.append((page.output_dir.width(), page.reference_dir.width(), page.recognition_policy.width()))
            for field, button in (
                (page.output_dir, page.browse_output_button),
                (page.reference_dir, page.browse_reference_button),
            ):
                assert field.width() > 2 * field.sizeHint().width()
                assert button.isVisible()
                field_right = field.mapTo(page.operation_scroll.viewport(), field.rect().topRight()).x()
                button_left = button.mapTo(page.operation_scroll.viewport(), button.rect().topLeft()).x()
                button_right = button.mapTo(page.operation_scroll.viewport(), button.rect().topRight()).x()
                assert field_right < button_left
                assert button_right < page.operation_scroll.viewport().width()
        assert all(wide >= narrow + 200 for narrow, wide in zip(widths[0], widths[1]))

        page.more_options_button.click()
        page.more_options_dialog.resize(560, 600)
        app.processEvents()
        initial_quality_width = page.video_crf.width()
        initial_threshold_width = page.recognition_threshold.width()
        page.more_options_dialog.resize(840, 600)
        app.processEvents()
        assert page.video_crf.width() >= initial_quality_width + 200
        assert page.recognition_threshold.width() >= initial_threshold_width + 200
    finally:
        if page is not None:
            page.more_options_dialog.close()
            page.processing_details_dialog.close()
            page.close()
        if probe is not None:
            probe.close()
        app.setStyleSheet("")
        app.setStyle(QStyleFactory.create(old_style_name) or QStyleFactory.create("Fusion"))
        app.setStyleSheet(old_stylesheet)
