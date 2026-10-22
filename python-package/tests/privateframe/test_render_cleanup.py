from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest

from insightface.app.privateframe import artifact_render
from insightface.app.privateframe.artifact_render import RenderTarget


def _prepare_render(monkeypatch):
    monkeypatch.setattr(
        artifact_render,
        "probe_video",
        lambda _path: SimpleNamespace(width=2, height=2, fps=30.0),
    )
    monkeypatch.setattr(
        artifact_render,
        "iter_oriented_frames",
        lambda _path: iter([(0, 0.0, 0, np.zeros((2, 2, 3), dtype=np.uint8))]),
    )
    return {
        "source_video": {
            "metadata": {"width": 2, "height": 2, "fps": 30.0, "frame_count": 1}
        },
        "observations": [],
    }


@pytest.mark.parametrize("backend", ["pyav", "ffmpeg"])
@pytest.mark.parametrize("failure_phase", ["write", "second_writer"])
def test_keyboard_interrupt_aborts_registered_writers_and_preserves_outputs(
    monkeypatch, tmp_path, backend, failure_phase
):
    result = _prepare_render(monkeypatch)
    failure = KeyboardInterrupt("interrupted rendering")
    destinations = [tmp_path / "redacted.mp4", tmp_path / "debug.mp4"]
    for destination in destinations:
        destination.write_bytes(b"existing output")
    registered = []

    class Writer:
        def __init__(self, destination, *_args):
            if failure_phase == "second_writer" and registered:
                raise failure
            self.temporary = destination.with_suffix(".partial.mp4")
            self.temporary.write_bytes(b"partial video")
            self.aborted = False
            registered.append(self)

        def write(self, _frame):
            raise failure

        def abort(self):
            self.aborted = True
            self.temporary.unlink()

    monkeypatch.setattr(
        artifact_render, "_PyAVWriter" if backend == "pyav" else "_FFmpegWriter", Writer
    )
    with pytest.raises(KeyboardInterrupt) as caught:
        artifact_render.render_artifacts(
            source=tmp_path / "input.mp4",
            targets=[RenderTarget(mode, path) for mode, path in zip(("redacted", "debug"), destinations)],
            settings={"backend": backend, "audio": {}},
            analysis_result=result,
        )

    assert caught.value is failure
    assert all(writer.aborted and not writer.temporary.exists() for writer in registered)
    assert all(path.read_bytes() == b"existing output" for path in destinations)


def test_cleanup_failure_does_not_mask_interruption_or_skip_other_writers(
    monkeypatch, tmp_path
):
    result = _prepare_render(monkeypatch)
    failure = KeyboardInterrupt("original interruption")
    cleaned = []

    class Writer:
        def __init__(self, destination, *_args):
            self.name = destination.name

        def write(self, _frame):
            raise failure

        def abort(self):
            cleaned.append(self.name)
            if self.name == "first.mp4":
                raise OSError("cleanup failed")

    monkeypatch.setattr(artifact_render, "_PyAVWriter", Writer)
    with pytest.raises(KeyboardInterrupt) as caught:
        artifact_render.render_artifacts(
            source=tmp_path / "input.mp4",
            targets=[RenderTarget("redacted", tmp_path / name) for name in ("first.mp4", "second.mp4")],
            settings={"backend": "pyav", "audio": {}},
            analysis_result=result,
        )

    assert caught.value is failure
    assert cleaned == ["first.mp4", "second.mp4"]


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("failure_phase", ["open", "add_stream"])
def test_partial_pyav_constructor_removes_temporary_and_closes_container(
    monkeypatch, tmp_path, failure_type, failure_phase
):
    failure = failure_type("encoder initialization failed")
    temporary = tmp_path / ".temporary.mp4"
    destination = tmp_path / "output.mp4"
    destination.write_bytes(b"previous video")
    closed = []

    class Container:
        def add_stream(self, *_args, **_kwargs):
            raise failure

        def close(self):
            closed.append(True)
            raise OSError("close must not replace original failure")

    def open_container(*_args, **_kwargs):
        temporary.write_bytes(b"partial video")
        if failure_phase == "open":
            raise failure
        return Container()

    monkeypatch.setattr(artifact_render, "temporary_video_path", lambda _path: temporary)
    monkeypatch.setattr(artifact_render.av, "open", open_container)
    with pytest.raises(failure_type) as caught:
        artifact_render._PyAVWriter(
            destination, tmp_path / "input.mp4", 2, 2, 30.0, {"encoder": "fake"}, "none"
        )

    assert caught.value is failure
    assert closed == ([True] if failure_phase == "add_stream" else [])
    assert not temporary.exists()
    assert destination.read_bytes() == b"previous video"


@pytest.mark.parametrize("failure_type", [OSError, KeyboardInterrupt])
def test_audio_open_failure_closes_already_opened_input(monkeypatch, tmp_path, failure_type):
    failure = failure_type("audio input failed")
    closed = []

    def open_container(*_args, **_kwargs):
        if not getattr(open_container, "opened", False):
            open_container.opened = True
            return SimpleNamespace(close=lambda: closed.append("video"))
        raise failure

    monkeypatch.setattr(artifact_render.av, "open", open_container)
    with pytest.raises(failure_type) as caught:
        artifact_render._copy_audio(
            tmp_path / "silent.mp4", tmp_path / "source.mp4", tmp_path / "muxed.mp4",
            requested_mode="copy", maximum_duration=1.0,
        )

    assert caught.value is failure
    assert closed == ["video"]


def test_audio_cleanup_closes_all_containers_without_masking_mux_failure(monkeypatch, tmp_path):
    failure = RuntimeError("mux failed")
    closed = []
    video = SimpleNamespace(
        streams=SimpleNamespace(video=[object()]), close=lambda: closed.append("video")
    )
    audio = SimpleNamespace(
        streams=SimpleNamespace(audio=[object()]), close=lambda: closed.append("audio")
    )

    def close_output():
        closed.append("output")
        raise OSError("output close failed")

    containers = iter([video, audio, SimpleNamespace(close=close_output)])
    monkeypatch.setattr(artifact_render.av, "open", lambda *_args, **_kwargs: next(containers))

    def copy_stream(*_args):
        raise failure

    monkeypatch.setattr(artifact_render, "_stream_from_template", copy_stream)
    with pytest.raises(RuntimeError) as caught:
        artifact_render._copy_audio(
            tmp_path / "silent.mp4", tmp_path / "source.mp4", tmp_path / "muxed.mp4",
            requested_mode="copy", maximum_duration=1.0,
        )

    assert caught.value is failure
    assert closed == ["output", "audio", "video"]


def test_ffmpeg_abort_kills_process_closes_pipes_and_removes_temporary(tmp_path):
    events = []
    writer = object.__new__(artifact_render._FFmpegWriter)
    writer.temporary = tmp_path / ".partial.mp4"
    writer.temporary.write_bytes(b"partial video")
    writer.process = SimpleNamespace(
        poll=lambda: None,
        kill=lambda: events.append("kill"),
        wait=lambda: events.append("wait"),
        stdin=io.BytesIO(),
        stderr=io.BytesIO(),
    )

    writer.abort()

    assert events == ["kill", "wait"]
    assert writer.process.stdin.closed and writer.process.stderr.closed
    assert not writer.temporary.exists()
