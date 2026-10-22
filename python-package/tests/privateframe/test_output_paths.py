from pathlib import Path

from insightface.app.privateframe import default_output_paths


def test_default_output_paths_keep_public_results_paired(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    paths = default_output_paths("source/camera.clip.mp4", "exports")

    output = (tmp_path / "exports").resolve()
    assert paths.source == (tmp_path / "source/camera.clip.mp4").resolve()
    assert paths.output_dir == output
    assert paths.result_json == output / "camera.clip_privateframe.json"
    assert paths.result_video == output / "camera.clip_privateframe.mp4"
    assert paths.debug_video == output / "camera.clip_privateframe_debug.mp4"
    assert paths.workdir == output / ".camera.clip_privateframe_work"


def test_default_output_paths_use_the_source_directory_when_unspecified(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.mov"

    paths = default_output_paths(source)

    assert paths.output_dir == tmp_path
    assert paths.result_json == tmp_path / "input_privateframe.json"
    assert paths.result_video == tmp_path / "input_privateframe.mp4"
