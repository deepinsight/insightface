from __future__ import annotations

from pathlib import Path

import pytest
from insightface_server.config import (
    DEFAULT_CPU_INFERENCE_MAX_CONCURRENCY,
    DEFAULT_CUDA_INFERENCE_MAX_CONCURRENCY,
    DEFAULT_DETECTOR_INPUT_SIZES,
    DEFAULT_DETECTOR_NMS_THRESHOLD,
    DEFAULT_DETECTOR_THRESHOLD,
    DEFAULT_MAX_DETECTED_FACES,
    DetectionProfile,
    ServerFileConfig,
    Settings,
    load_detector_input_sizes,
    load_server_config,
    normalize_detector_input_sizes,
)


def test_detector_input_sizes_default_and_custom_toml(tmp_path: Path) -> None:
    assert load_detector_input_sizes(None) == ((96, 96), (512, 512))
    assert DEFAULT_DETECTOR_INPUT_SIZES == ((96, 96), (512, 512))

    config = tmp_path / "server.toml"
    config.write_text(
        "[detection]\ninput_sizes = [[128, 128], [640, 640]]\n",
        encoding="utf-8",
    )

    assert load_detector_input_sizes(config) == ((128, 128), (640, 640))


def test_web_ui_defaults_enabled_and_can_be_disabled_in_toml(tmp_path: Path) -> None:
    assert load_server_config(None) == ServerFileConfig()

    config = tmp_path / "server.toml"
    config.write_text(
        "[detection]\n"
        "input_sizes = [[96, 96]]\n"
        "threshold = 0.42\n"
        "nms_threshold = 0.35\n"
        'single_face_selection = "center_largest"\n'
        "max_detected_faces = 25\n"
        "\n[web]\ndisabled = true\n",
        encoding="utf-8",
    )

    assert load_server_config(config) == ServerFileConfig(
        detection=DetectionProfile(
            input_sizes=((96, 96),),
            threshold=0.42,
            nms_threshold=0.35,
            single_face_selection="center_largest",
        ),
        max_detected_faces=25,
        web_ui_disabled=True,
    )


def test_settings_loads_detector_sizes_once_from_selected_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "server.toml"
    config.write_text(
        "[detection]\ninput_sizes = [[96, 96], [512, 512]]\n"
        "\n[web]\ndisabled = true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("INSIGHTFACE_CONFIG_FILE", str(config))

    settings = Settings.from_env()

    assert settings.config_file == config
    assert settings.detector_input_sizes == ((96, 96), (512, 512))
    assert settings.detector_threshold == DEFAULT_DETECTOR_THRESHOLD
    assert settings.detector_nms_threshold == DEFAULT_DETECTOR_NMS_THRESHOLD
    assert settings.max_detected_faces == DEFAULT_MAX_DETECTED_FACES
    assert settings.web_ui_disabled is True
    assert settings.inference_max_concurrency == DEFAULT_CPU_INFERENCE_MAX_CONCURRENCY

    config.write_text(
        "[detection]\ninput_sizes = [[128, 128]]\n",
        encoding="utf-8",
    )
    assert settings.detector_input_sizes == ((96, 96), (512, 512))


def test_inference_concurrency_defaults_by_provider_and_supports_startup_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("INSIGHTFACE_CONFIG_FILE", raising=False)
    monkeypatch.delenv("INSIGHTFACE_INFERENCE_MAX_CONCURRENCY", raising=False)
    monkeypatch.setenv("INSIGHTFACE_EXECUTION_PROVIDER", "CPUExecutionProvider")
    assert (
        Settings.from_env().inference_max_concurrency
        == DEFAULT_CPU_INFERENCE_MAX_CONCURRENCY
    )

    monkeypatch.setenv("INSIGHTFACE_EXECUTION_PROVIDER", "CUDAExecutionProvider")
    assert (
        Settings.from_env().inference_max_concurrency
        == DEFAULT_CUDA_INFERENCE_MAX_CONCURRENCY
    )

    config = tmp_path / "server.toml"
    config.write_text("[inference]\nmax_concurrency = 11\n", encoding="utf-8")
    monkeypatch.setenv("INSIGHTFACE_CONFIG_FILE", str(config))
    assert Settings.from_env().inference_max_concurrency == 11

    monkeypatch.setenv("INSIGHTFACE_INFERENCE_MAX_CONCURRENCY", "3")
    assert Settings.from_env().inference_max_concurrency == 3

    monkeypatch.setenv("INSIGHTFACE_INFERENCE_MAX_CONCURRENCY", "auto")
    assert (
        Settings.from_env().inference_max_concurrency
        == DEFAULT_CUDA_INFERENCE_MAX_CONCURRENCY
    )

    monkeypatch.setenv("INSIGHTFACE_INFERENCE_MAX_CONCURRENCY", "many")
    with pytest.raises(ValueError, match="INSIGHTFACE_INFERENCE_MAX_CONCURRENCY"):
        Settings.from_env()


@pytest.mark.parametrize(
    "document",
    [
        "[inference]\nmax_concurrency = 0\n",
        "[inference]\nmax_concurrency = 257\n",
        "[inference]\nmax_concurrency = true\n",
        "[inference]\nworkers = 4\n",
    ],
)
def test_inference_concurrency_file_validation(
    tmp_path: Path, document: str
) -> None:
    config = tmp_path / "server.toml"
    config.write_text(document, encoding="utf-8")
    with pytest.raises(ValueError, match="inference"):
        load_server_config(config)


def test_detection_profile_defaults_and_selection_validation() -> None:
    assert DetectionProfile().as_dict() == {
        "input_sizes": [[96, 96], [512, 512]],
        "threshold": 0.5,
        "nms_threshold": 0.4,
        "single_face_selection": "largest",
    }
    with pytest.raises(ValueError, match="single_face_selection"):
        DetectionProfile.from_mapping({"single_face_selection": "nearest"})


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([], "non-empty"),
        ([[100, 96]], "multiples of 32"),
        ([[96, 96], [96, 96]], "duplicates"),
        ([[True, 96]], "integers"),
        ([[2048, 2048], [32, 32]], "combined detector input pixels"),
    ],
)
def test_detector_input_size_validation(value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_detector_input_sizes(value)


def test_detector_config_rejects_missing_invalid_and_unknown_files(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Unable to read"):
        load_detector_input_sizes(tmp_path / "missing.toml")

    invalid = tmp_path / "invalid.toml"
    invalid.write_text("[detection\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid TOML"):
        load_detector_input_sizes(invalid)

    unknown = tmp_path / "unknown.toml"
    unknown.write_text("[detection]\nresolution = 512\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Unsupported \[detection\] setting"):
        load_detector_input_sizes(unknown)

    invalid_web = tmp_path / "invalid-web.toml"
    invalid_web.write_text("[web]\ndisabled = 'no'\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\[web\]\.disabled must be a boolean"):
        load_server_config(invalid_web)

    unknown_web = tmp_path / "unknown-web.toml"
    unknown_web.write_text("[web]\nenabled = true\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Unsupported \[web\] setting"):
        load_server_config(unknown_web)
