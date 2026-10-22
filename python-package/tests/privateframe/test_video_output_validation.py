from __future__ import annotations

from copy import deepcopy

import pytest

from insightface.app.privateframe.config import validate_video_output


def _video_output() -> dict[str, object]:
    return {
        "backend": "pyav",
        "encoder": "libx264",
        "pixel_format": "yuv420p",
        "preset": "medium",
        "rate_control": {"mode": "crf", "quality": 18},
        "keyframe_interval": 60,
        "faststart": True,
        "audio": {"debug": "none", "redacted": "aac", "bitrate": "192k"},
    }


@pytest.mark.parametrize(
    "value",
    ["definitely-not-a-bitrate", "0", "-1k", 0, 1.5, True],
)
def test_audio_bitrate_must_match_the_renderer_grammar(value) -> None:
    settings = _video_output()
    settings["audio"]["bitrate"] = value

    with pytest.raises((TypeError, ValueError), match="audio.bitrate"):
        validate_video_output(settings)


def test_audio_bitrate_must_fit_ffmpeg_integer_range() -> None:
    settings = _video_output()
    settings["audio"]["bitrate"] = 10**200

    with pytest.raises(ValueError, match="audio.bitrate"):
        validate_video_output(settings)


@pytest.mark.parametrize("value", [192_000, "192000", "192k", "1.5m"])
def test_supported_audio_bitrate_forms_are_accepted(value) -> None:
    settings = _video_output()
    settings["audio"]["bitrate"] = value

    validate_video_output(settings)


@pytest.mark.parametrize("value", [-1, 2_147_483_648])
def test_keyframe_interval_must_fit_ffmpeg_integer_range(value) -> None:
    settings = _video_output()
    settings["keyframe_interval"] = value

    with pytest.raises(ValueError, match="keyframe_interval"):
        validate_video_output(settings)


@pytest.mark.parametrize("value", [True, 1.5, "60"])
def test_keyframe_interval_must_be_an_integer(value) -> None:
    settings = _video_output()
    settings["keyframe_interval"] = value

    with pytest.raises(TypeError, match="keyframe_interval"):
        validate_video_output(settings)


def test_vbr_optional_bitrates_use_the_same_grammar() -> None:
    settings = _video_output()
    settings["rate_control"] = {
        "mode": "vbr",
        "bitrate": "4m",
        "max_bitrate": "definitely-not-a-bitrate",
    }

    with pytest.raises((TypeError, ValueError), match="max_bitrate"):
        validate_video_output(deepcopy(settings))
