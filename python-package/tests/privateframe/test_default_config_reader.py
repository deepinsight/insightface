from __future__ import annotations

import pytest
import yaml

from insightface.app.privateframe.base_config import read_default_config


def test_default_reader_reloads_yaml_and_returns_independent_values(tmp_path):
    source = tmp_path / "base.yaml"
    raw = read_default_config()
    raw["render"]["video_output"]["rate_control"]["quality"] = 21
    source.write_text(yaml.safe_dump(raw), encoding="utf-8")
    first = read_default_config(source)
    assert first["render"]["video_output"]["rate_control"]["quality"] == 21
    first["render"]["video_output"]["rate_control"]["quality"] = 7
    assert read_default_config(source)["render"]["video_output"]["rate_control"]["quality"] == 21

    raw["render"]["video_output"]["rate_control"]["quality"] = 22
    source.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert read_default_config(source)["render"]["video_output"]["rate_control"]["quality"] == 22


@pytest.mark.parametrize("content", ["[]", "schema_version: true", "schema_version: 2"])
def test_default_reader_rejects_invalid_base_document(tmp_path, content):
    source = tmp_path / "base.yaml"
    source.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version: 1"):
        read_default_config(source)
