from __future__ import annotations

import pytest
from insightface_server.config import Settings

SEARCH_ENVIRONMENT = (
    "INSIGHTFACE_DEFAULT_THRESHOLD",
    "INSIGHTFACE_COLLECTION_DEFAULT_SEARCH_PROFILE",
    "INSIGHTFACE_COLLECTION_DEFAULT_CAPACITY_ROWS",
    "INSIGHTFACE_COLLECTION_MAX_CAPACITY_ROWS",
    "INSIGHTFACE_COLLECTION_DEFAULT_MAX_FACES_PER_PERSON",
    "INSIGHTFACE_COLLECTION_DEFAULT_LOAD_POLICY",
)

SUPPORTED_PROFILES = (
    "fp32_v1",
    "fp16_v1",
    "bf16_v1",
    "int8_x1000_v1",
    "int8_x736_v1",
)


def test_search_collection_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in SEARCH_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.default_threshold == 0.4
    assert settings.default_search_profile == "fp32_v1"
    assert settings.default_search_capacity_rows == 100_000
    assert settings.max_search_capacity_rows == 10_000_000
    assert settings.default_max_faces_per_person == 20
    assert settings.default_search_load_policy == "lazy"


def test_search_collection_environment_is_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "INSIGHTFACE_COLLECTION_DEFAULT_SEARCH_PROFILE", "int8_x1000_rerank_v1"
    )
    with pytest.raises(ValueError, match="DEFAULT_SEARCH_PROFILE"):
        Settings.from_env()

    monkeypatch.setenv("INSIGHTFACE_COLLECTION_DEFAULT_SEARCH_PROFILE", "bf16_v1")
    monkeypatch.setenv("INSIGHTFACE_COLLECTION_DEFAULT_LOAD_POLICY", "warm")
    with pytest.raises(ValueError, match="DEFAULT_LOAD_POLICY"):
        Settings.from_env()


@pytest.mark.parametrize("profile", SUPPORTED_PROFILES)
def test_all_public_search_profiles_are_valid_configuration(
    monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    monkeypatch.setenv("INSIGHTFACE_COLLECTION_DEFAULT_SEARCH_PROFILE", profile)

    assert Settings.from_env().default_search_profile == profile


def test_threshold_must_be_nonnegative_raw_cosine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INSIGHTFACE_DEFAULT_THRESHOLD", "-0.01")
    with pytest.raises(ValueError, match="INSIGHTFACE_DEFAULT_THRESHOLD"):
        Settings.from_env()


def test_deployment_capacity_limit_is_configurable_and_bounds_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INSIGHTFACE_COLLECTION_DEFAULT_CAPACITY_ROWS", "125")
    monkeypatch.setenv("INSIGHTFACE_COLLECTION_MAX_CAPACITY_ROWS", "250")

    settings = Settings.from_env()

    assert settings.default_search_capacity_rows == 125
    assert settings.max_search_capacity_rows == 250

    monkeypatch.setenv("INSIGHTFACE_COLLECTION_DEFAULT_CAPACITY_ROWS", "251")
    with pytest.raises(ValueError, match="DEFAULT_CAPACITY_ROWS must not exceed"):
        Settings.from_env()
