from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path

import pytest
from insightface_server import models_cli
from insightface_server.licensing import ModelLicense


class _NonInteractiveInput(io.StringIO):
    def isatty(self) -> bool:
        return False


def test_install_requires_explicit_noninteractive_acceptance(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(models_cli.sys, "stdin", _NonInteractiveInput())
    result = models_cli.main(
        ["--models-dir", str(tmp_path), "install", "buffalo_l"]
    )
    captured = capsys.readouterr()
    assert result == 2
    assert "non-commercial research use only" in captured.out
    assert "add --accept-license" in captured.err


def test_successful_install_prints_license_after_result(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(models_cli, "install_package", lambda *_args, **_kwargs: "installed")
    result = models_cli.main(
        [
            "--models-dir",
            str(tmp_path),
            "install",
            "buffalo_l",
            "--accept-license",
        ]
    )
    captured = capsys.readouterr()
    assert result == 0
    assert "is installed and verified" in captured.out
    assert captured.out.rfind("LICENSE NOTICE") > captured.out.find("is installed and verified")
    assert "Commercial use requires a separate license" in captured.out
    assert "https://www.insightface.ai" in captured.out


def test_info_reports_pinned_source_and_hashes(capsys) -> None:
    assert models_cli.main(["info", "buffalo_l"]) == 0
    output = capsys.readouterr().out
    assert "/releases/download/v0.7/buffalo_l.zip" in output
    assert "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f" in output
    assert "det_10g.onnx SHA-256" in output
    assert "w600k_r50.onnx SHA-256" in output


def test_list_reports_every_supported_package(tmp_path: Path, capsys) -> None:
    assert models_cli.main(["--models-dir", str(tmp_path), "list"]) == 0
    output = capsys.readouterr().out
    for name in ("buffalo_l", "buffalo_m", "buffalo_sc", "antelopev2"):
        assert f"{name}\tv0.7\tnot installed" in output


@pytest.mark.parametrize(
    ("name", "detector", "recognizer"),
    (
        ("buffalo_m", "det_2.5g.onnx", "w600k_r50.onnx"),
        ("buffalo_sc", "det_500m.onnx", "w600k_mbf.onnx"),
        ("antelopev2", "scrfd_10g_bnkps.onnx", "glintr100.onnx"),
    ),
)
def test_info_reports_new_catalog_packages(
    name: str, detector: str, recognizer: str, capsys
) -> None:
    assert models_cli.main(["info", name]) == 0
    output = capsys.readouterr().out
    assert f"/releases/download/v0.7/{name}.zip" in output
    assert f"{detector} SHA-256" in output
    assert f"{recognizer} SHA-256" in output
    assert "Commercial use requires a separate license" in output


def test_verify_prints_installed_package_license(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        models_cli,
        "verify_installed",
        lambda _models_dir: (
            "buffalo_l",
            (
                {"file": "det_10g.onnx", "sha256": "a" * 64},
                {"file": "w600k_r50.onnx", "sha256": "b" * 64},
            ),
            ModelLicense(
                license_id="buffalo_l-public-v1",
                issuer="InsightFace",
                model_id="buffalo_l",
                grant="non-commercial",
                valid_from=datetime(2026, 7, 22, tzinfo=UTC),
                valid_until=None,
            ),
        ),
    )
    assert models_cli.main(["verify", "buffalo_l"]) == 0
    output = capsys.readouterr().out
    assert "Installed package: buffalo_l" in output
    assert "LICENSE VERIFIED" in output
    assert "Issuer: InsightFace" in output
    assert "Model ID: buffalo_l" in output
    assert "Signature: VALID" in output
    assert "Commercial use: NOT PERMITTED" in output
