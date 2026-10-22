from __future__ import annotations

import json

import pytest

from insightface.app.privateframe import cli


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (ValueError("invalid built-in config"), "invalid_config"),
        (FileNotFoundError("missing bundled YAML"), "file_not_found"),
    ],
)
def test_describe_failure_still_returns_one_error_json(monkeypatch, capsys, error, expected_code):
    def broken_contract(_parser):
        raise error

    monkeypatch.setattr(cli, "build_describe_payload", broken_contract)

    assert cli.main(["describe"]) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert payload["command"] == "describe"
    assert payload["error"]["stage"] == "describe"
    assert payload["error"]["code"] == expected_code


@pytest.mark.parametrize("internal_failure", [False, True])
def test_dry_run_distinguishes_unready_input_from_failed_diagnostics(
    monkeypatch, tmp_path, capsys, internal_failure
):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fixture; media inspection is stubbed")
    check = {
        "name": "doctor.internal.environment" if internal_failure else "runtime.pyav",
        "ok": False,
        "severity": "error",
        "message": "Environment inspection failed" if internal_failure else "PyAV is unavailable",
        "details": {},
    }
    monkeypatch.setattr(
        "insightface.app.privateframe.doctor.run_doctor",
        lambda **_kwargs: {
            "ok": not internal_failure,
            "ready": False,
            "checks": [check],
        },
    )
    exit_code = cli.main([
        "analyze", "--input", str(source),
        "--output-dir", str(tmp_path / "output"), "--dry-run",
    ])
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    payload = json.loads(captured.out)
    assert exit_code == (1 if internal_failure else 0)
    assert payload["ok"] is (not internal_failure)
    assert payload["ready"] is False
    assert payload["dry_run"] is True
    assert check in payload["checks"]
    assert payload["diagnostics"]["ready"] is False
    if internal_failure:
        assert payload["error"]["code"] == "operation_failed"
        assert payload["error"]["stage"] == "preflight"
    else:
        assert "error" not in payload
    assert not (tmp_path / "output").exists()
