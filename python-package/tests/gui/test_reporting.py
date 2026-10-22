from pathlib import Path

import pytest

from insightface.gui.core.models import EvaluationResult
from insightface.gui.core.reporting import generate_html_report, generate_markdown_report, write_reports


def _result():
    return EvaluationResult(
        scenario="KYC / 1:1 Verification",
        model_name="buffalo_l",
        provider="CPU",
        threshold=0.5,
        dataset_summary={"total_pairs": 2},
        metrics={"accuracy": 1.0, "TAR@FAR=1e-2": 0.98},
        errors=[],
        latency={"average_ms": 1.2},
        license_status="Research / Non-commercial",
        created_at="2026-05-19T00:00:00Z",
        raw_results=[{"image": "a.jpg", "similarity": 0.9}],
        threshold_recommendation=0.42,
    )


def test_reporting_markdown_html():
    result = _result()
    md = generate_markdown_report(result, language="en")
    html = generate_html_report(result, language="en")
    assert "InsightFace Enterprise Evaluation Report" in md
    assert "Commercial Licensing Next Steps" in md
    assert "report interpretation" in md
    assert "https://www.insightface.ai/contact" in md
    assert "| Field | Value |" in md
    assert "<html" in html


def test_reporting_markdown_is_localized():
    md = generate_markdown_report(_result(), language="zh")
    assert "InsightFace 企业评测报告" in md
    assert "执行摘要" in md
    assert "本报告不构成法律建议" in md
    assert "报告解读" in md
    assert "https://www.insightface.ai/contact" in md


def test_write_reports_outputs_formatted_pdf(tmp_path):
    pytest.importorskip("reportlab")
    paths = write_reports(_result(), tmp_path, language="zh")

    assert paths["markdown"].endswith(".md")
    assert paths["html"].endswith(".html")
    assert paths["pdf"].endswith(".pdf")
    assert Path(paths["markdown"]).exists()
    assert Path(paths["html"]).exists()
    assert Path(paths["pdf"]).exists()


def test_write_reports_sets_primary_report_path_to_pdf(tmp_path):
    pytest.importorskip("reportlab")
    result = EvaluationResult(
        scenario="KYC / 1:1 Verification",
        model_name="buffalo_l",
        provider="CPU",
        threshold=0.5,
        dataset_summary={"total_pairs": 2},
        metrics={"accuracy": 1.0},
        errors=[],
        latency={"average_ms": 1.2},
        license_status="Research / Non-commercial",
        created_at="2026-05-19T00:00:00Z",
    )
    paths = write_reports(result, tmp_path)
    assert result.report_path == paths["pdf"]
