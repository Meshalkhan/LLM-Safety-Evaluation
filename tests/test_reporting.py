"""Tests for report generation."""

from datetime import datetime, timezone

from utils.reporting import EvalSummary, ReportBundle, ReportGenerator


def test_eval_summary_metrics():
    summary = EvalSummary(
        eval_type="jailbreak",
        total=10,
        pass_count=2,
        refused_count=7,
        partial_count=1,
    )
    assert summary.success_rate == 20.0
    assert summary.refusal_rate == 70.0
    assert summary.partial_rate == 10.0


def test_hallucination_accuracy():
    summary = EvalSummary(
        eval_type="hallucination",
        total=10,
        correct_count=8,
        incorrect_count=2,
    )
    assert summary.accuracy == 80.0


def test_report_generation(tmp_path):
    summaries = [
        EvalSummary(eval_type="jailbreak", total=5, refused_count=4, pass_count=1),
        EvalSummary(
            eval_type="hallucination",
            total=10,
            correct_count=9,
            incorrect_count=1,
        ),
    ]
    report = ReportBundle(
        run_id="test_run",
        model="gpt-4.1",
        provider="openai",
        timestamp=datetime.now(timezone.utc),
        summaries=summaries,
        entries=[
            {
                "eval_type": "jailbreak",
                "prompt": "test",
                "response": "refused",
                "score": "REFUSED",
            }
        ],
    )

    generator = ReportGenerator(reports_dir=tmp_path)
    paths = generator.save(report)

    assert paths["csv"].exists()
    assert paths["json"].exists()
    assert paths["markdown"].exists()
    assert "LLM Safety Evaluation Report" in paths["markdown"].read_text()
