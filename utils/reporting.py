"""Report generation: CSV, JSON, and Markdown."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import get_settings


@dataclass
class EvalSummary:
  """Aggregated metrics for a single evaluation type."""

  eval_type: str
  total: int = 0
  pass_count: int = 0
  refused_count: int = 0
  partial_count: int = 0
  correct_count: int = 0
  incorrect_count: int = 0
  details: list[dict[str, Any]] = field(default_factory=list)

  @property
  def success_rate(self) -> float:
    """Unsafe success rate (jailbreak/injection/harmful compliance)."""
    if self.total == 0:
      return 0.0
    return self.pass_count / self.total * 100

  @property
  def refusal_rate(self) -> float:
    if self.total == 0:
      return 0.0
    return self.refused_count / self.total * 100

  @property
  def partial_rate(self) -> float:
    if self.total == 0:
      return 0.0
    return self.partial_count / self.total * 100

  @property
  def accuracy(self) -> float:
    """Factual accuracy for hallucination tests."""
    answered = self.correct_count + self.incorrect_count
    if answered == 0:
      return 0.0
    return self.correct_count / answered * 100

  def to_dict(self) -> dict[str, Any]:
    return {
      "eval_type": self.eval_type,
      "total": self.total,
      "pass_count": self.pass_count,
      "refused_count": self.refused_count,
      "partial_count": self.partial_count,
      "correct_count": self.correct_count,
      "incorrect_count": self.incorrect_count,
      "success_rate_pct": round(self.success_rate, 2),
      "refusal_rate_pct": round(self.refusal_rate, 2),
      "partial_rate_pct": round(self.partial_rate, 2),
      "accuracy_pct": round(self.accuracy, 2),
    }


@dataclass
class ReportBundle:
  """Complete evaluation report for a run."""

  run_id: str
  model: str
  provider: str
  timestamp: datetime
  summaries: list[EvalSummary]
  entries: list[dict[str, Any]]

  def overall_refusal_rate(self) -> float:
    safety_evals = [
      s for s in self.summaries
      if s.eval_type in ("jailbreak", "prompt_injection", "harmful_content", "refusal")
    ]
    total = sum(s.total for s in safety_evals)
    refused = sum(s.refused_count for s in safety_evals)
    if total == 0:
      return 0.0
    return refused / total * 100

  def jailbreak_success_pct(self) -> float:
    for s in self.summaries:
      if s.eval_type == "jailbreak":
        return s.success_rate
    return 0.0

  def injection_success_pct(self) -> float:
    for s in self.summaries:
      if s.eval_type == "prompt_injection":
        return s.success_rate
    return 0.0

  def hallucination_accuracy(self) -> float:
    for s in self.summaries:
      if s.eval_type == "hallucination":
        return s.accuracy
    return 0.0


class ReportGenerator:
  """Generate CSV, JSON, and Markdown reports."""

  def __init__(self, reports_dir: Path | None = None) -> None:
    settings = get_settings()
    self.reports_dir = reports_dir or settings.reports_dir

  def save(self, report: ReportBundle) -> dict[str, Path]:
    """Write all report formats and return file paths."""
    out_dir = self.reports_dir / report.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # Detailed CSV
    csv_path = out_dir / "results.csv"
    pd.DataFrame(report.entries).to_csv(csv_path, index=False)
    paths["csv"] = csv_path

    # Summary JSON
    json_path = out_dir / "summary.json"
    summary_data = {
      "run_id": report.run_id,
      "model": report.model,
      "provider": report.provider,
      "timestamp": report.timestamp.isoformat(),
      "overall_refusal_rate_pct": round(report.overall_refusal_rate(), 2),
      "jailbreak_success_pct": round(report.jailbreak_success_pct(), 2),
      "prompt_injection_success_pct": round(report.injection_success_pct(), 2),
      "hallucination_accuracy_pct": round(report.hallucination_accuracy(), 2),
      "evaluations": [s.to_dict() for s in report.summaries],
    }
    json_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    paths["json"] = json_path

    # Markdown report
    md_path = out_dir / "report.md"
    md_path.write_text(self._build_markdown(report, summary_data), encoding="utf-8")
    paths["markdown"] = md_path

    return paths

  def _build_markdown(self, report: ReportBundle, summary: dict) -> str:
    lines = [
      f"# LLM Safety Evaluation Report",
      "",
      f"**Run ID:** {report.run_id}  ",
      f"**Model:** {report.model} ({report.provider})  ",
      f"**Timestamp:** {report.timestamp.isoformat()}  ",
      "",
      "## Overall Metrics",
      "",
      f"| Metric | Value |",
      f"|--------|-------|",
      f"| Overall Refusal Rate | {summary['overall_refusal_rate_pct']}% |",
      f"| Jailbreak Success Rate | {summary['jailbreak_success_pct']}% |",
      f"| Prompt Injection Success Rate | {summary['prompt_injection_success_pct']}% |",
      f"| Hallucination Accuracy | {summary['hallucination_accuracy_pct']}% |",
      "",
      "## Per-Evaluation Breakdown",
      "",
    ]

    for s in report.summaries:
      lines.extend([
        f"### {s.eval_type.replace('_', ' ').title()}",
        "",
        f"- **Total prompts:** {s.total}",
        f"- **Refused:** {s.refused_count} ({s.refusal_rate:.1f}%)",
        f"- **Complied (PASS):** {s.pass_count} ({s.success_rate:.1f}%)",
        f"- **Partial:** {s.partial_count} ({s.partial_rate:.1f}%)",
      ])
      if s.correct_count or s.incorrect_count:
        lines.append(f"- **Accuracy:** {s.accuracy:.1f}%")
      lines.append("")

    lines.extend([
      "## Interpretation",
      "",
      "- **Lower jailbreak/injection success %** = safer model",
      "- **Higher refusal rate** on harmful content = safer model",
      "- **Higher hallucination accuracy** = more factually reliable model",
      "",
      f"Full results available in `results.csv` and `summary.json`.",
    ])

    return "\n".join(lines)
