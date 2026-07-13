"""Matplotlib visualizations for evaluation results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.reporting import ReportBundle


def generate_charts(report: ReportBundle, output_dir: Path) -> list[Path]:
  """Generate bar charts for evaluation metrics."""
  output_dir.mkdir(parents=True, exist_ok=True)
  paths: list[Path] = []

  # Chart 1: Per-eval score distribution
  if report.summaries:
    fig, ax = plt.subplots(figsize=(10, 6))
    eval_names = [s.eval_type.replace("_", "\n") for s in report.summaries]
    refused = [s.refused_count for s in report.summaries]
    passed = [s.pass_count for s in report.summaries]
    partial = [s.partial_count for s in report.summaries]

    x = range(len(eval_names))
    width = 0.25
    ax.bar([i - width for i in x], refused, width, label="Refused", color="#2ecc71")
    ax.bar(x, passed, width, label="Complied (Unsafe)", color="#e74c3c")
    ax.bar([i + width for i in x], partial, width, label="Partial", color="#f39c12")

    ax.set_xlabel("Evaluation Type")
    ax.set_ylabel("Count")
    ax.set_title(f"Safety Evaluation Results — {report.model}")
    ax.set_xticks(list(x))
    ax.set_xticklabels(eval_names, fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    p1 = output_dir / "score_distribution.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths.append(p1)

  # Chart 2: Key metrics overview
  fig, ax = plt.subplots(figsize=(8, 5))
  metrics = {
    "Refusal\nRate": report.overall_refusal_rate(),
    "Jailbreak\nSuccess": report.jailbreak_success_pct(),
    "Injection\nSuccess": report.injection_success_pct(),
    "Hallucination\nAccuracy": report.hallucination_accuracy(),
  }
  colors = ["#2ecc71", "#e74c3c", "#e74c3c", "#3498db"]
  bars = ax.bar(metrics.keys(), metrics.values(), color=colors, width=0.5)
  ax.set_ylabel("Percentage (%)")
  ax.set_title(f"Key Safety Metrics — {report.model}")
  ax.set_ylim(0, 100)
  ax.grid(axis="y", alpha=0.3)

  for bar, val in zip(bars, metrics.values()):
    ax.text(
      bar.get_x() + bar.get_width() / 2,
      bar.get_height() + 1,
      f"{val:.1f}%",
      ha="center",
      va="bottom",
      fontsize=10,
    )

  fig.tight_layout()
  p2 = output_dir / "key_metrics.png"
  fig.savefig(p2, dpi=150)
  plt.close(fig)
  paths.append(p2)

  return paths
