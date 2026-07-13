"""CLI entry point for running LLM safety evaluations."""

import argparse
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from config import EvalType, get_settings
from evaluators.hallucination import HallucinationEvaluator
from evaluators.harmful_content import HarmfulContentEvaluator
from evaluators.jailbreak import JailbreakEvaluator
from evaluators.prompt_injection import PromptInjectionEvaluator
from evaluators.refusal import RefusalEvaluator
from models.factory import create_model, detect_provider
from utils.logging_utils import EvalLogger, setup_logging
from utils.reporting import ReportBundle, ReportGenerator
from utils.visualization import generate_charts

load_dotenv()

console = Console()
logger = setup_logging()

EVALUATOR_MAP = {
  "jailbreak": JailbreakEvaluator,
  "prompt_injection": PromptInjectionEvaluator,
  "harmful_content": HarmfulContentEvaluator,
  "hallucination": HallucinationEvaluator,
  "refusal": RefusalEvaluator,
}

ALL_EVALS: list[EvalType] = [
  "jailbreak",
  "prompt_injection",
  "harmful_content",
  "hallucination",
  "refusal",
]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="LLM Safety Evaluation Framework",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python run_eval.py --model gpt-4.1 --eval jailbreak
  python run_eval.py --model gpt-4.1 --eval all
  python run_eval.py --model claude-sonnet-4-20250514 --provider anthropic --eval all
  python run_eval.py --model gemini-2.0-flash --provider gemini --eval hallucination --temperature 0.0
    """,
  )
  parser.add_argument(
    "--model",
    type=str,
    default=None,
    help="Model identifier (e.g. gpt-4.1, claude-sonnet-4-20250514)",
  )
  parser.add_argument(
    "--provider",
    type=str,
    choices=["openai", "anthropic", "gemini"],
    default=None,
    help="LLM provider (auto-detected from model name if omitted)",
  )
  parser.add_argument(
    "--eval",
    type=str,
    required=True,
    choices=["jailbreak", "prompt_injection", "harmful_content", "hallucination", "refusal", "all"],
    help="Evaluation type to run",
  )
  parser.add_argument(
    "--temperature",
    type=float,
    default=None,
    help="Sampling temperature (default: 0.0)",
  )
  parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Override reports output directory",
  )
  return parser.parse_args()


def print_summary_table(report: ReportBundle) -> None:
  table = Table(title="Evaluation Summary", show_header=True, header_style="bold cyan")
  table.add_column("Evaluation", style="bold")
  table.add_column("Total", justify="right")
  table.add_column("Refused", justify="right", style="green")
  table.add_column("Complied", justify="right", style="red")
  table.add_column("Partial", justify="right", style="yellow")
  table.add_column("Accuracy", justify="right", style="blue")

  for s in report.summaries:
    table.add_row(
      s.eval_type,
      str(s.total),
      str(s.refused_count),
      str(s.pass_count),
      str(s.partial_count),
      f"{s.accuracy:.1f}%" if (s.correct_count + s.incorrect_count) > 0 else "—",
    )

  console.print(table)
  console.print()
  console.print(f"[bold]Overall Refusal Rate:[/bold] {report.overall_refusal_rate():.1f}%")
  console.print(f"[bold]Jailbreak Success:[/bold] {report.jailbreak_success_pct():.1f}%")
  console.print(f"[bold]Injection Success:[/bold] {report.injection_success_pct():.1f}%")
  console.print(f"[bold]Hallucination Accuracy:[/bold] {report.hallucination_accuracy():.1f}%")


def main() -> None:
  args = parse_args()
  settings = get_settings()

  model_name = args.model or settings.default_model
  provider = args.provider or detect_provider(model_name)
  eval_types: list[str] = ALL_EVALS if args.eval == "all" else [args.eval]

  run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
  reports_dir = settings.reports_dir
  if args.output_dir:
    from pathlib import Path
    reports_dir = Path(args.output_dir)

  console.print(f"\n[bold cyan]LLM Safety Evaluation Framework[/bold cyan]")
  console.print(f"Model: [green]{model_name}[/green] | Provider: [green]{provider}[/green]")
  console.print(f"Evaluations: [yellow]{', '.join(eval_types)}[/yellow]")
  console.print(f"Run ID: [dim]{run_id}[/dim]\n")

  try:
    model = create_model(
      model=model_name,
      provider=provider,
      temperature=args.temperature,
    )
  except ValueError as e:
    console.print(f"[bold red]Error:[/bold red] {e}")
    sys.exit(1)

  eval_logger = EvalLogger(run_id=run_id, reports_dir=reports_dir)
  summaries = []
  all_entries: list[dict] = []

  for eval_type in eval_types:
    evaluator_cls = EVALUATOR_MAP[eval_type]
    evaluator = evaluator_cls(model=model, logger=eval_logger)
    logger.info("Starting %s evaluation...", eval_type)
    summary = evaluator.run()
    summaries.append(summary)
    all_entries.extend(summary.details)
    logger.info(
      "%s complete — refused: %d, complied: %d, partial: %d",
      eval_type,
      summary.refused_count,
      summary.pass_count,
      summary.partial_count,
    )

  report = ReportBundle(
    run_id=run_id,
    model=model_name,
    provider=provider,
    timestamp=datetime.now(timezone.utc),
    summaries=summaries,
    entries=all_entries,
  )

  generator = ReportGenerator(reports_dir=reports_dir)
  paths = generator.save(report)
  chart_paths = generate_charts(report, paths["csv"].parent)

  console.print()
  print_summary_table(report)
  console.print()
  console.print("[bold green]Reports generated:[/bold green]")
  for name, path in paths.items():
    console.print(f"  {name}: {path}")
  for cp in chart_paths:
    console.print(f"  chart: {cp}")
  console.print(f"  log: {eval_logger.log_file}")


if __name__ == "__main__":
  main()
