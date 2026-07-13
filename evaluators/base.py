"""Base evaluator with shared run logic."""

from abc import ABC, abstractmethod
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from models.base import BaseLLM
from utils.logging_utils import EvalLogger
from utils.reporting import EvalSummary

console = Console()


class BaseEvaluator(ABC):
  """Abstract base for all safety evaluators."""

  eval_type: str = "base"

  def __init__(self, model: BaseLLM, logger: EvalLogger) -> None:
    self.model = model
    self.logger = logger

  @abstractmethod
  def load_prompts(self) -> list[dict[str, Any]]:
    """Load evaluation prompts from dataset."""

  @abstractmethod
  def score_response(self, response: str, prompt_data: dict[str, Any]) -> str:
    """Score a single model response."""

  def run(self) -> EvalSummary:
    """Execute the full evaluation pipeline."""
    prompts = self.load_prompts()
    summary = EvalSummary(eval_type=self.eval_type, total=len(prompts))

    with Progress(
      SpinnerColumn(),
      TextColumn("[progress.description]{task.description}"),
      console=console,
    ) as progress:
      task = progress.add_task(
        f"Running {self.eval_type} eval ({len(prompts)} prompts)...",
        total=len(prompts),
      )

      for item in prompts:
        prompt_text = item["prompt"]
        response_obj = self.model.send(prompt_text)
        score = self.score_response(response_obj.text, item)

        detail = {
          "eval_type": self.eval_type,
          "prompt": prompt_text,
          "response": response_obj.text,
          "score": score,
          "category": item.get("category", ""),
          "id": item.get("id", ""),
          "model": self.model.model,
          "provider": self.model.provider,
          "latency_ms": response_obj.latency_ms,
        }
        summary.details.append(detail)

        self._update_counts(summary, score)
        self.logger.log_entry(
          eval_type=self.eval_type,
          prompt=prompt_text,
          response=response_obj.text,
          model=self.model.model,
          provider=self.model.provider,
          score=score,
          metadata={"category": item.get("category", ""), "id": item.get("id", "")},
          timestamp=response_obj.timestamp,
        )
        progress.advance(task)

    return summary

  def _update_counts(self, summary: EvalSummary, score: str) -> None:
    if score == "PASS":
      summary.pass_count += 1
    elif score == "REFUSED":
      summary.refused_count += 1
    elif score == "PARTIAL":
      summary.partial_count += 1
    elif score == "CORRECT":
      summary.correct_count += 1
    elif score == "INCORRECT":
      summary.incorrect_count += 1
