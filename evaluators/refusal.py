"""Refusal consistency evaluation — measures refusal rate over repeated requests."""

from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from config import get_settings
from evaluators.base import BaseEvaluator
from utils.dataset_loader import load_dataset
from utils.scoring import is_refusal
from utils.reporting import EvalSummary

console = Console()


class RefusalEvaluator(BaseEvaluator):
  """Ask the same harmful request N times and measure refusal consistency."""

  eval_type = "refusal"

  def load_prompts(self) -> list[dict[str, Any]]:
    df = load_dataset("system_override")
    return df.to_dict(orient="records")

  def score_response(self, response: str, prompt_data: dict[str, Any]) -> str:
    return "REFUSED" if is_refusal(response) else "PASS"

  def run(self) -> EvalSummary:
    """Override base run to repeat each prompt multiple times."""
    settings = get_settings()
    repetitions = settings.refusal_repetitions
    base_prompts = self.load_prompts()
    total_calls = len(base_prompts) * repetitions

    summary = EvalSummary(eval_type=self.eval_type, total=total_calls)

    with Progress(
      SpinnerColumn(),
      TextColumn("[progress.description]{task.description}"),
      console=console,
    ) as progress:
      task = progress.add_task(
        f"Running refusal consistency ({total_calls} calls)...",
        total=total_calls,
      )

      for item in base_prompts:
        prompt_text = item["prompt"]
        for rep in range(1, repetitions + 1):
          response_obj = self.model.send(prompt_text)
          score = self.score_response(response_obj.text, item)

          detail = {
            "eval_type": self.eval_type,
            "prompt": prompt_text,
            "response": response_obj.text,
            "score": score,
            "category": item.get("category", ""),
            "id": item.get("id", ""),
            "repetition": rep,
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
            metadata={
              "category": item.get("category", ""),
              "id": item.get("id", ""),
              "repetition": rep,
            },
            timestamp=response_obj.timestamp,
          )
          progress.advance(task)

    return summary
