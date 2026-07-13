"""Prompt injection evaluation — tests resistance to instruction hijacking."""

from typing import Any

from evaluators.base import BaseEvaluator
from utils.dataset_loader import load_dataset
from utils.scoring import score_prompt_injection


class PromptInjectionEvaluator(BaseEvaluator):
  """Measure whether the model follows malicious injected instructions."""

  eval_type = "prompt_injection"

  def load_prompts(self) -> list[dict[str, Any]]:
    df = load_dataset("prompt_injection")
    return df.to_dict(orient="records")

  def score_response(self, response: str, prompt_data: dict[str, Any]) -> str:
    injection_goal = prompt_data.get("injection_goal", "")
    return score_prompt_injection(response, injection_goal).value
