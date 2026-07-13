"""Harmful content evaluation — tests refusal of dangerous requests."""

from typing import Any

from evaluators.base import BaseEvaluator
from utils.dataset_loader import load_dataset
from utils.scoring import score_harmful_content


class HarmfulContentEvaluator(BaseEvaluator):
  """Record whether the model refuses harmful content requests."""

  eval_type = "harmful_content"

  def load_prompts(self) -> list[dict[str, Any]]:
    df = load_dataset("harmless")
    return df.to_dict(orient="records")

  def score_response(self, response: str, prompt_data: dict[str, Any]) -> str:
    return score_harmful_content(response).value
