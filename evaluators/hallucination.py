"""Hallucination evaluation — tests factual accuracy on known-answer questions."""

from typing import Any

from evaluators.base import BaseEvaluator
from utils.dataset_loader import load_dataset
from utils.scoring import score_factual


class HallucinationEvaluator(BaseEvaluator):
  """Ask factual questions and compare responses to expected answers."""

  eval_type = "hallucination"

  def load_prompts(self) -> list[dict[str, Any]]:
    df = load_dataset("factual")
    return df.to_dict(orient="records")

  def score_response(self, response: str, prompt_data: dict[str, Any]) -> str:
    expected = prompt_data.get("expected_answer", "")
    return score_factual(response, expected).value
