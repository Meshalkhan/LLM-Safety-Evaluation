"""Jailbreak evaluation — tests model resistance to adversarial prompts."""

from typing import Any

from evaluators.base import BaseEvaluator
from utils.dataset_loader import load_dataset
from utils.scoring import SafetyScore, score_jailbreak


class JailbreakEvaluator(BaseEvaluator):
  """Run jailbreak prompts and score PASS / REFUSED / PARTIAL."""

  eval_type = "jailbreak"

  def load_prompts(self) -> list[dict[str, Any]]:
    df = load_dataset("jailbreak")
    return df.to_dict(orient="records")

  def score_response(self, response: str, prompt_data: dict[str, Any]) -> str:
    return score_jailbreak(response).value
