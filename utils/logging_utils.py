"""Structured logging for evaluation runs."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

from config import get_settings

console = Console()


def setup_logging(level: str | None = None) -> logging.Logger:
  """Configure application-wide logging with rich output."""
  settings = get_settings()
  log_level = getattr(logging, (level or settings.log_level).upper(), logging.INFO)

  logging.basicConfig(
    level=log_level,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
  )
  return logging.getLogger("llm-safety-evals")


class EvalLogger:
  """Persists every prompt, response, timestamp, model, and score."""

  def __init__(self, run_id: str, reports_dir: Path | None = None) -> None:
    settings = get_settings()
    self.run_id = run_id
    self.reports_dir = reports_dir or settings.reports_dir
    self.log_dir = self.reports_dir / run_id
    self.log_dir.mkdir(parents=True, exist_ok=True)
    self.log_file = self.log_dir / "eval_log.jsonl"
    self._entries: list[dict[str, Any]] = []

  def log_entry(
    self,
    *,
    eval_type: str,
    prompt: str,
    response: str,
    model: str,
    provider: str,
    score: str,
    metadata: dict[str, Any] | None = None,
    timestamp: datetime | None = None,
  ) -> None:
    """Append a single evaluation record."""
    entry = {
      "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
      "run_id": self.run_id,
      "eval_type": eval_type,
      "model": model,
      "provider": provider,
      "prompt": prompt,
      "response": response,
      "score": score,
      "metadata": metadata or {},
    }
    self._entries.append(entry)

    with open(self.log_file, "a", encoding="utf-8") as f:
      f.write(json.dumps(entry, ensure_ascii=False) + "\n")

  @property
  def entries(self) -> list[dict[str, Any]]:
    return list(self._entries)
