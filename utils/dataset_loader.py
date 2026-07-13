"""Dataset loading utilities."""

from pathlib import Path

import pandas as pd

from config import DATASETS_DIR


def load_dataset(name: str) -> pd.DataFrame:
  """Load a CSV dataset by name (without extension)."""
  path = DATASETS_DIR / f"{name}.csv"
  if not path.exists():
    raise FileNotFoundError(f"Dataset not found: {path}")
  return pd.read_csv(path)
