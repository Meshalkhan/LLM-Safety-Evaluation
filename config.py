"""Application configuration loaded from environment variables."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

ProviderType = Literal["openai", "anthropic", "gemini"]
EvalType = Literal[
    "jailbreak",
    "prompt_injection",
    "harmful_content",
    "hallucination",
    "refusal",
    "all",
]


class Settings(BaseSettings):
    """Runtime settings sourced from .env and environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""

    default_model: str = "gpt-4.1"
    default_temperature: float = 0.0
    default_provider: ProviderType = "openai"

    log_level: str = "INFO"
    reports_dir: Path = Field(default=REPORTS_DIR)

    refusal_repetitions: int = 10


def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
