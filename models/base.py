"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ModelResponse:
    """Structured response from an LLM call."""

    text: str
    model: str
    provider: str
    prompt: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    system_prompt: str | None = None
    tokens_used: int | None = None
    latency_ms: float | None = None


class BaseLLM(ABC):
    """Unified interface for all LLM providers."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    @abstractmethod
    def provider(self) -> str:
        """Return the provider identifier."""

    @abstractmethod
    def send(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> ModelResponse:
        """Send a prompt and return the model response."""
