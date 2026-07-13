"""Anthropic Claude model wrapper."""

import time

import anthropic

from config import get_settings
from models.base import BaseLLM, ModelResponse


class AnthropicModel(BaseLLM):
    """Wrapper around the Anthropic Messages API."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        api_key: str | None = None,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        settings = get_settings()
        key = api_key or settings.anthropic_api_key
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is required for Anthropic models")
        self._client = anthropic.Anthropic(api_key=key)

    @property
    def provider(self) -> str:
        return "anthropic"

    def send(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> ModelResponse:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        start = time.perf_counter()
        response = self._client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        tokens = None
        if response.usage:
            tokens = response.usage.input_tokens + response.usage.output_tokens

        return ModelResponse(
            text=text,
            model=self.model,
            provider=self.provider,
            prompt=prompt,
            system_prompt=system_prompt,
            tokens_used=tokens,
            latency_ms=latency_ms,
        )
