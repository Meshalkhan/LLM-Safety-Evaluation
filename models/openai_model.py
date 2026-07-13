"""OpenAI model wrapper."""

import time

from openai import OpenAI

from config import get_settings
from models.base import BaseLLM, ModelResponse


class OpenAIModel(BaseLLM):
    """Wrapper around the OpenAI Chat Completions API."""

    def __init__(
        self,
        model: str = "gpt-4.1",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        api_key: str | None = None,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        settings = get_settings()
        key = api_key or settings.openai_api_key
        if not key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI models")
        self._client = OpenAI(api_key=key)

    @property
    def provider(self) -> str:
        return "openai"

    def send(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> ModelResponse:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        text = completion.choices[0].message.content or ""
        tokens = completion.usage.total_tokens if completion.usage else None

        return ModelResponse(
            text=text,
            model=self.model,
            provider=self.provider,
            prompt=prompt,
            system_prompt=system_prompt,
            tokens_used=tokens,
            latency_ms=latency_ms,
        )
