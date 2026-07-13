"""Google Gemini model wrapper."""

import time

import google.generativeai as genai

from config import get_settings
from models.base import BaseLLM, ModelResponse


class GeminiModel(BaseLLM):
    """Wrapper around the Google Generative AI API."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        api_key: str | None = None,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        settings = get_settings()
        key = api_key or settings.google_api_key
        if not key:
            raise ValueError("GOOGLE_API_KEY is required for Gemini models")
        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(
            model_name=self.model,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )

    @property
    def provider(self) -> str:
        return "gemini"

    def send(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> ModelResponse:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        start = time.perf_counter()
        response = self._model.generate_content(full_prompt)
        latency_ms = (time.perf_counter() - start) * 1000

        text = response.text if response.text else ""

        return ModelResponse(
            text=text,
            model=self.model,
            provider=self.provider,
            prompt=prompt,
            system_prompt=system_prompt,
            latency_ms=latency_ms,
        )
