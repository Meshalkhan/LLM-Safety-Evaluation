"""Factory for creating LLM model instances."""

from config import ProviderType, get_settings
from models.base import BaseLLM

PROVIDER_DEFAULTS: dict[str, str] = {
    "openai": "gpt-4.1",
    "anthropic": "claude-sonnet-4-20250514",
    "gemini": "gemini-2.0-flash",
}

MODEL_PROVIDER_MAP: dict[str, ProviderType] = {
    "gpt-": "openai",
    "o1": "openai",
    "o3": "openai",
    "claude": "anthropic",
    "gemini": "gemini",
}


def detect_provider(model: str) -> ProviderType:
    """Infer provider from model name prefix."""
    model_lower = model.lower()
    for prefix, provider in MODEL_PROVIDER_MAP.items():
        if model_lower.startswith(prefix):
            return provider
    return get_settings().default_provider


def create_model(
    model: str | None = None,
    provider: ProviderType | None = None,
    temperature: float | None = None,
    max_tokens: int = 1024,
) -> BaseLLM:
    """Create an LLM wrapper for the given provider and model."""
    settings = get_settings()
    resolved_model = model or settings.default_model
    resolved_provider = provider or detect_provider(resolved_model)
    resolved_temp = temperature if temperature is not None else settings.default_temperature

    if resolved_provider == "openai":
        from models.openai_model import OpenAIModel

        return OpenAIModel(
            model=resolved_model,
            temperature=resolved_temp,
            max_tokens=max_tokens,
        )
    if resolved_provider == "anthropic":
        from models.anthropic_model import AnthropicModel

        return AnthropicModel(
            model=resolved_model,
            temperature=resolved_temp,
            max_tokens=max_tokens,
        )
    if resolved_provider == "gemini":
        from models.gemini_model import GeminiModel

        return GeminiModel(
            model=resolved_model,
            temperature=resolved_temp,
            max_tokens=max_tokens,
        )
    raise ValueError(f"Unsupported provider: {resolved_provider}")
