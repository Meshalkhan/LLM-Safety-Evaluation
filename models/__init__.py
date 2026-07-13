"""LLM model wrappers with a unified interface."""

from models.base import BaseLLM, ModelResponse

__all__ = ["BaseLLM", "ModelResponse", "create_model"]


def create_model(*args, **kwargs):
    from models.factory import create_model as _create_model

    return _create_model(*args, **kwargs)
