"""
LLM abstraction.

The RAG pipeline talks to `LLMProvider.generate(...)` only -- it never
imports a provider SDK directly, so swapping providers means adding one
class here, not touching retrieval, chunking, or the pipeline.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from config import settings
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)


class LLMConfigurationError(RuntimeError):
    """Raised when the configured provider/API key is missing or invalid."""


class LLMGenerationError(RuntimeError):
    """Raised when a configured provider fails to produce an answer."""


class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        question: str,
        context: str,
        conversation_history: list[dict] | None = None,
    ) -> str:
        """Return a grounded answer string, given the question and retrieved context."""
        raise NotImplementedError


def _to_openai_messages(conversation_history: list[dict] | None) -> list[dict]:
    """Map our {role, content} history onto OpenAI-style chat messages (user/assistant only)."""
    if not conversation_history:
        return []
    messages = []
    for turn in conversation_history:
        role = turn.get("role") if turn.get("role") in ("user", "assistant") else "user"
        content = turn.get("content", "")
        if content:
            messages.append({"role": role, "content": content})
    return messages


class GroqLLM(LLMProvider):
    """
    Groq (GroqCloud, https://groq.com) -- fast inference for open models
    (Llama, Mixtral, etc.) behind an OpenAI-compatible API. Uses the
    `openai` SDK pointed at Groq's base URL rather than a Groq-specific
    client, since Groq's API is a drop-in OpenAI chat-completions surface.
    """

    BASE_URL = "https://api.groq.com/openai/v1"
    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        self.api_key = api_key or settings.llm_api_key
        self.model_name = model_name or settings.llm_model or self.DEFAULT_MODEL

        if not self.api_key:
            raise LLMConfigurationError(
                "No API key found. Set LLM_API_KEY (or GROQ_API_KEY) in .env, or pass api_key= explicitly."
            )

        from openai import OpenAI

        self._client = OpenAI(api_key=self.api_key, base_url=self.BASE_URL)

    def generate(
        self,
        question: str,
        context: str,
        conversation_history: list[dict] | None = None,
    ) -> str:
        if not question or not question.strip():
            raise ValueError("generate() requires a non-empty question")

        user_prompt = build_user_prompt(question, context, conversation_history=None)
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + _to_openai_messages(conversation_history)
            + [{"role": "user", "content": user_prompt}]
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
        except Exception as exc:  # noqa: BLE001 - any SDK/network failure -> one clear error type
            logger.error("Groq generation failed: %s", exc)
            raise LLMGenerationError(f"LLM request failed: {exc}") from exc

        text = response.choices[0].message.content if response.choices else None
        if not text or not text.strip():
            raise LLMGenerationError("LLM returned an empty response")

        return text


_PROVIDERS = {
    "groq": GroqLLM,
}


def get_llm_provider(provider_name: str | None = None, **kwargs) -> LLMProvider:
    """Factory: build the configured LLMProvider. Raises LLMConfigurationError if unset/unsupported."""
    name = (provider_name or settings.llm_provider or "none").strip().lower()

    if name not in _PROVIDERS:
        raise LLMConfigurationError(
            f"No usable LLM provider configured (LLM_PROVIDER={name!r}). "
            f"Set LLM_PROVIDER to one of {sorted(_PROVIDERS)} and provide LLM_API_KEY (or GROQ_API_KEY) in .env."
        )

    return _PROVIDERS[name](**kwargs)
