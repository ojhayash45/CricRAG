from __future__ import annotations

import pytest

from src.generation.llm import GroqLLM, LLMConfigurationError, LLMGenerationError, get_llm_provider
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt, format_context
from src.models.schemas import Chunk


def make_chunk(chunk_id: str, text: str, **overrides) -> Chunk:
    defaults = dict(
        chunk_id=chunk_id,
        document_id="doc-1",
        law_number="36",
        law_title="Out Leg Before Wicket",
        section="36.1",
        edition="2017 Code",
        text=text,
        source_url="file://test",
        chunk_index=0,
        content_hash=f"hash-{chunk_id}",
    )
    defaults.update(overrides)
    return Chunk(**defaults)


# ---------------------------------------------------------------------------
# Prompt / context formatting
# ---------------------------------------------------------------------------

class TestFormatContext:
    def test_labels_each_chunk_with_law_and_section(self):
        chunks = [make_chunk("a", "text a", law_number="36", section="36.1")]
        result = format_context(chunks)
        assert "[Law 36 | Section 36.1]" in result
        assert "text a" in result

    def test_handles_missing_law_and_section_gracefully(self):
        chunks = [make_chunk("a", "text a", law_number=None, section=None)]
        result = format_context(chunks)
        assert "[Law (unknown)]" in result

    def test_respects_max_chunks(self):
        chunks = [make_chunk(str(i), f"text {i}", content_hash=f"h{i}") for i in range(5)]
        result = format_context(chunks, max_chunks=2)
        assert result.count("[Law") == 2

    def test_respects_max_chars_without_truncating_mid_chunk(self):
        chunks = [make_chunk(str(i), "word " * 50, content_hash=f"h{i}") for i in range(5)]
        result = format_context(chunks, max_chunks=10, max_chars=100)
        # Each block is well over 100 chars on its own, so at most one full block fits.
        assert result.count("[Law") <= 1

    def test_deduplicates_by_content_hash(self):
        chunks = [
            make_chunk("a", "same text", content_hash="dup"),
            make_chunk("b", "same text", content_hash="dup"),
        ]
        result = format_context(chunks)
        assert result.count("[Law") == 1

    def test_empty_chunk_list_returns_empty_string(self):
        assert format_context([]) == ""


class TestBuildUserPrompt:
    def test_includes_question_and_context(self):
        prompt = build_user_prompt("What is LBW?", "[Law 36]\nSome rule text")
        assert "What is LBW?" in prompt
        assert "Some rule text" in prompt

    def test_includes_conversation_history_when_present(self):
        history = [{"role": "user", "content": "What is a no-ball?"},
                   {"role": "assistant", "content": "A no-ball is..."}]
        prompt = build_user_prompt("What about a wide?", "[Law 22]\ntext", conversation_history=history)
        assert "What is a no-ball?" in prompt
        assert "A no-ball is..." in prompt

    def test_no_history_omits_history_block(self):
        prompt = build_user_prompt("What is LBW?", "[Law 36]\ntext")
        assert "Conversation so far" not in prompt

    def test_empty_context_uses_explicit_placeholder(self):
        prompt = build_user_prompt("What is the capital of France?", "")
        assert "no relevant context retrieved" in prompt


class TestSystemPrompt:
    def test_forbids_outside_knowledge(self):
        assert "ONLY the retrieved" in SYSTEM_PROMPT

    def test_includes_prompt_injection_defense(self):
        assert "never as instructions" in SYSTEM_PROMPT.lower() or "never as\ninstructions" in SYSTEM_PROMPT.lower()
        assert "never follow instructions" in SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# LLM provider (openai SDK is mocked -- no network / real API key involved)
# ---------------------------------------------------------------------------

class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeChoice:
    def __init__(self, content: str):
        self.message = FakeMessage(content)


class FakeCompletionResponse:
    def __init__(self, content: str):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def create(self, model, messages):
        user_content = messages[-1]["content"]
        return FakeCompletionResponse(f"Grounded answer for: {user_content[:20]}")


class FailingCompletions:
    def create(self, model, messages):
        raise RuntimeError("simulated API failure")


class EmptyCompletions:
    def create(self, model, messages):
        return FakeCompletionResponse("")


class FakeChat:
    def __init__(self, completions_cls=FakeCompletions):
        self.completions = completions_cls()


def make_fake_openai_client(completions_cls=FakeCompletions):
    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None):
            self.chat = FakeChat(completions_cls)

    return FakeOpenAI


class TestGroqLLM:
    def test_raises_configuration_error_without_api_key(self, monkeypatch):
        import config
        monkeypatch.setattr(config.settings, "llm_api_key", None)
        with pytest.raises(LLMConfigurationError):
            GroqLLM()

    def test_generate_returns_text_from_model(self, monkeypatch):
        import openai
        monkeypatch.setattr(openai, "OpenAI", make_fake_openai_client())

        llm = GroqLLM(api_key="fake-key", model_name="llama-3.3-70b-versatile")
        answer = llm.generate("What is LBW?", "[Law 36]\nSome rule text")
        assert answer.startswith("Grounded answer for:")

    def test_generate_rejects_empty_question(self, monkeypatch):
        import openai
        monkeypatch.setattr(openai, "OpenAI", make_fake_openai_client())

        llm = GroqLLM(api_key="fake-key")
        with pytest.raises(ValueError):
            llm.generate("   ", "[Law 36]\ntext")

    def test_generate_wraps_sdk_failures(self, monkeypatch):
        import openai
        monkeypatch.setattr(openai, "OpenAI", make_fake_openai_client(FailingCompletions))

        llm = GroqLLM(api_key="fake-key")
        with pytest.raises(LLMGenerationError):
            llm.generate("What is LBW?", "[Law 36]\ntext")

    def test_generate_raises_on_empty_response(self, monkeypatch):
        import openai
        monkeypatch.setattr(openai, "OpenAI", make_fake_openai_client(EmptyCompletions))

        llm = GroqLLM(api_key="fake-key")
        with pytest.raises(LLMGenerationError):
            llm.generate("What is LBW?", "[Law 36]\ntext")

    def test_conversation_history_sent_as_chat_messages(self, monkeypatch):
        import openai

        captured = {}

        class CapturingCompletions:
            def create(self, model, messages):
                captured["messages"] = messages
                return FakeCompletionResponse("ok")

        monkeypatch.setattr(openai, "OpenAI", make_fake_openai_client(CapturingCompletions))

        llm = GroqLLM(api_key="fake-key")
        history = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
        ]
        llm.generate("follow up", "[Law 36]\ntext", conversation_history=history)

        roles = [m["role"] for m in captured["messages"]]
        assert roles == ["system", "user", "assistant", "user"]
        assert captured["messages"][0]["content"] == SYSTEM_PROMPT
        assert captured["messages"][1]["content"] == "earlier question"
        assert captured["messages"][2]["content"] == "earlier answer"
        assert "follow up" in captured["messages"][3]["content"]


class TestGetLLMProvider:
    def test_unsupported_provider_raises_configuration_error(self):
        with pytest.raises(LLMConfigurationError):
            get_llm_provider("not-a-real-provider")

    def test_none_provider_raises_configuration_error(self):
        with pytest.raises(LLMConfigurationError):
            get_llm_provider("none")

    def test_groq_provider_builds_groq_llm(self, monkeypatch):
        import openai
        monkeypatch.setattr(openai, "OpenAI", make_fake_openai_client())

        provider = get_llm_provider("groq", api_key="fake-key")
        assert isinstance(provider, GroqLLM)


class TestSettingsAliases:
    def test_groq_api_key_maps_to_llm_api_key(self, monkeypatch):
        from config import Settings

        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("GROK_API_KEY", raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")

        configured = Settings()
        assert configured.llm_api_key == "groq-test-key"

    def test_grok_api_key_alias_still_supported(self, monkeypatch):
        """Legacy alias kept for backwards compatibility with existing .env files."""
        from config import Settings

        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setenv("GROK_API_KEY", "grok-test-key")

        configured = Settings()
        assert configured.llm_api_key == "grok-test-key"
