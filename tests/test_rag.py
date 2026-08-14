from __future__ import annotations

import pytest

from src.generation.llm import LLMProvider
from src.models.schemas import Chunk
from src.pipeline.rag_pipeline import INSUFFICIENT_CONTEXT_MESSAGE, RAGPipeline, compute_confidence
from src.retrieval.retriever import RetrievalResult
from src.vectorstore.faiss_store import SearchResult


def make_chunk(chunk_id: str, **overrides) -> Chunk:
    defaults = dict(
        chunk_id=chunk_id,
        document_id="doc-1",
        law_number="36",
        law_title="Out Leg Before Wicket",
        section="36.1",
        edition="2017 Code",
        text=f"text for {chunk_id}",
        source_url="file://test",
        chunk_index=0,
        content_hash=f"hash-{chunk_id}",
    )
    defaults.update(overrides)
    return Chunk(**defaults)


class FakeRetriever:
    """Stand-in for Retriever -- returns a canned RetrievalResult, no embedding/FAISS involved."""

    def __init__(self, result: RetrievalResult):
        self._result = result
        self.last_call = None

    def search(self, query, top_k=None, score_threshold=None):
        self.last_call = (query, top_k, score_threshold)
        return self._result


class FakeLLM(LLMProvider):
    def __init__(self, answer_text: str = "Grounded answer."):
        self.answer_text = answer_text
        self.calls = []

    def generate(self, question, context, conversation_history=None):
        self.calls.append((question, context, conversation_history))
        return self.answer_text


def sufficient_result(top_score: float = 0.8) -> RetrievalResult:
    chunk_a = make_chunk("a", law_number="36", section="36.1")
    chunk_b = make_chunk("b", law_number="36", section="36.1")  # same law/section, lower score
    raw = [
        SearchResult(chunk=chunk_a, score=top_score),
        SearchResult(chunk=chunk_b, score=top_score - 0.1),
    ]
    return RetrievalResult(query="q", top_k=5, score_threshold=0.35, raw_results=raw, filtered_results=raw)


def insufficient_result() -> RetrievalResult:
    chunk_a = make_chunk("a")
    raw = [SearchResult(chunk=chunk_a, score=0.1)]
    return RetrievalResult(query="q", top_k=5, score_threshold=0.35, raw_results=raw, filtered_results=[])


class TestRAGPipelineGroundedPath:
    def test_returns_llm_answer_when_relevance_sufficient(self):
        retriever = FakeRetriever(sufficient_result())
        llm = FakeLLM("LBW means...")
        pipeline = RAGPipeline(retriever, llm)

        response = pipeline.answer("What is LBW?")
        assert response.answer == "LBW means..."
        assert response.grounded is True
        assert len(llm.calls) == 1

    def test_passes_question_and_context_and_history_to_llm(self):
        retriever = FakeRetriever(sufficient_result())
        llm = FakeLLM()
        pipeline = RAGPipeline(retriever, llm)
        history = [{"role": "user", "content": "earlier question"}]

        pipeline.answer("What about a wide?", conversation_history=history)
        question, context, passed_history = llm.calls[0]
        assert question == "What about a wide?"
        assert "[Law 36" in context
        assert passed_history == history

    def test_sources_are_deduplicated_by_law_and_section(self):
        retriever = FakeRetriever(sufficient_result())
        llm = FakeLLM()
        pipeline = RAGPipeline(retriever, llm)

        response = pipeline.answer("What is LBW?")
        # Both chunks share (law_number=36, section=36.1) -> one deduped source, highest score kept.
        assert len(response.sources) == 1
        assert response.sources[0].score == pytest.approx(0.8)

    def test_retrieval_always_uses_current_question_not_history(self):
        retriever = FakeRetriever(sufficient_result())
        llm = FakeLLM()
        pipeline = RAGPipeline(retriever, llm)
        history = [{"role": "user", "content": "an old question"}]

        pipeline.answer("current question", conversation_history=history)
        assert retriever.last_call[0] == "current question"

    def test_raises_when_no_llm_provider_configured(self):
        retriever = FakeRetriever(sufficient_result())
        pipeline = RAGPipeline(retriever, llm_provider=None)
        with pytest.raises(RuntimeError):
            pipeline.answer("What is LBW?")

    def test_empty_question_raises_value_error(self):
        retriever = FakeRetriever(sufficient_result())
        pipeline = RAGPipeline(retriever, FakeLLM())
        with pytest.raises(ValueError):
            pipeline.answer("   ")


class TestRAGPipelineInsufficientRelevance:
    def test_returns_fallback_message_without_calling_llm(self):
        retriever = FakeRetriever(insufficient_result())
        llm = FakeLLM()
        pipeline = RAGPipeline(retriever, llm)

        response = pipeline.answer("What is the capital of France?")
        assert response.answer == INSUFFICIENT_CONTEXT_MESSAGE
        assert response.grounded is False
        assert response.sources == []
        assert response.confidence == "insufficient"
        assert len(llm.calls) == 0

    def test_still_returns_raw_retrieved_chunks_for_debugging(self):
        retriever = FakeRetriever(insufficient_result())
        pipeline = RAGPipeline(retriever, FakeLLM())

        response = pipeline.answer("out of domain question")
        assert len(response.retrieved_chunks) == 1

    def test_out_of_domain_question_does_not_require_llm_provider(self):
        retriever = FakeRetriever(insufficient_result())
        pipeline = RAGPipeline(retriever, llm_provider=None)
        response = pipeline.answer("Who won the 2019 World Cup?")
        assert response.answer == INSUFFICIENT_CONTEXT_MESSAGE


class TestComputeConfidence:
    def test_below_threshold_is_insufficient(self):
        assert compute_confidence(0.2, score_threshold=0.35) == "insufficient"

    def test_none_score_is_insufficient(self):
        assert compute_confidence(None, score_threshold=0.35) == "insufficient"

    def test_just_above_threshold_is_low(self):
        assert compute_confidence(0.36, score_threshold=0.35) == "low"

    def test_mid_range_is_medium(self):
        assert compute_confidence(0.5, score_threshold=0.35) == "medium"

    def test_high_score_is_high(self):
        assert compute_confidence(0.9, score_threshold=0.35) == "high"
