"""
RAGPipeline: the single orchestration point tying retrieval and
generation together.

    question
        -> Retriever.search()          (embed, FAISS search, threshold filter)
        -> insufficient relevance? -> return the fixed fallback message, no LLM call
        -> format_context()             (bounded, labeled context block)
        -> LLMProvider.generate()       (grounded answer)
        -> sources built from retrieved chunk metadata (not parsed from LLM text)

Retrieval and generation stay logically separate: this class only wires
them together and applies policy (the relevance gate, source dedup,
confidence banding). It does not know how embeddings or FAISS work, and
it does not know which LLM SDK is in use.

Follow-up questions: conversation_history is passed to the LLM so it can
resolve references ("what about that?"), but retrieval always runs fresh
against the current question text only -- history never substitutes for
or overrides source grounding. Query rewriting for follow-ups is not
implemented yet; it can be added later without changing this class's
interface.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config import settings
from src.generation.llm import LLMProvider
from src.generation.prompts import format_context
from src.retrieval.retriever import Retriever
from src.vectorstore.faiss_store import SearchResult

logger = logging.getLogger(__name__)

INSUFFICIENT_CONTEXT_MESSAGE = (
    "I couldn't find sufficient information about this question in the indexed cricket rules."
)


@dataclass
class Source:
    law_number: str | None
    law_title: str | None
    section: str | None
    source_url: str | None
    score: float


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: list[Source]
    retrieved_chunks: list[SearchResult] = field(default_factory=list)
    confidence: str = "insufficient"  # "high" | "medium" | "low" | "insufficient"
    grounded: bool = False  # True only when an LLM actually produced the answer


def compute_confidence(top_score: float | None, score_threshold: float) -> str:
    """
    Map the best filtered similarity score to a coarse relevance label.
    This reflects retrieval relevance only, not factual certainty about the
    generated answer.
    """
    if top_score is None or top_score < score_threshold:
        return "insufficient"
    if top_score >= settings.confidence_high_threshold:
        return "high"
    if top_score >= settings.confidence_medium_threshold:
        return "medium"
    return "low"


def dedupe_sources(results: list[SearchResult]) -> list[Source]:
    """One Source per (law_number, section), keeping the highest score, in descending score order."""
    best: dict[tuple[str | None, str | None], Source] = {}
    for r in results:
        key = (r.chunk.law_number, r.chunk.section)
        existing = best.get(key)
        if existing is None or r.score > existing.score:
            best[key] = Source(
                law_number=r.chunk.law_number,
                law_title=r.chunk.law_title,
                section=r.chunk.section,
                source_url=r.chunk.source_url,
                score=r.score,
            )
    return sorted(best.values(), key=lambda s: s.score, reverse=True)


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm_provider: LLMProvider | None = None):
        self.retriever = retriever
        self.llm_provider = llm_provider

    def answer(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> RAGResponse:
        if not question or not question.strip():
            raise ValueError("question must not be empty")

        retrieval = self.retriever.search(question, top_k=top_k, score_threshold=score_threshold)

        if not retrieval.has_sufficient_relevance:
            logger.info("Insufficient relevance for question %r; returning fallback message", question)
            return RAGResponse(
                question=question,
                answer=INSUFFICIENT_CONTEXT_MESSAGE,
                sources=[],
                retrieved_chunks=retrieval.raw_results,
                confidence="insufficient",
                grounded=False,
            )

        if self.llm_provider is None:
            raise RuntimeError(
                "RAGPipeline has no configured LLM provider; cannot generate an answer. "
                "Pass an LLMProvider (see src.generation.llm.get_llm_provider)."
            )

        context = format_context([r.chunk for r in retrieval.filtered_results])
        answer_text = self.llm_provider.generate(question, context, conversation_history)

        top_score = retrieval.filtered_results[0].score
        confidence = compute_confidence(top_score, retrieval.score_threshold)
        sources = dedupe_sources(retrieval.filtered_results)

        return RAGResponse(
            question=question,
            answer=answer_text,
            sources=sources,
            retrieved_chunks=retrieval.raw_results,
            confidence=confidence,
            grounded=True,
        )
