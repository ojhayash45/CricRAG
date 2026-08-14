#!/usr/bin/env python
"""
Evaluation harness.

    python evaluation/evaluate.py

Runs evaluation/questions.json against whatever is ACTUALLY in
vector_store/ right now (build it first with scripts/ingest.py). Every
number printed is measured this run -- nothing here is a hard-coded or
assumed score, per the project's "do not fabricate evaluation numbers"
rule.

Metrics:
  - Retrieval Recall@K       -- retrieval only, no LLM required.
  - Citation Accuracy        -- retrieval only. Because RAGPipeline builds
                                 citations directly from retrieved-chunk
                                 metadata (never from LLM free text), this
                                 is structurally close to 100% by
                                 construction -- it exists as a regression
                                 guard, not a discovery metric.
  - Out-of-domain rejection  -- retrieval only.
    - Supported Answer Rate    -- requires a configured LLM (LLM_PROVIDER=groq +
                                 LLM_API_KEY or GROQ_API_KEY in .env). If none is
                                 configured, this metric is reported as
                                 SKIPPED rather than faked. When it does
                                 run, "supported" is measured with a
                                 simple lexical-overlap heuristic between
                                 the answer and the retrieved context --
                                 an approximation, not a semantic judgment
                                 (see README limitations).
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from src.embeddings.embedding_service import EmbeddingService  # noqa: E402
from src.generation.llm import LLMConfigurationError, get_llm_provider  # noqa: E402
from src.generation.prompts import format_context  # noqa: E402
from src.pipeline.rag_pipeline import RAGPipeline, dedupe_sources  # noqa: E402
from src.retrieval.retriever import Retriever  # noqa: E402
from src.vectorstore.faiss_store import FAISSStore  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("evaluate")

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def _matches_expectation(chunk, expected_laws: list[str], expected_sections: list[str]) -> bool:
    if expected_laws and chunk.law_number not in expected_laws:
        return False
    if expected_sections and chunk.section not in expected_sections:
        return False
    return True


def recall_at_k(retriever: Retriever, questions: list[dict]) -> tuple[float, int]:
    """Fraction of in-domain questions where an expected law/section appears in the raw top-K."""
    evaluated = [q for q in questions if q.get("expected_law_numbers")]
    if not evaluated:
        return 0.0, 0

    hits = 0
    for q in evaluated:
        result = retriever.search(q["question"])
        expected_laws = q["expected_law_numbers"]
        expected_sections = q.get("expected_sections") or []
        if any(_matches_expectation(r.chunk, expected_laws, expected_sections) for r in result.raw_results):
            hits += 1
    return hits / len(evaluated), len(evaluated)


def out_of_domain_rejection(retriever: Retriever, questions: list[dict]) -> tuple[float, int]:
    """Fraction of expect_sufficient_relevance=False questions correctly returning no filtered results."""
    evaluated = [q for q in questions if q.get("expect_sufficient_relevance") is False]
    if not evaluated:
        return 0.0, 0

    correct = 0
    for q in evaluated:
        result = retriever.search(q["question"])
        if not result.has_sufficient_relevance:
            correct += 1
    return correct / len(evaluated), len(evaluated)


def citation_accuracy(retriever: Retriever, questions: list[dict]) -> tuple[float, int]:
    """
    Fraction of in-domain, sufficiently-relevant questions where every built citation's
    (law, section) genuinely appears among that question's retrieved chunks.
    """
    evaluated = [q for q in questions if q.get("expect_sufficient_relevance") and q.get("expected_law_numbers")]
    if not evaluated:
        return 0.0, 0

    correct = 0
    for q in evaluated:
        result = retriever.search(q["question"])
        if not result.filtered_results:
            continue  # can't check citations for a question with no retrieved evidence
        sources = dedupe_sources(result.filtered_results)
        retrieved_keys = {(r.chunk.law_number, r.chunk.section) for r in result.filtered_results}
        if all((s.law_number, s.section) in retrieved_keys for s in sources):
            correct += 1
    return correct / len(evaluated), len(evaluated)


_WORD_RE = re.compile(r"[a-z]{4,}")


def _content_words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def supported_answer_rate(pipeline: RAGPipeline, questions: list[dict]) -> tuple[float | None, int]:
    """
    LLM-gated. For grounded answers, measures the fraction of the answer's
    content words that also appear in the retrieved context -- a coarse,
    explainable lexical-overlap proxy for faithfulness, not a semantic
    entailment check. Requires >=50% overlap to count as "supported".
    """
    if pipeline.llm_provider is None:
        return None, 0

    evaluated = [q for q in questions if q.get("expect_sufficient_relevance")]
    if not evaluated:
        return 0.0, 0

    supported = 0
    checked = 0
    for q in evaluated:
        history = q.get("conversation_history")
        try:
            response = pipeline.answer(q["question"], conversation_history=history)
        except Exception as exc:  # noqa: BLE001 - one bad question shouldn't abort the whole eval run
            logger.warning("Skipping %s during faithfulness check: %s", q["id"], exc)
            continue
        if not response.grounded:
            continue

        checked += 1
        context = format_context([r.chunk for r in response.retrieved_chunks])
        answer_words = _content_words(response.answer)
        context_words = _content_words(context)
        if not answer_words:
            continue
        overlap = len(answer_words & context_words) / len(answer_words)
        if overlap >= 0.5:
            supported += 1

    if checked == 0:
        return None, 0
    return supported / checked, checked


def main() -> int:
    questions = load_questions()

    try:
        store = FAISSStore.load()
    except FileNotFoundError as exc:
        print(f"No vector index found: {exc}")
        print("Run `python scripts/ingest.py` first.")
        return 1

    embedder = EmbeddingService()
    retriever = Retriever(embedder, store)

    try:
        llm_provider = get_llm_provider()
    except LLMConfigurationError:
        llm_provider = None
    pipeline = RAGPipeline(retriever, llm_provider)

    recall, recall_n = recall_at_k(retriever, questions)
    citation, citation_n = citation_accuracy(retriever, questions)
    rejection, rejection_n = out_of_domain_rejection(retriever, questions)
    faithfulness, faithfulness_n = supported_answer_rate(pipeline, questions)

    print()
    print(f"Questions evaluated: {len(questions)}")
    print(f"Retrieval Recall@{settings.top_k}: {recall * 100:.1f}% ({recall_n} question(s))")
    print(f"Citation Accuracy: {citation * 100:.1f}% ({citation_n} question(s))")
    if faithfulness is None:
        print("Supported Answer Rate: SKIPPED (no LLM provider configured -- set LLM_PROVIDER=groq plus LLM_API_KEY or GROQ_API_KEY in .env)")
    else:
        print(f"Supported Answer Rate: {faithfulness * 100:.1f}% ({faithfulness_n} question(s), lexical-overlap heuristic)")
    print(f"Out-of-domain rejection: {rejection * 100:.1f}% ({rejection_n} question(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
