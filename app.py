"""
Streamlit UI for the Cricket Laws RAG system.

This file only orchestrates: it loads cached resources (embedding model,
FAISS index, LLM client), wires them into a RAGPipeline per query, and
renders the result. All actual RAG logic lives in src/ -- nothing here
recomputes embeddings or rebuilds the index except in response to an
explicit sidebar button press.
"""
from __future__ import annotations

import logging
from datetime import datetime

import streamlit as st

from config import settings
from src.embeddings.embedding_service import EmbeddingService
from src.generation.llm import LLMConfigurationError, LLMGenerationError, LLMProvider, get_llm_provider
from src.pipeline.ingest_runner import rebuild_index_from_saved_chunks, run_ingestion
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.retriever import Retriever
from src.vectorstore.factory import VectorStore, load_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

st.set_page_config(page_title="Cricket Laws RAG", page_icon="🏏", layout="wide")

CONFIDENCE_LABELS = {
    "high": "🟢 High relevance",
    "medium": "🟡 Medium relevance",
    "low": "🟠 Low relevance",
    "insufficient": "⚪ Insufficient relevance",
}


# ---------------------------------------------------------------------------
# Cached resources -- loaded once per process, not on every rerun/query.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@st.cache_resource(show_spinner="Loading vector index...")
def get_vector_store() -> VectorStore | None:
    return load_vector_store()


@st.cache_resource(show_spinner="Connecting to LLM provider...")
def get_llm_provider_cached() -> LLMProvider | None:
    try:
        return get_llm_provider()
    except LLMConfigurationError as exc:
        logger.warning("LLM provider not configured: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # list[dict]: role, content, sources, confidence, retrieved_chunks, grounded
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

vector_store = get_vector_store()
llm_provider = get_llm_provider_cached()

with st.sidebar:
    st.header("Configuration")
    top_k = st.number_input("Top K", min_value=1, max_value=20, value=settings.top_k, step=1)
    similarity_threshold = st.slider(
        "Similarity threshold", min_value=-1.0, max_value=1.0,
        value=float(settings.similarity_threshold), step=0.01,
        help="Minimum cosine similarity for a chunk to count as relevant. "
             "Uncalibrated starting point -- tune against evaluation/questions.json.",
    )
    st.caption(f"Embedding model: `{settings.embedding_model}`")
    st.caption(f"LLM: `{settings.llm_provider}` / `{settings.llm_model or '(default)'}`")
    st.session_state.debug_mode = st.checkbox("Show retrieval details", value=st.session_state.debug_mode)

    st.divider()
    st.header("Knowledge Base")
    if vector_store is not None and vector_store.size > 0:
        chunks = vector_store.chunks
        law_numbers = {c.law_number for c in chunks if c.law_number}
        editions = {c.edition for c in chunks if c.edition}
        st.metric("Indexed chunks", vector_store.size)
        st.metric("Laws covered", len(law_numbers))
        st.caption(f"Edition: {', '.join(sorted(editions)) if editions else 'N/A'}")
        if settings.database_url:
            st.caption("Backend: Postgres (persists across restarts)")
        else:
            index_path = settings.vector_store_dir / "index.faiss"
            if index_path.exists():
                mtime = datetime.fromtimestamp(index_path.stat().st_mtime)
                st.caption(f"Last indexed: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.warning("No index found yet.")

    st.divider()
    st.header("Controls")
    if st.button("Rebuild index", help="Re-run full ingestion: load corpus, chunk, embed, re-index."):
        with st.spinner("Rebuilding index from data/raw/ ..."):
            try:
                stats = run_ingestion()
                get_vector_store.clear()
                st.success(
                    f"Rebuilt: {stats.documents_loaded} document(s), "
                    f"{stats.chunks_created} chunk(s), {stats.embeddings_generated} embedding(s)."
                )
            except RuntimeError as exc:
                logger.exception("Rebuild index failed")
                st.error(f"Rebuild failed: {exc}")
        st.rerun()

    if st.button("Reindex from saved chunks", help="Re-embed already-parsed chunks (e.g. after changing the embedding model)."):
        with st.spinner("Re-embedding saved chunks ..."):
            try:
                count = rebuild_index_from_saved_chunks()
                get_vector_store.clear()
                st.success(f"Reindexed {count} chunk(s).")
            except RuntimeError as exc:
                logger.exception("Reindex failed")
                st.error(f"Reindex failed: {exc}")
        st.rerun()

    if st.button("Reload index from disk", help="Pick up a vector store built externally (e.g. via scripts/ingest.py)."):
        get_vector_store.clear()
        st.rerun()

    if st.button("Clear cache"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("🏏 Cricket Laws RAG")
st.caption("Ask questions about the Laws of Cricket using a retrieval-augmented knowledge base.")

status_cols = st.columns(2)
with status_cols[0]:
    if vector_store is not None and vector_store.size > 0:
        st.success(f"Index loaded ({vector_store.size} chunks)")
    else:
        st.error("No index loaded -- run `python scripts/ingest.py` or click **Rebuild index** in the sidebar.")
with status_cols[1]:
    if llm_provider is not None:
        st.success("LLM configured")
    else:
        st.warning("LLM not configured -- set `LLM_PROVIDER=groq` and `LLM_API_KEY` (or `GROQ_API_KEY`) in `.env` to enable answers.")


def render_sources(sources) -> None:
    if not sources:
        return
    st.markdown("**Relevant Sources**" if len(sources) > 1 else "**Relevant Law**")
    for i, src in enumerate(sources, 1):
        law_label = f"Law {src.law_number}" if src.law_number else "Law (unknown)"
        if src.law_title:
            law_label += f" — {src.law_title}"
        if src.section:
            law_label += f" | Section {src.section}"
        prefix = f"{i}. " if len(sources) > 1 else ""
        line = f"{prefix}{law_label}"
        if src.source_url and src.source_url.startswith("http"):
            line += f" — [View official source]({src.source_url})"
        st.markdown(line)


def render_debug(retrieved_chunks) -> None:
    if not st.session_state.debug_mode or not retrieved_chunks:
        return
    with st.expander(f"Retrieved Context ({len(retrieved_chunks)} chunk(s), debug)"):
        for i, r in enumerate(retrieved_chunks, 1):
            st.markdown(
                f"**{i}. Law {r.chunk.law_number or '?'} | Section {r.chunk.section or '-'}** "
                f"— similarity score: `{r.score:.3f}`"
            )
            st.text(r.chunk.text)


def render_assistant_entry(entry: dict) -> None:
    st.markdown(entry["content"])
    if not entry.get("error"):
        st.caption(CONFIDENCE_LABELS.get(entry.get("confidence", "insufficient"), ""))
    render_sources(entry.get("sources", []))
    render_debug(entry.get("retrieved_chunks", []))


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_entry(message)
        else:
            st.markdown(message["content"])


def build_conversation_history(messages: list[dict], max_turns: int = 6) -> list[dict]:
    """Last few turns only, as plain role/content pairs -- resolves references, never a source of facts."""
    recent = messages[-max_turns:]
    return [{"role": m["role"], "content": m["content"]} for m in recent]


can_ask = vector_store is not None and vector_store.size > 0
question = st.chat_input(
    "Ask a question about the Laws of Cricket..." if can_ask else "Build the index first (see sidebar)",
    disabled=not can_ask,
)

if question:
    question = question.strip()
    if len(question) > settings.max_question_length:
        st.warning(f"Question is too long (max {settings.max_question_length} characters). Please shorten it.")
    else:
        history_for_llm = build_conversation_history(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant rules and generating an answer..."):
                embedder = get_embedding_service()
                retriever = Retriever(embedder, vector_store, top_k=int(top_k), score_threshold=float(similarity_threshold))
                pipeline = RAGPipeline(retriever, llm_provider)

                entry = None
                try:
                    response = pipeline.answer(question, conversation_history=history_for_llm)
                    entry = {
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources,
                        "confidence": response.confidence,
                        "retrieved_chunks": response.retrieved_chunks,
                        "grounded": response.grounded,
                    }
                except RuntimeError:
                    logger.exception("RAGPipeline could not produce an answer")
                    message = (
                        "No LLM provider is configured, so I can't generate an answer -- "
                        "set `LLM_PROVIDER=groq` and `LLM_API_KEY` (or `GROQ_API_KEY`) in `.env` and restart the app."
                    )
                    st.error(message)
                    entry = {"role": "assistant", "content": message, "sources": [], "error": True,
                              "retrieved_chunks": [], "grounded": False}
                except (LLMConfigurationError, LLMGenerationError):
                    logger.exception("LLM generation failed")
                    message = "The assistant couldn't generate an answer right now. Please try again shortly."
                    st.error(message)
                    entry = {"role": "assistant", "content": message, "sources": [], "error": True,
                              "retrieved_chunks": [], "grounded": False}
                except Exception:  # noqa: BLE001 - last-resort guard so the UI never shows a raw traceback
                    logger.exception("Unexpected error while answering question")
                    message = "Something went wrong while answering your question. Please try again."
                    st.error(message)
                    entry = {"role": "assistant", "content": message, "sources": [], "error": True,
                              "retrieved_chunks": [], "grounded": False}

                render_assistant_entry(entry)
                st.session_state.messages.append(entry)
