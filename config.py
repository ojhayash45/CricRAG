"""
Central configuration for the Cricket Laws RAG system.

All tunables live here and are overridable via environment variables / a
.env file (see .env.example). Nothing here should be hard-coded elsewhere.

Values such as CHUNK_SIZE, TOP_K and SIMILARITY_THRESHOLD are *starting
points*, not universal truths -- they should be revisited once the
evaluation harness (Phase 11) produces real numbers.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence-Transformers model name used for embeddings.",
    )
    embedding_batch_size: int = 32

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    llm_provider: str = Field(default="none", description="Provider key, e.g. 'groq' or 'none'.")
    llm_model: str = Field(default="", description="Provider-specific model name.")
    llm_api_key: str | None = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices("LLM_API_KEY", "GROQ_API_KEY", "GROK_API_KEY"),
        description="API key for the configured provider. Accepts LLM_API_KEY, GROQ_API_KEY, or GROK_API_KEY.",
    )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    top_k: int = 5
    similarity_threshold: float = Field(
        default=0.35,
        description=(
            "Minimum similarity score for a retrieved chunk to be considered "
            "relevant. This is NOT empirically calibrated yet -- it must be "
            "tuned against evaluation/questions.json before being trusted."
        ),
    )

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    chunk_size: int = Field(default=650, description="Approx. tokens (word-based proxy) per chunk.")
    chunk_overlap: int = Field(default=80, description="Approx. tokens (word-based proxy) of overlap.")

    # ------------------------------------------------------------------
    # Answer confidence bands (retrieval relevance, NOT factual certainty --
    # see README/limitations. Uncalibrated starting points.)
    # ------------------------------------------------------------------
    confidence_medium_threshold: float = Field(
        default=0.45,
        description="Filtered top score at/above this -> 'medium' relevance (must be > similarity_threshold).",
    )
    confidence_high_threshold: float = Field(
        default=0.6,
        description="Filtered top score at/above this -> 'high' relevance (must be > confidence_medium_threshold).",
    )

    # ------------------------------------------------------------------
    # Context window control (generation)
    # ------------------------------------------------------------------
    max_context_chunks: int = Field(
        default=8,
        description="Hard cap on how many retrieved chunks are ever sent to the LLM.",
    )
    max_context_chars: int = Field(
        default=6000,
        description="Hard cap on total formatted-context length sent to the LLM.",
    )

    # ------------------------------------------------------------------
    # UI / input validation
    # ------------------------------------------------------------------
    max_question_length: int = Field(
        default=500,
        description="Reject questions longer than this (characters) instead of sending them to the LLM.",
    )

    # ------------------------------------------------------------------
    # Vector store backend
    # ------------------------------------------------------------------
    database_url: str | None = Field(
        default=None,
        repr=False,
        description=(
            "Postgres connection string (e.g. Neon, with pgvector enabled). "
            "When set, the app stores/searches chunks in Postgres instead of "
            "local FAISS index files -- the only backend that survives a "
            "restart on a host with no persistent disk (e.g. Render's free "
            "tier). When unset, falls back to local FAISS files."
        ),
    )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    chunks_data_dir: Path = BASE_DIR / "data" / "chunks"
    vector_store_dir: Path = BASE_DIR / "vector_store"


settings = Settings()
