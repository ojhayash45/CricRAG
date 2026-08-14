from __future__ import annotations

import numpy as np
import pytest

from src.embeddings.embedding_service import EmbeddingService

# Loading a real Sentence-Transformers model is slow (model download/init),
# so we share one instance across this module's tests.


@pytest.fixture(scope="module")
def service() -> EmbeddingService:
    return EmbeddingService()


class TestEmbeddingService:
    def test_exposes_positive_dimension(self, service):
        assert isinstance(service.dimension, int)
        assert service.dimension > 0

    def test_embed_query_returns_vector_of_expected_dimension(self, service):
        vector = service.embed_query("What is a no-ball?")
        assert vector.shape == (service.dimension,)
        assert vector.dtype == np.float32

    def test_embed_documents_returns_matrix_of_expected_shape(self, service):
        texts = ["What is a no-ball?", "When is a batter out LBW?", "What is a wide?"]
        matrix = service.embed_documents(texts)
        assert matrix.shape == (len(texts), service.dimension)
        assert matrix.dtype == np.float32

    def test_normalized_vectors_have_unit_norm(self, service):
        vector = service.embed_query("Can a batter be timed out?")
        assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-3)

    def test_similar_texts_are_closer_than_unrelated_ones(self, service):
        lbw_a = service.embed_query("When is a batter out leg before wicket?")
        lbw_b = service.embed_query("What are the conditions for an LBW dismissal?")
        unrelated = service.embed_query("What is the capital of France?")

        sim_related = float(np.dot(lbw_a, lbw_b))
        sim_unrelated = float(np.dot(lbw_a, unrelated))
        assert sim_related > sim_unrelated

    def test_embed_query_rejects_empty_text(self, service):
        with pytest.raises(ValueError):
            service.embed_query("")
        with pytest.raises(ValueError):
            service.embed_query("   ")

    def test_embed_documents_rejects_empty_list(self, service):
        with pytest.raises(ValueError):
            service.embed_documents([])

    def test_embed_documents_rejects_blank_entry(self, service):
        with pytest.raises(ValueError):
            service.embed_documents(["valid text", "   "])
