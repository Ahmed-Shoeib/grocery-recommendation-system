import pytest

from recommendation.retrieval.index.factory import build_vector_index, candidate_pool_size
from recommendation.retrieval.index.faiss_index import FaissVectorIndex
from recommendation.utils.config import RetrievalConfig


def test_build_vector_index_faiss_backend_returns_faiss_index():
    index = build_vector_index(RetrievalConfig(backend="faiss"))
    assert isinstance(index, FaissVectorIndex)


def test_build_vector_index_scann_backend_returns_scann_index():
    # Selecting the backend never requires the `scann` package to be
    # installed (it's imported lazily inside `ScannVectorIndex.build`) -
    # this must succeed on native Windows too, where `scann` is absent.
    from recommendation.retrieval.index.scann_index import ScannVectorIndex

    index = build_vector_index(RetrievalConfig(backend="scann"))
    assert isinstance(index, ScannVectorIndex)


def test_candidate_pool_size_uses_multiplier():
    config = RetrievalConfig(candidate_pool_multiplier=5, min_candidate_pool=1)
    assert candidate_pool_size(config, limit=10, catalog_size=1000) == 50


def test_candidate_pool_size_respects_floor():
    config = RetrievalConfig(candidate_pool_multiplier=5, min_candidate_pool=50)
    assert candidate_pool_size(config, limit=1, catalog_size=1000) == 50


def test_candidate_pool_size_capped_by_catalog_size():
    config = RetrievalConfig(candidate_pool_multiplier=5, min_candidate_pool=50)
    assert candidate_pool_size(config, limit=10, catalog_size=30) == 30


def test_candidate_pool_size_rejects_non_positive_limit():
    config = RetrievalConfig()
    with pytest.raises(ValueError):
        candidate_pool_size(config, limit=0, catalog_size=100)


def test_candidate_pool_size_rejects_non_positive_catalog_size():
    config = RetrievalConfig()
    with pytest.raises(ValueError):
        candidate_pool_size(config, limit=10, catalog_size=0)
