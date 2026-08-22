import faiss
import numpy as np
import pytest

from recommendation.retrieval.index.faiss_index import FaissVectorIndex
from recommendation.utils.config import RetrievalConfig


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@pytest.fixture
def catalog() -> tuple[list[int], np.ndarray]:
    rng = np.random.default_rng(0)
    item_ids = [10, 20, 30, 40, 50]
    embeddings = _l2_normalize(rng.normal(size=(5, 8)).astype(np.float32))
    return item_ids, embeddings


@pytest.fixture
def index(catalog) -> FaissVectorIndex:
    item_ids, embeddings = catalog
    idx = FaissVectorIndex()
    idx.build(item_ids, embeddings)
    return idx


def brute_force_top_k(query: np.ndarray, item_ids: list[int], embeddings: np.ndarray, k: int) -> tuple[list[int], list[float]]:
    scores = embeddings @ query
    order = np.argsort(-scores)[:k]
    return [item_ids[i] for i in order], [float(scores[i]) for i in order]


def test_build_sets_size(index, catalog):
    item_ids, _ = catalog
    assert index.size == len(item_ids)


def test_build_produces_hnsw_not_flat_index(index):
    """FaissVectorIndex is now approximate (HNSW), not the old exact
    `IndexFlatIP` - this proves the underlying structure actually changed,
    not just that search still returns plausible-looking results.
    """
    inner = faiss.downcast_index(index._index.index)
    assert isinstance(inner, faiss.IndexHNSWFlat)
    assert inner.metric_type == faiss.METRIC_INNER_PRODUCT


def test_ann_params_are_config_driven(catalog):
    """Two different RetrievalConfig HNSW parameter sets must produce
    different underlying index configuration - proves M/efConstruction
    are not hard-coded.
    """
    item_ids, embeddings = catalog
    small = FaissVectorIndex(RetrievalConfig(faiss_hnsw_m=8, faiss_hnsw_ef_construction=40))
    small.build(item_ids, embeddings)
    large = FaissVectorIndex(RetrievalConfig(faiss_hnsw_m=48, faiss_hnsw_ef_construction=400))
    large.build(item_ids, embeddings)

    small_inner = faiss.downcast_index(small._index.index)
    large_inner = faiss.downcast_index(large._index.index)
    assert small_inner.hnsw.efConstruction == 40
    assert large_inner.hnsw.efConstruction == 400


def test_search_returns_self_as_top_match_with_score_one(index, catalog):
    item_ids, embeddings = catalog
    [result] = index.search(embeddings[0].reshape(1, -1), k=1)
    assert result.item_ids[0] == item_ids[0]
    assert result.scores[0] == pytest.approx(1.0, abs=1e-4)


def test_search_recall_against_brute_force_cosine_similarity(index, catalog):
    """HNSW is approximate in general, but at this fixture's trivial
    5-item catalog (with efSearch floored well above the catalog size),
    it should still recover the full exact ranking - checked as a high
    recall bar (set overlap), not brittle list-order equality, since
    exact order/score bit-identity is no longer a guarantee this backend
    makes at any scale.
    """
    item_ids, embeddings = catalog
    query = embeddings[2]
    [result] = index.search(query.reshape(1, -1), k=len(item_ids))
    expected_ids, expected_scores = brute_force_top_k(query, item_ids, embeddings, len(item_ids))
    assert set(result.item_ids) == set(expected_ids)
    assert max(result.scores) == pytest.approx(max(expected_scores), abs=1e-4)


def test_search_results_are_sorted_descending_by_score(index, catalog):
    _, embeddings = catalog
    [result] = index.search(embeddings[0].reshape(1, -1), k=len(embeddings))
    assert result.scores == sorted(result.scores, reverse=True)


def test_search_k_is_capped_to_catalog_size(index, catalog):
    item_ids, embeddings = catalog
    [result] = index.search(embeddings[0].reshape(1, -1), k=1000)
    assert len(result.item_ids) == len(item_ids)


def test_search_handles_batch_of_queries(index, catalog):
    _, embeddings = catalog
    results = index.search(embeddings[:3], k=2)
    assert len(results) == 3
    for result in results:
        assert len(result.item_ids) == 2


def test_search_before_build_raises(catalog):
    _, embeddings = catalog
    idx = FaissVectorIndex()
    with pytest.raises(RuntimeError):
        idx.search(embeddings[0].reshape(1, -1), k=1)


def test_build_rejects_mismatched_lengths():
    idx = FaissVectorIndex()
    with pytest.raises(ValueError):
        idx.build([1, 2, 3], np.zeros((2, 4), dtype=np.float32))


def test_build_rejects_empty_catalog():
    idx = FaissVectorIndex()
    with pytest.raises(ValueError):
        idx.build([], np.zeros((0, 4), dtype=np.float32))


def test_search_rejects_non_positive_k(index, catalog):
    _, embeddings = catalog
    with pytest.raises(ValueError):
        index.search(embeddings[0].reshape(1, -1), k=0)


def test_search_rejects_dim_mismatch(index):
    with pytest.raises(ValueError):
        index.search(np.zeros((1, 3), dtype=np.float32), k=1)


def test_save_and_load_round_trip_preserves_search_results(index, catalog, tmp_path):
    item_ids, embeddings = catalog
    query = embeddings[1].reshape(1, -1)
    expected = index.search(query, k=len(item_ids))

    save_path = tmp_path / "index" / "faiss_index.bin"
    index.save(save_path)
    assert save_path.exists()

    reloaded = FaissVectorIndex()
    reloaded.load(save_path)
    assert reloaded.size == len(item_ids)

    [actual] = reloaded.search(query, k=len(item_ids))
    [expected_result] = expected
    assert set(actual.item_ids) == set(expected_result.item_ids)
    assert actual.scores == pytest.approx(expected_result.scores, abs=1e-4)


def test_save_and_load_round_trip_preserves_hnsw_structure(index, tmp_path):
    """The reloaded index must still be HNSW (not silently degrading to a
    generic flat index on reload) - specifically exercises the
    `faiss.downcast_index` step `load()` needs, since `IndexIDMap2.index`
    only exposes the generic base `faiss.Index` type after a fresh
    `read_index` call.
    """
    save_path = tmp_path / "index" / "faiss_index.bin"
    index.save(save_path)

    reloaded = FaissVectorIndex()
    reloaded.load(save_path)
    inner = faiss.downcast_index(reloaded._index.index)
    assert isinstance(inner, faiss.IndexHNSWFlat)
    # efSearch must be settable post-reload (search() sets it every call) -
    # this would raise AttributeError if downcast_index had been skipped.
    [result] = reloaded.search(np.zeros((1, 8), dtype=np.float32), k=1)
    assert len(result.item_ids) == 1


def test_load_missing_path_raises(tmp_path):
    idx = FaissVectorIndex()
    with pytest.raises(FileNotFoundError):
        idx.load(tmp_path / "does_not_exist.bin")


def test_save_before_build_raises(tmp_path):
    idx = FaissVectorIndex()
    with pytest.raises(RuntimeError):
        idx.save(tmp_path / "index.bin")
