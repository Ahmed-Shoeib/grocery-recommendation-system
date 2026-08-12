import numpy as np
import pytest

from recommendation.retrieval.index.faiss_index import FaissVectorIndex


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


def test_search_returns_self_as_top_match_with_score_one(index, catalog):
    item_ids, embeddings = catalog
    [result] = index.search(embeddings[0].reshape(1, -1), k=1)
    assert result.item_ids[0] == item_ids[0]
    assert result.scores[0] == pytest.approx(1.0, abs=1e-5)


def test_search_matches_brute_force_cosine_similarity(index, catalog):
    item_ids, embeddings = catalog
    query = embeddings[2]
    [result] = index.search(query.reshape(1, -1), k=len(item_ids))
    expected_ids, expected_scores = brute_force_top_k(query, item_ids, embeddings, len(item_ids))
    assert result.item_ids == expected_ids
    assert result.scores == pytest.approx(expected_scores, abs=1e-5)


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
    assert actual.item_ids == expected_result.item_ids
    assert actual.scores == pytest.approx(expected_result.scores, abs=1e-5)


def test_load_missing_path_raises(tmp_path):
    idx = FaissVectorIndex()
    with pytest.raises(FileNotFoundError):
        idx.load(tmp_path / "does_not_exist.bin")


def test_save_before_build_raises(tmp_path):
    idx = FaissVectorIndex()
    with pytest.raises(RuntimeError):
        idx.save(tmp_path / "index.bin")
