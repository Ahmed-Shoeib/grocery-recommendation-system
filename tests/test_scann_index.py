"""ScaNN `VectorIndex` correctness tests.

`scann` has no Windows wheel, so this entire file is skipped (not failed)
on native Windows via `importorskip` below. It runs for real inside the
Linux/Docker image built from the repo-root `Dockerfile`, which is the
only place this project's primary/production retrieval backend is
actually exercised - see docs/data-mapping.md section 10.

Fixture catalog bumped from 5 to 300 synthetic items (2026-08-22, ANN
migration): ScaNN's tree partitioning + asymmetric-hashing quantization
need enough points for `.tree()`/`.score_ah()` to do something
non-degenerate - at 5 items, `num_leaves` collapses to 1 and the whole
catalog is effectively one partition, which wouldn't actually exercise
the approximate path this file is meant to verify.
"""

import numpy as np
import pytest

pytest.importorskip("scann")

from recommendation.retrieval.index.scann_index import ScannVectorIndex  # noqa: E402
from recommendation.utils.config import RetrievalConfig  # noqa: E402


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@pytest.fixture
def catalog() -> tuple[list[int], np.ndarray]:
    rng = np.random.default_rng(0)
    item_ids = list(range(10, 3010, 10))
    embeddings = _l2_normalize(rng.normal(size=(300, 8)).astype(np.float32))
    return item_ids, embeddings


@pytest.fixture
def index(catalog) -> ScannVectorIndex:
    item_ids, embeddings = catalog
    idx = ScannVectorIndex()
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
    assert result.scores[0] == pytest.approx(1.0, abs=1e-4)


def test_search_recall_against_brute_force_cosine_similarity(index, catalog):
    """ScaNN is now approximate (tree + AH + reorder), so its top-k is no
    longer guaranteed to exactly reproduce brute-force order/scores -
    checked as a high recall bar (set overlap over a generous top-30 out
    of 300) rather than brittle exact equality.
    """
    item_ids, embeddings = catalog
    query = embeddings[2]
    k = 30
    [result] = index.search(query.reshape(1, -1), k=k)
    expected_ids, _ = brute_force_top_k(query, item_ids, embeddings, k)
    overlap = len(set(result.item_ids) & set(expected_ids))
    assert overlap / k >= 0.7  # generous bar for a 300-item fixture catalog, not a tuned production threshold


def test_build_uses_tree_and_ah_not_brute_force(catalog, monkeypatch):
    """Proves the builder chain actually changed from
    `.score_brute_force(...)` to `.tree(...).score_ah(...).reorder(...)`
    - intercepts ScaNN's own builder (not our code) so this is a direct
    check of what `ScannVectorIndex.build` calls, independent of the
    installed scann version's actual search-quality behavior.
    """
    import scann

    item_ids, embeddings = catalog
    calls: list[tuple[str, dict]] = []

    class _FakeBuilder:
        def tree(self, **kwargs):
            calls.append(("tree", kwargs))
            return self

        def score_ah(self, **kwargs):
            calls.append(("score_ah", kwargs))
            return self

        def score_brute_force(self, *args, **kwargs):
            calls.append(("score_brute_force", kwargs))
            return self

        def reorder(self, *args, **kwargs):
            calls.append(("reorder", {"args": args}))
            return self

        def build(self):
            calls.append(("build", {}))
            return None

    monkeypatch.setattr(scann.scann_ops_pybind, "builder", lambda *a, **kw: _FakeBuilder())

    idx = ScannVectorIndex()
    idx.build(item_ids, embeddings)

    method_names = [name for name, _ in calls]
    assert method_names == ["tree", "score_ah", "reorder", "build"]
    assert "score_brute_force" not in method_names


def test_tiny_catalog_falls_back_to_brute_force_instead_of_crashing():
    """ScaNN's asymmetric hashing (`hash_type="lut16"`, the default) has
    a hard, non-configurable floor: `num_clusters_per_block` is fixed at
    16, and AH training raises `RuntimeError` if the catalog has fewer
    than 16 points - confirmed against the real installed scann package.
    Below that floor, `build()` must fall back to
    `.score_brute_force(...)` (exact search) rather than crash - this is
    a real production edge case (a brand-new/tiny catalog), not just a
    test-fixture concern.
    """
    rng = np.random.default_rng(0)
    item_ids = [10, 20, 30, 40, 50]
    embeddings = _l2_normalize(rng.normal(size=(5, 8)).astype(np.float32))

    idx = ScannVectorIndex()
    idx.build(item_ids, embeddings)  # must not raise

    [result] = idx.search(embeddings[0].reshape(1, -1), k=1)
    assert result.item_ids[0] == item_ids[0]
    assert result.scores[0] == pytest.approx(1.0, abs=1e-4)


def test_ann_params_are_config_driven(catalog, monkeypatch):
    """Two different RetrievalConfig ScaNN parameter sets must produce
    different builder kwargs - proves leaf count / AH block size are not
    hard-coded.
    """
    import scann

    item_ids, embeddings = catalog
    captured: dict = {}

    class _FakeBuilder:
        def tree(self, **kwargs):
            captured["tree"] = kwargs
            return self

        def score_ah(self, **kwargs):
            captured["score_ah"] = kwargs
            return self

        def reorder(self, *args, **kwargs):
            captured["reorder"] = args
            return self

        def build(self):
            return None

    monkeypatch.setattr(scann.scann_ops_pybind, "builder", lambda *a, **kw: _FakeBuilder())

    small_cfg = RetrievalConfig(scann_leaves_multiplier=1.0, scann_min_points_per_leaf=5, scann_ah_dims_per_block=1, scann_reorder_k=50)
    idx_small = ScannVectorIndex(small_cfg)
    idx_small.build(item_ids, embeddings)
    small_tree, small_ah, small_reorder = dict(captured["tree"]), dict(captured["score_ah"]), captured["reorder"]

    captured.clear()
    large_cfg = RetrievalConfig(scann_leaves_multiplier=5.0, scann_min_points_per_leaf=5, scann_ah_dims_per_block=4, scann_reorder_k=250)
    idx_large = ScannVectorIndex(large_cfg)
    idx_large.build(item_ids, embeddings)
    large_tree, large_ah, large_reorder = dict(captured["tree"]), dict(captured["score_ah"]), captured["reorder"]

    assert small_tree["num_leaves"] != large_tree["num_leaves"]
    assert small_ah["dimensions_per_block"] != large_ah["dimensions_per_block"]
    assert small_reorder != large_reorder


def test_search_results_are_sorted_descending_by_score(index, catalog):
    _, embeddings = catalog
    [result] = index.search(embeddings[0].reshape(1, -1), k=len(embeddings))
    assert result.scores == sorted(result.scores, reverse=True)


def test_search_k_is_capped_to_catalog_size(index, catalog):
    """Never returns MORE than the catalog holds, and never a phantom/
    NaN-scored entry - but unlike FAISS (whose `efSearch` floor makes a
    k=catalog_size request reliably reach the whole graph), ScaNN's tree
    partitioning only visits `num_leaves_to_search` of `num_leaves`
    partitions (fixed at build time), so it can legitimately return FEWER
    than the full catalog even when asked for everything - confirmed via
    a real Docker/CI run, where a naive "must equal catalog size"
    assertion failed on real embeddings despite `search()` behaving
    correctly (see its own docstring on why NaN-scored slots are
    filtered rather than kept). `EligibilityRestrictedIndex`'s widening
    loop is designed around exactly this - see its module docstring.
    """
    item_ids, embeddings = catalog
    [result] = index.search(embeddings[0].reshape(1, -1), k=1000)
    assert 0 < len(result.item_ids) <= len(item_ids)
    assert not any(np.isnan(s) for s in result.scores)


def test_search_never_returns_nan_scores_or_phantom_ids(index, catalog):
    """Regression test for a real bug found in Docker/CI: requesting more
    neighbors than the tree-partitioned search actually visited caused
    ScaNN to pad the tail with NaN scores (and along with them,
    non-matching ids) rather than truncating - which silently NaN-
    poisoned ranker training end-to-end via the retrieval_score feature.
    `item_ids` and `scores` must always be the same (possibly truncated)
    length, with no NaN anywhere.
    """
    item_ids, embeddings = catalog
    results = index.search(embeddings, k=len(item_ids))
    for result in results:
        assert len(result.item_ids) == len(result.scores)
        assert not any(np.isnan(s) for s in result.scores)


def test_search_handles_batch_of_queries(index, catalog):
    _, embeddings = catalog
    results = index.search(embeddings[:3], k=2)
    assert len(results) == 3
    for result in results:
        assert len(result.item_ids) == 2


def test_search_before_build_raises(catalog):
    _, embeddings = catalog
    idx = ScannVectorIndex()
    with pytest.raises(RuntimeError):
        idx.search(embeddings[0].reshape(1, -1), k=1)


def test_build_rejects_mismatched_lengths():
    idx = ScannVectorIndex()
    with pytest.raises(ValueError):
        idx.build([1, 2, 3], np.zeros((2, 4), dtype=np.float32))


def test_build_rejects_empty_catalog():
    idx = ScannVectorIndex()
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
    k = 20
    expected = index.search(query, k=k)

    save_path = tmp_path / "scann_index"
    index.save(save_path)
    assert save_path.exists()

    reloaded = ScannVectorIndex()
    reloaded.load(save_path)
    assert reloaded.size == len(item_ids)

    [actual] = reloaded.search(query, k=k)
    [expected_result] = expected
    # Approximate search over a reloaded (re-deserialized) searcher isn't
    # guaranteed bit-identical to the pre-save searcher - reloading must
    # still recover a near-identical top-k, not an exact one.
    overlap = len(set(actual.item_ids) & set(expected_result.item_ids))
    assert overlap / k >= 0.9


def test_load_missing_path_raises(tmp_path):
    idx = ScannVectorIndex()
    with pytest.raises(FileNotFoundError):
        idx.load(tmp_path / "does_not_exist")


def test_save_before_build_raises(tmp_path):
    idx = ScannVectorIndex()
    with pytest.raises(RuntimeError):
        idx.save(tmp_path / "scann_index")


def test_faiss_and_scann_agree_on_top_k_recall(catalog):
    """Cross-backend consistency check: both are now independently
    approximate over the same normalized-inner-product space (FAISS
    HNSW, ScaNN tree+AH+reorder), so for identical input they're expected
    to substantially agree on the top-k, not reproduce each other's exact
    ranking/scores.
    """
    from recommendation.retrieval.index.faiss_index import FaissVectorIndex

    item_ids, embeddings = catalog
    faiss_index = FaissVectorIndex()
    faiss_index.build(item_ids, embeddings)

    scann_index = ScannVectorIndex()
    scann_index.build(item_ids, embeddings)

    query = embeddings[3].reshape(1, -1)
    k = 20
    [faiss_result] = faiss_index.search(query, k=k)
    [scann_result] = scann_index.search(query, k=k)

    overlap = len(set(faiss_result.item_ids) & set(scann_result.item_ids))
    assert overlap / k >= 0.6  # generous bar - two different ANN algorithms, not required to match exactly
    assert faiss_result.item_ids[0] == scann_result.item_ids[0]  # both should still recover the true nearest neighbor
