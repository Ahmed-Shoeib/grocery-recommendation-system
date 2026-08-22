"""STEP 7 Docker/Linux ScaNN verification against the ALREADY-TRAINED
`models/sqlite_baseline/` artifacts (produced by
`scripts/train_sqlite_pipeline.py`, docs/data-mapping.md section 16/17) -
this file loads and exercises them, it never trains or retrains anything.

Entirely skipped (not failed) unless BOTH:
  - `scann` is importable (Linux/Docker only - no Windows wheel, see
    `retrieval.index.scann_index` module docstring), and
  - `models/sqlite_baseline/` actually exists on disk (bind-mounted into
    the Docker image via `docker-compose.yml`'s `./models:/app/models`
    pattern, or an equivalent `docker run -v`).

So this is inert on native Windows dev and on any CI job that hasn't run
`scripts/train_sqlite_pipeline.py` first - exactly the same
"importorskip + artifact-presence skip" pattern `test_scann_index.py`
already uses for the backend import half of this.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

pytest.importorskip("scann")

from recommendation.utils.config import get_config, resolve_path  # noqa: E402

ARTIFACT_ROOT = resolve_path(get_config().paths.models_dir) / "sqlite_baseline"

pytestmark = pytest.mark.skipif(
    not (ARTIFACT_ROOT / "two_tower" / "item_embeddings.npz").exists(),
    reason="models/sqlite_baseline artifacts not present - run scripts/train_sqlite_pipeline.py first",
)


@pytest.fixture(scope="module")
def two_tower_artifacts():
    from recommendation.retrieval.two_tower.serialization import load_two_tower_artifacts

    return load_two_tower_artifacts(ARTIFACT_ROOT / "two_tower")


@pytest.fixture(scope="module")
def ranker_artifacts():
    from recommendation.ranking.serialization import load_ranker_artifacts

    return load_ranker_artifacts(ARTIFACT_ROOT / "ranker")


# --- artifact sanity --------------------------------------------------------

def test_item_embeddings_shape_matches_step7_catalog(two_tower_artifacts):
    assert two_tower_artifacts.item_embeddings.shape == (1200, 128)
    assert len(two_tower_artifacts.item_ids) == 1200


def test_no_duplicate_item_ids(two_tower_artifacts):
    assert len(set(two_tower_artifacts.item_ids)) == len(two_tower_artifacts.item_ids)


def test_ranker_feature_dimension_is_29(ranker_artifacts):
    assert len(ranker_artifacts.feature_names) == 29


# --- ScaNN import + index construction --------------------------------------

def test_scann_actually_imports():
    import scann  # noqa: F401 - the import itself is the assertion


def test_scann_index_holds_full_catalog_before_eligibility_restriction(two_tower_artifacts):
    """The raw index always holds the FULL 1,200-item catalog - hard
    pre-retrieval eligibility is applied by
    `retrieval.index.eligibility_filter.EligibilityRestrictedIndex`, a
    query-time wrapper around an already-built index (see that module's
    docstring): eligibility never touches the index's own contents, only
    what a given `search()` call is allowed to return. This test proves
    that architecture claim directly against the STEP 7 artifacts rather
    than just citing the docstring.
    """
    from recommendation.retrieval.index.scann_index import ScannVectorIndex

    idx = ScannVectorIndex()
    idx.build(two_tower_artifacts.item_ids, two_tower_artifacts.item_embeddings)
    assert idx.size == 1200


def test_scann_product_id_mapping_is_correct(two_tower_artifacts):
    """Querying with an item's OWN embedding must return that exact item
    as the top-1 match with score ~1.0 - proves the id<->row mapping
    ScaNN was built with is not shuffled, over a real sample of the
    STEP 7 catalog (not just a handful of synthetic ids).
    """
    from recommendation.retrieval.index.scann_index import ScannVectorIndex

    idx = ScannVectorIndex()
    idx.build(two_tower_artifacts.item_ids, two_tower_artifacts.item_embeddings)

    rng = np.random.default_rng(0)
    sample_positions = rng.choice(len(two_tower_artifacts.item_ids), size=25, replace=False)
    for pos in sample_positions:
        query = two_tower_artifacts.item_embeddings[pos]
        [result] = idx.search(query.reshape(1, -1), k=1)
        assert result.item_ids[0] == two_tower_artifacts.item_ids[pos]
        assert result.scores[0] == pytest.approx(1.0, abs=1e-3)


# --- ScaNN vs FAISS ANN recall agreement on the STEP 7 embeddings ----------

def test_scann_and_faiss_agree_on_step7_embeddings(two_tower_artifacts):
    """Both backends now do genuinely APPROXIMATE search over these
    L2-normalized embeddings (`ScannVectorIndex`: tree partitioning +
    asymmetric hashing + reorder; `FaissVectorIndex`: HNSW) - for
    identical input they're expected to substantially agree on the
    neighbor set (high recall overlap, both should recover the true
    nearest neighbor), not reproduce each other's exact ranking/scores,
    over a representative random sample of the real 1,200-item STEP 7
    catalog (not the small synthetic fixture `test_scann_index.py`
    already covers).
    """
    from recommendation.retrieval.index.faiss_index import FaissVectorIndex
    from recommendation.retrieval.index.scann_index import ScannVectorIndex

    item_ids = two_tower_artifacts.item_ids
    embeddings = two_tower_artifacts.item_embeddings

    faiss_index = FaissVectorIndex()
    faiss_index.build(item_ids, embeddings)
    scann_index = ScannVectorIndex()
    scann_index.build(item_ids, embeddings)

    rng = np.random.default_rng(1)
    sample_positions = rng.choice(len(item_ids), size=25, replace=False)
    queries = embeddings[sample_positions]

    k = 20
    faiss_results = faiss_index.search(queries, k=k)
    scann_results = scann_index.search(queries, k=k)
    overlaps = []
    for faiss_result, scann_result in zip(faiss_results, scann_results):
        overlaps.append(len(set(faiss_result.item_ids) & set(scann_result.item_ids)) / k)
        assert faiss_result.item_ids[0] == scann_result.item_ids[0]  # both recover the true nearest neighbor (self-match)
    assert (sum(overlaps) / len(overlaps)) >= 0.6  # generous bar - two different ANN algorithms, not required to match exactly


# --- eligibility still excludes inactive/out-of-stock ----------------------

def test_pre_retrieval_eligibility_excludes_inactive_and_out_of_stock(two_tower_artifacts):
    from recommendation.data.adapters.sqlite_factory import build_sqlite_adapters
    from recommendation.features.product_features import build_product_features
    from recommendation.retrieval.index.eligibility_filter import EligibilityRestrictedIndex
    from recommendation.retrieval.index.scann_index import ScannVectorIndex
    from recommendation.serving.eligibility import apply_eligibility, build_eligibility_rules

    config = get_config()
    bundle = build_sqlite_adapters()
    products = bundle.products.list_products()
    product_features = build_product_features(products, [], [], [])
    rules = build_eligibility_rules(config.eligibility)
    result = apply_eligibility([p.id for p in products], product_features, rules)
    eligible_ids = set(result.eligible_ids)
    ineligible_ids = {p.id for p in products} - eligible_ids
    assert ineligible_ids, "expected at least one inactive/out-of-stock product in this dataset"

    idx = ScannVectorIndex()
    idx.build(two_tower_artifacts.item_ids, two_tower_artifacts.item_embeddings)
    restricted = EligibilityRestrictedIndex(idx)

    query = two_tower_artifacts.item_embeddings[0]
    [restricted_result] = restricted.search(query.reshape(1, -1), k=len(eligible_ids), allowed_ids=eligible_ids)
    assert not (set(restricted_result.item_ids) & ineligible_ids)


# --- full SQLite-trained pipeline produces recommendations with ScaNN -----

def test_full_pipeline_produces_valid_recommendations_with_scann(two_tower_artifacts, ranker_artifacts):
    from recommendation.data.adapters.engagement import build_engagement_profile
    from recommendation.data.adapters.sqlite_factory import build_sqlite_adapters
    from recommendation.embeddings.encoder import SentenceTransformerEncoder
    from recommendation.embeddings.product_embeddings import get_or_compute_product_embeddings
    from recommendation.features.price import build_price_catalog_context
    from recommendation.features.product_features import build_product_features
    from recommendation.features.user_features import build_user_features
    from recommendation.retrieval.index.scann_index import ScannVectorIndex
    from recommendation.serving.pipeline import generate_recommendations

    config = get_config()
    bundle = build_sqlite_adapters()
    products = bundle.products.list_products()
    product_lookup = {p.id: p for p in products}
    product_features = build_product_features(products, [], [], [])
    price_context = build_price_catalog_context(products)

    st_encoder = SentenceTransformerEncoder(
        config.embedding.sentence_transformer_model, device=config.embedding.device, batch_size=config.embedding.encode_batch_size
    )
    cache, _ = get_or_compute_product_embeddings(products, st_encoder, resolve_path("data/processed/product_embeddings_sqlite.npz"))
    product_embeddings = cache.as_dict()

    idx = ScannVectorIndex()
    idx.build(two_tower_artifacts.item_ids, two_tower_artifacts.item_embeddings)

    reference_time = datetime.now()
    tested_any_nonempty = False
    for user_id in bundle.users.list_user_ids()[:8]:
        profile = build_engagement_profile(
            user_id, bundle.users, bundle.purchases, bundle.cart, bundle.clicks, bundle.search, bundle.chatbot, bundle.reviews
        )
        user_features = build_user_features(
            profile, product_lookup, product_embeddings, config.features,
            reference_time=reference_time, price_context=price_context,
        )
        result = generate_recommendations(
            user_features, product_features, product_embeddings, two_tower_artifacts.item_ids,
            two_tower_artifacts.encoder, two_tower_artifacts.user_tower, ranker_artifacts.model,
            idx, config, top_n=10,
        )
        assert len(result.product_ids) <= 10
        assert len(result.product_ids) == len(set(result.product_ids))  # no duplicates
        tested_any_nonempty = tested_any_nonempty or len(result.product_ids) > 0

    assert tested_any_nonempty
