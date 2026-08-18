"""The recommendation service: everything the API needs to serve
requests, loaded ONCE (via `build_recommendation_service`, called from
`api.app`'s FastAPI `lifespan`) and reused across every request - never
rebuilt per-request. `RecommendationService` is a plain dataclass with no
FastAPI dependency, so it can be constructed directly in tests (real or
fake artifacts) and injected into `api.app.create_app(service=...)`
without touching the app's startup wiring - the dependency-injection
seam that keeps the pipeline testable.

Delegates every recommendation decision to `serving.pipeline.recommend`
- no recommendation logic duplicated here. The only thing this layer
adds is the API-specific "does this user exist at all" check
(`UnknownUserError`), which is deliberately NOT part of the pipeline
(the pipeline's own leave-one-out evaluation and the dashboard/other
future callers may have different notions of "known user").

STEP 9 (docs/data-mapping.md section 18): FastAPI is now the ONLY process
that constructs a `RecommendationService` - the Streamlit dashboard is a
plain HTTP client of the API (`ui.api_client.RecommendationApiClient`,
`ui.service_loader.load_api_client`) and no longer imports this module,
loads model artifacts, or touches SQLite/synthetic adapters itself. This
class stays here (not moved to `serving`) since `api.routes` is still its
only real caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf

from recommendation.api.errors import UnknownUserError
from recommendation.data.adapters.base import AdapterBundle
from recommendation.data.adapters.factory import build_synthetic_adapters
from recommendation.data.adapters.sqlite_factory import build_sqlite_adapters
from recommendation.data.schemas.engagement import EngagementProfile
from recommendation.data.schemas.product import Product
from recommendation.data.synthetic.dataset import generate_synthetic_dataset
from recommendation.data.synthetic.validation import validate_dataset
from recommendation.embeddings.encoder import SentenceTransformerEncoder
from recommendation.features.pipeline import run_feature_pipeline
from recommendation.features.price import PriceCatalogContext, build_price_catalog_context
from recommendation.features.product_features import ProductFeatures
from recommendation.features.user_features import build_user_text_embeddings
from recommendation.retrieval.index.base import VectorIndex
from recommendation.retrieval.index.factory import build_vector_index
from recommendation.retrieval.two_tower.feature_encoding import TwoTowerFeatureEncoder
from recommendation.retrieval.two_tower.serialization import load_two_tower_artifacts
from recommendation.ranking.serialization import load_ranker_artifacts
from recommendation.serving.pipeline import RecommendationResult
from recommendation.serving.pipeline import recommend as run_pipeline
from recommendation.serving.startup_validation import (
    load_or_raise,
    require_artifact_dir,
    validate_ranker_artifacts,
    validate_two_tower_artifacts,
    validate_vector_index_compatibility,
)
from recommendation.utils.config import AppConfig, resolve_path
from recommendation.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RecommendationService:
    product_lookup: dict[int, Product]
    product_features: dict[int, ProductFeatures]
    product_embeddings: dict[int, np.ndarray]
    text_embeddings: dict[str, np.ndarray]
    all_item_ids: list[int]
    tt_encoder: TwoTowerFeatureEncoder
    user_tower: tf.keras.Model
    ranker_model: tf.keras.Model
    vector_index: VectorIndex
    bundle: AdapterBundle
    config: AppConfig
    # STEP 6 (docs/data-mapping.md section 15): catalog-only price stats,
    # built ONCE at service load (same lifecycle as product_features) and
    # reused for every request - never a leakage surface.
    price_context: PriceCatalogContext = field(default_factory=lambda: PriceCatalogContext(0.0, 0.0, (0.0, 0.0)))
    ranker_model_version: str = "unknown"
    two_tower_model_version: str = "unknown"
    # Populated from the Phase 3 feature pipeline's own output - not
    # needed for serving a single recommendation request (that only
    # needs the ONE requested user's profile, looked up on demand via
    # `bundle`), but the dashboard's user-selection list and its offline
    # metrics section (`ui.metrics`) need every user's engagement data at
    # once, and recomputing it per-user would be wasteful when it's
    # already sitting in the feature pipeline result.
    engagement_profiles: dict[int, EngagementProfile] = field(default_factory=dict)

    def is_known_user(self, user_id: int) -> bool:
        return self.bundle.users.get_user_profile(user_id) is not None

    def recommend(self, user_id: int, limit: int) -> RecommendationResult:
        if not self.is_known_user(user_id):
            raise UnknownUserError(user_id)
        return run_pipeline(
            user_id,
            self.bundle,
            self.product_lookup,
            self.product_features,
            self.product_embeddings,
            self.text_embeddings,
            self.all_item_ids,
            self.tt_encoder,
            self.user_tower,
            self.ranker_model,
            self.vector_index,
            self.config,
            limit,
            price_context=self.price_context,
        )

    def readiness_checks(self) -> dict[str, bool]:
        return {
            "catalog_loaded": len(self.product_lookup) > 0,
            "two_tower_loaded": self.user_tower is not None,
            "ranker_loaded": self.ranker_model is not None,
            "vector_index_loaded": self.vector_index.size > 0,
        }


def build_recommendation_service(config: AppConfig) -> RecommendationService:
    """Real production wiring for LIVE serving - loads the ALREADY-TRAINED
    artifacts (never retrains) and builds the Phase 5 VectorIndex.

    STEP 9 (docs/data-mapping.md section 18): `config.paths.data_source`
    selects BOTH the adapter bundle AND the matching trained-artifact
    directory together, as one unit - "sqlite" (the current POC default)
    uses `build_sqlite_adapters` + the approved STEP 7/8 RECENCY+PRICE
    artifacts under `{models_dir}/sqlite_baseline/`; "synthetic" uses the
    original synthetic V1 generator + `{models_dir}/` directly, kept only
    for backward compatibility.

    Raises `serving.startup_validation.ArtifactValidationError` (a plain
    `RuntimeError`) on any missing/corrupt/incompatible artifact - by
    design, this must fail loudly here rather than let a mismatched
    ranker/Two-Tower/index combination silently serve wrong
    recommendations (Phase 10). This is also what correctly REJECTS the
    old synthetic V1 artifacts now that STEP 6 changed feature dimensions
    (7/8/23 -> 9/9/29): `validate_ranker_artifacts` compares the SAVED
    `RANKING_FEATURE_NAMES` snapshot against the currently-running code's
    list, so a `data_source="synthetic"` selection pointed at those
    pre-STEP-6 artifacts fails startup immediately with a clear message,
    rather than serving broken recommendations - no new validation logic
    was needed for this, the existing check already covers it. Artifacts
    are checked and loaded FIRST, before adapter/dataset construction or
    Sentence Transformer loading, so a missing/incompatible artifact fails
    in milliseconds rather than after several seconds of otherwise-wasted
    startup work.
    """
    if config.paths.data_source == "sqlite":
        models_root = resolve_path(config.paths.models_dir) / "sqlite_baseline"
    elif config.paths.data_source == "synthetic":
        models_root = resolve_path(config.paths.models_dir)
    else:
        raise ValueError(f"unknown paths.data_source: {config.paths.data_source!r}")

    two_tower_dir = models_root / "two_tower"
    ranker_dir = models_root / "ranker"
    require_artifact_dir(two_tower_dir, "Two-Tower")
    require_artifact_dir(ranker_dir, "Ranker")

    two_tower_artifacts = load_or_raise("Two-Tower", two_tower_dir, load_two_tower_artifacts)
    validate_two_tower_artifacts(two_tower_artifacts, config)
    ranker_artifacts = load_or_raise("ranker", ranker_dir, load_ranker_artifacts)
    validate_ranker_artifacts(ranker_artifacts)

    vector_index = build_vector_index(config.retrieval)
    vector_index.build(two_tower_artifacts.item_ids, two_tower_artifacts.item_embeddings)
    validate_vector_index_compatibility(vector_index.size, len(two_tower_artifacts.item_ids))

    if config.paths.data_source == "sqlite":
        bundle = build_sqlite_adapters()
        # STEP 7's dedicated cache for the 1,200-product SQLite catalog
        # (content-hash validated, never conflated with the 50-product
        # synthetic V1 cache at the default `config.embedding.cache_path`
        # - see docs/data-mapping.md section 16). A local config copy, not
        # a shared-module change, keeps this override scoped to serving
        # this one data source.
        feature_config = config.model_copy(
            update={"embedding": config.embedding.model_copy(update={"cache_path": "data/processed/product_embeddings_sqlite.npz"})}
        )
    else:
        dataset = generate_synthetic_dataset(config)
        issues = validate_dataset(dataset)
        if issues:
            raise RuntimeError(f"Dataset validation failed: {issues[:5]}")
        bundle = build_synthetic_adapters(dataset, config.synthetic_data)
        feature_config = config

    st_encoder = SentenceTransformerEncoder(
        config.embedding.sentence_transformer_model, device=config.embedding.device, batch_size=config.embedding.encode_batch_size
    )
    feature_result = run_feature_pipeline(bundle, feature_config, encoder=st_encoder)
    text_embeddings = build_user_text_embeddings(list(feature_result.engagement_profiles.values()), st_encoder)
    products = bundle.products.list_products()
    product_lookup = {p.id: p for p in products}
    price_context = build_price_catalog_context(products)

    logger.info(
        "Recommendation service built: data_source=%s %d products, %d users, retrieval backend=%s",
        config.paths.data_source, len(product_lookup), len(bundle.users.list_user_ids()), config.retrieval.backend,
    )

    return RecommendationService(
        product_lookup=product_lookup,
        product_features=feature_result.product_features,
        product_embeddings=feature_result.product_embeddings.as_dict(),
        text_embeddings=text_embeddings,
        all_item_ids=two_tower_artifacts.item_ids,
        tt_encoder=two_tower_artifacts.encoder,
        user_tower=two_tower_artifacts.user_tower,
        ranker_model=ranker_artifacts.model,
        vector_index=vector_index,
        bundle=bundle,
        config=config,
        price_context=price_context,
        ranker_model_version=str(ranker_artifacts.metadata.get("model_version", "unknown")),
        two_tower_model_version=str(two_tower_artifacts.metadata.get("model_version", "unknown")),
        engagement_profiles=feature_result.engagement_profiles,
    )
