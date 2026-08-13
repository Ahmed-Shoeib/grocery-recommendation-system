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

Despite the module path, `RecommendationService`/`build_recommendation
_service` are shared by the Phase 9 Streamlit dashboard too
(`ui.service_loader`) - it reuses this exact class in-process (no HTTP
hop to the API) rather than duplicating a second copy of "load the
Two-Tower/ranker/VectorIndex once" wiring. Left here rather than moved
to `serving` to avoid touching already-shipped Phase 8 call sites for a
Phase 9 change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf

from recommendation.api.errors import UnknownUserError
from recommendation.data.adapters.base import AdapterBundle
from recommendation.data.adapters.factory import build_synthetic_adapters
from recommendation.data.schemas.engagement import EngagementProfile
from recommendation.data.schemas.product import Product
from recommendation.data.synthetic.dataset import generate_synthetic_dataset
from recommendation.data.synthetic.validation import validate_dataset
from recommendation.embeddings.encoder import SentenceTransformerEncoder
from recommendation.features.pipeline import run_feature_pipeline
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
        )

    def readiness_checks(self) -> dict[str, bool]:
        return {
            "catalog_loaded": len(self.product_lookup) > 0,
            "two_tower_loaded": self.user_tower is not None,
            "ranker_loaded": self.ranker_model is not None,
            "vector_index_loaded": self.vector_index.size > 0,
        }


def build_recommendation_service(config: AppConfig) -> RecommendationService:
    """Real production wiring: generates the (synthetic V1) dataset,
    loads the ALREADY-TRAINED Phase 4/6 artifacts (never retrains), and
    builds the Phase 5 VectorIndex - mirrors `scripts/run_pipeline.py`.

    Raises `serving.startup_validation.ArtifactValidationError` (a plain
    `RuntimeError`) on any missing/corrupt/incompatible artifact - by
    design, this must fail loudly here rather than let a mismatched
    ranker/Two-Tower/index combination silently serve wrong
    recommendations (Phase 10). Artifacts are checked and loaded FIRST,
    before dataset generation or Sentence Transformer loading, so a
    missing/incompatible artifact fails in milliseconds rather than
    after several seconds of otherwise-wasted startup work.
    """
    two_tower_dir = resolve_path(config.paths.models_dir) / "two_tower"
    ranker_dir = resolve_path(config.paths.models_dir) / "ranker"
    require_artifact_dir(two_tower_dir, "Two-Tower")
    require_artifact_dir(ranker_dir, "Ranker")

    two_tower_artifacts = load_or_raise("Two-Tower", two_tower_dir, load_two_tower_artifacts)
    validate_two_tower_artifacts(two_tower_artifacts, config)
    ranker_artifacts = load_or_raise("ranker", ranker_dir, load_ranker_artifacts)
    validate_ranker_artifacts(ranker_artifacts)

    vector_index = build_vector_index(config.retrieval)
    vector_index.build(two_tower_artifacts.item_ids, two_tower_artifacts.item_embeddings)
    validate_vector_index_compatibility(vector_index.size, len(two_tower_artifacts.item_ids))

    dataset = generate_synthetic_dataset(config)
    issues = validate_dataset(dataset)
    if issues:
        raise RuntimeError(f"Dataset validation failed: {issues[:5]}")
    bundle = build_synthetic_adapters(dataset, config.synthetic_data)

    st_encoder = SentenceTransformerEncoder(
        config.embedding.sentence_transformer_model, device=config.embedding.device, batch_size=config.embedding.encode_batch_size
    )
    feature_result = run_feature_pipeline(bundle, config, encoder=st_encoder)
    text_embeddings = build_user_text_embeddings(list(feature_result.engagement_profiles.values()), st_encoder)
    product_lookup = {p.id: p for p in bundle.products.list_products()}

    logger.info(
        "Recommendation service built: %d products, %d users, retrieval backend=%s",
        len(product_lookup), len(bundle.users.list_user_ids()), config.retrieval.backend,
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
        ranker_model_version=str(ranker_artifacts.metadata.get("model_version", "unknown")),
        two_tower_model_version=str(two_tower_artifacts.metadata.get("model_version", "unknown")),
        engagement_profiles=feature_result.engagement_profiles,
    )
