"""The recommendation service: everything the API needs to serve
requests, loaded ONCE via `build_recommendation_service` (called from
`api.app`'s FastAPI `lifespan`) and reused across every request. A plain
dataclass with no FastAPI dependency, so it can be built directly in
tests (real or fake artifacts) and injected into
`api.app.create_app(service=...)` - the dependency-injection seam that
keeps the pipeline testable. Delegates every recommendation decision to
`serving.pipeline.recommend`; the only logic added here is the
API-specific "does this user exist at all" check (`UnknownUserError`),
deliberately NOT part of the pipeline since other callers (leave-one-out
evaluation, the dashboard) may have different notions of "known user".

FastAPI is the only process that constructs a `RecommendationService` -
the Streamlit dashboard is a plain HTTP client of the API
(`ui.api_client.RecommendationApiClient`) and never loads model artifacts
or touches SQLite/synthetic adapters itself. This class stays here (not
moved to `serving`) since `api.routes` is still its only real caller.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

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
    validate_retrieval_config,
    validate_two_tower_artifacts,
    validate_vector_index_compatibility,
)
from recommendation.utils.config import AppConfig, resolve_path
from recommendation.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class _DataSnapshot:
    """Everything `RecommendationService` needs that is derived from the
    adapter bundle (i.e. from `User_events` + Product/User/Review) -
    deliberately excludes trained model artifacts (Two-Tower/ranker/
    VectorIndex), which never change on a refresh.
    """

    bundle: AdapterBundle
    product_lookup: dict[int, Product]
    product_features: dict[int, ProductFeatures]
    product_embeddings: dict[int, np.ndarray]
    text_embeddings: dict[str, np.ndarray]
    price_context: PriceCatalogContext
    engagement_profiles: dict[int, EngagementProfile]


def _load_data_snapshot(config: AppConfig, encoder: SentenceTransformerEncoder) -> _DataSnapshot:
    """Builds the adapter bundle plus everything derived from it, for
    `config.paths.data_source`. Shared by `build_recommendation_service`
    (process startup) and `RecommendationService.maybe_refresh` (periodic
    reload of a running service) so the two can never drift apart. Never
    touches Two-Tower/ranker/VectorIndex artifacts.

    `db_path` is passed explicitly (rather than the previous implicit
    reliance on the process-wide cached `get_config()` inside
    `build_sqlite_adapters()`) so this function's behavior is fully
    determined by the `config` argument it's given - required for
    `maybe_refresh` to be reload-able/testable against a specific
    `paths.data_sqlite`, and behaviorally identical in production, where
    `config` already IS `get_config()`.
    """
    if config.paths.data_source == "sqlite":
        bundle = build_sqlite_adapters(resolve_path(config.paths.data_sqlite))
        # STEP 7's dedicated cache for the 1,200-product SQLite catalog
        # (content-hash validated, never conflated with the 50-product
        # synthetic V1 cache at the default `config.embedding.cache_path`
        # - see docs/data-mapping.md section 16).
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

    feature_result = run_feature_pipeline(bundle, feature_config, encoder=encoder)
    text_embeddings = build_user_text_embeddings(list(feature_result.engagement_profiles.values()), encoder)
    products = bundle.products.list_products()
    product_lookup = {p.id: p for p in products}
    price_context = build_price_catalog_context(products)

    return _DataSnapshot(
        bundle=bundle,
        product_lookup=product_lookup,
        product_features=feature_result.product_features,
        product_embeddings=feature_result.product_embeddings.as_dict(),
        text_embeddings=text_embeddings,
        price_context=price_context,
        engagement_profiles=feature_result.engagement_profiles,
    )


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
    # `bundle`), but the dashboard's user-selection list needs every
    # user's engagement data at once, and recomputing it per-user would
    # be wasteful when it's already sitting in the feature pipeline result.
    engagement_profiles: dict[int, EngagementProfile] = field(default_factory=dict)
    # STEP 9 offline-metrics fix: the raw `metadata.json` dicts saved
    # alongside the loaded ranker/Two-Tower artifacts (run_id, dataset
    # fingerprint, ...) - kept as-is (not just the single `model_version`
    # string already extracted above) so `GET /v1/metrics/offline` can
    # confirm a persisted evaluation report was produced against THIS
    # exact trained run before serving it, without re-hashing the dataset
    # or reloading artifacts a second time.
    ranker_metadata: dict = field(default_factory=dict)
    two_tower_metadata: dict = field(default_factory=dict)
    # Freshness fix: the already-loaded encoder, reused by `maybe_refresh`
    # so a reload never re-downloads/re-instantiates the Sentence
    # Transformer model. `None` only for tests that never call refresh.
    sentence_encoder: SentenceTransformerEncoder | None = None
    _data_loaded_at: float = field(default_factory=time.monotonic, init=False, repr=False, compare=False)
    _refresh_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def is_known_user(self, user_id: int) -> bool:
        return self.bundle.users.get_user_profile(user_id) is not None

    def maybe_refresh(self, *, force: bool = False) -> bool:
        """Reloads the DATA LAYER ONLY (`User_events` + Product/User/Review
        adapters and everything derived from them: product_features,
        engagement profiles, product embeddings/lookup, price context) so
        rows the backend team writes into `User_events` after this service
        started become visible to a RUNNING process - without a restart.

        Two-Tower/ranker/VectorIndex artifacts are never touched here.
        TTL-gated by `config.refresh.interval_seconds` (default 30s)
        rather than reloaded on every request, to keep the SQLite-read +
        feature-recompute cost off the hot path for most calls.
        `force=True` bypasses both the TTL and the disabled-by-config
        (`interval_seconds <= 0`) check - used by tests and by an
        explicit caller that wants an immediate reload.

        Returns True if a reload actually happened.
        """
        interval = self.config.refresh.interval_seconds
        if not force:
            if interval <= 0:
                return False
            if (time.monotonic() - self._data_loaded_at) < interval:
                return False
            if self.config.paths.data_source != "sqlite":
                # The synthetic generator is deterministic (fixed seed) -
                # there is no live external data source to become fresh.
                return False

        if not self._refresh_lock.acquire(blocking=False):
            # Another thread is already refreshing - this call proceeds
            # with the current (still valid) data rather than blocking on
            # a redundant reload.
            return False
        try:
            if not force and (time.monotonic() - self._data_loaded_at) < interval:
                return False  # a concurrent refresh already happened while we waited for the lock
            if self.sentence_encoder is None:
                raise RuntimeError("RecommendationService.sentence_encoder must be set to refresh data")
            snapshot = _load_data_snapshot(self.config, self.sentence_encoder)
            self.bundle = snapshot.bundle
            self.product_lookup = snapshot.product_lookup
            self.product_features = snapshot.product_features
            self.product_embeddings = snapshot.product_embeddings
            self.text_embeddings = snapshot.text_embeddings
            self.price_context = snapshot.price_context
            self.engagement_profiles = snapshot.engagement_profiles
            self._data_loaded_at = time.monotonic()
            logger.info(
                "recommendation service data refreshed: data_source=%s %d products, %d users",
                self.config.paths.data_source, len(self.product_lookup), len(self.bundle.users.list_user_ids()),
            )
            return True
        finally:
            self._refresh_lock.release()

    def recommend(self, user_id: int, limit: int) -> RecommendationResult:
        self.maybe_refresh()
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


def resolve_models_root(config: AppConfig) -> Path:
    """The same `data_source` -> trained-artifact-directory branch
    `build_recommendation_service` uses, extracted so other read-only
    callers (the persisted offline-evaluation-report loader) resolve the
    identical directory without duplicating - or silently drifting from -
    this logic.
    """
    if config.paths.data_source == "sqlite":
        return resolve_path(config.paths.models_dir) / "sqlite_baseline"
    elif config.paths.data_source == "synthetic":
        return resolve_path(config.paths.models_dir)
    else:
        raise ValueError(f"unknown paths.data_source: {config.paths.data_source!r}")


def build_recommendation_service(config: AppConfig) -> RecommendationService:
    """Real production wiring for LIVE serving - loads the ALREADY-TRAINED
    artifacts (never retrains) and builds the Phase 5 VectorIndex.

    `config.paths.data_source` selects BOTH the adapter bundle AND the
    matching trained-artifact directory together, as one unit (docs/
    data-mapping.md section 18) - "sqlite" (the current default) uses
    `build_sqlite_adapters` + the recency+price artifacts under
    `{models_dir}/sqlite_baseline/`; "synthetic" uses the original
    synthetic generator + `{models_dir}/` directly, kept only for
    backward compatibility.

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
    models_root = resolve_models_root(config)

    two_tower_dir = models_root / "two_tower"
    ranker_dir = models_root / "ranker"
    require_artifact_dir(two_tower_dir, "Two-Tower")
    require_artifact_dir(ranker_dir, "Ranker")

    two_tower_artifacts = load_or_raise("Two-Tower", two_tower_dir, load_two_tower_artifacts)
    validate_two_tower_artifacts(two_tower_artifacts, config)
    ranker_artifacts = load_or_raise("ranker", ranker_dir, load_ranker_artifacts)
    validate_ranker_artifacts(ranker_artifacts)
    validate_retrieval_config(config.retrieval)

    vector_index = build_vector_index(config.retrieval)
    vector_index.build(two_tower_artifacts.item_ids, two_tower_artifacts.item_embeddings)
    validate_vector_index_compatibility(vector_index.size, len(two_tower_artifacts.item_ids))

    st_encoder = SentenceTransformerEncoder(
        config.embedding.sentence_transformer_model, device=config.embedding.device, batch_size=config.embedding.encode_batch_size
    )
    snapshot = _load_data_snapshot(config, st_encoder)

    logger.info(
        "Recommendation service built: data_source=%s %d products, %d users, retrieval backend=%s",
        config.paths.data_source, len(snapshot.product_lookup), len(snapshot.bundle.users.list_user_ids()), config.retrieval.backend,
    )

    return RecommendationService(
        product_lookup=snapshot.product_lookup,
        product_features=snapshot.product_features,
        product_embeddings=snapshot.product_embeddings,
        text_embeddings=snapshot.text_embeddings,
        all_item_ids=two_tower_artifacts.item_ids,
        tt_encoder=two_tower_artifacts.encoder,
        user_tower=two_tower_artifacts.user_tower,
        ranker_model=ranker_artifacts.model,
        vector_index=vector_index,
        bundle=snapshot.bundle,
        config=config,
        price_context=snapshot.price_context,
        ranker_model_version=str(ranker_artifacts.metadata.get("model_version", "unknown")),
        two_tower_model_version=str(two_tower_artifacts.metadata.get("model_version", "unknown")),
        engagement_profiles=snapshot.engagement_profiles,
        ranker_metadata=ranker_artifacts.metadata,
        two_tower_metadata=two_tower_artifacts.metadata,
        sentence_encoder=st_encoder,
    )
