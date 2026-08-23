"""Typed configuration loading.

All tunables (paths, hyperparameters, thresholds, blend weights) live in
configs/*.yaml rather than being hard-coded, so later phases only ever edit
config, never scatter magic numbers through model/feature code.

Phase 10: a small, explicitly-documented set of environment variables can
override individual settings on top of whichever YAML file loads - for
container/deployment knobs that are awkward to bake into a config file
(where artifacts are mounted, which retrieval backend to force, log
verbosity) without introducing a generic "any field via env var" system.
See `_ENV_OVERRIDES` below for the full list. `RECS_CONFIG_PATH` (whole-
file selection) predates these and is handled separately in
`load_config`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"


class PathsConfig(BaseModel):
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    data_synthetic: str = "data/synthetic"
    models_dir: str = "models"
    # Backend-shaped synthetic integration/experimentation dataset (see
    # scripts/generate_backend_shaped_sqlite.py) - read-only. Default
    # consumed by adapters.sqlite_factory.build_sqlite_adapters when no
    # explicit path is passed.
    data_sqlite: str = "data/sqlite/backend_shaped_synthetic.db"
    # STEP 9 (docs/data-mapping.md section 18): which data source + trained
    # artifact set `api.dependencies.build_recommendation_service` uses for
    # LIVE serving. "sqlite" -> data_sqlite via build_sqlite_adapters,
    # artifacts from `{models_dir}/sqlite_baseline/` (the current STEP 7/8
    # RECENCY+PRICE pipeline). "synthetic" -> the original synthetic V1
    # generator, artifacts from `{models_dir}/` directly - kept only for
    # backward compatibility (its artifacts predate STEP 6's dimension
    # change and are therefore currently dimension-incompatible; startup
    # validation - `serving.startup_validation.validate_ranker_artifacts` -
    # rejects them loudly rather than serving silently wrong results, see
    # that module's docstring). Training scripts
    # (scripts/train_two_tower.py, train_ranker.py, train_sqlite_pipeline.py)
    # are unaffected by this flag - they take their adapter bundle as an
    # explicit argument, never read this field.
    data_source: Literal["synthetic", "sqlite"] = "sqlite"


class SyntheticDataConfig(BaseModel):
    num_products: int = 50
    num_users: int = 300
    random_seed: int = 42

    # Fraction of users generated with preferred_category/age_group left
    # unset, simulating users created before the backend migration lands.
    incomplete_profile_fraction: float = 0.05
    # Probability a user's preferred_category is drawn from their latent
    # persona's dominant categories rather than uniformly at random.
    preferred_category_alignment_prob: float = 0.8

    # Orders (previous purchases signal).
    order_status_weights: dict[str, float] = Field(
        default_factory=lambda: {"DELIVERED": 0.65, "COMPLETED": 0.15, "CANCELLED": 0.10, "PENDING": 0.10}
    )
    order_counted_statuses: list[str] = Field(default_factory=lambda: ["DELIVERED", "COMPLETED"])
    order_count_lambda: float = 2.5
    order_max_count: int = 8
    order_items_min: int = 1
    order_items_max: int = 5
    order_quantity_max: int = 3
    order_date_lookback_days: int = 180

    # Cart (add-to-cart habit signal).
    cart_nonempty_prob: float = 0.7
    cart_items_max: int = 6
    cart_quantity_max: int = 3

    # Reviews (auxiliary signal).
    review_prob: float = 0.35

    # Click (synthetic-only signal in V1; fifth engagement signal - see
    # docs/data-mapping.md section 4). Deliberately a higher-volume, lower-
    # intent signal than search: `replace=True` sampling (a user can click
    # the same product more than once) with a higher default lambda.
    click_count_lambda: float = 4.0
    click_max_count: int = 15

    # Search (synthetic-only signal).
    search_count_lambda: float = 2.5
    search_max_count: int = 10
    search_product_term_prob: float = 0.7

    # Chatbot context (synthetic-only signal).
    chatbot_context_prob: float = 0.4
    chatbot_mentions_max: int = 4

    # Brand affinity (orthogonal to persona).
    brand_affinity_prob: float = 0.35
    brand_affinity_bonus: float = 2.0

    # Shared affinity-scoring noise (keeps interactions correlated but not deterministic).
    affinity_noise_scale: float = 0.4


class EmbeddingConfig(BaseModel):
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    # .npz (+ a .meta.json sidecar) rather than parquet: numpy is already a
    # core dependency and IndexFlatIP/the Two-Tower pipeline want a plain
    # ndarray, so a parquet round-trip (and the pyarrow dependency it would
    # add) buys nothing here. See docs/data-mapping.md Phase 3 notes.
    cache_path: str = "data/processed/product_embeddings.npz"
    device: str = "cpu"
    encode_batch_size: int = 32


class RecencyConfig(BaseModel):
    """Configures the exponential half-life recency decay applied inside
    feature construction (`features.recency.effective_weight`) -
    `effective_weight = signal_weight * 0.5 ** (age_days / half_life_days)`.
    See `FeatureConfig` and docs/data-mapping.md section 14.

    `half_life_days=21` (three weeks) is an initial, explicitly
    unbenchmarked baseline chosen for grocery's typical weekly/biweekly
    replenishment cadence, not a tuned value - it should be re-evaluated
    once real usage data is available, exactly like `click_weight` below.
    """

    enabled: bool = True
    half_life_days: float = 21.0


class FeatureConfig(BaseModel):
    """Weights for blending the five V1 engagement signals (+ the confirmed
    preferredCategory attribute) into user category/brand affinity
    distributions and the user semantic embedding. Kept configurable per
    docs/data-mapping.md - never hard-coded in the feature-building code.

    `click_weight` is a placeholder default, not a tuned value: CLICK is a
    newly-added, weaker-intent signal (view-only, no explicit commitment
    like a purchase or cart add) with no real usage data to benchmark
    against yet. It is deliberately the smallest weight here so it can't
    dominate category/brand affinity or the semantic embedding by
    accident. All signal weights - not just this one - should be
    re-benchmarked once the real `User_events` dataset is available (see
    docs/data-mapping.md section 4).
    """

    click_weight: float = 0.1
    purchase_weight: float = 0.45
    cart_weight: float = 0.25
    search_weight: float = 0.15
    chatbot_weight: float = 0.15
    preferred_category_weight: float = 0.5

    max_top_categories: int = 5
    max_top_brands: int = 5

    recency: RecencyConfig = Field(default_factory=RecencyConfig)


class TwoTowerConfig(BaseModel):
    projection_dims: list[int] = Field(default_factory=lambda: [256, 128])
    output_dim: int = 128
    learning_rate: float = 0.001
    # 50-product catalog: a large batch already covers most/all of it as
    # in-batch negatives, so this is deliberately smaller than a typical
    # large-catalog two-tower setup (still config-overridable).
    batch_size: int = 64
    epochs: int = 20
    dropout_rate: float = 0.1
    # In-batch softmax temperature (logits = cosine_sim / temperature).
    temperature: float = 0.05
    category_embedding_dim: int = 16
    brand_embedding_dim: int = 16
    age_group_embedding_dim: int = 8
    # STEP 6 (docs/data-mapping.md section 15): BUDGET/MID/PREMIUM + an
    # "unknown/no profile" bucket - only 4 possible values, so a small
    # embedding is deliberate (matches age_group_embedding_dim's size,
    # not category/brand's larger vocab). A learned embedding, not a
    # 0/1/2 ordinal integer - see FeatureConfig-adjacent design note in
    # `retrieval.two_tower.feature_encoding`.
    price_tier_embedding_dim: int = 8
    # A user needs at least this many DISTINCT purchased products to be
    # eligible for the val/test leave-one-out holdout (1 test + 1 val +
    # >=1 remaining in train); users below this threshold contribute all
    # their purchases to train only and aren't evaluated.
    min_distinct_products_for_holdout: int = 3
    eval_k_values: list[int] = Field(default_factory=lambda: [5, 10, 20])
    early_stopping_patience: int = 3
    random_seed: int = 42


class RetrievalConfig(BaseModel):
    """`"scann"` is the primary/production ANN backend (Linux/Docker-only,
    see configs/docker.yaml); `"faiss"` is the native-Windows dev
    fallback used by the default config.base.yaml - see
    docs/data-mapping.md section 10. Both backends now do genuine
    approximate nearest-neighbor search (FAISS: HNSW; ScaNN: tree
    partitioning + asymmetric-hashing quantization + reordering) rather
    than exact brute-force - the fields below tune that approximation and
    are deliberately catalog-size-relative where it matters (e.g. ScaNN's
    leaf count), not literals tuned for one specific catalog size.
    """

    backend: Literal["faiss", "scann"] = "faiss"
    candidate_pool_multiplier: int = 5
    min_candidate_pool: int = 50

    # --- FAISS HNSW (see retrieval.index.faiss_index) ---
    faiss_hnsw_m: int = 32  # bi-directional links per graph node; higher = better recall, more memory
    faiss_hnsw_ef_construction: int = 200  # build-time candidate list size; higher = better graph quality, one-time cost
    faiss_hnsw_ef_search: int = 128  # query-time candidate list size floor; search() also floors this at k*2 per call

    # --- ScaNN tree + asymmetric hashing + reorder (see retrieval.index.scann_index) ---
    scann_leaves_multiplier: float = 2.0  # num_leaves ~= multiplier * sqrt(catalog_size) - scales with catalog, not a literal
    # Density floor, not a leaf-count floor: num_leaves is capped so each
    # partition averages at least this many catalog points. An earlier
    # version used an unconditional MINIMUM leaf count instead - that
    # padded a mid-size catalog (e.g. 50 items) up to 20 leaves regardless
    # (~2.5 points/leaf), which made ScaNN's kmeans tree partitioner
    # produce empty/zero-norm centroids and silently return NaN scores
    # for items in the broken partition - confirmed via a real Docker/CI
    # run, where it NaN-poisoned ranker training end-to-end. This field
    # scales num_leaves DOWN for small catalogs instead.
    scann_min_points_per_leaf: int = 20
    scann_max_leaves: int = 2000
    # Fraction of leaves searched per query - recall/latency knob. 0.3 is
    # empirically validated (not a guess) against the real trained
    # 1,200-item/128-D catalog in Docker/CI: at 0.1-0.15 the tree
    # partitioning missed the query's own point's leaf often enough to
    # fail self-match and drop Recall@10 to ~0.88; 0.3 recovers self-match
    # 200/200 and Recall@10=0.956 / Recall@20=0.919 over 200 sampled
    # queries (measured with scann_min_points_per_leaf=20 below) - see
    # scripts/evaluate_ann_recall.py to re-measure after any change to
    # this, scann_leaves_multiplier, or scann_min_points_per_leaf.
    scann_leaves_to_search_fraction: float = 0.3
    scann_ah_dims_per_block: int = 2  # asymmetric-hashing quantization block size (an embedding-dim property, not catalog-size-dependent)
    scann_aq_threshold: float = 0.2  # anisotropic quantization threshold, ScaNN's recommended default for inner-product search
    scann_training_sample_size: int = 100_000  # capped at catalog_size at build time if the catalog is smaller
    # Reordering depth is fixed at BUILD time (ScaNN API constraint, not
    # adjustable per query) - it re-scores the AH-quantized top
    # `scann_reorder_k` candidates with exact float embeddings before
    # returning, recovering most of the recall AH quantization would
    # otherwise cost. Must stay >= the largest k ever passed to search()
    # in this deployment, i.e. >= min_candidate_pool * candidate_pool_multiplier
    # (times the eligibility oversample factor below), or queries asking
    # for more than this many candidates silently can't get exact-reorder
    # scores for all of them.
    scann_reorder_k: int = 200

    # --- Eligibility-restricted search oversampling (see
    # retrieval.index.eligibility_filter.EligibilityRestrictedIndex) ---
    # Requesting the whole index per query (the old behavior) defeats an
    # ANN backend's sublinear-search benefit. Instead: ask for
    # k * eligibility_oversample_factor (floored at
    # eligibility_oversample_floor), filter to allowed_ids, and if still
    # short of k, widen by eligibility_widen_multiplier and retry, up to
    # eligibility_max_widen_attempts times - each attempt is capped at the
    # index's actual size, so this always terminates in a bounded number
    # of search calls. With these defaults, applied to this project's
    # pool sizes, the sequence happens to reach the full catalog by the
    # final attempt for a thin/scattered eligible set - but that's a
    # property of these specific numbers, not a guarantee: a more
    # aggressive config can plateau below full coverage first, honestly
    # under-filling rather than falling back to an unbounded full scan.
    eligibility_oversample_factor: int = 3
    eligibility_oversample_floor: int = 20
    eligibility_widen_multiplier: int = 2
    eligibility_max_widen_attempts: int = 4


class SparseBlendConfig(BaseModel):
    personalized: float = 0.6
    preferred_category: float = 0.2
    popularity: float = 0.2


class ColdStartConfig(BaseModel):
    strong_history_min_signals: int = 5
    sparse_history_min_signals: int = 1
    sparse_blend: SparseBlendConfig = Field(default_factory=SparseBlendConfig)
    no_history_fallback_order: list[str] = Field(
        default_factory=lambda: ["preferred_category", "category_popularity", "global_popularity"]
    )


class EligibilityConfig(BaseModel):
    require_active: bool = True
    require_in_stock: bool = True


class ReRankingConfig(BaseModel):
    """Continuous (never a hard cutoff) relevance-vs-diversity trade-off:
    `diversity_strength=0` reproduces the input ranking exactly (pure
    relevance order); higher values increasingly penalize repeated
    categories/brands. Because it's a score penalty rather than a quota,
    a candidate is never dropped purely for diversity reasons - it can
    only be reordered - so over-diversifying can push relevance down
    gradually and tunably, never catastrophically. See
    `recommendation.reranking.diversity`.
    """

    diversity_strength: float = 0.5
    category_repetition_penalty: float = 0.15
    brand_repetition_penalty: float = 0.08


class RankingConfig(BaseModel):
    hidden_units: list[int] = Field(default_factory=lambda: [128, 64])
    learning_rate: float = 0.001
    epochs: int = 10
    batch_size: int = 256
    dropout_rate: float = 0.1
    # Implicit-feedback negative sampling: how many non-purchased candidates
    # (drawn from what VectorIndex actually retrieves, not the raw catalog -
    # matches serving-time candidate distribution) to label per positive.
    negatives_per_positive: int = 4
    early_stopping_patience: int = 3
    eval_k_values: list[int] = Field(default_factory=lambda: [5, 10, 20])
    random_seed: int = 42


class ApiConfig(BaseModel):
    model_version: str = "v1"
    default_recommendation_count: int = 10
    max_recommendation_count: int = 50
    host: str = "0.0.0.0"
    port: int = 8000


class DashboardConfig(BaseModel):
    """STEP 9 (docs/data-mapping.md section 18): Streamlit is now a plain
    HTTP client of FastAPI - `api_base_url` is the ONE place that address
    is configured (never hard-coded per call site). The native-dev default
    below (`configs/base.yaml`) targets a locally-running
    `scripts/run_api.py`; `configs/docker.yaml` overrides it to the
    Compose service hostname (`http://api:8000`) since "localhost" inside
    the dashboard container would otherwise resolve to the dashboard
    container itself, not the separate api container.
    """

    api_base_url: str = "http://localhost:8000"
    request_timeout_seconds: float = 10.0


class AppConfig(BaseModel):
    random_seed: int = 42
    log_level: str = "INFO"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    synthetic_data: SyntheticDataConfig = Field(default_factory=SyntheticDataConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    two_tower: TwoTowerConfig = Field(default_factory=TwoTowerConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    cold_start: ColdStartConfig = Field(default_factory=ColdStartConfig)
    eligibility: EligibilityConfig = Field(default_factory=EligibilityConfig)
    reranking: ReRankingConfig = Field(default_factory=ReRankingConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)


# env var -> (dotted path into the raw config dict, caster). Deliberately
# a short, explicit list of operationally-relevant knobs, not a generic
# mechanism for overriding arbitrary fields - see module docstring.
_ENV_OVERRIDES: dict[str, tuple[tuple[str, ...], Callable[[str], object]]] = {
    "RECS_MODELS_DIR": (("paths", "models_dir"), str),
    "RECS_LOG_LEVEL": (("log_level",), str),
    "RECS_RETRIEVAL_BACKEND": (("retrieval", "backend"), str),
    "RECS_API_HOST": (("api", "host"), str),
    "RECS_API_PORT": (("api", "port"), int),
    "RECS_API_DEFAULT_TOP_N": (("api", "default_recommendation_count"), int),
    "RECS_API_MAX_TOP_N": (("api", "max_recommendation_count"), int),
    "RECS_DATA_SOURCE": (("paths", "data_source"), str),
    "RECS_DASHBOARD_API_BASE_URL": (("dashboard", "api_base_url"), str),
}


def _set_nested(d: dict, path: tuple[str, ...], value: object) -> None:
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value


def _apply_env_overrides(raw: dict) -> dict:
    for env_var, (path, caster) in _ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value is None:
            continue
        try:
            casted = caster(value)
        except ValueError as exc:
            raise ValueError(f"Invalid value for {env_var}={value!r}: {exc}") from exc
        _set_nested(raw, path, casted)
    return raw


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate the YAML config into an AppConfig.

    Resolution order: explicit `path` argument, then RECS_CONFIG_PATH env
    var, then configs/base.yaml - then the `_ENV_OVERRIDES` env vars are
    applied on top of whichever file loaded (highest precedence).
    """
    resolved = Path(path) if path is not None else Path(os.environ.get("RECS_CONFIG_PATH", DEFAULT_CONFIG_PATH))
    with resolved.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = _apply_env_overrides(raw)
    return AppConfig(**raw)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Process-wide cached config, read from the default/env-resolved path."""
    return load_config()


def resolve_path(relative_path: str | Path) -> Path:
    """Resolve a config-declared path (e.g. `config.embedding.cache_path`)
    against the repo root, so callers work the same regardless of the
    process's current working directory.
    """
    path = Path(relative_path)
    return path if path.is_absolute() else REPO_ROOT / path
