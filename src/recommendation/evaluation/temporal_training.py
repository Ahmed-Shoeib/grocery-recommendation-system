"""STEP 7: temporal-aware TRAINING example construction for the Two-Tower
model and the ranker, built on top of the STEP 5 temporal future-purchase
protocol (`evaluation.temporal_future_purchase`) instead of the original
non-temporal, product-id-based leave-one-out split
(`retrieval.two_tower.splitting`).

Why this module exists (docs/data-mapping.md section 16): the pre-existing
Two-Tower/ranker training-example builders
(`retrieval.two_tower.examples`, `ranking.examples`) are built around
`UserSplit` - a random, non-temporal per-product holdout with no notion of
"before"/"after". They cannot be reused as-is for temporally-consistent
training: every point-in-time user representation used as a TRAINING
input must be built from history strictly before that example's own
target time, or the model would implicitly see future information during
training (not just during evaluation, which STEP 5 already guarded).

Design (deliberately reuses, not reimplements, every piece of STEP 5/6
infrastructure - only the example-CONSTRUCTION glue below is new):

  - Splitting: `evaluation.temporal_future_purchase.build_temporal_splits`
    / `TemporalUserSplit`, UNCHANGED.
  - Point-in-time truncation: `build_point_in_time_engagement_profile`,
    UNCHANGED.
  - Feature construction (recency, price): `features.user_features
    .build_user_features(reference_time=..., price_context=...)`,
    UNCHANGED.

For a user with `TemporalUserSplit.val_cutoff` (`None` for
INSUFFICIENT_DEPTH/ENGAGEMENT_NO_PURCHASE/NO_HISTORY - see that class's
docstring):

  TRAIN positives = every PURCHASE event with `action_time < val_cutoff`
    (or, if `val_cutoff is None`, EVERY purchase the user has - this is
    exactly the INSUFFICIENT_DEPTH case: nothing is held out, so nothing
    needs excluding). Each becomes ONE training example whose reference
    time/point-in-time truncation is THAT PURCHASE'S OWN `action_time` -
    not a single shared per-user cutoff - so an earlier training purchase
    never sees a later training purchase, and no product-id exclusion
    (`exclude_product_ids`) is needed at all: strict time truncation
    already excludes exactly this one event while correctly KEEPING a
    genuinely earlier purchase of the same product as legitimate history
    (the repeat-purchase policy already established in STEP 5's protocol
    docstring) - time truncation subsumes and is MORE precise than the
    old product-id exclusion mechanism for this purpose.

  VAL / TEST cases = one point-in-time profile per evaluable user at
    `val_cutoff`/`test_cutoff`, targets = `val_target_ids`/`test_target_ids`
    (already leakage-safe by construction - STEP 5).

No future purchase is ever a training POSITIVE for a cutoff after its own
time (impossible by the `< val_cutoff` filter), and - the ranker-specific
risk this module also closes, docs/data-mapping.md section 16's negative-
sampling note - no future purchase is ever sampled as a NEGATIVE either:
`build_temporal_ranking_dataset` excludes every product this user EVER
purchases (past or future relative to any single example's cutoff, via
`all_purchased_by_user`) from that user's negative-candidate pool, so a
product the user simply hasn't bought YET (as of a training example's
cutoff) is never mislabeled "this user doesn't want this."
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import tensorflow as tf

from recommendation.data.adapters.base import ReviewAdapter, UserAdapter
from recommendation.data.schemas.events import ActionType, UserInteraction
from recommendation.data.schemas.product import Product
from recommendation.evaluation.retrieval_metrics import (
    mean_hit_rate_at_k,
    mean_ndcg_at_k,
    mean_precision_at_k,
    mean_reciprocal_rank,
    mean_recall_at_k,
)
from recommendation.evaluation.temporal_future_purchase import (
    TemporalUserSplit,
    build_point_in_time_engagement_profile,
)
from recommendation.features.price import PriceCatalogContext
from recommendation.features.product_features import ProductFeatures
from recommendation.features.user_features import UserFeatures, build_user_features
from recommendation.ranking.examples import RankingExample
from recommendation.ranking.features import build_ranking_feature_vector
from recommendation.ranking.model import build_ranker_model
from recommendation.retrieval.index.base import SearchResult, VectorIndex
from recommendation.retrieval.index.factory import build_vector_index, candidate_pool_size
from recommendation.retrieval.two_tower.evaluation import rank_all_items
from recommendation.retrieval.two_tower.examples import TrainingExample
from recommendation.retrieval.two_tower.feature_encoding import TwoTowerFeatureEncoder
from recommendation.retrieval.two_tower.model import TwoTowerModel, build_item_tower, build_user_tower
from recommendation.utils.config import FeatureConfig, RankingConfig, RetrievalConfig, TwoTowerConfig


@dataclass
class TemporalEvalCase:
    """One point-in-time evaluation example: `user_features` built from
    history strictly before `cutoff`, `target_ids` the held-out FUTURE
    PURCHASE product id(s) at/after `cutoff` (a set - a user may have
    purchased more than one product in the target window).
    """

    user_id: int
    cutoff: datetime
    target_ids: frozenset[int]
    user_features: UserFeatures


def all_purchased_product_ids_by_user(events_by_user: dict[int, list[UserInteraction]]) -> dict[int, frozenset[int]]:
    """Every product a user EVER purchases, across their entire event
    history (past AND future relative to any single cutoff) - used ONLY
    to keep a future purchase from being sampled as a ranker negative
    (see module docstring). Never used to build a point-in-time feature -
    that always goes through `build_point_in_time_engagement_profile`.
    """
    out: dict[int, frozenset[int]] = {}
    for user_id, events in events_by_user.items():
        out[user_id] = frozenset(e.product_id for e in events if e.action_type == ActionType.PURCHASE)
    return out


def build_temporal_two_tower_examples(
    events_by_user: dict[int, list[UserInteraction]],
    splits: dict[int, TemporalUserSplit],
    users_adapter: UserAdapter,
    reviews_adapter: ReviewAdapter,
    product_lookup: dict[int, Product],
    product_embeddings: dict[int, np.ndarray],
    feature_config: FeatureConfig,
    price_context: PriceCatalogContext | None,
) -> tuple[list[TrainingExample], list[TrainingExample], list[TemporalEvalCase], list[TemporalEvalCase]]:
    """Returns `(train_examples, val_loss_examples, val_cases, test_cases)`.

    `val_loss_examples` mirrors `train_examples`' shape (one
    `TrainingExample` per (user, val target product) pair, built from the
    SAME point-in-time profile as that user's `TemporalEvalCase` at
    `val_cutoff`) - it exists only to feed the Two-Tower's in-batch-
    softmax validation loss during training (early stopping), exactly
    like the pre-existing non-temporal protocol's val-loss examples.
    """
    train_examples: list[TrainingExample] = []
    val_loss_examples: list[TrainingExample] = []
    val_cases: list[TemporalEvalCase] = []
    test_cases: list[TemporalEvalCase] = []

    for user_id, split in splits.items():
        history = events_by_user.get(user_id, [])
        purchase_events = [e for e in history if e.action_type == ActionType.PURCHASE and e.action_time is not None]
        if not purchase_events:
            continue

        train_bound = split.val_cutoff  # None -> INSUFFICIENT_DEPTH: nothing held out, every purchase is train-eligible
        for e in purchase_events:
            if train_bound is not None and not (e.action_time < train_bound):
                continue
            profile = build_point_in_time_engagement_profile(user_id, history, e.action_time, users_adapter, reviews_adapter)
            user_features = build_user_features(
                profile, product_lookup, product_embeddings, feature_config,
                reference_time=e.action_time, price_context=price_context,
            )
            train_examples.append(TrainingExample(user_id=user_id, product_id=e.product_id, user_features=user_features))

        if split.is_val_evaluable:
            profile = build_point_in_time_engagement_profile(user_id, history, split.val_cutoff, users_adapter, reviews_adapter)
            user_features = build_user_features(
                profile, product_lookup, product_embeddings, feature_config,
                reference_time=split.val_cutoff, price_context=price_context,
            )
            val_cases.append(TemporalEvalCase(user_id, split.val_cutoff, split.val_target_ids, user_features))
            for pid in split.val_target_ids:
                val_loss_examples.append(TrainingExample(user_id=user_id, product_id=pid, user_features=user_features))

        if split.is_test_evaluable:
            profile = build_point_in_time_engagement_profile(user_id, history, split.test_cutoff, users_adapter, reviews_adapter)
            user_features = build_user_features(
                profile, product_lookup, product_embeddings, feature_config,
                reference_time=split.test_cutoff, price_context=price_context,
            )
            test_cases.append(TemporalEvalCase(user_id, split.test_cutoff, split.test_target_ids, user_features))

    return train_examples, val_loss_examples, val_cases, test_cases


@dataclass
class TemporalRetrievalReport:
    split_name: str
    num_cases: int
    recall_at_k: dict[int, float]
    hit_rate_at_k: dict[int, float]


def evaluate_temporal_retrieval(
    cases: list[TemporalEvalCase],
    split_name: str,
    user_tower: tf.keras.Model,
    item_embeddings: np.ndarray,
    item_ids: list[int],
    encoder: TwoTowerFeatureEncoder,
    k_values: list[int],
) -> TemporalRetrievalReport:
    """Brute-force (exact, full-catalog) Two-Tower-only retrieval quality -
    mirrors `retrieval.two_tower.evaluation.evaluate_split` but supports a
    multi-product `target_ids` set per case instead of one id. This is a
    TRAINING-time sanity check (does the retrieval embedding space alone
    separate a user from their future purchase?), not the primary offline
    metric - see `evaluate_primary_pipeline` for that (full pipeline:
    eligibility + ranker + re-ranking + final validation).
    """
    if not cases:
        return TemporalRetrievalReport(split_name=split_name, num_cases=0, recall_at_k={}, hit_rate_at_k={})

    user_batch = encoder.encode_user_batch([c.user_features for c in cases])
    user_embeddings = user_tower.predict(user_batch, verbose=0)
    rankings = rank_all_items(user_embeddings, item_embeddings, item_ids)
    relevant_sets = [set(c.target_ids) for c in cases]

    return TemporalRetrievalReport(
        split_name=split_name,
        num_cases=len(cases),
        recall_at_k={k: mean_recall_at_k(rankings, relevant_sets, k) for k in k_values},
        hit_rate_at_k={k: mean_hit_rate_at_k(rankings, relevant_sets, k) for k in k_values},
    )


# --- ranker training examples (temporal, leakage-safe negative sampling) --

def _score_and_rank(result: SearchResult, product_id: int) -> tuple[float, int]:
    if product_id in result.item_ids:
        rank = result.item_ids.index(product_id)
        return result.scores[rank], rank
    return (min(result.scores) if result.scores else 0.0), len(result.item_ids)


def _user_rng(seed: int, user_id: int, salt: int = 0) -> np.random.Generator:
    return np.random.default_rng([seed, user_id, salt % (2**32)])


def _sample_negatives(candidate_ids: list[int], exclude_ids: frozenset[int], num_needed: int, rng: np.random.Generator) -> list[int]:
    pool = [pid for pid in candidate_ids if pid not in exclude_ids]
    if len(pool) <= num_needed:
        return pool
    chosen = rng.choice(len(pool), size=num_needed, replace=False)
    return [pool[i] for i in chosen]


def build_temporal_ranking_dataset(
    train_examples: list[TrainingExample],
    val_cases: list[TemporalEvalCase],
    all_purchased_by_user: dict[int, frozenset[int]],
    product_features: dict[int, ProductFeatures],
    product_embeddings: dict[int, np.ndarray],
    tt_encoder: TwoTowerFeatureEncoder,
    user_tower: tf.keras.Model,
    vector_index: VectorIndex,
    pool_size: int,
    ranking_config: RankingConfig,
    include_price_features: bool = True,
) -> tuple[list[RankingExample], list[RankingExample]]:
    """Returns `(train_ranking_examples, val_loss_ranking_examples)`.

    Negative candidates are drawn from what `vector_index.search` actually
    retrieves for that example's own point-in-time query embedding (same
    "retrieved negatives, not raw catalog" philosophy as the non-temporal
    `ranking.examples.build_ranking_dataset`), MINUS `all_purchased_by_user
    [user_id]` - every product this user EVER purchases, past or future
    relative to this example's cutoff - so a future purchase can never
    become a mislabeled hard negative (module docstring).

    `include_price_features` (STEP 8, docs/data-mapping.md section 17):
    forwarded unchanged to `build_ranking_feature_vector` - `True`
    (default) reproduces STEP 6/7 exactly; `False` builds the pre-STEP-6
    23-feature vector for the controlled ablation's BASE condition.
    """

    def _feature_row(user_features: UserFeatures, pid: int, score: float, rank: int) -> np.ndarray:
        return build_ranking_feature_vector(
            user_features, product_features[pid], product_embeddings.get(pid), score, rank, pool_size, tt_encoder.max_price,
            include_price_features=include_price_features,
        )

    train_out: list[RankingExample] = []
    if train_examples:
        user_batch = tt_encoder.encode_user_batch([e.user_features for e in train_examples])
        embeddings = user_tower.predict(user_batch, verbose=0)
        results = vector_index.search(embeddings, k=pool_size)
        for ex, result in zip(train_examples, results):
            score, rank = _score_and_rank(result, ex.product_id)
            train_out.append(RankingExample(ex.user_id, ex.product_id, 1, _feature_row(ex.user_features, ex.product_id, score, rank)))

            exclude_ids = all_purchased_by_user.get(ex.user_id, frozenset())
            rng = _user_rng(ranking_config.random_seed, ex.user_id, salt=ex.product_id)
            for pid in _sample_negatives(result.item_ids, exclude_ids, ranking_config.negatives_per_positive, rng):
                score_n, rank_n = _score_and_rank(result, pid)
                train_out.append(RankingExample(ex.user_id, pid, 0, _feature_row(ex.user_features, pid, score_n, rank_n)))

    val_out: list[RankingExample] = []
    if val_cases:
        user_batch = tt_encoder.encode_user_batch([c.user_features for c in val_cases])
        embeddings = user_tower.predict(user_batch, verbose=0)
        results = vector_index.search(embeddings, k=pool_size)
        for case, result in zip(val_cases, results):
            exclude_ids = all_purchased_by_user.get(case.user_id, frozenset())
            for pid in case.target_ids:
                score, rank = _score_and_rank(result, pid)
                val_out.append(RankingExample(case.user_id, pid, 1, _feature_row(case.user_features, pid, score, rank)))

            rng = _user_rng(ranking_config.random_seed, case.user_id, salt=1)
            num_needed = ranking_config.negatives_per_positive * max(len(case.target_ids), 1)
            for pid in _sample_negatives(result.item_ids, exclude_ids, num_needed, rng):
                score, rank = _score_and_rank(result, pid)
                val_out.append(RankingExample(case.user_id, pid, 0, _feature_row(case.user_features, pid, score, rank)))

    return train_out, val_out


# --- primary offline evaluation: full serving pipeline -------------------

@dataclass
class PrimaryEvalReport:
    split_name: str
    num_cases: int
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    hit_rate_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    mrr: float
    mean_distinct_categories: float
    catalog_coverage: float
    mean_fill_rate: float


def evaluate_primary_pipeline(
    cases: list[TemporalEvalCase],
    split_name: str,
    product_features: dict[int, ProductFeatures],
    product_embeddings: dict[int, np.ndarray],
    all_item_ids: list[int],
    tt_encoder: TwoTowerFeatureEncoder,
    user_tower: tf.keras.Model,
    ranker_model: tf.keras.Model,
    vector_index: VectorIndex,
    config,
    top_n: int,
    k_values: list[int],
    eligible_catalog_size: int,
) -> PrimaryEvalReport:
    """Runs the SAME full serving pipeline
    (`serving.pipeline.generate_recommendations`: hard pre-retrieval
    eligibility -> Two-Tower/VectorIndex retrieval -> ranker ->
    re-ranking -> final eligibility validation -> Top-N) once per
    evaluation case, so the primary offline metric reflects what would
    actually be served - see docs/data-mapping.md section 16, "preserve
    the already-confirmed architecture."
    """
    from recommendation.serving.pipeline import generate_recommendations

    if not cases:
        return PrimaryEvalReport(split_name, 0, {}, {}, {}, {}, 0.0, 0.0, 0.0, 0.0)

    rankings: list[list[int]] = []
    relevant_sets: list[set[int]] = []
    distinct_categories: list[int] = []
    recommended_union: set[int] = set()
    fill_rates: list[float] = []

    for case in cases:
        result = generate_recommendations(
            case.user_features, product_features, product_embeddings, all_item_ids,
            tt_encoder, user_tower, ranker_model, vector_index, config, top_n,
        )
        rankings.append(result.product_ids)
        relevant_sets.append(set(case.target_ids))
        recommended_union.update(result.product_ids)
        fill_rates.append(result.fill_rate)
        cats = {product_features[pid].category_name for pid in result.product_ids if pid in product_features and product_features[pid].category_name}
        distinct_categories.append(len(cats))

    coverage = len(recommended_union) / eligible_catalog_size if eligible_catalog_size else 0.0

    return PrimaryEvalReport(
        split_name=split_name,
        num_cases=len(cases),
        precision_at_k={k: mean_precision_at_k(rankings, relevant_sets, k) for k in k_values},
        recall_at_k={k: mean_recall_at_k(rankings, relevant_sets, k) for k in k_values},
        hit_rate_at_k={k: mean_hit_rate_at_k(rankings, relevant_sets, k) for k in k_values},
        ndcg_at_k={k: mean_ndcg_at_k(rankings, relevant_sets, k) for k in k_values},
        mrr=mean_reciprocal_rank(rankings, relevant_sets),
        mean_distinct_categories=float(np.mean(distinct_categories)) if distinct_categories else 0.0,
        catalog_coverage=coverage,
        mean_fill_rate=float(np.mean(fill_rates)) if fill_rates else 0.0,
    )


# --- shared train-one-condition orchestration (STEP 8 controlled ablation) -

def _set_seeds(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _fit_two_tower_encoder(
    products: list[Product], users_adapter: UserAdapter, embedding_dim: int, include_price_features: bool
) -> TwoTowerFeatureEncoder:
    observed_age_groups = sorted(
        {p.age_group for uid in users_adapter.list_user_ids() if (p := users_adapter.get_user_profile(uid)) and p.age_group}
    )
    return TwoTowerFeatureEncoder.fit(
        category_names=[p.category_name for p in products if p.category_name],
        brand_names=[p.brand for p in products if p.brand],
        age_groups=observed_age_groups,
        prices=[p.price for p in products],
        embedding_dim=embedding_dim,
        include_price_features=include_price_features,
    )


def _examples_to_arrays(
    examples: list[TrainingExample],
    encoder: TwoTowerFeatureEncoder,
    product_features: dict[int, ProductFeatures],
    product_embeddings: dict[int, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    user_batch = encoder.encode_user_batch([e.user_features for e in examples])
    item_batch = encoder.encode_item_batch([e.product_id for e in examples], product_features, product_embeddings)
    return user_batch, item_batch


@dataclass
class TrainedCondition:
    """Everything one BASE/RECENCY+PRICE ablation condition (docs/data-
    mapping.md section 17) needs for evaluation - trained fresh by
    `train_temporal_condition`, or assembled from already-loaded STEP 7
    artifacts by the ablation script when reuse is legitimate (identical
    data/split/seed/config/procedure - see that module's docstring).
    """

    label: str
    encoder: TwoTowerFeatureEncoder
    user_tower: tf.keras.Model
    item_tower: tf.keras.Model
    ranker_model: tf.keras.Model
    item_ids: list[int]
    item_embeddings: np.ndarray
    vector_index: VectorIndex
    pool_size: int
    tt_history: dict
    ranker_history: dict
    num_tt_train_examples: int
    num_tt_val_loss_examples: int
    num_ranker_train_examples: int
    num_ranker_train_positive: int
    num_ranker_train_negative: int
    val_retrieval_report: TemporalRetrievalReport
    test_retrieval_report: TemporalRetrievalReport
    # The exact point-in-time eval cases this condition's `user_features`
    # were built from - needed by the caller for the PRIMARY full-pipeline
    # evaluation (`evaluate_primary_pipeline`). Built from the SAME
    # `splits`/target_ids for every condition (docs/data-mapping.md
    # section 17's "same evaluation population" requirement) - only the
    # `user_features` inside each case differ (recency/price config).
    val_cases: list[TemporalEvalCase]
    test_cases: list[TemporalEvalCase]


def train_temporal_condition(
    label: str,
    events_by_user: dict[int, list[UserInteraction]],
    splits: dict[int, TemporalUserSplit],
    users_adapter: UserAdapter,
    reviews_adapter: ReviewAdapter,
    products: list[Product],
    product_lookup: dict[int, Product],
    product_features: dict[int, ProductFeatures],
    product_embeddings: dict[int, np.ndarray],
    feature_config: FeatureConfig,
    price_context: PriceCatalogContext | None,
    include_price_features: bool,
    two_tower_config: TwoTowerConfig,
    ranking_config: RankingConfig,
    retrieval_config: RetrievalConfig,
    embedding_dim: int,
    default_recommendation_count: int,
    k_values: list[int],
) -> TrainedCondition:
    """Trains ONE full condition (Two-Tower from scratch, retrieval index,
    ranker from scratch) against the SAME `events_by_user`/`splits`/
    `product_*` a caller supplies - every controllable variable OTHER than
    `feature_config`/`price_context`/`include_price_features` must be
    identical between two calls for a fair ablation (docs/data-mapping.md
    section 17's fairness requirements) - this function does not enforce
    that itself, the CALLER (`scripts/run_ablation.py`) is responsible for
    passing identical `two_tower_config`/`ranking_config`/`retrieval_config`/
    seeds across conditions.

    `price_context=None` for a condition that should derive NO price
    preference information from purchase history at all (BASE) - not just
    an encoder that ignores it (see `features.price`/`build_user_features`'s
    own opt-in-via-`price_context` design, reused unchanged here).
    """
    _set_seeds(two_tower_config.random_seed)
    train_examples, val_loss_examples, val_cases, test_cases = build_temporal_two_tower_examples(
        events_by_user, splits, users_adapter, reviews_adapter, product_lookup, product_embeddings, feature_config, price_context
    )
    if not train_examples:
        raise ValueError(f"[{label}] no temporal training examples constructed")

    encoder = _fit_two_tower_encoder(products, users_adapter, embedding_dim, include_price_features)

    train_user_batch, train_item_batch = _examples_to_arrays(train_examples, encoder, product_features, product_embeddings)
    val_user_batch, val_item_batch = (
        _examples_to_arrays(val_loss_examples, encoder, product_features, product_embeddings) if val_loss_examples else (None, None)
    )

    effective_batch_size = min(two_tower_config.batch_size, len(train_examples))
    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_user_batch, train_item_batch))
        .shuffle(buffer_size=len(train_examples), seed=two_tower_config.random_seed)
        .batch(effective_batch_size, drop_remainder=True)
    )
    val_ds = None
    if val_loss_examples:
        val_batch_size = min(two_tower_config.batch_size, len(val_loss_examples))
        val_ds = tf.data.Dataset.from_tensor_slices((val_user_batch, val_item_batch)).batch(val_batch_size, drop_remainder=True)

    user_tower = build_user_tower(encoder, two_tower_config)
    item_tower = build_item_tower(encoder, two_tower_config)
    tt_model = TwoTowerModel(user_tower=user_tower, item_tower=item_tower, temperature=two_tower_config.temperature)
    tt_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=two_tower_config.learning_rate))

    callbacks = []
    if val_ds is not None:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=two_tower_config.early_stopping_patience, restore_best_weights=True)
        )
    fit_kwargs = {"epochs": two_tower_config.epochs, "verbose": 0, "callbacks": callbacks}
    if val_ds is not None:
        fit_kwargs["validation_data"] = val_ds
    tt_history = tt_model.fit(train_ds, **fit_kwargs)

    item_ids = [p.id for p in products]
    item_batch_full = encoder.encode_item_batch(item_ids, product_features, product_embeddings)
    item_embeddings = np.asarray(item_tower.predict(item_batch_full, verbose=0))

    val_retrieval_report = evaluate_temporal_retrieval(val_cases, f"{label}/val", user_tower, item_embeddings, item_ids, encoder, k_values)
    test_retrieval_report = evaluate_temporal_retrieval(test_cases, f"{label}/test", user_tower, item_embeddings, item_ids, encoder, k_values)

    vector_index = build_vector_index(retrieval_config)
    vector_index.build(item_ids, item_embeddings)

    _set_seeds(ranking_config.random_seed)
    all_purchased_by_user = all_purchased_product_ids_by_user(events_by_user)
    pool_size = candidate_pool_size(retrieval_config, limit=default_recommendation_count, catalog_size=len(item_ids))

    rk_train_examples, rk_val_loss_examples = build_temporal_ranking_dataset(
        train_examples, val_cases, all_purchased_by_user, product_features, product_embeddings,
        encoder, user_tower, vector_index, pool_size, ranking_config, include_price_features=include_price_features,
    )
    if not rk_train_examples:
        raise ValueError(f"[{label}] no ranker training examples constructed")

    X_train = np.stack([e.features for e in rk_train_examples])
    y_train = np.array([e.label for e in rk_train_examples], dtype=np.float32)
    num_pos = int(y_train.sum())
    num_neg = len(rk_train_examples) - num_pos

    X_val = y_val = None
    if rk_val_loss_examples:
        X_val = np.stack([e.features for e in rk_val_loss_examples])
        y_val = np.array([e.label for e in rk_val_loss_examples], dtype=np.float32)

    ranker_model = build_ranker_model(input_dim=X_train.shape[1], config=ranking_config)
    ranker_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=ranking_config.learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")],
    )
    rk_callbacks = []
    rk_fit_kwargs = {"epochs": ranking_config.epochs, "batch_size": min(ranking_config.batch_size, len(X_train)), "verbose": 0}
    if X_val is not None and len(X_val) > 0:
        rk_callbacks.append(
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=ranking_config.early_stopping_patience, restore_best_weights=True)
        )
        rk_fit_kwargs["validation_data"] = (X_val, y_val)
    rk_fit_kwargs["callbacks"] = rk_callbacks
    ranker_history = ranker_model.fit(X_train, y_train, **rk_fit_kwargs)

    return TrainedCondition(
        label=label,
        encoder=encoder,
        user_tower=user_tower,
        item_tower=item_tower,
        ranker_model=ranker_model,
        item_ids=item_ids,
        item_embeddings=item_embeddings,
        vector_index=vector_index,
        pool_size=pool_size,
        tt_history=tt_history.history,
        ranker_history=ranker_history.history,
        num_tt_train_examples=len(train_examples),
        num_tt_val_loss_examples=len(val_loss_examples),
        num_ranker_train_examples=len(rk_train_examples),
        num_ranker_train_positive=num_pos,
        num_ranker_train_negative=num_neg,
        val_retrieval_report=val_retrieval_report,
        test_retrieval_report=test_retrieval_report,
        val_cases=val_cases,
        test_cases=test_cases,
    )
