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
from recommendation.retrieval.index.base import SearchResult, VectorIndex
from recommendation.retrieval.two_tower.evaluation import rank_all_items
from recommendation.retrieval.two_tower.examples import TrainingExample
from recommendation.retrieval.two_tower.feature_encoding import TwoTowerFeatureEncoder
from recommendation.utils.config import FeatureConfig, RankingConfig


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
) -> tuple[list[RankingExample], list[RankingExample]]:
    """Returns `(train_ranking_examples, val_loss_ranking_examples)`.

    Negative candidates are drawn from what `vector_index.search` actually
    retrieves for that example's own point-in-time query embedding (same
    "retrieved negatives, not raw catalog" philosophy as the non-temporal
    `ranking.examples.build_ranking_dataset`), MINUS `all_purchased_by_user
    [user_id]` - every product this user EVER purchases, past or future
    relative to this example's cutoff - so a future purchase can never
    become a mislabeled hard negative (module docstring).
    """

    def _feature_row(user_features: UserFeatures, pid: int, score: float, rank: int) -> np.ndarray:
        return build_ranking_feature_vector(
            user_features, product_features[pid], product_embeddings.get(pid), score, rank, pool_size, tt_encoder.max_price,
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