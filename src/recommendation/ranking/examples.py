"""Builds ranking training/validation examples and evaluation cases from
retrieval-stage output (the trained Two-Tower model + a Phase 5
`VectorIndex`), applying the same target-leakage discipline as Phase 4's
`retrieval.two_tower.examples` (leave-one-out per positive) and reusing
the SAME `UserSplit`s as Two-Tower training so held-out val/test targets
are identical between retrieval and ranking evaluation - required for the
ranker-vs-raw-retrieval-score baseline comparison to be apples-to-apples.

Three embedding/retrieval passes, each batched (one Keras `predict` +
one `VectorIndex.search` call, regardless of dataset size):

1. "shared" query embeddings, one per trainable user, excluding only
   that user's held-out val/test products - used to source TRAIN
   negatives (labels sampled from what the VectorIndex actually
   retrieves, not the raw catalog, so training candidate distribution
   matches serving-time candidate distribution).
2. "leave-one-out" query embeddings, one per (user, train-positive)
   pair, additionally excluding that specific positive - used only to
   compute a leakage-safe retrieval_score/rank FEATURE for TRAIN
   positives (a positive's own presence in the user's history must not
   inflate its own retrieval similarity).
3. Two-Tower's own `EvalCase` embeddings (evaluable users, excluding
   both val AND test) - reused directly for the VAL-LOSS positive/
   negatives (val_product_id only, never test_product_id) and for the
   final ranking-quality evaluation candidate pool (both val and test
   reports) - see `ranking.evaluation`.

Negatives always come from what `VectorIndex.search` actually retrieves,
minus known positives and held-out ids - never from the raw catalog -
the retrieved-negatives pattern used to reduce train/serve skew in
real two-stage retrieval-then-rank systems.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from recommendation.data.schemas.engagement import EngagementProfile
from recommendation.data.schemas.product import Product
from recommendation.features.product_features import ProductFeatures
from recommendation.features.user_features import UserFeatures, build_user_features
from recommendation.ranking.features import build_ranking_feature_vector
from recommendation.retrieval.index.base import SearchResult, VectorIndex
from recommendation.retrieval.two_tower.examples import build_eval_cases
from recommendation.retrieval.two_tower.feature_encoding import TwoTowerFeatureEncoder
from recommendation.retrieval.two_tower.splitting import UserSplit
from recommendation.utils.config import FeatureConfig, RankingConfig


@dataclass
class RankingExample:
    user_id: int
    product_id: int
    label: int  # 1 = user purchased it (train-visible), 0 = sampled non-purchased candidate
    features: np.ndarray


@dataclass
class RankingEvalCase:
    user_id: int
    val_product_id: int
    test_product_id: int
    candidate_ids: list[int]  # VectorIndex retrieval order (= the baseline ranking)
    retrieval_scores: list[float]  # aligned with candidate_ids
    features: np.ndarray  # shape (len(candidate_ids), num_ranking_features), row-aligned


def _user_rng(seed: int, user_id: int, salt: int = 0) -> np.random.Generator:
    return np.random.default_rng([seed, user_id, salt])


def _sample_negatives(
    candidate_ids: list[int], exclude_ids: set[int], num_needed: int, rng: np.random.Generator
) -> list[int]:
    pool = [pid for pid in candidate_ids if pid not in exclude_ids]
    if len(pool) <= num_needed:
        return pool
    chosen = rng.choice(len(pool), size=num_needed, replace=False)
    return [pool[i] for i in chosen]


def _score_and_rank(result: SearchResult, product_id: int) -> tuple[float, int]:
    """Returns (score, 0-indexed rank) of `product_id` within a
    VectorIndex search result. Falls back to a worst-case rank/score if
    absent (only possible when the candidate pool is smaller than the
    full catalog - see docs/data-mapping.md section 10 catalog-scale note).
    """
    if product_id in result.item_ids:
        rank = result.item_ids.index(product_id)
        return result.scores[rank], rank
    return (min(result.scores) if result.scores else 0.0), len(result.item_ids)


def build_ranking_dataset(
    engagement_profiles: dict[int, EngagementProfile],
    splits: dict[int, UserSplit],
    product_lookup: dict[int, Product],
    product_features: dict[int, ProductFeatures],
    product_embeddings: dict[int, np.ndarray],
    text_embeddings: dict[str, np.ndarray],
    feature_config: FeatureConfig,
    tt_encoder: TwoTowerFeatureEncoder,
    user_tower: tf.keras.Model,
    vector_index: VectorIndex,
    pool_size: int,
    ranking_config: RankingConfig,
) -> tuple[list[RankingExample], list[RankingExample], list[RankingEvalCase]]:
    trainable = {uid: split for uid, split in splits.items() if split.train_product_ids}

    def _features_for(user_id: int, exclude: frozenset[int]) -> UserFeatures:
        return build_user_features(
            engagement_profiles[user_id],
            product_lookup,
            product_embeddings,
            feature_config,
            text_embeddings=text_embeddings,
            exclude_product_ids=exclude,
        )

    def _feature_row(user_features: UserFeatures, product_id: int, score: float, rank: int) -> np.ndarray:
        return build_ranking_feature_vector(
            user_features,
            product_features[product_id],
            product_embeddings.get(product_id),
            score,
            rank,
            pool_size,
            tt_encoder.max_price,
        )

    # --- pass 1: shared (held-out-excluded) embeddings for every trainable user ---
    shared_uids = list(trainable.keys())
    shared_features = {uid: _features_for(uid, trainable[uid].held_out_ids) for uid in shared_uids}
    shared_results: dict[int, SearchResult] = {}
    if shared_uids:
        shared_batch = tt_encoder.encode_user_batch([shared_features[uid] for uid in shared_uids])
        shared_embeddings = user_tower.predict(shared_batch, verbose=0)
        shared_results = dict(zip(shared_uids, vector_index.search(shared_embeddings, k=pool_size)))

    # --- pass 2: leave-one-out embeddings, one per (user, train-positive) pair ---
    loo_pairs: list[tuple[int, int]] = [(uid, pid) for uid, split in trainable.items() for pid in split.train_product_ids]
    loo_features = [_features_for(uid, frozenset(trainable[uid].held_out_ids | {pid})) for uid, pid in loo_pairs]
    loo_results: list[SearchResult] = []
    if loo_pairs:
        loo_batch = tt_encoder.encode_user_batch(loo_features)
        loo_embeddings = user_tower.predict(loo_batch, verbose=0)
        loo_results = vector_index.search(loo_embeddings, k=pool_size)

    # --- pass 3: Two-Tower's own eval cases (evaluable users, val+test excluded) ---
    tt_eval_cases = build_eval_cases(
        engagement_profiles, splits, product_lookup, product_embeddings, feature_config, text_embeddings
    )
    eval_results: list[SearchResult] = []
    if tt_eval_cases:
        eval_batch = tt_encoder.encode_user_batch([c.user_features for c in tt_eval_cases])
        eval_embeddings = user_tower.predict(eval_batch, verbose=0)
        eval_results = vector_index.search(eval_embeddings, k=pool_size)

    # --- assemble TRAIN examples ---
    train_examples: list[RankingExample] = []

    for (uid, pid), user_features, result in zip(loo_pairs, loo_features, loo_results):
        score, rank = _score_and_rank(result, pid)
        train_examples.append(RankingExample(uid, pid, 1, _feature_row(user_features, pid, score, rank)))

    for uid, split in trainable.items():
        result = shared_results[uid]
        user_features = shared_features[uid]
        exclude_ids = set(split.train_product_ids) | set(split.held_out_ids)
        num_needed = ranking_config.negatives_per_positive * len(split.train_product_ids)
        for pid in _sample_negatives(result.item_ids, exclude_ids, num_needed, _user_rng(ranking_config.random_seed, uid)):
            score, rank = _score_and_rank(result, pid)
            train_examples.append(RankingExample(uid, pid, 0, _feature_row(user_features, pid, score, rank)))

    # --- assemble VAL-LOSS examples + final RankingEvalCases ---
    val_loss_examples: list[RankingExample] = []
    ranking_eval_cases: list[RankingEvalCase] = []

    for case, result in zip(tt_eval_cases, eval_results):
        split = splits[case.user_id]
        user_features = case.user_features

        val_score, val_rank = _score_and_rank(result, case.val_product_id)
        val_loss_examples.append(
            RankingExample(case.user_id, case.val_product_id, 1, _feature_row(user_features, case.val_product_id, val_score, val_rank))
        )

        exclude_ids = set(split.train_product_ids) | split.held_out_ids
        negatives = _sample_negatives(
            result.item_ids, exclude_ids, ranking_config.negatives_per_positive, _user_rng(ranking_config.random_seed, case.user_id, salt=1)
        )
        for pid in negatives:
            score, rank = _score_and_rank(result, pid)
            val_loss_examples.append(RankingExample(case.user_id, pid, 0, _feature_row(user_features, pid, score, rank)))

        case_features = np.stack(
            [_feature_row(user_features, cid, s, r) for r, (cid, s) in enumerate(zip(result.item_ids, result.scores))]
        )
        ranking_eval_cases.append(
            RankingEvalCase(
                user_id=case.user_id,
                val_product_id=case.val_product_id,
                test_product_id=case.test_product_id,
                candidate_ids=list(result.item_ids),
                retrieval_scores=list(result.scores),
                features=case_features,
            )
        )

    return train_examples, val_loss_examples, ranking_eval_cases
