"""Exact (brute-force) retrieval evaluation for the Two-Tower model.

Brute-force, not ANN: FAISS/ScaNN retrieval lives in `retrieval.index`.
At ~50 catalog items this is trivial (one (n_users x 128) @ (128 x
n_items) matmul) and gives an exact upper bound the ANN retrieval
backends can be compared against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from recommendation.evaluation.retrieval_metrics import mean_hit_rate_at_k, mean_recall_at_k
from recommendation.retrieval.two_tower.examples import EvalCase
from recommendation.retrieval.two_tower.feature_encoding import TwoTowerFeatureEncoder


@dataclass
class RetrievalEvalReport:
    split_name: str  # "val" or "test"
    num_cases: int
    recall_at_k: dict[int, float]
    hit_rate_at_k: dict[int, float]


def rank_all_items(user_embeddings: np.ndarray, item_embeddings: np.ndarray, item_ids: list[int]) -> list[list[int]]:
    """For each user embedding, return `item_ids` sorted by descending
    cosine similarity (both inputs are already L2-normalized, so this is
    a plain dot product).
    """
    scores = user_embeddings @ item_embeddings.T  # (n_users, n_items)
    order = np.argsort(-scores, axis=1)
    return [[item_ids[i] for i in row] for row in order]


def evaluate_split(
    cases: list[EvalCase],
    target_attr: str,  # "val_product_id" or "test_product_id"
    split_name: str,
    user_tower: tf.keras.Model,
    item_embeddings: np.ndarray,
    item_ids: list[int],
    encoder: TwoTowerFeatureEncoder,
    k_values: list[int],
) -> RetrievalEvalReport:
    if not cases:
        return RetrievalEvalReport(split_name=split_name, num_cases=0, recall_at_k={}, hit_rate_at_k={})

    user_batch = encoder.encode_user_batch([c.user_features for c in cases])
    user_embeddings = user_tower.predict(user_batch, verbose=0)

    rankings = rank_all_items(user_embeddings, item_embeddings, item_ids)
    relevant_sets = [{getattr(c, target_attr)} for c in cases]

    recall = {k: mean_recall_at_k(rankings, relevant_sets, k) for k in k_values}
    hit_rate = {k: mean_hit_rate_at_k(rankings, relevant_sets, k) for k in k_values}
    return RetrievalEvalReport(split_name=split_name, num_cases=len(cases), recall_at_k=recall, hit_rate_at_k=hit_rate)
