"""Ranking-quality evaluation over the SAME retrieved candidate pool for
every case, comparing the trained ranker's re-ordering against the
"do-nothing" baseline of just keeping the VectorIndex's own retrieval-
score ordering. Reuses `evaluation.retrieval_metrics` unchanged (it was
built generic enough for exactly this) and the same val/test leave-one-
out targets `retrieval.two_tower.evaluation` reports Recall@K/HitRate@K
against, so ranker vs. Two-Tower-only-retrieval is an apples-to-apples
comparison, not two different tasks.

At the current ~50-item catalog, `pool_size` (`retrieval.index.factory
.candidate_pool_size`) equals the full catalog, so ranking here is
effectively "re-rank the whole catalog," not "re-rank a genuinely
truncated ANN shortlist" - a real test of ranking-under-truncation needs
a larger catalog. See docs/data-mapping.md section 10.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from recommendation.evaluation.retrieval_metrics import (
    mean_hit_rate_at_k,
    mean_ndcg_at_k,
    mean_precision_at_k,
    mean_reciprocal_rank,
    mean_recall_at_k,
)
from recommendation.ranking.examples import RankingEvalCase


@dataclass
class RankingEvalReport:
    split_name: str
    num_cases: int
    ndcg_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    hit_rate_at_k: dict[int, float]
    mrr: float


def baseline_scores(case: RankingEvalCase) -> np.ndarray:
    """The retrieval-only baseline: candidates are already retrieval-score-
    sorted (`VectorIndex.search`'s own output order), so scoring by
    `retrieval_scores` unchanged reproduces that ordering exactly.
    """
    return np.asarray(case.retrieval_scores)


def _rank_by_score(candidate_ids: list[int], scores: np.ndarray) -> list[int]:
    order = np.argsort(-scores)
    return [candidate_ids[i] for i in order]


def evaluate_ranking(
    cases: list[RankingEvalCase],
    target_attr: str,  # "val_product_id" or "test_product_id"
    split_name: str,
    score_fn: Callable[[RankingEvalCase], np.ndarray],
    k_values: list[int],
) -> RankingEvalReport:
    if not cases:
        return RankingEvalReport(split_name=split_name, num_cases=0, ndcg_at_k={}, precision_at_k={}, recall_at_k={}, hit_rate_at_k={}, mrr=0.0)

    rankings = [_rank_by_score(case.candidate_ids, score_fn(case)) for case in cases]
    relevant_sets = [{getattr(case, target_attr)} for case in cases]

    return RankingEvalReport(
        split_name=split_name,
        num_cases=len(cases),
        ndcg_at_k={k: mean_ndcg_at_k(rankings, relevant_sets, k) for k in k_values},
        precision_at_k={k: mean_precision_at_k(rankings, relevant_sets, k) for k in k_values},
        recall_at_k={k: mean_recall_at_k(rankings, relevant_sets, k) for k in k_values},
        hit_rate_at_k={k: mean_hit_rate_at_k(rankings, relevant_sets, k) for k in k_values},
        mrr=mean_reciprocal_rank(rankings, relevant_sets),
    )
