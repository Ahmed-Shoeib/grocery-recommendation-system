"""Full-pipeline evaluation (Phase 7): reuses the SAME leave-one-out
`EvalCase`s (`retrieval.two_tower.examples.build_eval_cases`) as Phase 4/6
- one `generate_recommendations` call per case (using
`EvalCase.user_features`, which already excludes both held-out val AND
test products) serves both the val report and the test report, exactly
like Phase 4/6's own evaluation reused eval cases for both splits.

Reports the standard ranking metrics (reused unchanged from
`evaluation.retrieval_metrics`) plus pipeline-specific ones requested for
Phase 7: catalog coverage, intra-list category diversity, duplicate rate
(expected to be exactly 0 - re-ranking dedupes unconditionally; kept as a
live sanity check, not an assumption), fill rate (share of requested
slots actually filled after eligibility filtering), and the cold-start
tier distribution across evaluated users.
"""

from __future__ import annotations

from dataclasses import dataclass

from recommendation.evaluation.retrieval_metrics import (
    mean_hit_rate_at_k,
    mean_ndcg_at_k,
    mean_precision_at_k,
    mean_reciprocal_rank,
    mean_recall_at_k,
)
from recommendation.features.product_features import ProductFeatures
from recommendation.retrieval.two_tower.examples import EvalCase
from recommendation.serving.pipeline import RecommendationResult


@dataclass
class PipelineEvalReport:
    split_name: str
    num_cases: int
    ndcg_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    hit_rate_at_k: dict[int, float]
    mrr: float
    catalog_coverage: float
    mean_distinct_categories: float
    duplicate_rate: float
    fill_rate: float
    tier_counts: dict[str, int]


def _rankings_for(results: list[RecommendationResult], use_pre_rerank: bool) -> list[list[int]]:
    return [r.pre_rerank_product_ids if use_pre_rerank else r.product_ids for r in results]


def evaluate_pipeline(
    eval_cases: list[EvalCase],
    results: list[RecommendationResult],
    target_attr: str,  # "val_product_id" or "test_product_id"
    split_name: str,
    k_values: list[int],
    catalog_size: int,
    product_features: dict[int, ProductFeatures],
    use_pre_rerank: bool = False,
) -> PipelineEvalReport:
    """`use_pre_rerank=True` scores the ranker-only order (before
    re-ranking/eligibility) from the SAME pipeline run - an apples-to-
    apples "Phase 6 ranker equivalent" baseline for comparison, since it
    shares identical retrieval and ranker scoring with the full-pipeline
    report, differing only in whether re-ranking/eligibility ran.
    """
    if not eval_cases:
        return PipelineEvalReport(split_name, 0, {}, {}, {}, {}, 0.0, 0.0, 0.0, 0.0, 0.0, {})

    rankings = _rankings_for(results, use_pre_rerank)
    relevant_sets = [{getattr(case, target_attr)} for case in eval_cases]

    all_recommended: set[int] = set()
    distinct_category_counts: list[int] = []
    duplicate_flags: list[float] = []
    fill_ratios: list[float] = []
    tier_counts: dict[str, int] = {}

    for result, ranking in zip(results, rankings):
        # List-quality metrics (coverage/diversity/duplicates/fill) describe
        # the FINAL N-item deliverable, not the internal candidate pool -
        # `ranking` here can be the full pool (`use_pre_rerank=True`), so
        # truncate to what would actually have been returned as the top-N
        # for a fair baseline-vs-full-pipeline comparison on this basis.
        final_slice = ranking[: result.requested_n]
        all_recommended.update(final_slice)
        categories = {product_features[pid].category_name for pid in final_slice if product_features[pid].category_name}
        distinct_category_counts.append(len(categories))
        duplicate_flags.append(0.0 if len(set(final_slice)) == len(final_slice) else 1.0)
        fill_ratios.append(len(final_slice) / result.requested_n if result.requested_n > 0 else 0.0)
        tier_counts[result.tier.value] = tier_counts.get(result.tier.value, 0) + 1

    return PipelineEvalReport(
        split_name=split_name,
        num_cases=len(eval_cases),
        ndcg_at_k={k: mean_ndcg_at_k(rankings, relevant_sets, k) for k in k_values},
        precision_at_k={k: mean_precision_at_k(rankings, relevant_sets, k) for k in k_values},
        recall_at_k={k: mean_recall_at_k(rankings, relevant_sets, k) for k in k_values},
        hit_rate_at_k={k: mean_hit_rate_at_k(rankings, relevant_sets, k) for k in k_values},
        mrr=mean_reciprocal_rank(rankings, relevant_sets),
        catalog_coverage=len(all_recommended) / catalog_size if catalog_size > 0 else 0.0,
        mean_distinct_categories=sum(distinct_category_counts) / len(distinct_category_counts) if distinct_category_counts else 0.0,
        duplicate_rate=sum(duplicate_flags) / len(duplicate_flags) if duplicate_flags else 0.0,
        fill_rate=sum(fill_ratios) / len(fill_ratios) if fill_ratios else 0.0,
        tier_counts=tier_counts,
    )
