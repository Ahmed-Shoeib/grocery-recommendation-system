"""Neural ranking of retrieved candidates (Phase 6).

Scores the candidates Phase 5's `VectorIndex.search` actually returns
(`ranking.examples`), using enriched per-candidate features - retrieval
score/rank, category/brand affinity match, semantic similarity, price/
discount/popularity/rating, stock/active (`ranking.features`) - through a
plain MLP (`ranking.model`). Business rules/eligibility (isActive/
stockQuantity) still run AFTER ranking and re-ranking in V1 (see
docs/data-mapping.md section 5), not as a pre-filter here - no candidate
is dropped before this stage; stock/active are available as features but
never used to exclude a candidate in this module.
"""
