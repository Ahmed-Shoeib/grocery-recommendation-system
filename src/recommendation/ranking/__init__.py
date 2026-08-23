"""Neural ranking of retrieved candidates.

Scores the candidates `VectorIndex.search` returns (`ranking.examples`)
using enriched per-candidate features - retrieval score/rank,
category/brand affinity match, semantic similarity, price/discount/
popularity/rating, stock/active (`ranking.features`) - through a plain
MLP (`ranking.model`). Eligibility filtering (`serving.eligibility`) is
the serving pipeline's job, not the ranker's: stock/active are available
as features but never used to exclude a candidate in this module.
"""
