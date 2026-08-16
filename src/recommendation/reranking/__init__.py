"""Duplicate removal and category/brand diversity re-ranking (Phase 7).

`candidates.RankedCandidate` is the shared type produced by
`serving.pipeline` (from personalized ranking or a cold-start fallback/
blend source - see `serving.fallback`) and consumed by
`diversity.rerank`. The three-level personalization strategy (strong/
sparse/no-history) and the eligibility check (isActive, stockQuantity)
live in `serving.cold_start`/`serving.fallback`/`serving.eligibility` and
are orchestrated by `serving.pipeline.generate_recommendations` - this
package only transforms an already-scored candidate list, it doesn't
look up business/catalog data itself. Since Phase 11, eligibility is a
HARD pre-retrieval gate applied before this package ever sees a
candidate (every id reaching `rerank` is already eligible), plus a final
lightweight re-validation applied after it (docs/data-mapping.md
section 5) - `rerank` itself is unaffected either way, it never filters.
"""
