"""Duplicate removal and category/brand diversity re-ranking (Phase 7).

`candidates.RankedCandidate` is the shared type produced by
`serving.pipeline` (from personalized ranking or a cold-start fallback/
blend source - see `serving.fallback`) and consumed by
`diversity.rerank`. The three-level personalization strategy (strong/
sparse/no-history) and the final business-rules/eligibility check
(isActive, stockQuantity), applied LAST after this stage (docs
/data-mapping.md section 5), live in `serving.cold_start`/
`serving.fallback`/`serving.eligibility` and are orchestrated by
`serving.pipeline.generate_recommendations` - this package only
transforms an already-scored candidate list, it doesn't look up
business/catalog data itself.
"""
