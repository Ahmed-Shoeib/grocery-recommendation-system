"""Shared inference/service layer used by both the API and the dashboard.

`pipeline.generate_recommendations` runs the full serving pipeline (hard
pre-retrieval eligibility -> Two-Tower -> VectorIndex -> Neural Ranker ->
Re-ranking -> remaining business rules -> final lightweight eligibility
validation -> Final Top-N - see docs/data-mapping.md section 5) as a pure
function of an already-built `UserFeatures`; `pipeline.recommend(user_id,
..., limit, context=None)` is the adapter-backed convenience wrapper.
`context` is an extensibility hook for future recommendation surfaces
(home, product detail, cart, search), unused by any current model.

Three-level cold-start tiering lives in `cold_start.py`, non-personalized
fallback candidate sources in `fallback.py`, and the eligibility policy
(applied both pre-retrieval and as a final validation) in
`eligibility.py` - all orchestrated by `pipeline.py`, never duplicated
inline.
"""
