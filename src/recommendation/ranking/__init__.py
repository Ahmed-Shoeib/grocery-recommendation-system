"""Neural ranking of retrieved candidates (Phase 6).

Consumes enriched candidate features (two-tower score, category/brand
affinity, purchase/cart/search/chatbot affinity, review signal, price,
discount, popularity) and produces fine-grained scores. Business
rules/eligibility (isActive/stockQuantity) run AFTER ranking and
re-ranking in V1 (see docs/data-mapping.md section 5), not as a pre-filter
here - no candidate is dropped before this stage.
"""
