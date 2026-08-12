"""Lightweight re-ranking, cold-start blending, and business rules / final
eligibility validation (Phase 7).

Handles duplicate removal, category/brand diversity, the three-level
personalization strategy (strong/sparse/no-history blending), and -
running LAST, after re-ranking (docs/data-mapping.md section 5) - the
business-rules/eligibility check (isActive, stockQuantity) before
returning Top-N. No earlier stage (retrieval, ranking) filters on these
fields in V1.
"""
