"""Lightweight re-ranking, cold-start blending, and final validation (Phase 7).

Handles duplicate removal, category/brand diversity, the three-level
personalization strategy (strong/sparse/no-history blending), and a final
availability safety check before returning Top-N.
"""
