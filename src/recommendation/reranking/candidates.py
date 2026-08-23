"""Shared candidate type threaded through the serving pipeline: produced
by `serving.pipeline` (from personalized ranking or a fallback/blend
source), consumed by `reranking.diversity`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RankedCandidate:
    product_id: int
    score: float
    source: str  # "personalized" | "preferred_category" | "category_popularity" | "global_popularity"
