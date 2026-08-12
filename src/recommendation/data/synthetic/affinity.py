"""Persona -> product affinity scoring and weighted sampling.

Shared by order, cart, search, and chatbot generation so all four V1
signals are driven by the same underlying (noisy) user preference
structure - which is what makes them realistically correlated with each
other, not just individually correlated with the persona.
"""

from __future__ import annotations

import numpy as np

from recommendation.data.synthetic.personas import Persona
from recommendation.data.synthetic.raw_schemas import RawProduct

BASELINE_SCORE = 1.0


def product_affinity_scores(
    products: list[RawProduct],
    category_name_by_id: dict[int, str],
    tags_by_product_id: dict[int, list[str]],
    persona: Persona,
    preferred_brand: str | None,
    brand_affinity_bonus: float,
    rng: np.random.Generator,
    noise_scale: float,
) -> np.ndarray:
    """One affinity score per product, for a single user.

    score = baseline + persona category bias + persona tag bias
            + brand-affinity bonus (if applicable) + per-product Gaussian noise
    """
    scores = np.empty(len(products), dtype=float)
    noise = rng.normal(loc=0.0, scale=noise_scale, size=len(products))
    for i, product in enumerate(products):
        category_name = category_name_by_id.get(product.category_id, "")
        tags = tags_by_product_id.get(product.id, [])
        score = BASELINE_SCORE
        score += persona.category_weights.get(category_name, 0.0)
        score += sum(persona.tag_weights.get(t, 0.0) for t in tags)
        if preferred_brand is not None and product.brand == preferred_brand:
            score += brand_affinity_bonus
        scores[i] = score
    return scores + noise


def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = scores / temperature
    scaled = scaled - scaled.max()
    exp = np.exp(scaled)
    return exp / exp.sum()


def sample_product_ids(
    rng: np.random.Generator,
    product_ids: list[int],
    scores: np.ndarray,
    k: int,
    replace: bool = False,
) -> list[int]:
    """Weighted sample of `k` product ids using softmax(scores) as the
    sampling distribution. Never raises for k > len(product_ids) when
    replace=False - it silently clamps, since callers use random k draws.
    """
    if k <= 0 or not product_ids:
        return []
    k = min(k, len(product_ids)) if not replace else k
    probs = softmax(scores)
    idx = rng.choice(len(product_ids), size=k, replace=replace, p=probs)
    return [product_ids[i] for i in np.atleast_1d(idx)]
