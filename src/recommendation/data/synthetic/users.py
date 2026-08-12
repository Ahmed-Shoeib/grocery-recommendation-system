"""Synthetic user generation with latent persona assignment.

Produces `RawUser` rows (the ERD-shaped table, including the confirmed
`preferred_category_id` / `age_group` fields) plus a parallel
`UserLatentProfile` per user - the persona/brand-affinity/activity-level
assignment used only during generation (never exposed to adapters,
features, or models) to keep the four V1 signals statistically correlated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recommendation.data.synthetic.catalog import ALL_BRANDS, CATEGORY_ID_BY_NAME
from recommendation.data.synthetic.personas import AGE_GROUPS, PERSONAS, Persona
from recommendation.data.synthetic.raw_schemas import RawUser
from recommendation.utils.config import SyntheticDataConfig


@dataclass(frozen=True)
class UserLatentProfile:
    """Generation-only latent state for one user. Not a canonical/backend
    concept - exists so interaction generators (orders/cart/search/chatbot)
    can share one consistent, noisy preference structure per user, and so
    tests can verify persona correlation actually holds.
    """

    user_id: int
    persona_key: str
    preferred_brand: str | None
    activity_level: float  # >0, scales event counts (purchase/cart/search frequency)


def _weighted_choice(rng: np.random.Generator, keys: list[str], weights: dict[str, float]) -> str:
    w = np.array([weights.get(k, 0.0) for k in keys], dtype=float)
    w = np.clip(w, 0.05, None)  # floor so nothing is ever truly impossible
    probs = w / w.sum()
    return keys[int(rng.choice(len(keys), p=probs))]


def generate_users(
    n: int,
    rng: np.random.Generator,
    config: SyntheticDataConfig,
) -> tuple[list[RawUser], dict[int, UserLatentProfile]]:
    users: list[RawUser] = []
    latent_profiles: dict[int, UserLatentProfile] = {}

    n_incomplete = int(round(n * config.incomplete_profile_fraction))
    incomplete_user_ids = set(rng.choice(np.arange(1, n + 1), size=n_incomplete, replace=False)) if n_incomplete else set()

    for i in range(n):
        user_id = i + 1
        persona: Persona = PERSONAS[int(rng.integers(0, len(PERSONAS)))]

        age_group = _weighted_choice(rng, AGE_GROUPS, persona.age_group_weights)

        if rng.random() < config.preferred_category_alignment_prob:
            category_name = persona.dominant_categories[int(rng.integers(0, len(persona.dominant_categories)))]
        else:
            category_name = list(CATEGORY_ID_BY_NAME.keys())[int(rng.integers(0, len(CATEGORY_ID_BY_NAME)))]
        preferred_category_id = CATEGORY_ID_BY_NAME[category_name]

        preferred_brand = None
        if rng.random() < config.brand_affinity_prob:
            preferred_brand = ALL_BRANDS[int(rng.integers(0, len(ALL_BRANDS)))]

        activity_level = float(rng.uniform(0.3, 1.0))

        is_incomplete = user_id in incomplete_user_ids
        users.append(
            RawUser(
                id=user_id,
                first_name=f"User{user_id}",
                last_name="Synthetic",
                email=f"user{user_id}@synthetic.invalid",
                preferred_category_id=None if is_incomplete else preferred_category_id,
                age_group=None if is_incomplete else age_group,
            )
        )
        latent_profiles[user_id] = UserLatentProfile(
            user_id=user_id,
            persona_key=persona.key,
            preferred_brand=preferred_brand,
            activity_level=activity_level,
        )

    return users, latent_profiles
