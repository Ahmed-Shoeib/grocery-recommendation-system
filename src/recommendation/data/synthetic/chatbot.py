"""Synthetic chatbot-context generation.

No chatbot entity exists in the ERD (docs/data-mapping.md section 4), so
this produces canonical `ChatbotContextRecord`s directly. A future real
chatbot backend (API response, DB entity, or event summary) only needs to
implement `ChatbotContextAdapter`
(`recommendation.data.adapters.chatbot_adapter`); this generator is not
part of that interface. `summary` text is plain, short, and template-based
so it stays a reasonable stand-in for Sentence Transformer encoding in
Phase 3, without pretending to be a real NLU-generated summary.
"""

from __future__ import annotations

import numpy as np

from recommendation.data.schemas.engagement import ChatbotContextRecord
from recommendation.data.synthetic.affinity import product_affinity_scores, sample_product_ids
from recommendation.data.synthetic.catalog import CATEGORY_NAME_BY_ID
from recommendation.data.synthetic.personas import PERSONA_BY_KEY
from recommendation.data.synthetic.raw_schemas import RawProduct, RawProductTag, RawTag, RawUser
from recommendation.data.synthetic.users import UserLatentProfile
from recommendation.utils.config import SyntheticDataConfig


def _tags_by_product_id(product_tags: list[RawProductTag], tags: list[RawTag]) -> dict[int, list[str]]:
    tag_name_by_id = {t.id: t.name for t in tags}
    out: dict[int, list[str]] = {}
    for pt in product_tags:
        out.setdefault(pt.product_id, []).append(tag_name_by_id[pt.tag_id])
    return out


def generate_chatbot_records(
    users: list[RawUser],
    latent_profiles: dict[int, UserLatentProfile],
    products: list[RawProduct],
    product_tags: list[RawProductTag],
    tags: list[RawTag],
    rng: np.random.Generator,
    config: SyntheticDataConfig,
) -> list[ChatbotContextRecord]:
    tags_by_product = _tags_by_product_id(product_tags, tags)
    product_ids = [p.id for p in products]

    records: list[ChatbotContextRecord] = []
    for user in users:
        if rng.random() >= config.chatbot_context_prob:
            continue

        latent = latent_profiles[user.id]
        persona = PERSONA_BY_KEY[latent.persona_key]
        scores = product_affinity_scores(
            products, CATEGORY_NAME_BY_ID, tags_by_product, persona,
            latent.preferred_brand, config.brand_affinity_bonus, rng, config.affinity_noise_scale,
        )
        num_mentions = int(rng.integers(1, config.chatbot_mentions_max + 1))
        mentioned_ids = sample_product_ids(rng, product_ids, scores, k=num_mentions, replace=False)

        category_name = persona.dominant_categories[int(rng.integers(0, len(persona.dominant_categories)))]
        adjectives = list(persona.adjectives)
        summary = f"User asked for {', '.join(adjectives)} {category_name.lower()} products."

        records.append(
            ChatbotContextRecord(
                user_id=user.id,
                mentioned_product_ids=mentioned_ids,
                preferred_category=category_name,
                product_interest=category_name,
                summary=summary,
                keywords=adjectives,
                source="synthetic",
            )
        )

    return records
