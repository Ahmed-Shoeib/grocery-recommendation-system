"""Synthetic search-history generation.

No `SearchHistory` table exists in the ERD (docs/data-mapping.md section
4), so this produces canonical `SearchRecord`s directly - there is no raw
backend shape to mirror. A future real search provider (API/table/event
pipeline) only needs to implement `SearchAdapter`
(`recommendation.data.adapters.search_adapter`); this generator is not
part of that interface.
"""

from __future__ import annotations

import numpy as np

from recommendation.data.schemas.engagement import SearchRecord
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


def generate_search_records(
    users: list[RawUser],
    latent_profiles: dict[int, UserLatentProfile],
    products: list[RawProduct],
    product_tags: list[RawProductTag],
    tags: list[RawTag],
    rng: np.random.Generator,
    config: SyntheticDataConfig,
) -> list[SearchRecord]:
    tags_by_product = _tags_by_product_id(product_tags, tags)
    product_ids = [p.id for p in products]
    product_by_id = {p.id: p for p in products}

    records: list[SearchRecord] = []
    for user in users:
        latent = latent_profiles[user.id]
        persona = PERSONA_BY_KEY[latent.persona_key]
        scores = product_affinity_scores(
            products, CATEGORY_NAME_BY_ID, tags_by_product, persona,
            latent.preferred_brand, config.brand_affinity_bonus, rng, config.affinity_noise_scale,
        )

        num_searches = min(int(rng.poisson(config.search_count_lambda * latent.activity_level)), config.search_max_count)
        for _ in range(num_searches):
            if rng.random() < config.search_product_term_prob:
                product_id = sample_product_ids(rng, product_ids, scores, k=1, replace=False)[0]
                product = product_by_id[product_id]
                records.append(
                    SearchRecord(
                        user_id=user.id,
                        search_term=product.name.lower(),
                        matched_product_id=product_id,
                        source="synthetic",
                    )
                )
            else:
                adjective = persona.adjectives[int(rng.integers(0, len(persona.adjectives)))]
                category_name = persona.dominant_categories[int(rng.integers(0, len(persona.dominant_categories)))]
                records.append(
                    SearchRecord(
                        user_id=user.id,
                        search_term=f"{adjective} {category_name.lower()}",
                        matched_product_id=None,
                        source="synthetic",
                    )
                )

    return records
