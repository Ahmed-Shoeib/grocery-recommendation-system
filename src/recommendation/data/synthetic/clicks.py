"""Synthetic click-event generation.

CLICK is the fifth V1 engagement signal (docs/data-mapping.md section 4).
No backend table distinct from the future `User_events` activity log
exists yet, so - exactly like search/chatbot - this produces canonical
`ClickRecord`s directly, driven by the same persona/affinity structure as
orders/carts/searches/chatbot so click behavior is realistically
correlated with a user's other engagement, not independently random. A
future real click provider (`User_events` rows with `action_type ==
CLICK`) only needs to implement `ClickAdapter`
(`recommendation.data.adapters.click_adapter`); this generator is not
part of that interface.
"""

from __future__ import annotations

import numpy as np

from recommendation.data.schemas.engagement import ClickRecord
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


def generate_click_records(
    users: list[RawUser],
    latent_profiles: dict[int, UserLatentProfile],
    products: list[RawProduct],
    product_tags: list[RawProductTag],
    tags: list[RawTag],
    rng: np.random.Generator,
    config: SyntheticDataConfig,
) -> list[ClickRecord]:
    tags_by_product = _tags_by_product_id(product_tags, tags)
    product_ids = [p.id for p in products]

    records: list[ClickRecord] = []
    for user in users:
        latent = latent_profiles[user.id]
        persona = PERSONA_BY_KEY[latent.persona_key]
        scores = product_affinity_scores(
            products, CATEGORY_NAME_BY_ID, tags_by_product, persona,
            latent.preferred_brand, config.brand_affinity_bonus, rng, config.affinity_noise_scale,
        )

        num_clicks = min(int(rng.poisson(config.click_count_lambda * latent.activity_level)), config.click_max_count)
        clicked_ids = sample_product_ids(rng, product_ids, scores, k=num_clicks, replace=True)
        for product_id in clicked_ids:
            records.append(ClickRecord(user_id=user.id, product_id=product_id, source="synthetic"))

    return records
