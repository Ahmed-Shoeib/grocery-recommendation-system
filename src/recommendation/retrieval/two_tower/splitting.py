"""Per-user leave-one-out train/val/test split over DISTINCT purchased
product ids.

NOT a temporal split - the legacy synthetic dataset has no usable
timestamps (docs/data-mapping.md section 1/7), so this uses a
config-seeded random holdout instead, which
is the standard evaluation protocol for implicit-feedback recommenders
without timestamps (leave-one-out, as in He et al.'s NCF and much of the
following literature) rather than a stand-in that quietly pretends to be
temporal.

Split unit is the DISTINCT product id, not the raw purchase record: a
user who bought the same product twice has both records governed by one
train/val/test assignment, matching how `build_user_features`'s
`exclude_product_ids` scrubs by product id (excluding a product removes
every record referencing it, not just one).

Users below `min_distinct_products_for_holdout` contribute all of their
purchases to train only and are excluded from evaluation - there is
nothing meaningful to hold out from a 1-2 item history without erasing
the only signal available for that user.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recommendation.data.schemas.engagement import EngagementProfile


@dataclass
class UserSplit:
    user_id: int
    train_product_ids: list[int]
    val_product_id: int | None
    test_product_id: int | None

    @property
    def held_out_ids(self) -> frozenset[int]:
        ids = set()
        if self.val_product_id is not None:
            ids.add(self.val_product_id)
        if self.test_product_id is not None:
            ids.add(self.test_product_id)
        return frozenset(ids)

    @property
    def is_evaluable(self) -> bool:
        return self.val_product_id is not None and self.test_product_id is not None


def build_user_splits(
    engagement_profiles: dict[int, EngagementProfile],
    min_distinct_products_for_holdout: int,
    random_seed: int,
) -> dict[int, UserSplit]:
    rng = np.random.default_rng(random_seed)
    splits: dict[int, UserSplit] = {}

    for user_id, profile in engagement_profiles.items():
        distinct_products = sorted({p.product_id for p in profile.purchases})
        if not distinct_products:
            splits[user_id] = UserSplit(user_id=user_id, train_product_ids=[], val_product_id=None, test_product_id=None)
            continue

        if len(distinct_products) < min_distinct_products_for_holdout:
            splits[user_id] = UserSplit(
                user_id=user_id, train_product_ids=distinct_products, val_product_id=None, test_product_id=None
            )
            continue

        shuffled = list(distinct_products)
        rng.shuffle(shuffled)
        test_id, val_id, *train_ids = shuffled
        splits[user_id] = UserSplit(
            user_id=user_id, train_product_ids=sorted(train_ids), val_product_id=val_id, test_product_id=test_id
        )

    return splits
