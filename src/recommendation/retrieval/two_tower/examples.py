"""Builds Two-Tower training pairs and evaluation-time user vectors from
`UserSplit` + `EngagementProfile` + engineered features, applying the
target-leakage exclusion at every step (docs/data-mapping.md section 12).

Two distinct exclusion levels, both via `build_user_features`'s
`exclude_product_ids`:

- Training examples: for user u's train-split product p, the features
  used to predict (u, p) exclude BOTH u's held-out val/test products
  (they must never influence training) AND p itself (leave-one-out, so
  the model can't just read "the label" out of its own input).
- Evaluation cases: one feature vector per evaluable user, excluding
  ONLY the held-out val+test products - representing what the model
  legitimately knows about that user before being asked to retrieve the
  held-out items.
"""

from __future__ import annotations

from dataclasses import dataclass

from recommendation.data.schemas.engagement import EngagementProfile
from recommendation.data.schemas.product import Product
from recommendation.features.price import PriceCatalogContext
from recommendation.features.user_features import UserFeatures, build_user_features
from recommendation.retrieval.two_tower.splitting import UserSplit
from recommendation.utils.config import FeatureConfig


@dataclass
class TrainingExample:
    user_id: int
    product_id: int
    user_features: UserFeatures


@dataclass
class EvalCase:
    user_id: int
    val_product_id: int
    test_product_id: int
    user_features: UserFeatures  # built with BOTH val and test excluded


def build_training_examples(
    engagement_profiles: dict[int, EngagementProfile],
    splits: dict[int, UserSplit],
    product_lookup: dict[int, Product],
    product_embeddings: dict[int, np.ndarray],
    feature_config: FeatureConfig,
    text_embeddings: dict[str, np.ndarray],
    price_context: PriceCatalogContext | None = None,
) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for user_id, split in splits.items():
        if not split.train_product_ids:
            continue
        profile = engagement_profiles[user_id]
        held_out = split.held_out_ids
        for product_id in split.train_product_ids:
            exclude = frozenset(held_out | {product_id})
            user_features = build_user_features(
                profile,
                product_lookup,
                product_embeddings,
                feature_config,
                text_embeddings=text_embeddings,
                exclude_product_ids=exclude,
                price_context=price_context,
            )
            examples.append(TrainingExample(user_id=user_id, product_id=product_id, user_features=user_features))
    return examples


def build_eval_cases(
    engagement_profiles: dict[int, EngagementProfile],
    splits: dict[int, UserSplit],
    product_lookup: dict[int, Product],
    product_embeddings: dict[int, np.ndarray],
    feature_config: FeatureConfig,
    text_embeddings: dict[str, np.ndarray],
    price_context: PriceCatalogContext | None = None,
) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for user_id, split in splits.items():
        if not split.is_evaluable:
            continue
        profile = engagement_profiles[user_id]
        user_features = build_user_features(
            profile,
            product_lookup,
            product_embeddings,
            feature_config,
            text_embeddings=text_embeddings,
            exclude_product_ids=split.held_out_ids,
            price_context=price_context,
        )
        cases.append(
            EvalCase(
                user_id=user_id,
                val_product_id=split.val_product_id,
                test_product_id=split.test_product_id,
                user_features=user_features,
            )
        )
    return cases
