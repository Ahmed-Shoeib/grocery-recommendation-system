"""Turns Phase 3's `UserFeatures`/`ProductFeatures` (+ Sentence Transformer
embeddings) into fixed-size numpy tensors the Keras towers consume.

`TwoTowerFeatureEncoder` is fit ONCE from the full product catalog (category
names, brands - static catalog metadata, not user behavior, so fitting on
the full catalog is not target leakage) plus the known age-group vocabulary,
then reused to encode every item/user example. It is serialized alongside
the model weights (Phase 5 needs the exact same vocab/normalization at
serving time) - see `serialization.py`.

Index 0 in every `Vocabulary` is reserved for "unknown/missing" so a
category, brand, or age group value not seen at fit time (or a genuinely
missing preferredCategory/ageGroup) degrades to a learnable "unknown"
embedding rather than raising.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from recommendation.features.product_features import ProductFeatures
from recommendation.features.user_features import UserFeatures


@dataclass
class Vocabulary:
    values: list[str] = field(default_factory=list)  # index i -> values[i-1]; index 0 = unknown
    index: dict[str, int] = field(default_factory=dict)

    @classmethod
    def fit(cls, values: list[str]) -> "Vocabulary":
        unique_sorted = sorted(set(v for v in values if v))
        return cls(values=unique_sorted, index={v: i + 1 for i, v in enumerate(unique_sorted)})

    @property
    def size(self) -> int:
        return len(self.values) + 1  # +1 for the unknown bucket at index 0

    def encode(self, value: str | None) -> int:
        if value is None:
            return 0
        return self.index.get(value, 0)

    def to_dict(self) -> dict:
        return {"values": self.values}

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        values = data["values"]
        return cls(values=values, index={v: i + 1 for i, v in enumerate(values)})


ITEM_NUMERIC_FEATURE_NAMES = [
    "normalized_price", "discount_fraction", "log_purchase_count",
    "log_cart_add_count", "log_review_count", "average_rating", "has_rating",
]
USER_NUMERIC_FEATURE_NAMES = [
    "log_purchase_count", "log_cart_item_count", "log_search_count", "log_total_engagement_events",
    "has_chatbot_context", "has_preferred_category", "has_age_group", "has_semantic_embedding",
]


@dataclass
class TwoTowerFeatureEncoder:
    embedding_dim: int
    category_vocab: Vocabulary
    brand_vocab: Vocabulary
    age_group_vocab: Vocabulary
    max_price: float  # catalog-level normalization stat, fit once (not leakage: item-side, static)

    @property
    def item_numeric_dim(self) -> int:
        return len(ITEM_NUMERIC_FEATURE_NAMES)

    @property
    def user_numeric_dim(self) -> int:
        return len(USER_NUMERIC_FEATURE_NAMES)

    @property
    def category_affinity_dim(self) -> int:
        return len(self.category_vocab.values)  # affinity vectors don't need an "unknown" slot

    @property
    def brand_affinity_dim(self) -> int:
        return len(self.brand_vocab.values)

    # --- fitting -----------------------------------------------------------

    @classmethod
    def fit(
        cls,
        category_names: list[str],
        brand_names: list[str],
        age_groups: list[str],
        prices: list[float],
        embedding_dim: int,
    ) -> "TwoTowerFeatureEncoder":
        return cls(
            embedding_dim=embedding_dim,
            category_vocab=Vocabulary.fit(category_names),
            brand_vocab=Vocabulary.fit(brand_names),
            age_group_vocab=Vocabulary.fit(age_groups),
            max_price=max(prices) if prices else 1.0,
        )

    # --- item encoding -------------------------------------------------------

    def encode_item(self, features: ProductFeatures, semantic_embedding: np.ndarray) -> dict[str, np.ndarray]:
        numeric = np.array(
            [
                min(features.effective_price / self.max_price, 1.0) if self.max_price > 0 else 0.0,
                (features.discount_percentage or 0.0) / 100.0,
                np.log1p(features.purchase_count),
                np.log1p(features.cart_add_count),
                np.log1p(features.review_count),
                features.average_rating if features.average_rating is not None else 0.0,
                1.0 if features.average_rating is not None else 0.0,
            ],
            dtype=np.float32,
        )
        return {
            "semantic_embedding": semantic_embedding.astype(np.float32),
            "category_id": np.int32(self.category_vocab.encode(features.category_name)),
            "brand_id": np.int32(self.brand_vocab.encode(features.brand)),
            "numeric": numeric,
        }

    def encode_item_batch(
        self, product_ids: list[int], product_features: dict[int, ProductFeatures], product_embeddings: dict[int, np.ndarray]
    ) -> dict[str, np.ndarray]:
        rows = [self.encode_item(product_features[pid], product_embeddings[pid]) for pid in product_ids]
        return self._stack(rows)

    # --- user encoding -------------------------------------------------------

    def encode_user(self, features: UserFeatures) -> dict[str, np.ndarray]:
        semantic = (
            features.semantic_embedding.astype(np.float32)
            if features.semantic_embedding is not None
            else np.zeros(self.embedding_dim, dtype=np.float32)
        )
        category_affinity = np.zeros(self.category_affinity_dim, dtype=np.float32)
        for name, weight in features.category_affinity.items():
            idx = self.category_vocab.encode(name)
            if idx > 0:  # unknown categories (idx 0) have no affinity-vector slot
                category_affinity[idx - 1] = weight

        brand_affinity = np.zeros(self.brand_affinity_dim, dtype=np.float32)
        for name, weight in features.brand_affinity.items():
            idx = self.brand_vocab.encode(name)
            if idx > 0:
                brand_affinity[idx - 1] = weight

        numeric = np.array(
            [
                np.log1p(features.purchase_count),
                np.log1p(features.cart_item_count),
                np.log1p(features.search_count),
                np.log1p(features.total_engagement_events),
                1.0 if features.has_chatbot_context else 0.0,
                1.0 if features.has_preferred_category else 0.0,
                1.0 if features.has_age_group else 0.0,
                1.0 if features.semantic_embedding is not None else 0.0,
            ],
            dtype=np.float32,
        )
        return {
            "semantic_embedding": semantic,
            "preferred_category_id": np.int32(self.category_vocab.encode(features.preferred_category)),
            "age_group_id": np.int32(self.age_group_vocab.encode(features.age_group)),
            "category_affinity": category_affinity,
            "brand_affinity": brand_affinity,
            "numeric": numeric,
        }

    def encode_user_batch(self, feature_list: list[UserFeatures]) -> dict[str, np.ndarray]:
        rows = [self.encode_user(f) for f in feature_list]
        return self._stack(rows)

    @staticmethod
    def _stack(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        if not rows:
            return {}
        return {key: np.stack([row[key] for row in rows]) for key in rows[0]}

    # --- serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "embedding_dim": self.embedding_dim,
            "max_price": self.max_price,
            "category_vocab": self.category_vocab.to_dict(),
            "brand_vocab": self.brand_vocab.to_dict(),
            "age_group_vocab": self.age_group_vocab.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TwoTowerFeatureEncoder":
        return cls(
            embedding_dim=data["embedding_dim"],
            max_price=data["max_price"],
            category_vocab=Vocabulary.from_dict(data["category_vocab"]),
            brand_vocab=Vocabulary.from_dict(data["brand_vocab"]),
            age_group_vocab=Vocabulary.from_dict(data["age_group_vocab"]),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TwoTowerFeatureEncoder":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
