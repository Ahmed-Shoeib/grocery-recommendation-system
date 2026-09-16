"""Turns `UserFeatures`/`ProductFeatures` (+ Sentence Transformer
embeddings) into fixed-size numpy tensors the Keras towers consume.

`TwoTowerFeatureEncoder` is fit ONCE from the full product catalog (category
names - static catalog metadata, not user behavior, so fitting on the full
catalog is not target leakage), then reused to encode every item/user
example. It is serialized alongside the model weights (serving needs the
exact same vocab/normalization at inference time) - see `serialization.py`.

Index 0 in every `Vocabulary` is reserved for "unknown/missing" so a
category value not seen at fit time degrades to a learnable "unknown"
embedding rather than raising.

PRODUCTION-SAFE FEATURE CONTRACT (docs/production-feature-parity-audit.md,
"production-safe feature contract redesign"): the real SQL Server
`Products` table has no `Brand`, `SalePrice`/`DiscountPercentage`, or
`isActive` column, and the real `Users` table has no `AgeGroup` column at
all. As of this redesign, the encoder and both towers no longer have ANY
input derived from those fields:

  REMOVED entirely: `brand_vocab`/`brand_id` (item categorical),
  `brand_affinity` (user vector), `age_group_vocab`/`age_group_id` (user
  categorical), `discount_fraction`/`is_discounted` (item numeric),
  `preferred_category_id` (user categorical - see below).

  `preferred_category_id` specifically is not just removed but REDESIGNED:
  the real backend models preferred/favorite categories as a LIST
  (`UserProfile.preferred_categories`), and that signal is now folded
  directly into the existing `category_affinity` vector
  (`features.user_features.build_user_features`) rather than encoded as a
  second, separate single-category embedding lookup - so multiple
  favorites contribute naturally, with no "pick one" reduction, and no
  extra Two-Tower input slot was needed to carry them.

  RETAINED unchanged: `category_id`/`category_affinity` (category is a
  real `CategoryId`-backed production field), `price_tier_id` (derived
  purely from `Product.Price`, itself real), and every remaining numeric
  feature below.

`CURRENT_CONTRACT_VERSION` is stamped on every newly-fit encoder and
persisted in `to_dict()`/checked on load, so a pre-redesign artifact
(which still has `brand_vocab`/`age_group_vocab` keys and a 9-wide item
numeric vector) is REJECTED at startup with a clear message
(`serving.startup_validation`) instead of silently producing wrong
predictions or a cryptic Keras input-shape error at first request.

**`production_safe_v2` (docs/data-mapping.md 19.15, the train-serve
learned-feature parity fix): `log_purchase_count`/`log_cart_add_count`
REMOVED from the item numeric vector** (9 -> 7 item numeric features).
The train-serve parity audit (19.14) found these two were genuine
learned-model inputs computed from training's COMPLETE SQLite purchase/
cart history, while live `backend_api` serving can only ever supply a
BOUNDED recent-window approximation of the real backend's 1.5M+-row
activity table (no aggregate endpoint, no delta filter exists to fix
this efficiently - see docs/data-mapping.md 19.14/19.15). Rather than
leave a silent training=lifetime/serving=recent-window mismatch on a
LEARNED input, they are removed from the model entirely - product
purchase/cart popularity remains available as a serving-only fallback
heuristic (`serving.fallback.global_popularity_ranking`/
`category_popularity_ranking`, which the bounded-window approximation is
perfectly fine for - a fallback ranking has no "training semantics" to
be unfaithful to), just never again as something the trained model
assumes is reproducible exactly. No dummy zero placeholder was kept in
their place - the vector is genuinely 7-wide, not 9-wide-with-two-zeros.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from recommendation.features.price import PRICE_TIERS
from recommendation.features.product_features import ProductFeatures
from recommendation.features.user_features import UserFeatures

# Bumped whenever the SET of Two-Tower inputs changes shape (not for every
# code change) - see module docstring. A pre-redesign encoder (still
# carrying `brand_vocab`/`age_group_vocab`) has no `contract_version` key
# at all, so `from_dict` stamps it "legacy_pre_production_safe_contract"
# rather than guessing - that string will never equal this constant, so
# `serving.startup_validation` always rejects it explicitly.
CURRENT_CONTRACT_VERSION = "production_safe_v2"
_LEGACY_CONTRACT_VERSION = "legacy_pre_production_safe_contract"


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


# `discount_fraction`/`is_discounted` were removed (no real SalePrice/
# DiscountPercentage in production - see module docstring).
# `log_purchase_count`/`log_cart_add_count` were removed in
# `production_safe_v2` (see module docstring) - the real backend cannot
# reproduce a true lifetime aggregate for these efficiently, so they no
# longer feed the model at all (they remain a serving-only fallback
# heuristic via `serving.fallback`, computed from `ProductFeatures`
# directly, never through this encoder).
ITEM_NUMERIC_FEATURE_NAMES_BASE = [
    "normalized_price", "log_review_count", "average_rating", "has_rating",
]
# docs/data-mapping.md section 15: `category_relative_price` is
# already a [0,1] percentile (no extra normalization needed). `price_tier`
# itself is NOT a numeric feature - it's a categorical (see `price_tier_id`/
# `price_tier_vocab` below), not an ordinal number.
ITEM_NUMERIC_FEATURE_NAMES_PRICE_EXTRA = ["category_relative_price"]
ITEM_NUMERIC_FEATURE_NAMES = ITEM_NUMERIC_FEATURE_NAMES_BASE + ITEM_NUMERIC_FEATURE_NAMES_PRICE_EXTRA

# `has_age_group` was removed (no real AgeGroup column in production - see
# module docstring); `has_preferred_category` stays (now means "has >=1
# preferred category" - see `features.user_features.UserFeatures`).
USER_NUMERIC_FEATURE_NAMES_BASE = [
    "log_purchase_count", "log_cart_item_count", "log_search_count", "log_total_engagement_events",
    "has_chatbot_context", "has_preferred_category", "has_semantic_embedding",
]
# The user's typical purchase price, normalized by the SAME catalog
# `max_price` the item tower uses (consistent scale on both towers - docs/
# data-mapping.md section 15). `0.0` when there is no `price_profile` at
# all (a call site that didn't pass `price_context`) - distinguishable
# from a real $0-normalized value via `price_tier_id`'s dedicated "unknown"
# embedding bucket in that case (every REAL profile, even a catalog-prior
# fallback, gets a real budget/mid/premium tier - only a missing
# `price_profile` itself maps to "unknown"), so no separate has-flag is
# needed here.
USER_NUMERIC_FEATURE_NAMES_PRICE_EXTRA = ["normalized_typical_price"]
USER_NUMERIC_FEATURE_NAMES = USER_NUMERIC_FEATURE_NAMES_BASE + USER_NUMERIC_FEATURE_NAMES_PRICE_EXTRA


@dataclass
class TwoTowerFeatureEncoder:
    embedding_dim: int
    category_vocab: Vocabulary
    max_price: float  # catalog-level normalization stat, fit once (not leakage: item-side, static)
    # A FIXED (not data-fit) 3-value vocabulary - see PRICE_TIERS
    # in `features.price` - defaulted so existing callers/tests
    # constructing a TwoTowerFeatureEncoder directly don't need updating.
    price_tier_vocab: Vocabulary = field(default_factory=lambda: Vocabulary.fit(PRICE_TIERS))
    # Reported/serialized metadata only (docs/data-mapping.md section 15):
    # the price-aware inputs (item category_relative_price, user
    # normalized_typical_price, the shared price_tier_id categorical input
    # on both towers) are always part of the encoded representation - this
    # field no longer branches encoding shape, it exists so the
    # currently-loaded encoder's price-aware status is readable for
    # provenance (`evaluation.offline_report`, `GET /v1/metrics/offline`,
    # the dashboard).
    include_price_features: bool = True
    # See module docstring - distinguishes a post-redesign encoder (no
    # brand/age-group inputs) from a legacy one at load time.
    contract_version: str = CURRENT_CONTRACT_VERSION

    @property
    def item_numeric_dim(self) -> int:
        return len(ITEM_NUMERIC_FEATURE_NAMES)

    @property
    def user_numeric_dim(self) -> int:
        return len(USER_NUMERIC_FEATURE_NAMES)

    @property
    def category_affinity_dim(self) -> int:
        return len(self.category_vocab.values)  # affinity vectors don't need an "unknown" slot

    # --- fitting -----------------------------------------------------------

    @classmethod
    def fit(
        cls,
        category_names: list[str],
        prices: list[float],
        embedding_dim: int,
        include_price_features: bool = True,
    ) -> "TwoTowerFeatureEncoder":
        return cls(
            embedding_dim=embedding_dim,
            category_vocab=Vocabulary.fit(category_names),
            max_price=max(prices) if prices else 1.0,
            include_price_features=include_price_features,
        )

    # --- item encoding -------------------------------------------------------

    def encode_item(self, features: ProductFeatures, semantic_embedding: np.ndarray) -> dict[str, np.ndarray]:
        # No `purchase_count`/`cart_add_count` here (production_safe_v2 -
        # see module docstring): those remain a serving-only fallback
        # heuristic (`serving.fallback`), never a learned input.
        numeric_values = [
            min(features.effective_price / self.max_price, 1.0) if self.max_price > 0 else 0.0,
            np.log1p(features.review_count),
            features.average_rating if features.average_rating is not None else 0.0,
            1.0 if features.average_rating is not None else 0.0,
            features.category_relative_price,
        ]

        return {
            "semantic_embedding": semantic_embedding.astype(np.float32),
            "category_id": np.int32(self.category_vocab.encode(features.category_name)),
            "numeric": np.array(numeric_values, dtype=np.float32),
            "price_tier_id": np.int32(self.price_tier_vocab.encode(features.price_tier)),
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
        # `features.preferred_categories` (a list) is folded in HERE, not
        # encoded as a separate `preferred_category_id` input - see module
        # docstring. That folding already happened upstream in
        # `features.user_features.build_user_features` (it adds each
        # preferred category's weight straight into `category_affinity`),
        # so this loop needs no special-casing at all.
        category_affinity = np.zeros(self.category_affinity_dim, dtype=np.float32)
        for name, weight in features.category_affinity.items():
            idx = self.category_vocab.encode(name)
            if idx > 0:  # unknown categories (idx 0) have no affinity-vector slot
                category_affinity[idx - 1] = weight

        normalized_typical_price = 0.0
        if features.price_profile is not None and self.max_price > 0:
            normalized_typical_price = min(features.price_profile.typical_price / self.max_price, 1.0)
        price_tier = features.price_profile.price_tier if features.price_profile is not None else None

        numeric_values = [
            np.log1p(features.purchase_count),
            np.log1p(features.cart_item_count),
            np.log1p(features.search_count),
            np.log1p(features.total_engagement_events),
            1.0 if features.has_chatbot_context else 0.0,
            1.0 if features.has_preferred_category else 0.0,
            1.0 if features.semantic_embedding is not None else 0.0,
            normalized_typical_price,
        ]
        return {
            "semantic_embedding": semantic,
            "category_affinity": category_affinity,
            "price_tier_id": np.int32(self.price_tier_vocab.encode(price_tier)),
            "numeric": np.array(numeric_values, dtype=np.float32),
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
            "price_tier_vocab": self.price_tier_vocab.to_dict(),
            "include_price_features": self.include_price_features,
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TwoTowerFeatureEncoder":
        # `price_tier_vocab` is a fixed vocabulary (PRICE_TIERS) - older
        # serialized encoders predating the price-aware features won't
        # have the key, so fall back to the same fixed default `fit()`/the
        # dataclass default, rather than a KeyError. A model retrained
        # since then always has the key (round-tripped via `to_dict`), so
        # this only matters for a pre-price-feature artifact, which is
        # already dimension-incompatible for other reasons (see
        # docs/data-mapping.md section 15's "model artifact compatibility"
        # note) and must be retrained regardless.
        price_tier_vocab = (
            Vocabulary.from_dict(data["price_tier_vocab"]) if "price_tier_vocab" in data else Vocabulary.fit(PRICE_TIERS)
        )
        # A pre-redesign encoder JSON has no `contract_version` key at all
        # (and still has now-removed `brand_vocab`/`age_group_vocab` keys,
        # simply ignored here) - stamped as the legacy marker so
        # `serving.startup_validation` rejects it explicitly rather than
        # this loader silently dropping brand/age-group data on the floor.
        contract_version = data.get("contract_version", _LEGACY_CONTRACT_VERSION)
        return cls(
            embedding_dim=data["embedding_dim"],
            max_price=data["max_price"],
            category_vocab=Vocabulary.from_dict(data["category_vocab"]),
            price_tier_vocab=price_tier_vocab,
            contract_version=contract_version,
            include_price_features=data.get("include_price_features", True),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TwoTowerFeatureEncoder":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
