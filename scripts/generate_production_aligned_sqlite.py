"""Generate a NEW, deterministic, PRODUCTION-DOMAIN-ALIGNED synthetic
SQLite database - `data/sqlite/production_aligned_training.db` -
mirroring `scripts/generate_backend_shaped_sqlite.py`'s ERD-shaped schema
and event-generation approach, but with a catalog domain that matches the
REAL backend's verified live contract instead of the original hand-authored
grocery taxonomy:

  - EXACTLY the 6 real backend category names (Fruits, Packages,
    vegetables, "test category", davidfr3f, david), flat (no ParentId).
  - Price/stock distributions calibrated to the real backend's live
    aggregate statistics (min/max/mean/median/quartiles), not copied
    individual values - see PRICE_RANGE_BY_CATEGORY / STOCK_RANGE below
    and the accompanying report for the exact real-vs-synthetic numbers.
  - Genuine multi-favorite user preferences (`UserFavoriteCategory` join
    table, 0-3 favorites per user) instead of a single
    `PreferredCategoryId` scalar - matching the real backend's
    `FavoriteCategory[]` join shape (docs/production-feature-parity-audit.md).
    `sqlite.loader.load_users` auto-detects this table and prefers it.

Honest caveat, recorded here rather than silently smoothed over: the real
backend's own price ($1-$198, mean ~$95) and stock (thousands of units,
effectively never zero) distributions, and three of its six category names
("test category", "davidfr3f", "david"), read like a dev/QA-seeded
environment, not curated production grocery data. This generator matches
that distribution anyway (see docs/production-feature-parity-audit.md for
why matching current live behavior - however synthetic-looking - is the
correct target for training-serving parity), rather than "fixing" it into
something that looks more like a realistic grocery store.

Does NOT modify or regenerate `data/sqlite/backend_shaped_synthetic.db`
(kept unchanged for historical comparison) or the original in-package
synthetic generator. Only generates and validates data - does not touch
recommendation model code, train anything, or rebuild any retrieval index.

Usage:
    python scripts/generate_production_aligned_sqlite.py [--seed 42] [--out PATH]
        [--num-users 1000] [--num-products 120]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from recommendation.synthetic.affinity import product_affinity_scores, sample_product_ids
from recommendation.synthetic.catalog import TAG_VOCABULARY, build_tags
from recommendation.synthetic.personas import AGE_GROUPS, Persona
from recommendation.synthetic.raw_schemas import RawProduct

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "sqlite" / "production_aligned_training.db"

REFERENCE_NOW = datetime(2026, 9, 15, 12, 0, 0)
EVENT_WINDOW_START = datetime(2026, 2, 15, 0, 0, 0)  # ~7-month window ending at REFERENCE_NOW

ACTION_TYPES = ("CLICK", "ADD_TO_CART", "PURCHASE", "SEARCH", "CHATBOT")

# ============================================================================
# REAL BACKEND DOMAIN - verified live 2026-09-15 (docs/production-feature-parity-audit.md)
# ============================================================================

# Exact real category names/casing - GET /api/categories.
REAL_CATEGORY_NAMES = ["Fruits", "Packages", "vegetables", "test category", "davidfr3f", "david"]


@dataclass
class GenConfig:
    seed: int = 42
    num_users: int = 1000
    num_products: int = 120

    no_history_fraction: float = 0.10
    sparse_fraction: float = 0.25

    incomplete_profile_fraction: float = 0.05
    preferred_category_alignment_prob: float = 0.8
    brand_affinity_prob: float = 0.35  # legacy/metadata-only signal, see module docstring
    brand_affinity_bonus: float = 2.0
    affinity_noise_scale: float = 0.4

    # Product generation - stock/out-of-stock calibrated to the real
    # backend's live distribution (100% in-stock observed, min=212,
    # max=14893) - a small nonzero out_of_stock_prob is kept anyway so
    # the eligibility gate stays genuinely exercised in training/eval,
    # rather than degenerating into a rule that never fires.
    out_of_stock_prob: float = 0.02
    inactive_prob: float = 0.06  # legacy/metadata-only field, not an ML input - see module docstring
    on_sale_prob: float = 0.35  # legacy/metadata-only (sale_price/discount_percentage), not an ML input

    p_search_entry: float = 0.50
    p_chatbot_entry: float = 0.20
    p_click_after_search: float = 0.55
    p_click_after_chatbot: float = 0.65
    p_cart_after_click: float = 0.40
    p_purchase_after_cart: float = 0.50

    review_prob: float = 0.40

    # Multi-favorite categories per user (section 11 - genuine list, not a
    # scalar): distribution over how many favorites a (non-incomplete) user
    # gets.
    favorite_count_choices: tuple[int, ...] = (1, 2, 3)
    favorite_count_probs: tuple[float, ...] = (0.55, 0.30, 0.15)

    out_path: Path = DEFAULT_OUT


# ============================================================================
# Production-domain category taxonomy (flat, no parent - real Categories
# table has no ParentId column at all)
# ============================================================================

_CATEGORY_ROWS: list[tuple[int, str]] = [(i + 1, name) for i, name in enumerate(REAL_CATEGORY_NAMES)]


@dataclass(frozen=True)
class ProdCategory:
    id: int
    name: str
    parent_id: None = None


def build_production_categories() -> list[ProdCategory]:
    return [ProdCategory(id=cid, name=name) for cid, name in _CATEGORY_ROWS]


# ============================================================================
# Word banks per real category. "test category"/"davidfr3f"/"david" read as
# dev/QA artifacts on the live backend (see module docstring) - deliberately
# generic placeholder naming for those three, not invented grocery content.
# ============================================================================

BASE_ITEMS_BY_CATEGORY: dict[str, list[str]] = {
    "Fruits": ["Apples", "Bananas", "Oranges", "Strawberries", "Grapes", "Blueberries", "Mangoes",
               "Pineapple", "Watermelon", "Peaches", "Pears", "Kiwi", "Plums", "Cherries", "Lemons",
               "Limes", "Raspberries", "Pomegranate"],
    "vegetables": ["Carrots", "Broccoli", "Spinach", "Tomatoes", "Cucumbers", "Bell Peppers", "Onions",
                   "Potatoes", "Lettuce", "Zucchini", "Cauliflower", "Green Beans", "Mushrooms",
                   "Sweet Potatoes", "Kale", "Celery", "Garlic", "Eggplant"],
    "Packages": ["Snack Box", "Cereal Pack", "Pasta Pack", "Rice Pack", "Cookie Pack", "Chip Pack",
                 "Juice Pack", "Granola Pack", "Trail Mix Pack", "Cracker Pack", "Cereal Bar Pack",
                 "Tea Bag Pack", "Coffee Pod Pack", "Noodle Pack", "Soup Pack"],
    "test category": ["Test Item", "Sample Product", "Demo Grocery Item", "QA Test Product", "Trial Item"],
    "davidfr3f": ["Davidfr3f Item", "Davidfr3f Product", "Davidfr3f Sample"],
    "david": ["David Item", "David Product", "David Sample"],
}

# Calibrated to the real backend's LIVE per-category price stats (verified
# 2026-09-15): distributional similarity, not copied individual values.
#   Fruits:         min=27.00 max=198.72 mean=117.95 (n=26)
#   Packages:       min=5.77  max=196.22 mean=90.33  (n=30)
#   vegetables:     min=17.68 max=188.62 mean=94.87  (n=23)
#   test category:  min=1.00  max=142.00 mean=48.00  (n=3)
#   davidfr3f:      min=1.00  max=1.00   mean=1.00   (n=1)
#   david:          min=1.00  max=1.00   mean=1.00   (n=2)
# davidfr3f/david have degenerate single-value real prices (n=1, n=2) - a
# small realistic-looking range is used instead of hard-coding $1.00 for
# every generated product, to avoid a degenerate all-identical-price
# training signal (see module docstring's "not overfitting exact values").
PRICE_RANGE_BY_CATEGORY: dict[str, tuple[float, float]] = {
    "Fruits": (25.0, 200.0),
    "vegetables": (15.0, 190.0),
    "Packages": (5.0, 200.0),
    "test category": (1.0, 145.0),
    "davidfr3f": (1.0, 10.0),
    "david": (1.0, 10.0),
}

# Calibrated to the real backend's LIVE stock stats (verified 2026-09-15):
# min=212 max=14893 mean=14184.7 median=14807.0, 0/85 out of stock.
STOCK_RANGE: tuple[int, int] = (200, 15000)

ADJECTIVES = ["Fresh", "Organic", "Premium", "Family-Size", "Classic", "Select", "Everyday", "Farm Fresh",
              "Seasonal", "Natural", "Value", "Signature", "Everyday Value", "Choice", "Grade A"]

SIZES = ["250g", "500g", "1kg", "2kg", "1L", "2L", "6-pack", "12ct", "4-pack", "300g", "400g", "900g",
         "750ml", "1.5L", "6ct", "Single", "Box"]

BENEFIT_PHRASES = ["rich in flavor", "perfect for everyday meals", "a family favorite",
                    "great for on-the-go", "ideal for healthy living", "a versatile staple",
                    "loved for its freshness", "packed with quality", "a satisfying everyday choice",
                    "crafted for consistent quality"]
USE_CASES = ["breakfast", "lunch", "dinner", "snacking", "meal prep", "the whole family",
             "quick weekday meals", "healthy routines", "everyday cooking", "family gatherings"]
CLOSING_PHRASES = ["Sourced with care.", "A customer favorite.", "Quality you can taste.",
                    "Perfect for your next grocery run.", "Stocked fresh every week.",
                    "A reliable everyday pick."]

# Legacy/metadata-only content (module docstring): populated so the fields
# physically exist (schema parity with backend_shaped_synthetic.db,
# no migration churn), but NO production model input consumes them - see
# docs/production-feature-parity-audit.md.
BRAND_POOL = ["GreenValley", "DairyBest", "PureFarm", "GoldenGrain", "HomeStyle", "NutriCo", "SnackWorks",
              "Value Choice", "BrewHouse", "SunHarvest", "FreshFields", "OceanCatch"]
INGREDIENT_WORDS = ["water", "sugar", "salt", "wheat flour", "vegetable oil", "natural flavoring",
                     "citric acid", "milk", "corn starch", "preservatives", "vinegar", "cane sugar"]

REVIEW_TEMPLATES = {
    "high": ["Great product, will buy again.", "Exactly what I was looking for.", "Really good quality.",
             "Exceeded my expectations."],
    "mid": ["Decent, does the job.", "Good but nothing special.", "Solid everyday choice.", "Would buy again."],
    "low": ["Not what I expected.", "Wouldn't repurchase.", "Below average quality.", "Disappointing."],
}

PAYMENT_METHODS = ["CREDIT_CARD", "DEBIT_CARD", "PAYPAL", "CASH_ON_DELIVERY"]


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# ============================================================================
# Production-domain personas - dominant categories drawn from the real 6,
# NOT the old 13-category taxonomy (recommendation.synthetic.personas is
# untouched - this is a self-contained, generator-local persona set, same
# pattern generate_backend_shaped_sqlite.py already uses for its own
# dataclasses rather than reusing recommendation.synthetic.raw_schemas
# directly).
# ============================================================================

PRODUCTION_PERSONAS: list[Persona] = [
    Persona(
        key="fruit_lover", label="Fruit-Lover", dominant_categories=["Fruits"],
        category_weights={"Fruits": 3.0, "vegetables": 0.8, "Packages": -0.2},
        tag_weights={"healthy": 2.0, "organic": 1.6, "low-sugar": 1.0},
        adjectives=["fresh", "organic", "seasonal"],
        age_group_weights={"18-24": 1.1, "25-34": 1.3, "35-44": 1.2, "45-54": 1.0, "55-64": 0.9, "65+": 0.8},
        quantity_bias=1.0,
    ),
    Persona(
        key="vegetable_lover", label="Vegetable-Lover", dominant_categories=["vegetables"],
        category_weights={"vegetables": 3.0, "Fruits": 0.8, "Packages": -0.2},
        tag_weights={"healthy": 2.0, "vegetarian": 1.8, "organic": 1.2},
        adjectives=["fresh", "organic", "farm fresh"],
        age_group_weights={"25-34": 1.3, "35-44": 1.4, "45-54": 1.2, "18-24": 0.9, "55-64": 1.0, "65+": 0.8},
        quantity_bias=1.0,
    ),
    Persona(
        key="produce_balanced", label="Produce-Balanced", dominant_categories=["Fruits", "vegetables"],
        category_weights={"Fruits": 1.8, "vegetables": 1.8, "Packages": 0.2},
        tag_weights={"healthy": 1.8, "organic": 1.0, "family-size": 0.6},
        adjectives=["fresh", "seasonal", "value"],
        age_group_weights={"25-34": 1.2, "35-44": 1.3, "45-54": 1.2, "18-24": 1.0, "55-64": 1.0, "65+": 0.9},
        quantity_bias=1.2,
    ),
    Persona(
        key="package_shopper", label="Package-Shopper", dominant_categories=["Packages"],
        category_weights={"Packages": 3.0, "test category": 0.3, "davidfr3f": 0.1, "david": 0.1},
        tag_weights={"family-size": 1.8, "budget-friendly": 1.6, "on-the-go": 1.0},
        adjectives=["family-size", "everyday", "value"],
        age_group_weights={"18-24": 1.2, "25-34": 1.3, "35-44": 1.4, "45-54": 1.1, "55-64": 0.9, "65+": 0.7},
        quantity_bias=1.4,
    ),
    Persona(
        key="generic_shopper", label="Generic-Shopper",
        dominant_categories=["Packages", "test category", "davidfr3f", "david"],
        category_weights={"Packages": 1.0, "test category": 1.2, "davidfr3f": 1.2, "david": 1.2,
                           "Fruits": 0.3, "vegetables": 0.3},
        tag_weights={"budget-friendly": 1.0},
        adjectives=["everyday", "value", "classic"],
        age_group_weights={"18-24": 1.0, "25-34": 1.0, "35-44": 1.0, "45-54": 1.0, "55-64": 1.0, "65+": 1.0},
        quantity_bias=1.0,
    ),
]
PRODUCTION_PERSONA_BY_KEY: dict[str, Persona] = {p.key: p for p in PRODUCTION_PERSONAS}


# ============================================================================
# Product generation
# ============================================================================

@dataclass
class GeneratedProduct:
    id: int
    category_id: int
    slug: str
    name: str
    description: str
    brand: str
    price: float
    sale_price: float | None
    discount_percentage: float | None
    stock_quantity: int
    ingredients: str
    is_active: bool
    product_image: str
    alt_text: str
    tags: list[str]


# Catalog-size-per-category allocation: floor of 10 (so even the sparsest
# real category - davidfr3f, n=1 live - has enough synthetic products for
# category-affinity/negative-sampling to mean something) + the remaining
# budget split proportionally to the REAL per-category product counts
# (Fruits 26, Packages 30, vegetables 23, test category 3, davidfr3f 1,
# david 2 - live 2026-09-15), so the catalog's category *shape* still
# resembles production even though every category has a training-viable
# minimum. See the accompanying report for the full "catalog scale
# strategy" writeup (task section 4).
_REAL_CATEGORY_COUNTS = {"Fruits": 26, "Packages": 30, "vegetables": 23, "test category": 3, "davidfr3f": 1, "david": 2}


def _category_product_counts(num_products: int, floor: int = 10) -> dict[str, int]:
    names = REAL_CATEGORY_NAMES
    n_cats = len(names)
    reserved = floor * n_cats
    remaining = max(0, num_products - reserved)
    real_total = sum(_REAL_CATEGORY_COUNTS.values())
    counts = {name: floor + round(remaining * _REAL_CATEGORY_COUNTS[name] / real_total) for name in names}
    # Rounding can land 1-2 off target; correct the largest category.
    diff = num_products - sum(counts.values())
    if diff != 0:
        largest = max(counts, key=counts.get)
        counts[largest] += diff
    return counts


def generate_products(rng: np.random.Generator, categories: list[ProdCategory], config: GenConfig) -> list[GeneratedProduct]:
    category_by_name = {c.name: c for c in categories}
    counts = _category_product_counts(config.num_products)

    products: list[GeneratedProduct] = []
    used_names: set[str] = set()
    used_slugs: set[str] = set()
    product_id = 1

    for cat_name in REAL_CATEGORY_NAMES:
        cat = category_by_name[cat_name]
        base_items = BASE_ITEMS_BY_CATEGORY[cat_name]
        lo, hi = PRICE_RANGE_BY_CATEGORY[cat_name]

        for _ in range(counts[cat_name]):
            base_item = base_items[int(rng.integers(0, len(base_items)))]
            adjective = ADJECTIVES[int(rng.integers(0, len(ADJECTIVES)))]
            size = SIZES[int(rng.integers(0, len(SIZES)))]

            name = f"{adjective} {base_item} {size}"
            attempts = 0
            while name in used_names and attempts < 5:
                size = SIZES[int(rng.integers(0, len(SIZES)))]
                adjective = ADJECTIVES[int(rng.integers(0, len(ADJECTIVES)))]
                name = f"{adjective} {base_item} {size}"
                attempts += 1
            if name in used_names:
                name = f"{name} #{product_id}"
            used_names.add(name)

            slug = _slugify(name)
            if slug in used_slugs:
                slug = f"{slug}-{product_id}"
            used_slugs.add(slug)

            price = round(float(rng.uniform(lo, hi)), 2)
            on_sale = rng.random() < config.on_sale_prob
            if on_sale:
                discount_frac = float(rng.uniform(0.05, 0.40))
                sale_price = round(price * (1 - discount_frac), 2)
                discount_percentage = round((price - sale_price) / price * 100, 1)
            else:
                sale_price = None
                discount_percentage = None

            is_active = rng.random() >= config.inactive_prob
            stock_lo, stock_hi = STOCK_RANGE
            stock_quantity = 0 if rng.random() < config.out_of_stock_prob else int(rng.integers(stock_lo, stock_hi))

            benefit = BENEFIT_PHRASES[int(rng.integers(0, len(BENEFIT_PHRASES)))]
            use_case = USE_CASES[int(rng.integers(0, len(USE_CASES)))]
            closing = CLOSING_PHRASES[int(rng.integers(0, len(CLOSING_PHRASES)))]
            description = f"{adjective} {base_item.lower()}, {benefit} - great for {use_case}. {closing}"

            n_ingredients = int(rng.integers(2, 5))
            ingredients = ", ".join(
                sorted(set(INGREDIENT_WORDS[i] for i in rng.choice(len(INGREDIENT_WORDS), size=n_ingredients, replace=False)))
            )
            brand = BRAND_POOL[int(rng.integers(0, len(BRAND_POOL)))]
            n_tags = int(rng.integers(1, 4))
            tags = list(rng.choice(TAG_VOCABULARY, size=n_tags, replace=False))

            products.append(
                GeneratedProduct(
                    id=product_id, category_id=cat.id, slug=slug, name=name, description=description,
                    brand=brand, price=price, sale_price=sale_price, discount_percentage=discount_percentage,
                    stock_quantity=stock_quantity, ingredients=ingredients, is_active=bool(is_active),
                    product_image=f"https://example-cdn.invalid/products/{slug}.jpg", alt_text=name, tags=tags,
                )
            )
            product_id += 1

    return products


# ============================================================================
# User generation (multi-favorite categories - section 11)
# ============================================================================

@dataclass
class GeneratedUser:
    id: int
    first_name: str
    last_name: str
    email: str
    phone_number: str
    role: str
    preferred_category_id: int | None  # legacy single-value column, kept for schema parity only (first favorite, or None)
    favorite_category_ids: list[int]  # genuine multi-favorite list - see UserFavoriteCategory
    age_group: str | None
    created_at: str
    updated_at: str


@dataclass
class UserLatentProfile:
    user_id: int
    persona_key: str
    preferred_brand: str | None
    activity_level: float
    history_tier: str


def _sample_favorite_categories(
    rng: np.random.Generator, persona: Persona, category_names: list[str], config: GenConfig
) -> list[str]:
    n = int(rng.choice(config.favorite_count_choices, p=config.favorite_count_probs))
    favorites: list[str] = []
    pool = list(category_names)
    for _ in range(n):
        if not pool:
            break
        if rng.random() < config.preferred_category_alignment_prob and persona.dominant_categories:
            candidates = [c for c in persona.dominant_categories if c in pool]
            choice = candidates[int(rng.integers(0, len(candidates)))] if candidates else pool[int(rng.integers(0, len(pool)))]
        else:
            choice = pool[int(rng.integers(0, len(pool)))]
        favorites.append(choice)
        pool.remove(choice)
    return favorites


def generate_users(
    rng: np.random.Generator, categories: list[ProdCategory], config: GenConfig
) -> tuple[list[GeneratedUser], dict[int, UserLatentProfile]]:
    category_id_by_name = {c.name: c.id for c in categories}
    category_names = list(category_id_by_name.keys())
    persona_keys = list(PRODUCTION_PERSONA_BY_KEY.keys())

    n = config.num_users
    n_no_history = int(round(n * config.no_history_fraction))
    n_sparse = int(round(n * config.sparse_fraction))
    n_strong = n - n_no_history - n_sparse
    tier_pool = ["no_history"] * n_no_history + ["sparse"] * n_sparse + ["strong"] * n_strong
    rng.shuffle(tier_pool)

    n_incomplete = int(round(n * config.incomplete_profile_fraction))
    incomplete_ids = set(rng.choice(np.arange(1, n + 1), size=n_incomplete, replace=False)) if n_incomplete else set()

    users: list[GeneratedUser] = []
    latent: dict[int, UserLatentProfile] = {}

    for i in range(n):
        user_id = i + 1
        persona = PRODUCTION_PERSONA_BY_KEY[persona_keys[int(rng.integers(0, len(persona_keys)))]]
        age_group = _weighted_choice(rng, AGE_GROUPS, persona.age_group_weights)

        is_incomplete = user_id in incomplete_ids
        favorites = [] if is_incomplete else _sample_favorite_categories(rng, persona, category_names, config)
        favorite_ids = [category_id_by_name[name] for name in favorites]

        preferred_brand = BRAND_POOL[int(rng.integers(0, len(BRAND_POOL)))] if rng.random() < config.brand_affinity_prob else None

        days_ago = int(rng.integers(30, 200))
        created_at = (REFERENCE_NOW - timedelta(days=days_ago)).isoformat()

        users.append(
            GeneratedUser(
                id=user_id, first_name=f"User{user_id}", last_name="Synthetic",
                email=f"user{user_id}@synthetic.invalid", phone_number=f"+1555{user_id:07d}", role="Customer",
                preferred_category_id=favorite_ids[0] if favorite_ids else None,
                favorite_category_ids=favorite_ids,
                age_group=None if is_incomplete else age_group,
                created_at=created_at, updated_at=created_at,
            )
        )
        latent[user_id] = UserLatentProfile(
            user_id=user_id, persona_key=persona.key, preferred_brand=preferred_brand,
            activity_level=float(rng.uniform(0.3, 1.0)), history_tier=tier_pool[i],
        )

    return users, latent


def _weighted_choice(rng: np.random.Generator, keys: list[str], weights: dict[str, float]) -> str:
    w = np.array([weights.get(k, 0.0) for k in keys], dtype=float)
    w = np.clip(w, 0.05, None)
    probs = w / w.sum()
    return keys[int(rng.choice(len(keys), p=probs))]


# ============================================================================
# User_events (same funnel-based, session-driven approach - unchanged
# probabilities, see module docstring section 9: do not let SEARCH dominate
# just because the real backend's own seeded action log happens to be ~87%
# SearchProduct - that is a known-skewed generated distribution, not
# something to imitate).
# ============================================================================

@dataclass
class Event:
    id: int
    user_id: int
    product_id: int
    action_type: str
    action_time: str


def _tags_by_product_id(products: list[GeneratedProduct]) -> dict[int, list[str]]:
    return {p.id: p.tags for p in products}


def generate_events(
    rng: np.random.Generator,
    users: list[GeneratedUser],
    latent: dict[int, UserLatentProfile],
    products: list[GeneratedProduct],
    categories: list[ProdCategory],
    config: GenConfig,
) -> list[Event]:
    category_name_by_id = {c.id: c.name for c in categories}
    tags_by_product = _tags_by_product_id(products)
    raw_products = [
        RawProduct(
            id=p.id, category_id=p.category_id, slug=p.slug, name=p.name, description=p.description,
            brand=p.brand, price=p.price, sale_price=p.sale_price, discount_percentage=p.discount_percentage,
            stock_quantity=p.stock_quantity, ingredients=p.ingredients, is_active=p.is_active,
            product_image=p.product_image, alt_text=p.alt_text,
        )
        for p in products
    ]
    product_ids = [p.id for p in raw_products]

    events: list[Event] = []
    event_id = 1
    window_span_seconds = int((REFERENCE_NOW - EVENT_WINDOW_START).total_seconds())

    for user in users:
        prof = latent[user.id]
        if prof.history_tier == "no_history":
            continue

        persona = PRODUCTION_PERSONA_BY_KEY[prof.persona_key]
        scores = product_affinity_scores(
            raw_products, category_name_by_id, tags_by_product, persona,
            prof.preferred_brand, config.brand_affinity_bonus, rng, config.affinity_noise_scale,
        )

        num_sessions = int(rng.integers(1, 3)) if prof.history_tier == "sparse" else int(rng.integers(8, 22))

        for _ in range(num_sessions):
            session_product_id = sample_product_ids(rng, product_ids, scores, k=1, replace=False)
            if not session_product_id:
                continue
            pid = session_product_id[0]

            session_start_offset = int(rng.integers(0, window_span_seconds))
            t = EVENT_WINDOW_START + timedelta(seconds=session_start_offset)

            entry_roll = rng.random()
            reached_click = False
            if entry_roll < config.p_search_entry:
                events.append(Event(event_id, user.id, pid, "SEARCH", t.isoformat())); event_id += 1
                if rng.random() < config.p_click_after_search:
                    t += timedelta(seconds=int(rng.integers(10, 300)))
                    reached_click = True
            elif entry_roll < config.p_search_entry + config.p_chatbot_entry:
                events.append(Event(event_id, user.id, pid, "CHATBOT", t.isoformat())); event_id += 1
                if rng.random() < config.p_click_after_chatbot:
                    t += timedelta(seconds=int(rng.integers(30, 600)))
                    reached_click = True
            else:
                reached_click = True

            if not reached_click:
                continue

            events.append(Event(event_id, user.id, pid, "CLICK", t.isoformat())); event_id += 1

            if rng.random() < config.p_cart_after_click:
                t += timedelta(seconds=int(rng.integers(60, 1200)))
                events.append(Event(event_id, user.id, pid, "ADD_TO_CART", t.isoformat())); event_id += 1

                if rng.random() < config.p_purchase_after_cart:
                    if rng.random() < 0.6:
                        t += timedelta(seconds=int(rng.integers(120, 1800)))
                    else:
                        t += timedelta(days=int(rng.integers(1, 4)))
                    if t > REFERENCE_NOW:
                        t = REFERENCE_NOW - timedelta(minutes=int(rng.integers(1, 60)))
                    events.append(Event(event_id, user.id, pid, "PURCHASE", t.isoformat())); event_id += 1

    events.sort(key=lambda e: (e.user_id, e.action_time))
    for i, e in enumerate(events, start=1):
        e.id = i
    return events


# ============================================================================
# Orders / Cart / Reviews derived from events (unchanged approach)
# ============================================================================

@dataclass
class GeneratedOrder:
    id: int
    user_id: int
    voucher_id: int | None
    address_id: int | None
    idempotence_key: str
    total_amount: float
    status: str
    payment_method: str
    creation_date: str
    delivery_date: str | None
    items: list[tuple[int, int, float]] = field(default_factory=list)


def derive_orders(
    rng: np.random.Generator, events: list[Event], products: list[GeneratedProduct],
    addresses_by_user: dict[int, int], voucher_ids: list[int],
) -> list[GeneratedOrder]:
    product_by_id = {p.id: p for p in products}
    purchases = [e for e in events if e.action_type == "PURCHASE"]

    groups: dict[tuple[int, str], list[Event]] = {}
    for e in purchases:
        day = e.action_time[:10]
        groups.setdefault((e.user_id, day), []).append(e)

    orders: list[GeneratedOrder] = []
    order_id = 1
    for (user_id, _day), group_events in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[1][0].action_time)):
        r = rng.random()
        status = "DELIVERED" if r < 0.75 else ("COMPLETED" if r < 0.90 else "PENDING")
        creation_dt = min(datetime.fromisoformat(e.action_time) for e in group_events)
        delivery_date = (creation_dt + timedelta(days=int(rng.integers(1, 8)))).isoformat() if status in ("DELIVERED", "COMPLETED") else None

        items = []
        total = 0.0
        for e in group_events:
            product = product_by_id[e.product_id]
            qty = max(1, int(rng.integers(1, 4)))
            items.append((product.id, qty, product.price))
            total += qty * product.price

        use_voucher = rng.random() < 0.15 and voucher_ids
        voucher_id = int(rng.choice(voucher_ids)) if use_voucher else None
        if voucher_id is not None:
            total = round(total * 0.9, 2)

        orders.append(
            GeneratedOrder(
                id=order_id, user_id=user_id, voucher_id=voucher_id, address_id=addresses_by_user.get(user_id),
                idempotence_key=f"idem-{order_id:08d}", total_amount=round(total, 2), status=status,
                payment_method=PAYMENT_METHODS[int(rng.integers(0, len(PAYMENT_METHODS)))],
                creation_date=creation_dt.isoformat(), delivery_date=delivery_date, items=items,
            )
        )
        order_id += 1

    all_user_ids = list({e.user_id for e in events}) or list(addresses_by_user.keys())
    n_extra = max(1, int(len(orders) * 0.06))
    for _ in range(n_extra):
        if not all_user_ids:
            break
        user_id = int(rng.choice(all_user_ids))
        product = products[int(rng.integers(0, len(products)))]
        qty = max(1, int(rng.integers(1, 3)))
        days_ago = int(rng.integers(0, int((REFERENCE_NOW - EVENT_WINDOW_START).days)))
        creation_dt = REFERENCE_NOW - timedelta(days=days_ago)
        status = "CANCELLED" if rng.random() < 0.6 else "PENDING"
        orders.append(
            GeneratedOrder(
                id=order_id, user_id=user_id, voucher_id=None, address_id=addresses_by_user.get(user_id),
                idempotence_key=f"idem-{order_id:08d}", total_amount=round(qty * product.price, 2), status=status,
                payment_method=PAYMENT_METHODS[int(rng.integers(0, len(PAYMENT_METHODS)))],
                creation_date=creation_dt.isoformat(), delivery_date=None, items=[(product.id, qty, product.price)],
            )
        )
        order_id += 1
    return orders


@dataclass
class GeneratedCart:
    id: int
    user_id: int
    items: list[tuple[int, int]] = field(default_factory=list)


def derive_carts(rng: np.random.Generator, users: list[GeneratedUser], events: list[Event]) -> list[GeneratedCart]:
    last_cart_add: dict[tuple[int, int], str] = {}
    last_purchase: dict[tuple[int, int], str] = {}
    for e in events:
        key = (e.user_id, e.product_id)
        if e.action_type == "ADD_TO_CART":
            if key not in last_cart_add or e.action_time > last_cart_add[key]:
                last_cart_add[key] = e.action_time
        elif e.action_type == "PURCHASE":
            if key not in last_purchase or e.action_time > last_purchase[key]:
                last_purchase[key] = e.action_time

    current_cart_items: dict[int, list[int]] = {}
    for (user_id, product_id), add_time in last_cart_add.items():
        purchase_time = last_purchase.get((user_id, product_id))
        if purchase_time is None or add_time > purchase_time:
            current_cart_items.setdefault(user_id, []).append(product_id)

    carts: list[GeneratedCart] = []
    for cart_id, user in enumerate(users, start=1):
        pids = current_cart_items.get(user.id, [])
        items = [(pid, max(1, int(rng.integers(1, 4)))) for pid in pids]
        carts.append(GeneratedCart(id=cart_id, user_id=user.id, items=items))
    return carts


@dataclass
class GeneratedReview:
    id: int
    user_id: int
    product_id: int
    rating: float
    comment: str
    creation_date: str


def derive_reviews(
    rng: np.random.Generator, events: list[Event], latent: dict[int, UserLatentProfile],
    products: list[GeneratedProduct], categories: list[ProdCategory], config: GenConfig,
) -> list[GeneratedReview]:
    category_name_by_id = {c.id: c.name for c in categories}
    product_by_id = {p.id: p for p in products}
    tags_by_product = _tags_by_product_id(products)

    purchases_by_pair: dict[tuple[int, int], str] = {}
    for e in events:
        if e.action_type == "PURCHASE":
            key = (e.user_id, e.product_id)
            if key not in purchases_by_pair or e.action_time < purchases_by_pair[key]:
                purchases_by_pair[key] = e.action_time

    reviews: list[GeneratedReview] = []
    review_id = 1
    for (user_id, product_id), purchase_time in purchases_by_pair.items():
        if rng.random() >= config.review_prob:
            continue
        prof = latent.get(user_id)
        if prof is None:
            continue
        persona = PRODUCTION_PERSONA_BY_KEY[prof.persona_key]
        product = product_by_id[product_id]
        category_name = category_name_by_id.get(product.category_id, "")
        product_tags = tags_by_product.get(product_id, [])

        affinity = 1.0 + persona.category_weights.get(category_name, 0.0)
        affinity += sum(persona.tag_weights.get(t, 0.0) for t in product_tags)
        normalized = 1.0 / (1.0 + np.exp(-affinity / 2.5))
        rating = 3.0 + normalized * 2.0 + float(rng.normal(0, 0.4))
        rating = float(np.clip(round(rating * 2) / 2, 1.0, 5.0))
        bucket = "high" if rating >= 4.5 else "mid" if rating >= 3.0 else "low"
        comment = REVIEW_TEMPLATES[bucket][int(rng.integers(0, len(REVIEW_TEMPLATES[bucket])))]

        days_after = int(rng.integers(1, 15))
        creation_date = (datetime.fromisoformat(purchase_time) + timedelta(days=days_after)).isoformat()
        if creation_date > REFERENCE_NOW.isoformat():
            creation_date = REFERENCE_NOW.isoformat()

        reviews.append(GeneratedReview(review_id, user_id, product_id, rating, comment, creation_date))
        review_id += 1

    return reviews


CITIES = ["Springfield", "Riverton", "Fairview", "Maplewood", "Georgetown", "Clinton", "Franklin", "Greenville"]
COUNTRIES = ["USA"]


def generate_addresses(rng: np.random.Generator, users: list[GeneratedUser]) -> tuple[list[dict], dict[int, int]]:
    addresses = []
    by_user: dict[int, int] = {}
    addr_id = 1
    for user in users:
        addresses.append({
            "id": addr_id, "user_id": user.id,
            "line1": f"{int(rng.integers(1, 9999))} {CITIES[int(rng.integers(0, len(CITIES)))]} St",
            "city": CITIES[int(rng.integers(0, len(CITIES)))],
            "postal_code": f"{int(rng.integers(10000, 99999))}", "country": COUNTRIES[0],
        })
        by_user[user.id] = addr_id
        addr_id += 1
    return addresses, by_user


def generate_vouchers(rng: np.random.Generator, n: int = 30) -> list[dict]:
    return [
        {
            "id": i + 1, "code": f"SAVE{int(rng.integers(10, 99))}-{i + 1:03d}",
            "discount_percentage": round(float(rng.uniform(5, 25)), 1),
            "expiry_date": (REFERENCE_NOW + timedelta(days=int(rng.integers(10, 300)))).isoformat(),
        }
        for i in range(n)
    ]


# ============================================================================
# SQLite schema - identical to generate_backend_shaped_sqlite.py PLUS a new
# UserFavoriteCategory join table (section 11 - genuine multi-favorite
# semantics; sqlite.loader.load_users auto-detects this table).
# ============================================================================

SCHEMA_SQL = """
CREATE TABLE Category (
    Id INTEGER PRIMARY KEY,
    ParentId INTEGER,
    Name TEXT NOT NULL,
    CreatedAt TEXT NOT NULL,
    FOREIGN KEY(ParentId) REFERENCES Category(Id)
);

CREATE TABLE Tag (
    Id INTEGER PRIMARY KEY,
    Name TEXT NOT NULL
);

CREATE TABLE Product (
    Id INTEGER PRIMARY KEY,
    CategoryId INTEGER NOT NULL,
    Slug TEXT NOT NULL,
    Name TEXT NOT NULL,
    Description TEXT,
    Brand TEXT,
    Price REAL NOT NULL,
    SalePrice REAL,
    DiscountPercentage REAL,
    StockQuantity INTEGER NOT NULL,
    Ingredients TEXT,
    isActive BOOLEAN NOT NULL,
    ProductImage TEXT,
    AltText TEXT,
    FOREIGN KEY(CategoryId) REFERENCES Category(Id)
);

CREATE TABLE ProductTags (
    Id INTEGER PRIMARY KEY,
    ProductId INTEGER NOT NULL,
    TagId INTEGER NOT NULL,
    FOREIGN KEY(ProductId) REFERENCES Product(Id),
    FOREIGN KEY(TagId) REFERENCES Tag(Id)
);

CREATE TABLE User (
    Id INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Email TEXT NOT NULL,
    PhoneNumber TEXT,
    HashedPassword TEXT,
    RefreshToken TEXT,
    Role TEXT NOT NULL,
    PreferredCategoryId INTEGER,
    AgeGroup TEXT,
    CreatedAt TEXT NOT NULL,
    UpdatedAt TEXT NOT NULL,
    FOREIGN KEY(PreferredCategoryId) REFERENCES Category(Id)
);

-- Genuine multi-favorite categories per user (section 11) - matches the
-- real backend's FavoriteCategory[] join shape. sqlite.loader.load_users
-- prefers this table when present; User.PreferredCategoryId above is kept
-- only as a legacy single-value column (first favorite, or NULL).
CREATE TABLE UserFavoriteCategory (
    Id INTEGER PRIMARY KEY,
    UserId INTEGER NOT NULL,
    CategoryId INTEGER NOT NULL,
    FOREIGN KEY(UserId) REFERENCES User(Id),
    FOREIGN KEY(CategoryId) REFERENCES Category(Id)
);

CREATE TABLE UserAddress (
    Id INTEGER PRIMARY KEY,
    UserId INTEGER NOT NULL,
    Line1 TEXT,
    City TEXT,
    PostalCode TEXT,
    Country TEXT,
    FOREIGN KEY(UserId) REFERENCES User(Id)
);

CREATE TABLE Voucher (
    Id INTEGER PRIMARY KEY,
    Code TEXT NOT NULL,
    DiscountPercentage REAL,
    ExpiryDate TEXT
);

CREATE TABLE Cart (
    Id INTEGER PRIMARY KEY,
    UserId INTEGER NOT NULL,
    FOREIGN KEY(UserId) REFERENCES User(Id)
);

CREATE TABLE Cart_Item (
    Id INTEGER PRIMARY KEY,
    CartId INTEGER NOT NULL,
    ProductId INTEGER NOT NULL,
    Quantity INTEGER NOT NULL,
    FOREIGN KEY(CartId) REFERENCES Cart(Id),
    FOREIGN KEY(ProductId) REFERENCES Product(Id)
);

CREATE TABLE "Order" (
    Id INTEGER PRIMARY KEY,
    UserId INTEGER NOT NULL,
    VoucherId INTEGER,
    AddressId INTEGER,
    IdempotenceKey TEXT,
    TotalAmount REAL,
    Status TEXT NOT NULL,
    PaymentMethod TEXT,
    CreationDate TEXT NOT NULL,
    DeliveryDate TEXT,
    FOREIGN KEY(UserId) REFERENCES User(Id),
    FOREIGN KEY(VoucherId) REFERENCES Voucher(Id),
    FOREIGN KEY(AddressId) REFERENCES UserAddress(Id)
);

CREATE TABLE Order_Item (
    Id INTEGER PRIMARY KEY,
    OrderId INTEGER NOT NULL,
    ProductId INTEGER NOT NULL,
    Quantity INTEGER NOT NULL,
    UnitPrice REAL NOT NULL,
    FOREIGN KEY(OrderId) REFERENCES "Order"(Id),
    FOREIGN KEY(ProductId) REFERENCES Product(Id)
);

CREATE TABLE Review (
    Id INTEGER PRIMARY KEY,
    UserId INTEGER NOT NULL,
    ProductId INTEGER NOT NULL,
    Rating REAL NOT NULL,
    Comment TEXT,
    CreationDate TEXT NOT NULL,
    FOREIGN KEY(UserId) REFERENCES User(Id),
    FOREIGN KEY(ProductId) REFERENCES Product(Id)
);

CREATE TABLE User_events (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    action_time TEXT NOT NULL,
    action_type TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES User(Id),
    FOREIGN KEY(product_id) REFERENCES Product(Id)
);

CREATE INDEX idx_user_events_user ON User_events(user_id);
CREATE INDEX idx_user_events_product ON User_events(product_id);
CREATE INDEX idx_user_events_type ON User_events(action_type);
CREATE INDEX idx_product_category ON Product(CategoryId);
CREATE INDEX idx_producttags_product ON ProductTags(ProductId);
CREATE INDEX idx_cartitem_cart ON Cart_Item(CartId);
CREATE INDEX idx_orderitem_order ON Order_Item(OrderId);
CREATE INDEX idx_review_user_product ON Review(UserId, ProductId);
CREATE INDEX idx_userfavcat_user ON UserFavoriteCategory(UserId);
"""


def write_database(
    out_path: Path, categories: list[ProdCategory], tags, products: list[GeneratedProduct],
    users: list[GeneratedUser], addresses: list[dict], vouchers: list[dict], carts: list[GeneratedCart],
    orders: list[GeneratedOrder], reviews: list[GeneratedReview], events: list[Event],
) -> None:
    if out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(str(out_path))
    try:
        cur = con.cursor()
        cur.executescript(SCHEMA_SQL)

        cur.executemany(
            "INSERT INTO Category (Id, ParentId, Name, CreatedAt) VALUES (?,?,?,?)",
            [(c.id, c.parent_id, c.name, REFERENCE_NOW.isoformat()) for c in categories],
        )
        cur.executemany("INSERT INTO Tag (Id, Name) VALUES (?,?)", [(t.id, t.name) for t in tags])

        cur.executemany(
            "INSERT INTO Product (Id, CategoryId, Slug, Name, Description, Brand, Price, SalePrice, "
            "DiscountPercentage, StockQuantity, Ingredients, isActive, ProductImage, AltText) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (p.id, p.category_id, p.slug, p.name, p.description, p.brand, p.price, p.sale_price,
                 p.discount_percentage, p.stock_quantity, p.ingredients, int(p.is_active), p.product_image, p.alt_text)
                for p in products
            ],
        )

        tag_id_by_name = {t.name: t.id for t in tags}
        pt_rows = []
        pt_id = 1
        for p in products:
            for tag_name in p.tags:
                pt_rows.append((pt_id, p.id, tag_id_by_name[tag_name]))
                pt_id += 1
        cur.executemany("INSERT INTO ProductTags (Id, ProductId, TagId) VALUES (?,?,?)", pt_rows)

        cur.executemany(
            "INSERT INTO User (Id, FirstName, LastName, Email, PhoneNumber, HashedPassword, RefreshToken, "
            "Role, PreferredCategoryId, AgeGroup, CreatedAt, UpdatedAt) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (u.id, u.first_name, u.last_name, u.email, u.phone_number, "synthetic_hash", None,
                 u.role, u.preferred_category_id, u.age_group, u.created_at, u.updated_at)
                for u in users
            ],
        )

        ufc_rows = []
        ufc_id = 1
        for u in users:
            for cid in u.favorite_category_ids:
                ufc_rows.append((ufc_id, u.id, cid))
                ufc_id += 1
        cur.executemany("INSERT INTO UserFavoriteCategory (Id, UserId, CategoryId) VALUES (?,?,?)", ufc_rows)

        cur.executemany(
            "INSERT INTO UserAddress (Id, UserId, Line1, City, PostalCode, Country) VALUES (?,?,?,?,?,?)",
            [(a["id"], a["user_id"], a["line1"], a["city"], a["postal_code"], a["country"]) for a in addresses],
        )
        cur.executemany(
            "INSERT INTO Voucher (Id, Code, DiscountPercentage, ExpiryDate) VALUES (?,?,?,?)",
            [(v["id"], v["code"], v["discount_percentage"], v["expiry_date"]) for v in vouchers],
        )

        cur.executemany("INSERT INTO Cart (Id, UserId) VALUES (?,?)", [(c.id, c.user_id) for c in carts])
        ci_rows = []
        ci_id = 1
        for c in carts:
            for product_id, qty in c.items:
                ci_rows.append((ci_id, c.id, product_id, qty))
                ci_id += 1
        cur.executemany("INSERT INTO Cart_Item (Id, CartId, ProductId, Quantity) VALUES (?,?,?,?)", ci_rows)

        cur.executemany(
            'INSERT INTO "Order" (Id, UserId, VoucherId, AddressId, IdempotenceKey, TotalAmount, Status, '
            "PaymentMethod, CreationDate, DeliveryDate) VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                (o.id, o.user_id, o.voucher_id, o.address_id, o.idempotence_key, o.total_amount, o.status,
                 o.payment_method, o.creation_date, o.delivery_date)
                for o in orders
            ],
        )
        oi_rows = []
        oi_id = 1
        for o in orders:
            for product_id, qty, unit_price in o.items:
                oi_rows.append((oi_id, o.id, product_id, qty, unit_price))
                oi_id += 1
        cur.executemany("INSERT INTO Order_Item (Id, OrderId, ProductId, Quantity, UnitPrice) VALUES (?,?,?,?,?)", oi_rows)

        cur.executemany(
            "INSERT INTO Review (Id, UserId, ProductId, Rating, Comment, CreationDate) VALUES (?,?,?,?,?,?)",
            [(r.id, r.user_id, r.product_id, r.rating, r.comment, r.creation_date) for r in reviews],
        )

        cur.executemany(
            "INSERT INTO User_events (id, user_id, product_id, action_time, action_type) VALUES (?,?,?,?,?)",
            [(e.id, e.user_id, e.product_id, e.action_time, e.action_type) for e in events],
        )

        con.commit()
    finally:
        con.close()


def generate_all(config: GenConfig):
    rng = np.random.default_rng(config.seed)
    categories = build_production_categories()
    tags = build_tags()

    products = generate_products(rng, categories, config)
    users, latent = generate_users(rng, categories, config)
    addresses, addresses_by_user = generate_addresses(rng, users)
    vouchers = generate_vouchers(rng)
    events = generate_events(rng, users, latent, products, categories, config)
    orders = derive_orders(rng, events, products, addresses_by_user, [v["id"] for v in vouchers])
    carts = derive_carts(rng, users, events)
    reviews = derive_reviews(rng, events, latent, products, categories, config)

    return {
        "categories": categories, "tags": tags, "products": products, "users": users, "latent": latent,
        "addresses": addresses, "vouchers": vouchers, "events": events, "orders": orders, "carts": carts,
        "reviews": reviews,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--num-users", type=int, default=1000)
    parser.add_argument("--num-products", type=int, default=120)
    args = parser.parse_args()

    config = GenConfig(seed=args.seed, num_users=args.num_users, num_products=args.num_products, out_path=Path(args.out))
    data = generate_all(config)
    write_database(
        config.out_path, data["categories"], data["tags"], data["products"], data["users"],
        data["addresses"], data["vouchers"], data["carts"], data["orders"], data["reviews"], data["events"],
    )
    print(f"Wrote {config.out_path} ({config.out_path.stat().st_size} bytes)")
    print(json.dumps({
        "users": len(data["users"]), "products": len(data["products"]), "events": len(data["events"]),
        "orders": len(data["orders"]), "carts": len(data["carts"]), "reviews": len(data["reviews"]),
        "categories": [c.name for c in data["categories"]],
    }, indent=2))


if __name__ == "__main__":
    main()
