"""Generate a NEW, deterministic, backend-shaped synthetic SQLite database.

Produces `data/sqlite/backend_shaped_synthetic.db` — an ERD-shaped relational
database (User, Category, Product, Tag, ProductTags, Cart, Cart_Item,
"Order", Order_Item, Review, UserAddress, Voucher, User_events) whose
User_events table follows the CONFIRMED future backend contract exactly:
one row per action (id, user_id, product_id, action_time, action_type),
action_type in {CLICK, ADD_TO_CART, PURCHASE, SEARCH, CHATBOT}.

This is NOT the previously-inspected data/sqlite/ecommerce.db POC (which
used a pre-aggregated BehavioralLog table, not a per-action log) and does
not modify it. It is also NOT the existing in-package synthetic generator
under `recommendation.data.synthetic` (which stays as-is for unit/
regression tests) - this script is a separate, larger, more varied dataset
meant for future adapter/integration experimentation.

ALL data produced by this script is SYNTHETIC. Nothing here is real user
or purchase data, and no documentation should ever describe it as such.

This script only generates and validates data - it does not touch any
recommendation model code, does not train anything, and does not rebuild
any retrieval index.

Usage:
    python scripts/generate_backend_shaped_sqlite.py [--seed 42] [--out PATH]
        [--num-users 1000] [--num-products 1200] [--check-determinism]
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

from recommendation.data.synthetic.affinity import product_affinity_scores, sample_product_ids
from recommendation.data.synthetic.catalog import TAG_VOCABULARY, build_categories, build_tags
from recommendation.data.synthetic.personas import AGE_GROUPS, PERSONA_BY_KEY, PERSONAS, Persona
from recommendation.data.synthetic.raw_schemas import RawProduct

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "sqlite" / "backend_shaped_synthetic.db"

REFERENCE_NOW = datetime(2026, 8, 17, 12, 0, 0)
EVENT_WINDOW_START = datetime(2026, 1, 15, 0, 0, 0)  # ~7-month window ending at REFERENCE_NOW

ACTION_TYPES = ("CLICK", "ADD_TO_CART", "PURCHASE", "SEARCH", "CHATBOT")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class GenConfig:
    seed: int = 42
    num_users: int = 1000
    num_products: int = 1200

    # User history-tier proportions (must sum to num_users at generation time).
    no_history_fraction: float = 0.10
    sparse_fraction: float = 0.25
    # remaining fraction is STRONG

    incomplete_profile_fraction: float = 0.05  # users with no PreferredCategoryId/AgeGroup
    preferred_category_alignment_prob: float = 0.8
    brand_affinity_prob: float = 0.35
    brand_affinity_bonus: float = 2.0
    affinity_noise_scale: float = 0.4

    # Product generation
    out_of_stock_prob: float = 0.12
    inactive_prob: float = 0.06
    on_sale_prob: float = 0.35

    # Funnel probabilities
    p_search_entry: float = 0.50
    p_chatbot_entry: float = 0.20
    # remaining probability mass = no precursor (direct browse -> click)
    p_click_after_search: float = 0.55
    p_click_after_chatbot: float = 0.65
    p_cart_after_click: float = 0.40
    p_purchase_after_cart: float = 0.50

    review_prob: float = 0.40

    out_path: Path = DEFAULT_OUT


# ============================================================================
# Word banks (category-name-keyed to match personas.py's category_weights)
# ============================================================================

BASE_ITEMS_BY_CATEGORY: dict[str, list[str]] = {
    "Dairy & Eggs": ["Whole Milk", "Skim Milk", "Greek Yogurt", "Vanilla Yogurt", "Cheddar Cheese",
                     "Mozzarella Cheese", "Butter", "Free-Range Eggs", "Cottage Cheese", "Sour Cream",
                     "Cream Cheese", "Kefir"],
    "Bakery": ["Whole Wheat Bread", "Sourdough Bread", "Bagels", "Croissants", "Dinner Rolls",
               "Baguette", "Blueberry Muffins", "Cinnamon Rolls", "Ciabatta Loaf"],
    "Breakfast & Cereal": ["Rolled Oats", "Honey Cereal", "Granola Clusters", "Pancake Mix", "Corn Flakes",
                           "Muesli", "Instant Oatmeal Packets", "Waffle Mix"],
    "Snacks": ["Potato Chips", "Trail Mix", "Chocolate Chip Cookies", "Protein Bars", "Popcorn",
               "Pretzel Sticks", "Tortilla Chips", "Fruit Snacks", "Crackers", "Rice Cakes"],
    "Beverages": ["Sparkling Water", "Iced Tea", "Energy Drink", "Coconut Water", "Lemonade"],
    "Coffee & Tea": ["Ground Coffee", "Green Tea Bags", "Instant Coffee", "Black Tea Bags",
                      "Espresso Pods", "Herbal Tea", "Cold Brew Concentrate"],
    "Juices & Sodas": ["Orange Juice", "Cola Soda", "Apple Juice", "Grape Soda", "Cranberry Juice",
                        "Root Beer", "Tomato Juice"],
    "Produce": ["Bananas", "Baby Spinach", "Avocados", "Roma Tomatoes", "Broccoli Crowns", "Gala Apples",
                "Carrots", "Red Onions", "Bell Peppers", "Strawberries", "Blueberries", "Sweet Potatoes",
                "Cucumbers", "Romaine Lettuce"],
    "Meat & Seafood": ["Chicken Breast", "Ground Beef", "Atlantic Salmon Fillet", "Peeled Shrimp",
                        "Pork Chops", "Turkey Breast", "Bacon", "Tilapia Fillet", "Beef Steak"],
    "Pantry & Staples": ["Basmati Rice", "Whole Wheat Pasta", "Canned Black Beans", "Extra Virgin Olive Oil",
                          "Peanut Butter", "White Rice", "Canned Tomatoes", "Lentils", "All-Purpose Flour",
                          "Brown Sugar", "Canned Chickpeas", "Quinoa"],
    "Frozen Foods": ["Frozen Mixed Berries", "Frozen Margherita Pizza", "Frozen Vegetable Medley",
                      "Frozen Chicken Nuggets", "Frozen French Fries", "Ice Cream", "Frozen Waffles",
                      "Frozen Burritos"],
    "Health & Protein": ["Whey Protein Powder", "Chia Seeds", "Almond Milk", "Protein Cookies", "Kombucha",
                          "Collagen Powder", "Multivitamin Gummies", "Protein Granola"],
    "Condiments & Sauces": ["Tomato Ketchup", "Hot Sauce", "Mayonnaise", "Soy Sauce", "BBQ Sauce",
                             "Mustard", "Salad Dressing", "Salsa", "Pasta Sauce"],
}

PRICE_RANGE_BY_CATEGORY: dict[str, tuple[float, float]] = {
    "Dairy & Eggs": (2.0, 10.0), "Bakery": (2.0, 8.0), "Breakfast & Cereal": (2.0, 9.0),
    "Snacks": (1.5, 9.0), "Beverages": (1.0, 6.0), "Coffee & Tea": (3.0, 15.0),
    "Juices & Sodas": (1.5, 6.0), "Produce": (1.0, 8.0), "Meat & Seafood": (4.0, 25.0),
    "Pantry & Staples": (1.0, 12.0), "Frozen Foods": (3.0, 10.0), "Health & Protein": (4.0, 35.0),
    "Condiments & Sauces": (1.5, 8.0),
}

ADJECTIVES = ["Organic", "Classic", "Premium", "Family-Size", "Low-Sugar", "Whole Grain", "Sea Salt",
              "Original", "Extra Virgin", "Free-Range", "Farm Fresh", "Natural", "Gourmet", "Reduced-Fat",
              "Spicy", "Sweet", "Unsalted", "Fresh", "Roasted", "Artisan"]

SIZES = ["250g", "500g", "1kg", "2kg", "1L", "2L", "6-pack", "12ct", "4-pack", "300g", "400g", "900g",
         "750ml", "1.5L", "6ct", "24ct"]

BRAND_POOL = ["GreenValley", "DairyBest", "PureFarm", "GoldenGrain", "HomeStyle", "NutriCo", "SnackWorks",
              "Value Choice", "BrewHouse", "SunHarvest", "FreshFields", "OceanCatch", "Nature's Best",
              "Harborline", "Golden Fields", "Rootwise", "Daily Harvest", "CrispCo", "Morning Ritual",
              "Wildmeadow"]

BENEFIT_PHRASES = ["rich in flavor", "perfect for everyday meals", "a family favorite",
                    "great for on-the-go snacking", "ideal for healthy living",
                    "a premium choice for special occasions", "budget-friendly and reliable",
                    "a versatile kitchen staple", "loved for its freshness",
                    "packed with quality ingredients", "a satisfying everyday choice",
                    "crafted for consistent quality"]
USE_CASES = ["breakfast", "lunch", "dinner", "snacking", "meal prep", "the whole family",
             "quick weekday meals", "weekend treats", "healthy routines", "everyday cooking",
             "packed lunches", "family gatherings"]
CLOSING_PHRASES = ["Sourced with care.", "A customer favorite.", "Made to satisfy.",
                    "Quality you can taste.", "Perfect for your next grocery run.", "Loved by shoppers.",
                    "A staple in many households.", "Consistently high quality.",
                    "Stocked fresh every week.", "A reliable everyday pick."]
INGREDIENT_WORDS = ["water", "sugar", "salt", "wheat flour", "vegetable oil", "natural flavoring",
                     "citric acid", "milk", "corn starch", "yeast", "baking soda", "spices",
                     "preservatives", "vinegar", "cane sugar"]

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


def generate_products(rng: np.random.Generator, categories, config: GenConfig) -> list[GeneratedProduct]:
    category_by_name = {c.name: c for c in categories}
    category_names = list(BASE_ITEMS_BY_CATEGORY.keys())
    n_cats = len(category_names)
    base_count = config.num_products // n_cats
    remainder = config.num_products - base_count * n_cats
    counts = {name: base_count + (1 if i < remainder else 0) for i, name in enumerate(category_names)}

    products: list[GeneratedProduct] = []
    used_names: set[str] = set()
    used_slugs: set[str] = set()
    product_id = 1

    for cat_name in category_names:
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
            stock_quantity = 0 if rng.random() < config.out_of_stock_prob else int(rng.integers(5, 300))

            benefit = BENEFIT_PHRASES[int(rng.integers(0, len(BENEFIT_PHRASES)))]
            use_case = USE_CASES[int(rng.integers(0, len(USE_CASES)))]
            closing = CLOSING_PHRASES[int(rng.integers(0, len(CLOSING_PHRASES)))]
            description = f"{adjective} {base_item.lower()}, {benefit} - great for {use_case}. {closing}"

            n_ingredients = int(rng.integers(2, 5))
            ingredients = ", ".join(
                sorted(set(INGREDIENT_WORDS[i] for i in rng.choice(len(INGREDIENT_WORDS), size=n_ingredients, replace=False)))
            )

            brand = BRAND_POOL[int(rng.integers(0, len(BRAND_POOL)))]

            n_tags = int(rng.integers(2, 5))
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
# User generation
# ============================================================================

@dataclass
class GeneratedUser:
    id: int
    first_name: str
    last_name: str
    email: str
    phone_number: str
    role: str
    preferred_category_id: int | None
    age_group: str | None
    created_at: str
    updated_at: str


@dataclass
class UserLatentProfile:
    user_id: int
    persona_key: str
    preferred_brand: str | None
    activity_level: float
    history_tier: str  # "no_history" | "sparse" | "strong"


def generate_users(rng: np.random.Generator, categories, config: GenConfig) -> tuple[list[GeneratedUser], dict[int, UserLatentProfile]]:
    category_id_by_name = {c.name: c.id for c in categories}
    category_names = list(category_id_by_name.keys())

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
        persona: Persona = PERSONAS[int(rng.integers(0, len(PERSONAS)))]

        age_group = _weighted_choice(rng, AGE_GROUPS, persona.age_group_weights)
        if rng.random() < config.preferred_category_alignment_prob:
            cat_name = persona.dominant_categories[int(rng.integers(0, len(persona.dominant_categories)))]
        else:
            cat_name = category_names[int(rng.integers(0, len(category_names)))]
        preferred_category_id = category_id_by_name[cat_name]

        preferred_brand = BRAND_POOL[int(rng.integers(0, len(BRAND_POOL)))] if rng.random() < config.brand_affinity_prob else None

        is_incomplete = user_id in incomplete_ids
        days_ago = int(rng.integers(30, 400))
        created_at = (REFERENCE_NOW - timedelta(days=days_ago)).isoformat()

        users.append(
            GeneratedUser(
                id=user_id, first_name=f"User{user_id}", last_name="Synthetic",
                email=f"user{user_id}@synthetic.invalid", phone_number=f"+1555{user_id:07d}",
                role="Customer",
                preferred_category_id=None if is_incomplete else preferred_category_id,
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
# User_events (funnel-based, session-driven)
# ============================================================================

@dataclass
class Event:
    id: int
    user_id: int
    product_id: int
    action_type: str
    action_time: str  # ISO8601


def _tags_by_product_id(products: list[GeneratedProduct]) -> dict[int, list[str]]:
    return {p.id: p.tags for p in products}


def generate_events(
    rng: np.random.Generator,
    users: list[GeneratedUser],
    latent: dict[int, UserLatentProfile],
    products: list[GeneratedProduct],
    categories,
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

        persona = PERSONA_BY_KEY[prof.persona_key]
        scores = product_affinity_scores(
            raw_products, category_name_by_id, tags_by_product, persona,
            prof.preferred_brand, config.brand_affinity_bonus, rng, config.affinity_noise_scale,
        )

        if prof.history_tier == "sparse":
            num_sessions = int(rng.integers(1, 3))
        else:  # strong
            num_sessions = int(rng.integers(8, 22))

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
                reached_click = True  # direct browse -> click

            if not reached_click:
                continue

            events.append(Event(event_id, user.id, pid, "CLICK", t.isoformat())); event_id += 1

            if rng.random() < config.p_cart_after_click:
                t += timedelta(seconds=int(rng.integers(60, 1200)))
                events.append(Event(event_id, user.id, pid, "ADD_TO_CART", t.isoformat())); event_id += 1

                if rng.random() < config.p_purchase_after_cart:
                    if rng.random() < 0.6:
                        t += timedelta(seconds=int(rng.integers(120, 1800)))  # same-session checkout
                    else:
                        t += timedelta(days=int(rng.integers(1, 4)))  # returns later to buy
                    if t > REFERENCE_NOW:
                        t = REFERENCE_NOW - timedelta(minutes=int(rng.integers(1, 60)))
                    events.append(Event(event_id, user.id, pid, "PURCHASE", t.isoformat())); event_id += 1

    events.sort(key=lambda e: (e.user_id, e.action_time))
    # renumber ids in final chronological-per-user order for readability (still unique/sequential)
    for i, e in enumerate(events, start=1):
        e.id = i
    return events


# ============================================================================
# Orders / Cart / Reviews derived from events (documented authoritative mapping)
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
    items: list[tuple[int, int, float]] = field(default_factory=list)  # (product_id, quantity, unit_price)


def derive_orders(
    rng: np.random.Generator,
    events: list[Event],
    products: list[GeneratedProduct],
    addresses_by_user: dict[int, int],
    voucher_ids: list[int],
) -> list[GeneratedOrder]:
    """Order/Order_Item is DERIVED from User_events PURCHASE rows: every
    PURCHASE event maps to exactly one OrderItem, grouped into one Order per
    (user, calendar day). This is the documented authoritative relationship
    - User_events is the source of truth for "this counts as a purchase
    engagement signal"; Order/Order_Item exists to keep the ERD's
    transactional tables relationally consistent with it, not as a second,
    independent purchase source. A small number of EXTRA orders (not linked
    to any PURCHASE event) are added with CANCELLED/PENDING status only,
    representing checkout attempts that did not culminate in a purchase
    signal - they never use DELIVERED/COMPLETED, so they can never be
    mistaken for a second purchase source.
    """
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
            unit_price = product.price
            items.append((product.id, qty, unit_price))
            total += qty * unit_price

        use_voucher = rng.random() < 0.15 and voucher_ids
        voucher_id = int(rng.choice(voucher_ids)) if use_voucher else None
        if voucher_id is not None:
            total = round(total * 0.9, 2)

        orders.append(
            GeneratedOrder(
                id=order_id, user_id=user_id, voucher_id=voucher_id,
                address_id=addresses_by_user.get(user_id), idempotence_key=f"idem-{order_id:08d}",
                total_amount=round(total, 2), status=status, payment_method=PAYMENT_METHODS[int(rng.integers(0, len(PAYMENT_METHODS)))],
                creation_date=creation_dt.isoformat(), delivery_date=delivery_date, items=items,
            )
        )
        order_id += 1

    # Extra non-purchase-linked orders: CANCELLED/PENDING only, never DELIVERED/COMPLETED.
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
                idempotence_key=f"idem-{order_id:08d}", total_amount=round(qty * product.price, 2),
                status=status, payment_method=PAYMENT_METHODS[int(rng.integers(0, len(PAYMENT_METHODS)))],
                creation_date=creation_dt.isoformat(), delivery_date=None,
                items=[(product.id, qty, product.price)],
            )
        )
        order_id += 1

    return orders


@dataclass
class GeneratedCart:
    id: int
    user_id: int
    items: list[tuple[int, int]] = field(default_factory=list)  # (product_id, quantity)


def derive_carts(rng: np.random.Generator, users: list[GeneratedUser], events: list[Event]) -> list[GeneratedCart]:
    """Cart/Cart_Item represents CURRENT cart state, derived as: for each
    user, products whose most recent relevant action is ADD_TO_CART with no
    LATER PURCHASE of that same product (i.e. still sitting unpurchased in
    the cart "now"). Every user gets a Cart row (even if empty), matching a
    real backend where a cart is created lazily but always addressable.
    """
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
    rng: np.random.Generator,
    events: list[Event],
    latent: dict[int, UserLatentProfile],
    products: list[GeneratedProduct],
    categories,
    config: GenConfig,
) -> list[GeneratedReview]:
    category_name_by_id = {c.id: c.name for c in categories}
    product_by_id = {p.id: p for p in products}
    tags_by_product = _tags_by_product_id(products)

    purchases_by_pair: dict[tuple[int, int], str] = {}
    for e in events:
        if e.action_type == "PURCHASE":
            key = (e.user_id, e.product_id)
            if key not in purchases_by_pair or e.action_time < purchases_by_pair[key]:
                purchases_by_pair[key] = e.action_time  # first purchase time for that pair

    reviews: list[GeneratedReview] = []
    review_id = 1
    for (user_id, product_id), purchase_time in purchases_by_pair.items():
        if rng.random() >= config.review_prob:
            continue
        prof = latent.get(user_id)
        if prof is None:
            continue
        persona = PERSONA_BY_KEY[prof.persona_key]
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


# ============================================================================
# UserAddress / Voucher (minimal, not consumed by the recommender)
# ============================================================================

CITIES = ["Springfield", "Riverton", "Fairview", "Maplewood", "Georgetown", "Clinton", "Franklin", "Greenville"]
COUNTRIES = ["USA"]


def generate_addresses(rng: np.random.Generator, users: list[GeneratedUser]) -> tuple[list[dict], dict[int, int]]:
    addresses = []
    by_user: dict[int, int] = {}
    addr_id = 1
    for user in users:
        if rng.random() >= 0.8:
            continue
        addresses.append({
            "id": addr_id, "user_id": user.id,
            "line1": f"{int(rng.integers(1, 9999))} {['Main','Oak','Pine','Maple','Elm'][int(rng.integers(0,5))]} St",
            "city": CITIES[int(rng.integers(0, len(CITIES)))],
            "postal_code": f"{int(rng.integers(10000, 99999))}",
            "country": COUNTRIES[0],
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
# SQLite schema + persistence
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

-- Confirmed future backend contract: one row = one action.
-- Column names intentionally lowercase snake_case, matching the literal
-- confirmed contract text (id, user_id, product_id, action_time,
-- action_type) - every other table above uses PascalCase to match
-- docs/erd.jpeg's existing convention. This casing difference is
-- deliberate, not an oversight - see the generation report, section P.
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
"""


def write_database(
    out_path: Path,
    categories,
    tags,
    products: list[GeneratedProduct],
    users: list[GeneratedUser],
    addresses: list[dict],
    vouchers: list[dict],
    carts: list[GeneratedCart],
    orders: list[GeneratedOrder],
    reviews: list[GeneratedReview],
    events: list[Event],
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


# ============================================================================
# Generation entrypoint
# ============================================================================

def generate_all(config: GenConfig):
    rng = np.random.default_rng(config.seed)
    categories = build_categories()
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
    parser.add_argument("--num-products", type=int, default=1200)
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
    }, indent=2))


if __name__ == "__main__":
    main()
