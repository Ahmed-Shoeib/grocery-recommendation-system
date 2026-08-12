"""Hand-authored grocery catalog: categories, tags, and ~50 products.

Product *content* (name, description, brand, category, tags, base price) is
authored, not randomly generated - the goal is a realistic, semantically
meaningful catalog (useful for Sentence Transformer text in Phase 3), not a
statistically generated one. Only operational fields that a real backend
would mutate at runtime - `stock_quantity`, `is_active`, promotional
`sale_price`/`discount_percentage` - are jittered by a seeded RNG, via
`build_catalog(rng)`, to keep the eligibility-relevant fields non-trivial
without hand-picking every value.

`config.synthetic_data.num_products` documents the intended catalog size;
this module is the source of truth for actual content and is asserted
against that count in tests.
"""

from __future__ import annotations

import re

import numpy as np

from recommendation.data.synthetic.raw_schemas import RawCategory, RawProduct, RawProductTag, RawTag

# --- Categories -------------------------------------------------------
# (id, name, parent_id)
_CATEGORY_ROWS: list[tuple[int, str, int | None]] = [
    (1, "Dairy & Eggs", None),
    (2, "Bakery", None),
    (3, "Breakfast & Cereal", None),
    (4, "Snacks", None),
    (5, "Beverages", None),
    (6, "Coffee & Tea", 5),
    (7, "Juices & Sodas", 5),
    (8, "Produce", None),
    (9, "Meat & Seafood", None),
    (10, "Pantry & Staples", None),
    (11, "Frozen Foods", None),
    (12, "Health & Protein", None),
    (13, "Condiments & Sauces", None),
]

# --- Tag vocabulary -----------------------------------------------------
TAG_VOCABULARY: list[str] = [
    "healthy", "high-protein", "low-sugar", "organic", "indulgent",
    "budget-friendly", "premium", "family-size", "vegetarian", "gluten-free",
    "breakfast-staple", "on-the-go", "kid-friendly", "whole-grain", "dairy-free",
]

# --- Products -------------------------------------------------------------
# (name, category_id, brand, price, sale_price, description, ingredients, tags)
_PRODUCT_ROWS: list[tuple[str, int, str, float, float | None, str, str, list[str]]] = [
    # Dairy & Eggs (1)
    ("Greek Yogurt Plain 500g", 1, "GreenValley", 4.29, None,
     "Thick strained plain Greek yogurt, a high-protein low-sugar staple for breakfast bowls.",
     "Milk, live active cultures.", ["healthy", "high-protein", "low-sugar"]),
    ("Whole Milk 1L", 1, "DairyBest", 2.49, None,
     "Fresh whole milk, a family breakfast-table staple.",
     "Pasteurized whole milk.", ["family-size", "breakfast-staple"]),
    ("Free-Range Eggs 12ct", 1, "PureFarm", 5.49, None,
     "A dozen free-range eggs from pasture-raised hens, a versatile high-protein staple.",
     "Free-range eggs.", ["healthy", "high-protein", "premium"]),
    ("Cheddar Cheese Block 400g", 1, "DairyBest", 5.99, None,
     "Aged sharp cheddar block, great for sandwiches and family snack boards.",
     "Milk, salt, enzymes, cheese cultures.", ["indulgent", "family-size"]),
    ("Unsalted Butter 250g", 1, "DairyBest", 3.79, None,
     "Creamy unsalted butter for baking and breakfast toast.",
     "Cream.", ["breakfast-staple", "indulgent"]),

    # Bakery (2)
    ("Whole Wheat Bread Loaf", 2, "GoldenGrain", 3.29, None,
     "Soft whole wheat sandwich bread made with whole-grain flour.",
     "Whole wheat flour, water, yeast, salt.", ["whole-grain", "healthy", "breakfast-staple"]),
    ("Sourdough Bread Loaf", 2, "GoldenGrain", 4.49, None,
     "Naturally leavened sourdough loaf with a crisp crust.",
     "Wheat flour, water, salt, sourdough starter.", ["premium", "breakfast-staple"]),
    ("Chocolate Croissants 4-pack", 2, "HomeStyle", 4.99, 3.99,
     "Buttery croissants filled with rich chocolate, a family weekend treat.",
     "Wheat flour, butter, chocolate, sugar, yeast.", ["indulgent", "kid-friendly"]),
    ("Everything Bagels 6-pack", 2, "GoldenGrain", 3.99, None,
     "Chewy bagels topped with a classic everything seasoning blend.",
     "Wheat flour, water, yeast, sesame, poppy seed, garlic, onion.", ["breakfast-staple", "family-size"]),

    # Breakfast & Cereal (3)
    ("Rolled Oats 1kg", 3, "NutriCo", 3.49, None,
     "Whole-grain rolled oats, a high-protein low-cost breakfast staple.",
     "100% whole grain rolled oats.", ["healthy", "whole-grain", "high-protein", "breakfast-staple"]),
    ("Honey Nut Cereal 500g", 3, "SnackWorks", 4.29, None,
     "Sweet honey and nut flavored cereal, a kid-friendly breakfast favorite.",
     "Corn, oats, honey, sugar, nuts.", ["kid-friendly", "breakfast-staple", "indulgent"]),
    ("Granola Clusters 400g", 3, "NutriCo", 4.99, None,
     "Baked whole-grain granola clusters with a light honey glaze, good on the go.",
     "Oats, honey, almonds, sunflower oil.", ["healthy", "whole-grain", "on-the-go"]),
    ("Pancake Mix 700g", 3, "HomeStyle", 3.99, None,
     "Fluffy pancake mix, a family breakfast favorite that just needs milk and eggs.",
     "Wheat flour, sugar, baking powder, salt.", ["family-size", "breakfast-staple", "kid-friendly"]),

    # Snacks (4)
    ("Sea Salt Potato Chips 200g", 4, "SnackWorks", 2.79, None,
     "Crunchy kettle-cooked potato chips with sea salt.",
     "Potatoes, sunflower oil, sea salt.", ["indulgent", "budget-friendly"]),
    ("Mixed Nuts Trail Mix 300g", 4, "NutriCo", 6.49, None,
     "A protein-rich mix of almonds, cashews, and walnuts for an on-the-go snack.",
     "Almonds, cashews, walnuts, sea salt.", ["healthy", "high-protein", "on-the-go", "premium"]),
    ("Chocolate Chip Cookies 250g", 4, "HomeStyle", 3.49, None,
     "Classic soft-baked chocolate chip cookies.",
     "Wheat flour, butter, chocolate chips, sugar.", ["indulgent", "kid-friendly"]),
    ("Protein Bars Box 6ct", 4, "NutriCo", 8.99, None,
     "High-protein low-sugar snack bars for a quick on-the-go boost.",
     "Whey protein, oats, nuts, natural sweetener.", ["high-protein", "healthy", "on-the-go", "low-sugar"]),
    ("Buttered Popcorn 150g", 4, "SnackWorks", 2.29, None,
     "Ready-to-eat buttered popcorn, a kid-friendly movie-night snack.",
     "Popcorn, butter, salt.", ["indulgent", "budget-friendly", "kid-friendly"]),
    ("Pretzel Sticks 250g", 4, "Value Choice", 1.99, None,
     "Crunchy salted pretzel sticks, a budget-friendly everyday snack.",
     "Wheat flour, salt, yeast.", ["budget-friendly", "on-the-go"]),

    # Coffee & Tea (6)
    ("Ground Coffee Medium Roast 340g", 6, "BrewHouse", 7.99, None,
     "Smooth medium roast ground coffee for a premium morning brew.",
     "100% Arabica ground coffee.", ["premium", "on-the-go"]),
    ("Green Tea Bags 20ct", 6, "PureFarm", 3.49, None,
     "Light, antioxidant-rich green tea bags.",
     "Green tea leaves.", ["healthy", "low-sugar"]),
    ("Instant Coffee Jar 200g", 6, "Value Choice", 4.49, None,
     "Budget-friendly instant coffee for a quick everyday cup.",
     "100% instant coffee granules.", ["budget-friendly", "on-the-go"]),

    # Juices & Sodas (7)
    ("Orange Juice 1L", 7, "SunHarvest", 3.99, None,
     "Freshly squeezed orange juice, a healthy family breakfast staple.",
     "100% orange juice.", ["healthy", "breakfast-staple", "family-size"]),
    ("Cola Soda 2L", 7, "SnackWorks", 2.49, None,
     "Classic cola soda, a family-size indulgent treat.",
     "Carbonated water, sugar, caramel color, caffeine.", ["indulgent", "budget-friendly", "family-size"]),
    ("Sparkling Water Lime 6-pack", 7, "PureFarm", 4.29, None,
     "Zero-sugar sparkling water with a hint of lime, refreshing on the go.",
     "Carbonated water, natural lime flavor.", ["healthy", "low-sugar", "on-the-go"]),

    # Produce (8)
    ("Bananas 1kg", 8, "SunHarvest", 1.49, None,
     "Fresh ripe bananas, a kid-friendly everyday family fruit.",
     "Bananas.", ["healthy", "kid-friendly", "family-size"]),
    ("Baby Spinach 200g", 8, "SunHarvest", 2.99, None,
     "Tender organic baby spinach leaves, great in salads and smoothies.",
     "Organic spinach.", ["healthy", "vegetarian", "organic"]),
    ("Avocados 4-pack", 8, "SunHarvest", 4.99, None,
     "Ripe creamy avocados, a premium healthy staple.",
     "Avocados.", ["healthy", "premium"]),
    ("Roma Tomatoes 500g", 8, "FreshFields", 2.29, None,
     "Fresh Roma tomatoes, ideal for sauces and salads.",
     "Tomatoes.", ["healthy", "vegetarian"]),
    ("Broccoli Crowns 500g", 8, "FreshFields", 2.79, None,
     "Fresh broccoli crowns, a low-sugar vegetarian side.",
     "Broccoli.", ["healthy", "vegetarian", "low-sugar"]),
    ("Gala Apples 1kg", 8, "SunHarvest", 3.29, None,
     "Crisp sweet Gala apples, a kid-friendly family fruit.",
     "Apples.", ["healthy", "kid-friendly", "family-size"]),

    # Meat & Seafood (9)
    ("Chicken Breast 1kg", 9, "FreshFields", 8.49, None,
     "Fresh boneless skinless chicken breast, a high-protein family staple.",
     "Chicken breast.", ["high-protein", "family-size"]),
    ("Ground Beef 500g", 9, "FreshFields", 6.49, None,
     "Fresh lean ground beef for family dinners.",
     "Beef.", ["high-protein", "family-size"]),
    ("Atlantic Salmon Fillet 400g", 9, "OceanCatch", 11.99, None,
     "Premium fresh Atlantic salmon fillet, rich in protein and omega-3s.",
     "Salmon.", ["healthy", "high-protein", "premium"]),
    ("Peeled Shrimp 400g", 9, "OceanCatch", 10.49, None,
     "Premium peeled and deveined shrimp, ready to cook.",
     "Shrimp.", ["high-protein", "premium"]),

    # Pantry & Staples (10)
    ("Basmati Rice 2kg", 10, "GoldenGrain", 6.99, None,
     "Long-grain basmati rice, a family-size budget-friendly pantry staple.",
     "Basmati rice.", ["family-size", "budget-friendly", "gluten-free"]),
    ("Whole Wheat Pasta 500g", 10, "GoldenGrain", 2.49, None,
     "Whole-grain pasta, a healthy budget-friendly dinner base.",
     "Whole wheat semolina.", ["whole-grain", "healthy", "budget-friendly"]),
    ("Canned Black Beans 400g", 10, "Value Choice", 1.29, None,
     "Ready-to-eat canned black beans, a budget-friendly vegetarian protein source.",
     "Black beans, water, salt.", ["vegetarian", "high-protein", "budget-friendly"]),
    ("Extra Virgin Olive Oil 500ml", 10, "PureFarm", 8.99, None,
     "Cold-pressed extra virgin olive oil, a premium healthy pantry staple.",
     "100% extra virgin olive oil.", ["healthy", "premium"]),
    ("Peanut Butter 500g", 10, "NutriCo", 4.49, None,
     "Creamy peanut butter, a kid-friendly high-protein budget staple.",
     "Peanuts, salt.", ["high-protein", "kid-friendly", "budget-friendly"]),

    # Frozen Foods (11)
    ("Frozen Mixed Berries 500g", 11, "SunHarvest", 4.49, None,
     "Flash-frozen mixed berries, a healthy low-sugar family smoothie staple.",
     "Strawberries, blueberries, raspberries.", ["healthy", "low-sugar", "family-size"]),
    ("Frozen Margherita Pizza", 11, "HomeStyle", 5.49, 4.49,
     "Oven-ready margherita pizza, a kid-friendly family dinner shortcut.",
     "Wheat flour, tomato sauce, mozzarella, basil.", ["indulgent", "family-size", "kid-friendly"]),
    ("Frozen Vegetable Medley 1kg", 11, "FreshFields", 3.99, None,
     "A budget-friendly frozen mix of carrots, peas, and corn for family meals.",
     "Carrots, peas, corn.", ["healthy", "vegetarian", "budget-friendly", "family-size"]),

    # Health & Protein (12)
    ("Whey Protein Powder 900g", 12, "NutriCo", 24.99, None,
     "Premium low-sugar whey protein powder for post-workout recovery.",
     "Whey protein concentrate, natural flavoring.", ["high-protein", "premium", "low-sugar"]),
    ("Chia Seeds 300g", 12, "PureFarm", 5.99, None,
     "Organic gluten-free chia seeds, rich in fiber and omega-3s.",
     "Organic chia seeds.", ["healthy", "organic", "gluten-free"]),
    ("Unsweetened Almond Milk 1L", 12, "NutriCo", 3.29, None,
     "Dairy-free unsweetened almond milk, low in sugar.",
     "Filtered water, almonds.", ["dairy-free", "low-sugar", "healthy"]),
    ("Low-Sugar Protein Cookies 6ct", 12, "NutriCo", 7.49, None,
     "High-protein low-sugar cookies for an on-the-go healthy treat.",
     "Whey protein, oats, natural sweetener.", ["high-protein", "low-sugar", "on-the-go"]),
    ("Ginger Kombucha 4-pack", 12, "PureFarm", 6.99, None,
     "Organic fermented ginger kombucha, a low-sugar gut-healthy beverage.",
     "Fermented tea, ginger.", ["healthy", "organic", "low-sugar"]),

    # Condiments & Sauces (13)
    ("Tomato Ketchup 500ml", 13, "Value Choice", 2.29, None,
     "Classic family-size tomato ketchup, a kid-friendly budget staple.",
     "Tomatoes, sugar, vinegar, salt.", ["budget-friendly", "family-size", "kid-friendly"]),
    ("Hot Sauce 250ml", 13, "HomeStyle", 3.99, None,
     "Small-batch premium hot sauce with a bold kick.",
     "Chili peppers, vinegar, salt.", ["premium", "indulgent"]),
]


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug


def build_categories() -> list[RawCategory]:
    return [RawCategory(id=cid, name=name, parent_id=parent_id) for cid, name, parent_id in _CATEGORY_ROWS]


def build_tags() -> list[RawTag]:
    return [RawTag(id=i + 1, name=name) for i, name in enumerate(TAG_VOCABULARY)]


def build_catalog(
    rng: np.random.Generator,
    out_of_stock_prob: float = 0.04,
    inactive_prob: float = 0.02,
    sale_jitter_prob: float = 0.15,
) -> tuple[list[RawProduct], list[RawProductTag]]:
    """Build the ~50-product catalog. Content (name/brand/price/tags) is
    fixed/authored; `stock_quantity`, `is_active`, and extra promotional
    `sale_price` assignment are jittered by `rng` so those operational
    fields aren't hand-picked while staying reproducible.
    """
    tag_id_by_name = {name: i + 1 for i, name in enumerate(TAG_VOCABULARY)}

    products: list[RawProduct] = []
    product_tags: list[RawProductTag] = []
    tag_row_id = 1

    for i, (name, category_id, brand, price, sale_price, description, ingredients, tags) in enumerate(_PRODUCT_ROWS):
        product_id = i + 1

        stock = 0 if rng.random() < out_of_stock_prob else int(rng.integers(5, 200))
        is_active = rng.random() >= inactive_prob

        effective_sale_price = sale_price
        if effective_sale_price is None and rng.random() < sale_jitter_prob:
            effective_sale_price = round(price * float(rng.uniform(0.75, 0.9)), 2)
        discount_percentage = (
            round((price - effective_sale_price) / price * 100, 1) if effective_sale_price is not None else None
        )

        products.append(
            RawProduct(
                id=product_id,
                category_id=category_id,
                slug=_slugify(name),
                name=name,
                description=description,
                brand=brand,
                price=price,
                sale_price=effective_sale_price,
                discount_percentage=discount_percentage,
                stock_quantity=stock,
                ingredients=ingredients,
                is_active=is_active,
                product_image=f"https://example-cdn.invalid/products/{_slugify(name)}.jpg",
                alt_text=name,
            )
        )
        for tag_name in tags:
            product_tags.append(RawProductTag(id=tag_row_id, product_id=product_id, tag_id=tag_id_by_name[tag_name]))
            tag_row_id += 1

    return products, product_tags


CATEGORY_NAME_BY_ID: dict[int, str] = {cid: name for cid, name, _ in _CATEGORY_ROWS}
CATEGORY_ID_BY_NAME: dict[str, int] = {name: cid for cid, name, _ in _CATEGORY_ROWS}
ALL_BRANDS: list[str] = sorted({row[2] for row in _PRODUCT_ROWS})
