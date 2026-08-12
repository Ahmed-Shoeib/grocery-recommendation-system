"""Latent grocery-shopper personas used to correlate synthetic interactions.

Each persona is a soft bias over product categories and attribute tags, not
a hard rule: `recommendation.data.synthetic.affinity` turns these weights
plus per-user Gaussian noise into a probability distribution over the
catalog, so e.g. a health-conscious user interacts with Greek yogurt/oats/
protein products *more often on average*, not exclusively. This is what
makes generated purchases/cart-adds/searches/chatbot mentions
statistically correlated instead of uniform-random, per the Phase 2
requirement.

Personas are a generation-time device only - they are never exposed to
adapters, features, or models, and have no equivalent in the real backend.
`age_group_weights` bias *data generation* realism; they must not be read
as a claim about which age groups actually prefer which products (see
docs/data-mapping.md section 2 on avoiding hard-coded age stereotypes in
the recommender itself).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    key: str
    label: str
    dominant_categories: list[str]
    category_weights: dict[str, float]
    tag_weights: dict[str, float]
    adjectives: list[str]
    age_group_weights: dict[str, float]
    quantity_bias: float = 1.0


AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

PERSONAS: list[Persona] = [
    Persona(
        key="health_conscious",
        label="Health-Conscious",
        dominant_categories=["Health & Protein", "Produce"],
        category_weights={
            "Health & Protein": 3.0, "Produce": 2.2, "Dairy & Eggs": 0.6,
            "Coffee & Tea": 0.3, "Snacks": -1.2, "Frozen Foods": -0.6, "Condiments & Sauces": -0.3,
        },
        tag_weights={
            "healthy": 2.5, "high-protein": 1.8, "low-sugar": 1.8, "organic": 1.6,
            "whole-grain": 1.2, "vegetarian": 0.6, "indulgent": -2.2, "budget-friendly": -0.2,
        },
        adjectives=["high-protein", "low-sugar", "organic"],
        age_group_weights={"18-24": 1.0, "25-34": 1.4, "35-44": 1.3, "45-54": 1.0, "55-64": 0.9, "65+": 0.7},
        quantity_bias=0.9,
    ),
    Persona(
        key="budget_oriented",
        label="Budget-Oriented",
        dominant_categories=["Pantry & Staples", "Frozen Foods"],
        category_weights={
            "Pantry & Staples": 2.4, "Frozen Foods": 1.6, "Snacks": 0.8,
            "Condiments & Sauces": 0.8, "Meat & Seafood": -0.6, "Health & Protein": -1.0,
        },
        tag_weights={
            "budget-friendly": 2.8, "family-size": 1.0, "indulgent": 0.2, "premium": -2.5, "organic": -1.2,
        },
        adjectives=["affordable", "everyday", "value"],
        age_group_weights={"18-24": 1.1, "25-34": 1.2, "35-44": 1.3, "45-54": 1.1, "55-64": 1.0, "65+": 1.0},
        quantity_bias=1.1,
    ),
    Persona(
        key="family_household",
        label="Family Household",
        dominant_categories=["Breakfast & Cereal", "Snacks", "Dairy & Eggs"],
        category_weights={
            "Breakfast & Cereal": 1.8, "Snacks": 1.6, "Dairy & Eggs": 1.4,
            "Bakery": 1.2, "Frozen Foods": 1.0, "Juices & Sodas": 0.8,
        },
        tag_weights={
            "family-size": 2.6, "kid-friendly": 2.2, "budget-friendly": 0.8, "indulgent": 0.5, "on-the-go": -0.3,
        },
        adjectives=["family-size", "kid-friendly", "everyday"],
        age_group_weights={"25-34": 1.2, "35-44": 1.8, "45-54": 1.4, "18-24": 0.5, "55-64": 0.6, "65+": 0.3},
        quantity_bias=1.6,
    ),
    Persona(
        key="snack_heavy",
        label="Snack-Heavy",
        dominant_categories=["Snacks", "Juices & Sodas"],
        category_weights={
            "Snacks": 2.8, "Juices & Sodas": 1.4, "Bakery": 0.8,
            "Frozen Foods": 0.6, "Health & Protein": -0.8, "Produce": -0.6,
        },
        tag_weights={
            "indulgent": 2.2, "on-the-go": 1.4, "kid-friendly": 0.8, "budget-friendly": 0.6,
            "healthy": -1.4, "low-sugar": -1.0,
        },
        adjectives=["indulgent", "on-the-go", "fun"],
        age_group_weights={"18-24": 1.8, "25-34": 1.4, "35-44": 0.9, "45-54": 0.6, "55-64": 0.4, "65+": 0.3},
        quantity_bias=1.0,
    ),
    Persona(
        key="dairy_heavy",
        label="Dairy-Heavy",
        dominant_categories=["Dairy & Eggs"],
        category_weights={"Dairy & Eggs": 3.0, "Breakfast & Cereal": 1.0, "Bakery": 0.8, "Health & Protein": 0.4},
        tag_weights={"breakfast-staple": 1.4, "indulgent": 0.6, "family-size": 0.5, "dairy-free": -3.0},
        adjectives=["creamy", "breakfast-staple", "fresh"],
        age_group_weights={"25-34": 1.2, "35-44": 1.4, "45-54": 1.2, "18-24": 0.9, "55-64": 0.9, "65+": 0.7},
        quantity_bias=1.2,
    ),
    Persona(
        key="vegetarian_oriented",
        label="Vegetarian-Oriented",
        dominant_categories=["Produce", "Pantry & Staples"],
        category_weights={
            "Produce": 2.6, "Pantry & Staples": 1.4, "Dairy & Eggs": 0.8,
            "Health & Protein": 0.8, "Meat & Seafood": -3.0,
        },
        tag_weights={"vegetarian": 2.6, "healthy": 1.4, "organic": 1.0, "whole-grain": 0.8, "gluten-free": 0.4},
        adjectives=["plant-based", "vegetarian", "fresh"],
        age_group_weights={"18-24": 1.3, "25-34": 1.5, "35-44": 1.1, "45-54": 0.9, "55-64": 0.7, "65+": 0.5},
        quantity_bias=1.0,
    ),
    Persona(
        key="breakfast_focused",
        label="Breakfast-Focused",
        dominant_categories=["Breakfast & Cereal", "Coffee & Tea"],
        category_weights={
            "Breakfast & Cereal": 2.8, "Coffee & Tea": 1.8, "Bakery": 1.4,
            "Dairy & Eggs": 1.0, "Juices & Sodas": 0.6,
        },
        tag_weights={"breakfast-staple": 2.4, "on-the-go": 1.0, "whole-grain": 0.6, "family-size": 0.4},
        adjectives=["breakfast-staple", "morning", "on-the-go"],
        age_group_weights={"25-34": 1.3, "35-44": 1.3, "45-54": 1.1, "18-24": 1.0, "55-64": 0.9, "65+": 0.8},
        quantity_bias=1.0,
    ),
    Persona(
        key="beverage_focused",
        label="Beverage-Focused",
        dominant_categories=["Coffee & Tea", "Juices & Sodas"],
        category_weights={"Coffee & Tea": 2.6, "Juices & Sodas": 2.4, "Snacks": 0.6, "Health & Protein": 0.4},
        tag_weights={"on-the-go": 1.6, "premium": 0.8, "low-sugar": 0.6, "budget-friendly": 0.4},
        adjectives=["refreshing", "on-the-go", "energizing"],
        age_group_weights={"18-24": 1.4, "25-34": 1.5, "35-44": 1.1, "45-54": 0.9, "55-64": 0.7, "65+": 0.6},
        quantity_bias=0.9,
    ),
]

PERSONA_BY_KEY: dict[str, Persona] = {p.key: p for p in PERSONAS}
