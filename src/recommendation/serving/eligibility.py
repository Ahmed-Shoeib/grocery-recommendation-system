"""Final business-rules/eligibility policy (Phase 7) - applied LAST, after
retrieval, ranking, and re-ranking (docs/data-mapping.md section 5). V1
uses exactly the fields the ERD actually has: `Product.isActive` and
`Product.stockQuantity`. No candidate is dropped before this stage.

A plain, ordered list of named predicate rules, not hard-coded boolean
logic inline - a future rule (a real `isDeleted`/soft-delete flag,
regional restrictions, other purchase-eligibility rules) is another
`EligibilityRule` appended in `build_eligibility_rules`, with zero
changes to `apply_eligibility` or its callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from recommendation.features.product_features import ProductFeatures
from recommendation.utils.config import EligibilityConfig


@dataclass
class EligibilityRule:
    name: str
    predicate: Callable[[ProductFeatures], bool]  # True = passes / eligible


@dataclass
class EligibilityResult:
    eligible_ids: list[int]
    excluded_ids: list[int]
    excluded_reasons: dict[int, list[str]] = field(default_factory=dict)


def build_eligibility_rules(config: EligibilityConfig) -> list[EligibilityRule]:
    rules: list[EligibilityRule] = []
    if config.require_active:
        rules.append(EligibilityRule("is_active", lambda pf: pf.is_active))
    if config.require_in_stock:
        rules.append(EligibilityRule("in_stock", lambda pf: pf.stock_quantity > 0))
    return rules


def apply_eligibility(
    candidate_ids: list[int],
    product_features: dict[int, ProductFeatures],
    rules: list[EligibilityRule],
) -> EligibilityResult:
    eligible: list[int] = []
    excluded: list[int] = []
    reasons: dict[int, list[str]] = {}

    for product_id in candidate_ids:
        pf = product_features.get(product_id)
        failed = [rule.name for rule in rules if pf is None or not rule.predicate(pf)]
        if failed:
            excluded.append(product_id)
            reasons[product_id] = failed
        else:
            eligible.append(product_id)

    return EligibilityResult(eligible_ids=eligible, excluded_ids=excluded, excluded_reasons=reasons)
