"""Hard global catalog-eligibility policy - `isActive`/`stockQuantity`,
the fields the ERD actually has. No `isDeleted` or similar field is
invented.

This one policy (`build_eligibility_rules` + `apply_eligibility`) is
applied at TWO points in `serving.pipeline.generate_recommendations`,
never a third stage in between:

1. **Pre-retrieval** (hard gate): before any candidate generation -
   personalized (Two-Tower/VectorIndex, via `retrieval.index
   .eligibility_filter.EligibilityRestrictedIndex`) or fallback
   (category/global popularity) - so inactive/out-of-stock products never
   become candidates, never get ranked, never get re-ranked.
2. **Final validation** (lightweight safety net): again, over the fully
   re-ranked pool, immediately before the Top-N is returned - defense in
   depth against a product becoming unavailable between pre-retrieval
   filtering and the final response, independent of whether stage 1
   already guarantees eligibility.

Both stages call this same module so the rules can never drift apart
between them. User-specific/list-specific business rules (a future
regional restriction, a purchase-eligibility rule tied to the requesting
user) are NOT pre-retrieval-hard-eligibility - they belong at the final
stage only, since they aren't global catalog facts (see
docs/data-mapping.md section 5).

A plain, ordered list of named predicate rules, not hard-coded boolean
logic inline - a future rule (a real `isDeleted`/soft-delete flag, other
GLOBAL catalog-eligibility rules) is another `EligibilityRule` appended
in `build_eligibility_rules`, with zero changes to `apply_eligibility`,
its callers, or which of the two pipeline stages it participates in.
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
