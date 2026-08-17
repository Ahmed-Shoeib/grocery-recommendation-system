"""Wires the concrete synthetic (in-memory) adapters together.

The only place in the codebase that knows both "we're on synthetic V1
data" and "here are the concrete adapter classes" - everything else
(feature engineering, models, API, dashboard) depends on the `AdapterBundle`
interface types, not on the fact that they happen to be in-memory/synthetic
today. Swapping to a real backend later means writing one new
`build_backend_adapters(...)`-style factory, not touching call sites.
"""

from __future__ import annotations

from recommendation.data.adapters.base import AdapterBundle
from recommendation.data.adapters.cart_adapter import InMemoryCartAdapter
from recommendation.data.adapters.chatbot_adapter import SyntheticChatbotAdapter
from recommendation.data.adapters.click_adapter import SyntheticClickAdapter
from recommendation.data.adapters.product_adapter import InMemoryProductCatalogAdapter
from recommendation.data.adapters.purchase_adapter import InMemoryPurchaseAdapter
from recommendation.data.adapters.review_adapter import InMemoryReviewAdapter
from recommendation.data.adapters.search_adapter import SyntheticSearchAdapter
from recommendation.data.adapters.user_adapter import InMemoryUserAdapter
from recommendation.data.synthetic.dataset import SyntheticDataset
from recommendation.utils.config import SyntheticDataConfig


def build_synthetic_adapters(dataset: SyntheticDataset, config: SyntheticDataConfig) -> AdapterBundle:
    return AdapterBundle(
        products=InMemoryProductCatalogAdapter(
            dataset.categories, dataset.tags, dataset.products, dataset.product_tags
        ),
        users=InMemoryUserAdapter(dataset.users, dataset.categories),
        purchases=InMemoryPurchaseAdapter(
            dataset.orders, dataset.order_items, counted_statuses=set(config.order_counted_statuses)
        ),
        cart=InMemoryCartAdapter(dataset.carts, dataset.cart_items),
        clicks=SyntheticClickAdapter(dataset.click_records),
        reviews=InMemoryReviewAdapter(dataset.reviews),
        search=SyntheticSearchAdapter(dataset.search_records),
        chatbot=SyntheticChatbotAdapter(dataset.chatbot_records),
    )
