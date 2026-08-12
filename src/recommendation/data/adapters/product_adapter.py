"""ProductCatalogAdapter backed by ERD entities Product, Category, Tag, ProductTags.

`InMemoryProductCatalogAdapter` performs the same join a real backend
implementation would run in SQL: Product -> Category (+ parent Category)
-> ProductTags -> Tag. A future SQL-backed adapter implementing
`ProductCatalogAdapter` can replace this one without any change to
callers, since both return `recommendation.data.schemas.product.Product`.
"""

from __future__ import annotations

from recommendation.data.adapters.base import ProductCatalogAdapter
from recommendation.data.schemas.product import Product
from recommendation.data.synthetic.raw_schemas import RawCategory, RawProduct, RawProductTag, RawTag


class InMemoryProductCatalogAdapter(ProductCatalogAdapter):
    def __init__(
        self,
        categories: list[RawCategory],
        tags: list[RawTag],
        products: list[RawProduct],
        product_tags: list[RawProductTag],
    ) -> None:
        category_by_id = {c.id: c for c in categories}
        tag_name_by_id = {t.id: t.name for t in tags}

        tags_by_product_id: dict[int, list[str]] = {}
        for pt in product_tags:
            tags_by_product_id.setdefault(pt.product_id, []).append(tag_name_by_id[pt.tag_id])

        self._products: dict[int, Product] = {}
        for raw_product in products:
            category = category_by_id.get(raw_product.category_id)
            parent_category_name = None
            if category is not None and category.parent_id is not None:
                parent = category_by_id.get(category.parent_id)
                parent_category_name = parent.name if parent is not None else None

            self._products[raw_product.id] = Product(
                id=raw_product.id,
                category_id=raw_product.category_id,
                slug=raw_product.slug,
                name=raw_product.name,
                description=raw_product.description,
                brand=raw_product.brand,
                price=raw_product.price,
                sale_price=raw_product.sale_price,
                discount_percentage=raw_product.discount_percentage,
                stock_quantity=raw_product.stock_quantity,
                ingredients=raw_product.ingredients,
                is_active=raw_product.is_active,
                product_image=raw_product.product_image,
                alt_text=raw_product.alt_text,
                tags=sorted(tags_by_product_id.get(raw_product.id, [])),
                category_name=category.name if category is not None else None,
                parent_category_name=parent_category_name,
            )

    def list_products(self) -> list[Product]:
        return list(self._products.values())

    def get_product(self, product_id: int) -> Product | None:
        return self._products.get(product_id)
