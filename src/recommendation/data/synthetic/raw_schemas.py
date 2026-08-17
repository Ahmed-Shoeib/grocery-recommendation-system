""""Raw" backend-table-shaped models.

These mirror `docs/erd.jpeg` field-for-field (Category, Product, Tag,
ProductTags, User, Cart, CartItem, Order, OrderItem, Review) and stand in
for what a real ORM row / DB query result would look like. They exist only
inside `recommendation.data.synthetic` and are consumed by the matching
`InMemory*Adapter` implementations in `recommendation.data.adapters`.

This is the seam a real backend integration replaces: swap the synthetic
generator + these raw models for actual DB queries / an API client
returning the same field shapes, keep the adapter classes (or write SQL-
backed ones with identical constructors), and every canonical schema,
feature, and model downstream is unaffected.

Excluded on purpose (per the V1 scope decision, docs/data-mapping.md
section 1/7): no impression fields, no session ids. CLICK is a supported
V1 engagement signal (docs/data-mapping.md section 4), but - like
search/chatbot - has no dedicated ERD raw table here; it is produced
directly as canonical `ClickRecord`s by `synthetic.clicks
.generate_click_records`, standing in for the future `User_events`
table's CLICK rows. Timestamp
fields that genuinely exist in the ERD (Order.CreationDate,
Order.DeliveryDate, Review.CreationDate) are kept here and threaded through
to the canonical schemas, but are never consumed by V1 feature engineering.

`preferred_category_id` / `age_group` on RawUser model the confirmed
(pending) backend addition to `User` (docs/data-mapping.md section 2).
`preferred_category_id` is modeled as an FK to Category, matching how the
rest of the schema references categories; the UserAdapter resolves it to
a category name for the canonical `UserProfile`.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class RawCategory(BaseModel):
    id: int
    name: str
    parent_id: int | None = None


class RawTag(BaseModel):
    id: int
    name: str


class RawProduct(BaseModel):
    id: int
    category_id: int
    slug: str
    name: str
    description: str | None = None
    brand: str | None = None
    price: float = Field(gt=0)
    sale_price: float | None = Field(default=None, gt=0)
    discount_percentage: float | None = Field(default=None, ge=0, le=100)
    stock_quantity: int = Field(ge=0, default=0)
    ingredients: str | None = None
    is_active: bool = True
    product_image: str | None = None
    alt_text: str | None = None


class RawProductTag(BaseModel):
    id: int
    product_id: int
    tag_id: int


class RawUser(BaseModel):
    id: int
    first_name: str
    last_name: str
    email: str
    preferred_category_id: int | None = None
    age_group: str | None = None


class RawCart(BaseModel):
    id: int
    user_id: int


class RawCartItem(BaseModel):
    id: int
    cart_id: int
    product_id: int
    quantity: int = Field(ge=1)


class RawOrder(BaseModel):
    id: int
    user_id: int
    status: str
    creation_date: datetime
    delivery_date: datetime | None = None


class RawOrderItem(BaseModel):
    id: int
    order_id: int
    product_id: int
    quantity: int = Field(ge=1)
    unit_price: float = Field(gt=0)


class RawReview(BaseModel):
    id: int
    user_id: int
    product_id: int
    rating: float = Field(ge=1, le=5)
    comment: str | None = None
    creation_date: datetime | None = None
