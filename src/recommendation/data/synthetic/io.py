"""Persist/load a SyntheticDataset as one JSON file per table.

Used by `scripts/generate_synthetic_dataset.py` and, later, by the API/
dashboard so training and serving read the same on-disk snapshot rather
than regenerating (non-deterministic wall-clock cost aside, regeneration
is deterministic given the same seed, but a fixed snapshot is still what a
real deployment would load from).
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from recommendation.data.schemas.engagement import ChatbotContextRecord, ClickRecord, SearchRecord
from recommendation.data.synthetic.dataset import SyntheticDataset
from recommendation.data.synthetic.raw_schemas import (
    RawCart,
    RawCartItem,
    RawCategory,
    RawOrder,
    RawOrderItem,
    RawProduct,
    RawProductTag,
    RawReview,
    RawTag,
    RawUser,
)

_TABLE_MODELS = {
    "categories": RawCategory,
    "tags": RawTag,
    "products": RawProduct,
    "product_tags": RawProductTag,
    "users": RawUser,
    "orders": RawOrder,
    "order_items": RawOrderItem,
    "carts": RawCart,
    "cart_items": RawCartItem,
    "reviews": RawReview,
    "click_records": ClickRecord,
    "search_records": SearchRecord,
    "chatbot_records": ChatbotContextRecord,
}


def save_dataset(dataset: SyntheticDataset, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for table_name in _TABLE_MODELS:
        rows = getattr(dataset, table_name)
        adapter = TypeAdapter(list[_TABLE_MODELS[table_name]])
        (out_dir / f"{table_name}.json").write_bytes(adapter.dump_json(rows, indent=2))
    (out_dir / "debug_user_personas.json").write_text(
        json.dumps(dataset.debug_user_personas, indent=2), encoding="utf-8"
    )


def load_dataset(in_dir: str | Path) -> SyntheticDataset:
    in_dir = Path(in_dir)
    tables: dict[str, list] = {}
    for table_name, model in _TABLE_MODELS.items():
        adapter = TypeAdapter(list[model])
        tables[table_name] = adapter.validate_json((in_dir / f"{table_name}.json").read_bytes())
    debug_user_personas = json.loads((in_dir / "debug_user_personas.json").read_text(encoding="utf-8"))
    debug_user_personas = {int(k): v for k, v in debug_user_personas.items()}
    return SyntheticDataset(**tables, debug_user_personas=debug_user_personas)
