"""Build the FINAL live-serving ANN for `data_source=backend_api` from the
REAL backend catalog - task section 20/21 of the production-aligned
retrain.

Does NOT train or retrain anything: loads the ALREADY-TRAINED
`models/backend_api/two_tower/` item tower + feature encoder (produced by
`scripts/train_backend_api_pipeline.py` against the SQLite production-
aligned dataset) and runs pure inference over the REAL backend catalog,
fetched read-only via `GET /api/ai/products` (+ `GET /api/ai/user-activities`
for real purchase/cart/review aggregate features, + `GET /api/categories`)
using the existing secure service-auth integration - never SQL Server,
never DB credentials.

What this produces:
  1. Production-safe product text + fresh Sentence-Transformer embeddings
     for every REAL backend product.
  2. Real `ProductFeatures` built from the catalog alone (price/stock/
     category - always real) via `GET /api/ai/products` +
     `GET /api/categories` ONLY. Deliberately does NOT call
     `GET /api/ai/user-activities` (via `adapters.backend_factory
     .build_backend_api_adapters`, which this script bypasses on purpose):
     that endpoint's cursor pagination hit `BackendPaginationError`
     (`exceeded 10000-page budget`) when this script first ran - the real
     `UserActivities` table has grown past ~1,000,000 rows since the last
     audit snapshot, which is itself a real production-readiness finding
     worth flagging on its own (see the accompanying report), not
     something this script papers over by raising the page cap and
     blocking for many minutes on a full-history pull it does not need.
     Purchase/cart aggregate features (`purchase_count`, `cart_add_count`)
     and review aggregates (`review_count`, `average_rating`) therefore
     degrade to their documented neutral defaults (0 / 0 / None) for every
     real product here - the SAME graceful degradation
     `features.product_features.build_product_features` already performs
     for any catalog with no purchase/cart/review history, not a new code
     path. Price/stock/category - the fields section 20 actually asked
     for - are always real.
  3. Real 128-D item-tower embeddings, keyed by the REAL backend ProductId
     (never a synthetic SQLite id, never slug identity).
  4. Overwrites `models/backend_api/two_tower/item_embeddings.npz` (+
     `.meta.json`-equivalent id list) with these real embeddings - this is
     the exact file `retrieval.two_tower.serialization.load_two_tower_artifacts`
     reads and `api.service.build_recommendation_service` feeds straight
     into `VectorIndex.build(...)` at live-serving startup, so overwriting
     it IS "building the final live-serving ANN" in this codebase's
     existing architecture (the index itself is always rebuilt in-memory
     from this file at process startup, never loaded from a serialized
     index file - see `api/service.py`).

Before overwriting, the PREVIOUS contents (the offline/training-catalog
embeddings `scripts/train_backend_api_pipeline.py` just wrote, over the
SQLite production-aligned catalog) are backed up to a clearly-labeled
sibling file (`item_embeddings_offline_training_catalog.npz` +
`.offline_training_catalog_ids.json`) so the two are never confused - see
task section 17. The offline evaluation numbers already computed and
persisted to `models/backend_api/offline_report.json` remain valid and
reproducible from `scripts/train_backend_api_pipeline.py`; this script
does not touch that file.

The FAISS `vector_index/faiss_index.bin` diagnostic artifact is rebuilt
the same way (backup + overwrite) for local manual-inspection convenience,
even though live serving never reads it directly (see module docstring
above) - kept for parity with the training script's own convention.

Usage:
    python scripts/build_live_backend_ann.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from recommendation.adapters.product_adapter import InMemoryProductCatalogAdapter
from recommendation.backend.client import BackendApiClient
from recommendation.backend.identity import ExternalIdentityResolver
from recommendation.backend.loader import load_backend_catalog
from recommendation.embeddings.encoder import SentenceTransformerEncoder
from recommendation.embeddings.product_embeddings import get_or_compute_product_embeddings
from recommendation.embeddings.text_builder import build_product_text
from recommendation.features.product_features import build_product_features
from recommendation.retrieval.index.factory import build_vector_index
from recommendation.retrieval.two_tower.serialization import load_two_tower_artifacts, save_two_tower_artifacts
from recommendation.serving.eligibility import apply_eligibility, build_eligibility_rules
from recommendation.config import get_config, resolve_path
from recommendation.logging import get_logger, setup_logging

logger = get_logger(__name__)


def _load_dotenv_if_present(repo_root: Path) -> None:
    env_path = repo_root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _load_dotenv_if_present(repo_root)

    config = get_config()
    setup_logging(config.log_level)

    models_root = resolve_path(config.paths.models_dir) / "backend_api"
    tt_dir = models_root / "two_tower"
    if not tt_dir.exists():
        raise SystemExit(f"{tt_dir} does not exist - run scripts/train_backend_api_pipeline.py first.")

    print("=" * 78)
    print("Loading TRAINED item tower + feature encoder (no training happens here)")
    print("=" * 78)
    artifacts = load_two_tower_artifacts(tt_dir)
    print(f"loaded from {tt_dir} (model_version={artifacts.metadata.get('model_version')}, "
          f"contract_version={artifacts.encoder.contract_version})")
    print(f"training catalog this tower was FIT against: {artifacts.metadata.get('training_dataset_num_products')} "
          f"products from {artifacts.metadata.get('training_data_source')}")

    print()
    print("=" * 78)
    print("Fetching REAL backend catalog (read-only GET /api/ai/products, /api/categories - see module docstring")
    print("re: why /api/ai/user-activities is deliberately NOT called here)")
    print("=" * 78)
    client = BackendApiClient(config.backend_api)
    resolver = ExternalIdentityResolver(resolve_path(config.paths.backend_identity_registry))
    catalog = load_backend_catalog(client, resolver)
    resolver.save()
    products_adapter = InMemoryProductCatalogAdapter(catalog.categories, catalog.tags, catalog.products, catalog.product_tags)
    products = products_adapter.list_products()
    print(f"real backend products: {len(products)}")
    if not products:
        raise SystemExit("Backend returned 0 products - refusing to build an empty live ANN.")

    real_category_names = sorted({p.category_name for p in products if p.category_name})
    print(f"real categories on these products: {real_category_names}")
    unknown_categories = [n for n in real_category_names if artifacts.encoder.category_vocab.encode(n) == 0]
    print(f"categories mapping to 'unknown' in the TRAINED vocabulary: {unknown_categories or '(none)'}")
    if unknown_categories:
        raise SystemExit(
            f"Real categories {unknown_categories} are not in the trained category vocabulary "
            f"({artifacts.encoder.category_vocab.values}) - the training dataset domain has drifted from "
            "production since training. Regenerate the production-aligned SQLite dataset and retrain."
        )

    sample = products[0]
    print(f"sample real product text (id={sample.id}): {build_product_text(sample)!r}")

    print()
    print("=" * 78)
    print("Building real ProductFeatures (catalog-only: price/stock/category real; purchase/cart/review")
    print("aggregates default to neutral - see module docstring) + fresh semantic embeddings")
    print("=" * 78)
    # purchase/cart/review aggregates default to neutral (0/0/None) - see
    # module docstring. price/stock/category (what section 20 asked for)
    # are always real.
    product_features = build_product_features(products, all_purchases=[], all_cart_items=[], all_reviews=[])

    # Distinct cache: never the SQLite training-catalog cache, and distinct
    # from any earlier experimental live-catalog cache.
    embed_cache_path = resolve_path("data/processed/product_embeddings_live_backend_catalog.npz")
    st_encoder = SentenceTransformerEncoder(
        config.embedding.sentence_transformer_model, device=config.embedding.device, batch_size=config.embedding.encode_batch_size
    )
    embedding_cache, embeddings_recomputed = get_or_compute_product_embeddings(products, st_encoder, embed_cache_path)
    product_embeddings = embedding_cache.as_dict()
    print(f"products with computed features: {len(product_features)}")
    print(f"products with semantic embeddings: {len(product_embeddings)} (recomputed={embeddings_recomputed})")
    assert set(product_features.keys()) == {p.id for p in products}, "product id / features mismatch"
    assert set(product_embeddings.keys()) == {p.id for p in products}, "product id / embedding mismatch"

    eligibility_rules = build_eligibility_rules(config.eligibility)
    eligible_result = apply_eligibility([p.id for p in products], product_features, eligibility_rules)
    print(f"eligible (stock>0) real products: {len(eligible_result.eligible_ids)} / {len(products)}")

    print()
    print("=" * 78)
    print("Encoding real catalog through the TRAINED item tower")
    print("=" * 78)
    item_ids = [p.id for p in products]
    assert len(set(item_ids)) == len(item_ids), "duplicate real ProductIds - refusing to build ANN"
    assert all(isinstance(pid, int) for pid in item_ids), "non-integer product id found - identity must be ProductId, never slug"

    item_batch = artifacts.encoder.encode_item_batch(item_ids, product_features, product_embeddings)
    live_item_embeddings = np.asarray(artifacts.item_tower.predict(item_batch, verbose=0))
    norms = np.linalg.norm(live_item_embeddings, axis=1)
    print(f"live item embeddings: {live_item_embeddings.shape}  norm range=[{norms.min():.4f}, {norms.max():.4f}]")
    assert live_item_embeddings.shape == (len(item_ids), config.two_tower.output_dim)

    print()
    print("=" * 78)
    print("Backing up the OFFLINE/TRAINING-CATALOG artifacts before overwriting (section 17 - never confuse the two)")
    print("=" * 78)
    offline_npz = tt_dir / "item_embeddings.npz"
    offline_backup_npz = tt_dir / "item_embeddings_offline_training_catalog.npz"
    if offline_npz.exists():
        shutil.copy2(offline_npz, offline_backup_npz)
        print(f"backed up {offline_npz.name} -> {offline_backup_npz.name} "
              f"({artifacts.metadata.get('training_dataset_num_products')} SQLite training-catalog products - "
              "OFFLINE EVALUATION ONLY, never live-serving)")

    print()
    print("=" * 78)
    print("Overwriting item_embeddings.npz with the REAL backend catalog (this IS the live-serving ANN source)")
    print("=" * 78)
    updated_metadata = dict(artifacts.metadata)
    updated_metadata.update({
        "ann_catalog_source": "live backend API",
        "ann_built_at_utc": datetime.now(timezone.utc).isoformat(),
        "ann_num_products": len(item_ids),
        "ann_num_eligible_products": len(eligible_result.eligible_ids),
        "ann_product_id_source": "real backend ProductId (GET /api/ai/products) - never slug, never synthetic id",
        "ann_offline_training_catalog_backup": str(offline_backup_npz),
        "training_dataset_num_products_NOTE": (
            f"the towers were TRAINED against {artifacts.metadata.get('training_dataset_num_products')} SQLite "
            "production-aligned products (see training_data_source/training_dataset_path above) - this ANN's "
            f"{len(item_ids)} items are the REAL backend catalog the trained tower now SERVES, encoded via pure "
            "inference (no training happened in this script)."
        ),
    })
    save_two_tower_artifacts(
        tt_dir, artifacts.user_tower, artifacts.item_tower, artifacts.encoder,
        item_ids, live_item_embeddings, updated_metadata,
    )
    print(f"saved live-catalog item_embeddings.npz + updated metadata.json to {tt_dir}")

    print()
    print("=" * 78)
    print("Rebuilding the diagnostic FAISS index over the real catalog (not read by live serving - see module docstring)")
    print("=" * 78)
    idx_dir = models_root / "vector_index"
    idx_path = idx_dir / ("faiss_index.bin" if config.retrieval.backend == "faiss" else "scann_index")
    offline_idx_backup = idx_dir / f"offline_training_catalog_{idx_path.name}"
    if idx_path.exists():
        shutil.copy2(idx_path, offline_idx_backup)
        print(f"backed up {idx_path.name} -> {offline_idx_backup.name}")
    live_index = build_vector_index(config.retrieval)
    live_index.build(item_ids, live_item_embeddings)
    idx_dir.mkdir(parents=True, exist_ok=True)
    live_index.save(idx_path)
    print(f"saved live-catalog diagnostic index to {idx_path} ({live_index.size} products)")

    print()
    print("=" * 78)
    print("Verification")
    print("=" * 78)
    reloaded = load_two_tower_artifacts(tt_dir)
    assert reloaded.item_ids == item_ids, "reloaded item_ids do not match the real catalog"
    assert set(reloaded.item_ids) == {p.id for p in products}
    assert np.allclose(reloaded.item_embeddings, live_item_embeddings), "reloaded embeddings do not match"
    print(f"reloaded item_embeddings.npz: {len(reloaded.item_ids)} real ProductIds, matches expected catalog exactly")
    print(f"no duplicate ids: {len(set(reloaded.item_ids)) == len(reloaded.item_ids)}")
    print(f"no synthetic/slug ids: all {len(reloaded.item_ids)} ids are real backend ProductId integers")

    with open(models_root / "live_ann_build_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "ann_catalog_source": "live backend API",
            "num_products": len(item_ids),
            "num_eligible_products": len(eligible_result.eligible_ids),
            "categories": real_category_names,
            "unknown_category_count": len(unknown_categories),
            "product_ids": sorted(item_ids),
        }, f, indent=2)
    print(f"\nwrote {models_root / 'live_ann_build_report.json'}")


if __name__ == "__main__":
    main()
