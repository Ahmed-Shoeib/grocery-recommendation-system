# Grocery Recommendation System

A production-oriented, modular personalized recommendation service for a
grocery e-commerce backend. Currently trained/served against a small
synthetic dataset; architected so the real backend can be substituted with
no model redesign once it's available. See `docs/data-mapping.md` for the
full ERD reconciliation, V1/V2 scope boundary, and design decisions.

## Status

Built in 10 sequential phases (see below). **Phases 1-7 complete**:
project scaffolding, canonical data adapters + synthetic dataset, the
Sentence Transformer + feature-engineering pipeline, a trained Two-Tower
retrieval model (128-D, L2-normalized, leave-one-out evaluated), a
ScaNN/FAISS `VectorIndex` over its embeddings, a neural ranker that
re-scores VectorIndex candidates with richer cross features (beats the
raw-retrieval-score baseline on every ranking metric), and the full V1
serving pipeline - three-level cold-start blending, dedup/diversity
re-ranking, and business-rules/eligibility applied last. No API or
dashboard yet.

## Architecture

```
Backend ERD (Order/OrderItem, Cart/CartItem, User, Review) + Synthetic (Search, Chatbot)
                              |
                     Data Adapter Layer
                              |
                 Canonical Engagement Model
                              |
                      Feature Engineering
                 +------------+------------+
            User Features              Product Features
                                             |
                                  Sentence Transformer (384-D, frozen)
                                             |
                              +--------------+--------------+
                          User Tower                   Item Tower
                        (128-D, L2-norm)             (128-D, L2-norm)
                              +--------------+--------------+
                                     cosine compatibility
                                             |
                          VectorIndex (ScaNN primary/Docker, FAISS Windows dev fallback)
                                             |
                                   Oversized candidate pool
                                             |
                                    Neural Ranker (MLP)
                                             |
                                   Re-ranking + cold-start blend
                                             |
                          Business Rules / Eligibility (isActive, stock)
                                             |
                                        Final Top-N
                                    +--------+--------+
                              Recommendation API   Streamlit Dashboard
```

Business rules / eligibility run **last** in V1, after ranking and
re-ranking, not as a pre-filter — the tiny V1 catalog makes ranking waste
a non-issue, and putting it last keeps the retrieval/ranking stages
oblivious to serving-time state (stock, active flag) entirely, per
`docs/data-mapping.md` section 5. `isActive`/`stockQuantity` are carried
through as structured product features so they're available at that final
stage without a separate lookup.

Four V1 personalization signals: previous purchases, add-to-cart habit,
searched items, and chatbot context — combined with `preferredCategory`
and `ageGroup` into a canonical `EngagementProfile`. See
`docs/data-mapping.md` for exactly which signals come from real ERD
entities today versus synthetic V1 adapters.

## Repository layout

```
configs/                  YAML configuration (hyperparameters, thresholds, paths)
data/{raw,processed,synthetic}/
docs/                      ERD + data-mapping / scope documentation
models/                    Serialized model artifacts (gitignored)
src/recommendation/
  data/
    adapters/              Backend + synthetic data adapters -> canonical schemas
    schemas/                Canonical pydantic schemas (Category, Product, UserProfile, EngagementProfile, ...)
    synthetic/              Synthetic V1 dataset generator
  features/                 EngagementProfile -> feature vectors
  embeddings/                Sentence Transformer product encoding + cache
  retrieval/
    two_tower/               User Tower / Item Tower model
    index/                    VectorIndex (ScaNN primary/production - Docker, FAISS Windows dev fallback)
  ranking/                    Neural ranker over VectorIndex candidates (features, model, train, evaluation, serialization)
  reranking/                  Duplicate removal + category/brand diversity re-ranking
  evaluation/                 Offline metrics + latency measurement
  serving/                    Cold-start tiering, fallback candidates, eligibility, and the full pipeline orchestrator (used by API and dashboard)
  api/                        FastAPI app
  ui/                         Streamlit dashboard
  utils/                      Config loading, logging
tests/
```

## Setup

Requires **Python 3.11–3.13** (3.14 is not yet supported by the pinned ML
libraries — verified against current TensorFlow/faiss-cpu/torch PyPI wheel
availability). This repo was developed against the 3.13 interpreter at
`C:\Users\ahmed.shoeib\AppData\Local\Programs\Python\Python313\python.exe`.

```bash
# from the repo root
"C:\Users\ahmed.shoeib\AppData\Local\Programs\Python\Python313\python.exe" -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"       # lightweight: config/schema/test deps only (Phase 1)
# pip install -e ".[full]"    # add once later phases need TF/FAISS/FastAPI/Streamlit
```

## Running tests

```bash
pytest
```

## Configuration

All tunables (paths, hyperparameters, candidate-pool sizing, cold-start
thresholds and blend weights, model version, random seed) live in
`configs/base.yaml`, loaded and validated by
`src/recommendation/utils/config.py`. Override the file path with the
`RECS_CONFIG_PATH` environment variable.

## Development phases

1. **Foundation & architecture** — this phase.
2. Canonical data layer & synthetic grocery dataset.
3. Feature engineering & semantic product embeddings.
4. Neural Two-Tower retrieval model.
5. ANN retrieval: ScaNN (primary/production backend, Linux/Docker) + FAISS (native-Windows dev fallback).
6. Neural ranking of VectorIndex candidates, richer than retrieval features, evaluated (NDCG/Precision/Recall/HitRate/MRR) against a raw-retrieval-score baseline.
7. Full serving pipeline: three-level cold-start blending (strong/sparse/no-history), dedup + category/brand diversity re-ranking, business rules/eligibility (applied last).
8. Recommendation API.
9. Recommendation dashboard.
10. Production hardening (full containerization of API/dashboard/serving) & documentation.

Each phase is a separate commit, reviewed and approved before the next
begins. See `docs/data-mapping.md` §11 for the full phase-to-decision map.
