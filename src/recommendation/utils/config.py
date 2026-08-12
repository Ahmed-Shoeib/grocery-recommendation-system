"""Typed configuration loading.

All tunables (paths, hyperparameters, thresholds, blend weights) live in
configs/*.yaml rather than being hard-coded, so later phases only ever edit
config, never scatter magic numbers through model/feature code.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"


class PathsConfig(BaseModel):
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    data_synthetic: str = "data/synthetic"
    models_dir: str = "models"


class SyntheticDataConfig(BaseModel):
    num_products: int = 50
    num_users: int = 300
    random_seed: int = 42


class EmbeddingConfig(BaseModel):
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    cache_path: str = "data/processed/product_embeddings.parquet"


class TwoTowerConfig(BaseModel):
    projection_dims: list[int] = Field(default_factory=lambda: [256, 128])
    output_dim: int = 128
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 20
    random_seed: int = 42


class RetrievalConfig(BaseModel):
    backend: Literal["faiss", "scann"] = "faiss"
    candidate_pool_multiplier: int = 5
    min_candidate_pool: int = 50


class SparseBlendConfig(BaseModel):
    personalized: float = 0.6
    preferred_category: float = 0.2
    popularity: float = 0.2


class ColdStartConfig(BaseModel):
    strong_history_min_signals: int = 5
    sparse_history_min_signals: int = 1
    sparse_blend: SparseBlendConfig = Field(default_factory=SparseBlendConfig)
    no_history_fallback_order: list[str] = Field(
        default_factory=lambda: ["preferred_category", "category_popularity", "global_popularity"]
    )


class EligibilityConfig(BaseModel):
    require_active: bool = True
    require_in_stock: bool = True


class RankingConfig(BaseModel):
    hidden_units: list[int] = Field(default_factory=lambda: [128, 64])
    learning_rate: float = 0.001
    epochs: int = 10
    batch_size: int = 256


class ApiConfig(BaseModel):
    model_version: str = "v1"
    default_recommendation_count: int = 10
    max_recommendation_count: int = 50


class DashboardConfig(BaseModel):
    api_base_url: str = "http://localhost:8000"


class AppConfig(BaseModel):
    random_seed: int = 42
    log_level: str = "INFO"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    synthetic_data: SyntheticDataConfig = Field(default_factory=SyntheticDataConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    two_tower: TwoTowerConfig = Field(default_factory=TwoTowerConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    cold_start: ColdStartConfig = Field(default_factory=ColdStartConfig)
    eligibility: EligibilityConfig = Field(default_factory=EligibilityConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate the YAML config into an AppConfig.

    Resolution order: explicit `path` argument, then RECS_CONFIG_PATH env
    var, then configs/base.yaml.
    """
    resolved = Path(path) if path is not None else Path(os.environ.get("RECS_CONFIG_PATH", DEFAULT_CONFIG_PATH))
    with resolved.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig(**raw)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Process-wide cached config, read from the default/env-resolved path."""
    return load_config()
