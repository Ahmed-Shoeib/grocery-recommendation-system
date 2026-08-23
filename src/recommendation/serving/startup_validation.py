"""Startup artifact/config validation.

Runs BEFORE a `RecommendationService` is considered usable
(`api.dependencies.build_recommendation_service` calls every function
here), so a missing, corrupt, or incompatible artifact fails loudly and
immediately at process startup - never silently serves recommendations
built from a ranker trained against a different feature schema, or a
Two-Tower whose embedding dimension doesn't match the running config.
This is deliberately conservative: better a container that refuses to
become ready than one that serves wrong recommendations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, TypeVar

from recommendation.ranking.features import RANKING_FEATURE_NAMES
from recommendation.ranking.serialization import RankerArtifacts
from recommendation.retrieval.two_tower.serialization import TwoTowerArtifacts
from recommendation.utils.config import AppConfig, RetrievalConfig

T = TypeVar("T")


class ArtifactValidationError(RuntimeError):
    """Trained artifacts are missing, corrupt, or incompatible with the
    current config/code - see the message for exactly which check failed
    and how to fix it (retrain vs. fix config vs. mount the volume).
    """


def require_artifact_dir(path: Path, name: str) -> None:
    if not path.is_dir():
        raise ArtifactValidationError(
            f"{name} artifacts not found at {path}. Run the corresponding training script "
            "(scripts/train_two_tower.py or scripts/train_ranker.py) first, or - in a "
            "container - mount a volume containing pre-trained artifacts at this path "
            "(models/ is intentionally not baked into the image or git)."
        )


def load_or_raise(name: str, path: Path, loader: Callable[[Path], T]) -> T:
    """Wraps a `load_*_artifacts` call: any failure (missing file inside
    an existing directory, truncated/corrupt file, incompatible Keras
    format) becomes one clear `ArtifactValidationError` instead of a raw
    Keras/JSON/numpy exception leaking out of startup.
    """
    try:
        return loader(path)
    except Exception as exc:  # noqa: BLE001 - artifact I/O is a system boundary; any failure must fail startup clearly
        raise ArtifactValidationError(f"Failed to load {name} artifacts from {path}: {exc}") from exc


def validate_two_tower_artifacts(artifacts: TwoTowerArtifacts, config: AppConfig) -> None:
    if not artifacts.item_ids:
        raise ArtifactValidationError("Two-Tower artifacts contain an empty catalog (0 items) - nothing to retrieve against.")
    if artifacts.item_embeddings.ndim != 2 or len(artifacts.item_ids) != artifacts.item_embeddings.shape[0]:
        raise ArtifactValidationError(
            f"Two-Tower artifacts are corrupt: {len(artifacts.item_ids)} item ids but "
            f"embeddings array shape {artifacts.item_embeddings.shape} (expected ({len(artifacts.item_ids)}, dim))."
        )

    actual_dim = artifacts.item_embeddings.shape[1]
    expected_dim = config.two_tower.output_dim
    if actual_dim != expected_dim:
        raise ArtifactValidationError(
            f"Two-Tower item embeddings have dimension {actual_dim}, but config.two_tower.output_dim="
            f"{expected_dim}. These artifacts were likely trained against a different config. "
            "Retrain (scripts/train_two_tower.py) or fix configs/*.yaml."
        )
    if artifacts.encoder.embedding_dim != config.embedding.embedding_dim:
        raise ArtifactValidationError(
            f"Two-Tower feature encoder expects {artifacts.encoder.embedding_dim}-D semantic embeddings, "
            f"but config.embedding.embedding_dim={config.embedding.embedding_dim}. Retrain or fix config."
        )


def validate_ranker_artifacts(artifacts: RankerArtifacts) -> None:
    if artifacts.feature_names != RANKING_FEATURE_NAMES:
        raise ArtifactValidationError(
            "Ranker artifacts were trained against a different feature schema than the running "
            f"code expects.\n  saved feature_names:    {artifacts.feature_names}\n"
            f"  expected feature_names: {RANKING_FEATURE_NAMES}\n"
            "Retrain the ranker (scripts/train_ranker.py) after any recommendation.ranking.features change."
        )
    expected_input_dim = len(RANKING_FEATURE_NAMES)
    actual_input_dim = artifacts.model.input_shape[-1]
    if actual_input_dim != expected_input_dim:
        raise ArtifactValidationError(
            f"Ranker model expects {actual_input_dim} input features but RANKING_FEATURE_NAMES "
            f"currently has {expected_input_dim}. Retrain the ranker (scripts/train_ranker.py)."
        )


def validate_vector_index_compatibility(vector_index_size: int, expected_catalog_size: int) -> None:
    if vector_index_size != expected_catalog_size:
        raise ArtifactValidationError(
            f"VectorIndex was built with {vector_index_size} items but the Two-Tower artifacts "
            f"describe a {expected_catalog_size}-item catalog - the index build is inconsistent "
            "with the embeddings it was built from."
        )


def validate_retrieval_config(config: RetrievalConfig) -> None:
    """Bounds sanity check on the ANN/eligibility-widening tuning fields
    (`faiss_hnsw_*`, `scann_*`, `eligibility_*` - see `RetrievalConfig`),
    so a misconfigured YAML (e.g. `eligibility_max_widen_attempts: 0`,
    which would silently make eligibility filtering never widen past its
    first, possibly-too-small oversample) fails loudly at startup instead
    of degrading recommendation quality/fill-rate at first request.
    Deliberately NOT a deep check of the on-disk index's actual ANN
    structure against this config - the index is rebuilt fresh from
    Two-Tower embeddings at every startup (`api.dependencies
    .build_recommendation_service` calls `vector_index.build(...)`, never
    `.load()`s a saved artifact for live serving), so there is no
    config/disk drift to catch there.
    """
    positive_int_fields = (
        "faiss_hnsw_m",
        "faiss_hnsw_ef_construction",
        "faiss_hnsw_ef_search",
        "scann_min_points_per_leaf",
        "scann_max_leaves",
        "scann_ah_dims_per_block",
        "scann_training_sample_size",
        "scann_reorder_k",
        "eligibility_oversample_factor",
        "eligibility_oversample_floor",
        "eligibility_widen_multiplier",
        "eligibility_max_widen_attempts",
    )
    for field_name in positive_int_fields:
        value = getattr(config, field_name)
        if value <= 0:
            raise ArtifactValidationError(f"config.retrieval.{field_name} must be positive, got {value}.")

    if not (0.0 < config.scann_leaves_to_search_fraction <= 1.0):
        raise ArtifactValidationError(
            f"config.retrieval.scann_leaves_to_search_fraction must be in (0, 1], got "
            f"{config.scann_leaves_to_search_fraction}."
        )
