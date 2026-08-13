import time

import numpy as np
import pytest

from recommendation.api.dependencies import build_recommendation_service
from recommendation.ranking.features import RANKING_FEATURE_NAMES
from recommendation.ranking.model import build_ranker_model
from recommendation.ranking.serialization import RankerArtifacts
from recommendation.retrieval.two_tower.feature_encoding import TwoTowerFeatureEncoder
from recommendation.retrieval.two_tower.serialization import TwoTowerArtifacts
from recommendation.serving.startup_validation import (
    ArtifactValidationError,
    load_or_raise,
    require_artifact_dir,
    validate_ranker_artifacts,
    validate_two_tower_artifacts,
    validate_vector_index_compatibility,
)
from recommendation.utils.config import AppConfig, RankingConfig


def _two_tower_artifacts(output_dim=8, encoder_embedding_dim=8, num_items=3) -> TwoTowerArtifacts:
    encoder = TwoTowerFeatureEncoder.fit([], [], [], [1.0], encoder_embedding_dim)
    return TwoTowerArtifacts(
        user_tower=None, item_tower=None, encoder=encoder,
        item_ids=list(range(num_items)),
        item_embeddings=np.zeros((num_items, output_dim), dtype=np.float32),
        metadata={"model_version": "test"},
    )


def _ranker_artifacts(feature_names=None, input_dim=None) -> RankerArtifacts:
    feature_names = feature_names if feature_names is not None else RANKING_FEATURE_NAMES
    input_dim = input_dim if input_dim is not None else len(RANKING_FEATURE_NAMES)
    model = build_ranker_model(input_dim=input_dim, config=RankingConfig(hidden_units=[4]))
    return RankerArtifacts(model=model, feature_names=feature_names, metadata={})


# --- require_artifact_dir ------------------------------------------------

def test_require_artifact_dir_raises_when_missing(tmp_path):
    with pytest.raises(ArtifactValidationError, match="not found"):
        require_artifact_dir(tmp_path / "does_not_exist", "Two-Tower")


def test_require_artifact_dir_passes_when_present(tmp_path):
    require_artifact_dir(tmp_path, "Two-Tower")  # tmp_path itself exists


# --- load_or_raise ---------------------------------------------------------

def test_load_or_raise_wraps_underlying_exception(tmp_path):
    def _boom(path):
        raise FileNotFoundError("nope")

    with pytest.raises(ArtifactValidationError, match="Failed to load"):
        load_or_raise("Two-Tower", tmp_path, _boom)


def test_load_or_raise_passes_through_result(tmp_path):
    assert load_or_raise("x", tmp_path, lambda p: 42) == 42


# --- validate_two_tower_artifacts ------------------------------------------

def test_two_tower_validation_passes_for_consistent_artifacts():
    artifacts = _two_tower_artifacts(output_dim=8, encoder_embedding_dim=8)
    config = AppConfig()
    config.two_tower.output_dim = 8
    config.embedding.embedding_dim = 8
    validate_two_tower_artifacts(artifacts, config)  # no raise


def test_two_tower_validation_rejects_empty_catalog():
    artifacts = _two_tower_artifacts(num_items=0)
    with pytest.raises(ArtifactValidationError, match="empty catalog"):
        validate_two_tower_artifacts(artifacts, AppConfig())


def test_two_tower_validation_rejects_output_dim_mismatch():
    artifacts = _two_tower_artifacts(output_dim=8, encoder_embedding_dim=8)
    config = AppConfig()
    config.two_tower.output_dim = 128
    config.embedding.embedding_dim = 8
    with pytest.raises(ArtifactValidationError, match="output_dim"):
        validate_two_tower_artifacts(artifacts, config)


def test_two_tower_validation_rejects_embedding_dim_mismatch():
    artifacts = _two_tower_artifacts(output_dim=8, encoder_embedding_dim=8)
    config = AppConfig()
    config.two_tower.output_dim = 8
    config.embedding.embedding_dim = 384
    with pytest.raises(ArtifactValidationError, match="semantic embeddings"):
        validate_two_tower_artifacts(artifacts, config)


def test_two_tower_validation_rejects_corrupt_shape_mismatch():
    artifacts = _two_tower_artifacts(num_items=3)
    artifacts.item_embeddings = np.zeros((5, 8), dtype=np.float32)  # 3 ids, 5 embedding rows
    with pytest.raises(ArtifactValidationError, match="corrupt"):
        validate_two_tower_artifacts(artifacts, AppConfig())


# --- validate_ranker_artifacts ---------------------------------------------

def test_ranker_validation_passes_for_consistent_artifacts():
    validate_ranker_artifacts(_ranker_artifacts())  # no raise


def test_ranker_validation_rejects_feature_schema_mismatch():
    artifacts = _ranker_artifacts(feature_names=["a", "b"], input_dim=2)
    with pytest.raises(ArtifactValidationError, match="feature schema"):
        validate_ranker_artifacts(artifacts)


def test_ranker_validation_rejects_input_dim_mismatch():
    # feature_names matches (so the schema check passes) but the model's
    # actual input layer was built with a different width - simulates a
    # corrupt/mismatched save.
    artifacts = _ranker_artifacts(feature_names=RANKING_FEATURE_NAMES, input_dim=len(RANKING_FEATURE_NAMES) + 1)
    with pytest.raises(ArtifactValidationError, match="input features"):
        validate_ranker_artifacts(artifacts)


# --- validate_vector_index_compatibility -----------------------------------

def test_vector_index_compatibility_passes_when_sizes_match():
    validate_vector_index_compatibility(50, 50)  # no raise


def test_vector_index_compatibility_rejects_size_mismatch():
    with pytest.raises(ArtifactValidationError, match="VectorIndex"):
        validate_vector_index_compatibility(49, 50)


# --- build_recommendation_service fails fast on missing artifacts ---------

def test_build_recommendation_service_fails_fast_on_missing_artifacts(tmp_path):
    """Artifact presence must be checked BEFORE dataset generation/Sentence
    Transformer loading, so a missing-artifacts failure is near-instant,
    not paid for after several seconds of otherwise-wasted startup work.
    """
    config = AppConfig()
    config.paths.models_dir = str(tmp_path)  # empty - no two_tower/ or ranker/ subdirs

    start = time.perf_counter()
    with pytest.raises(ArtifactValidationError, match="Two-Tower"):
        build_recommendation_service(config)
    elapsed = time.perf_counter() - start

    assert elapsed < 5.0  # generous bound; a real ST-encoder load alone takes several seconds
