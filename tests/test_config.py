from recommendation.utils.config import AppConfig, DEFAULT_CONFIG_PATH, load_config


def test_default_config_file_exists():
    assert DEFAULT_CONFIG_PATH.exists()


def test_load_config_returns_app_config():
    config = load_config()
    assert isinstance(config, AppConfig)


def test_load_config_defaults():
    config = load_config()
    assert config.embedding.embedding_dim == 384
    assert config.two_tower.output_dim == 128
    assert config.retrieval.backend in ("faiss", "scann")


def test_refresh_config_default_interval():
    config = load_config()
    assert config.refresh.interval_seconds == 30.0


def test_env_override_refresh_interval_seconds(monkeypatch):
    monkeypatch.setenv("RECS_REFRESH_INTERVAL_SECONDS", "5.5")
    config = load_config()
    assert config.refresh.interval_seconds == 5.5


def test_retrieval_ann_config_defaults():
    config = load_config()
    retrieval = config.retrieval
    assert retrieval.faiss_hnsw_m == 32
    assert retrieval.faiss_hnsw_ef_construction == 200
    assert retrieval.faiss_hnsw_ef_search == 128
    assert retrieval.scann_leaves_multiplier == 2.0
    assert retrieval.scann_min_points_per_leaf == 20
    assert retrieval.scann_max_leaves == 2000
    assert retrieval.scann_leaves_to_search_fraction == 0.3
    assert retrieval.scann_ah_dims_per_block == 2
    assert retrieval.scann_reorder_k == 200
    assert retrieval.eligibility_oversample_factor == 3
    assert retrieval.eligibility_oversample_floor == 20
    assert retrieval.eligibility_widen_multiplier == 2
    assert retrieval.eligibility_max_widen_attempts == 4


def test_base_and_docker_configs_define_identical_retrieval_ann_fields():
    """configs/base.yaml and configs/docker.yaml are full standalone
    copies, not overlays (see both files' header comments) - the new ANN
    tuning fields must be mirrored in both, not just the one loaded by
    default, or the two environments would silently diverge in tuning.
    """
    from recommendation.utils.config import REPO_ROOT

    base = load_config(REPO_ROOT / "configs" / "base.yaml")
    docker = load_config(REPO_ROOT / "configs" / "docker.yaml")
    assert base.retrieval.model_dump(exclude={"backend"}) == docker.retrieval.model_dump(exclude={"backend"})


def test_recency_config_defaults():
    config = load_config()
    assert config.features.recency.enabled is True
    assert config.features.recency.half_life_days == 21.0


def test_price_tier_embedding_dim_default():
    config = load_config()
    assert config.two_tower.price_tier_embedding_dim == 8


def test_cold_start_sparse_blend_sums_to_one():
    config = load_config()
    blend = config.cold_start.sparse_blend
    total = blend.personalized + blend.preferred_category + blend.popularity
    assert abs(total - 1.0) < 1e-6


def test_config_is_overridable_via_explicit_path(tmp_path):
    custom = tmp_path / "custom.yaml"
    custom.write_text("random_seed: 7\nlog_level: DEBUG\n", encoding="utf-8")
    config = load_config(custom)
    assert config.random_seed == 7
    assert config.log_level == "DEBUG"
    # Unspecified sections still fall back to their defaults.
    assert config.embedding.embedding_dim == 384


# --- Phase 10: env var overrides --------------------------------------

def test_env_override_models_dir(monkeypatch):
    monkeypatch.setenv("RECS_MODELS_DIR", "/mnt/custom-models")
    config = load_config()
    assert config.paths.models_dir == "/mnt/custom-models"


def test_env_override_log_level(monkeypatch):
    monkeypatch.setenv("RECS_LOG_LEVEL", "DEBUG")
    config = load_config()
    assert config.log_level == "DEBUG"


def test_env_override_retrieval_backend(monkeypatch):
    monkeypatch.setenv("RECS_RETRIEVAL_BACKEND", "scann")
    config = load_config()
    assert config.retrieval.backend == "scann"


def test_env_override_api_host_and_port(monkeypatch):
    monkeypatch.setenv("RECS_API_HOST", "127.0.0.1")
    monkeypatch.setenv("RECS_API_PORT", "9000")
    config = load_config()
    assert config.api.host == "127.0.0.1"
    assert config.api.port == 9000


def test_env_override_top_n_settings(monkeypatch):
    monkeypatch.setenv("RECS_API_DEFAULT_TOP_N", "5")
    monkeypatch.setenv("RECS_API_MAX_TOP_N", "20")
    config = load_config()
    assert config.api.default_recommendation_count == 5
    assert config.api.max_recommendation_count == 20


def test_env_override_invalid_int_raises_clear_error(monkeypatch):
    monkeypatch.setenv("RECS_API_PORT", "not-a-number")
    try:
        load_config()
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "RECS_API_PORT" in str(exc)


def test_no_env_overrides_leaves_defaults_untouched(monkeypatch):
    for var in ("RECS_MODELS_DIR", "RECS_LOG_LEVEL", "RECS_RETRIEVAL_BACKEND", "RECS_API_HOST", "RECS_API_PORT"):
        monkeypatch.delenv(var, raising=False)
    config = load_config()
    assert config.api.host == "0.0.0.0"
    assert config.api.port == 8000
    assert config.paths.models_dir == "models"


def test_explicit_path_config_also_receives_env_overrides(monkeypatch, tmp_path):
    custom = tmp_path / "custom.yaml"
    custom.write_text("random_seed: 7\n", encoding="utf-8")
    monkeypatch.setenv("RECS_LOG_LEVEL", "WARNING")
    config = load_config(custom)
    assert config.log_level == "WARNING"
    assert config.random_seed == 7
