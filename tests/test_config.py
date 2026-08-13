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
