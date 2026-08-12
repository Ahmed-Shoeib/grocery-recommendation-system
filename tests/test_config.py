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
