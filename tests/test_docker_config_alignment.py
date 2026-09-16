"""Docker/deployment config alignment (docs: docker-compose.yml's own
header comment): training must stay SQLite-only
(`data/sqlite/production_aligned_training.db`, never the live backend),
while production serving must use `data_source=backend_api` and resolve
its artifacts to `models/backend_api/` - with no mode where the two
concerns mix.

Parses the raw `docker-compose.yml` YAML directly (never `docker compose
config`, which resolves `.env` and would print real secret values into
test output/CI logs if it ever failed loudly) and exercises
`api.service.resolve_models_root` directly - no Docker installation
required to run these.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from recommendation.api.service import resolve_models_root
from recommendation.config import AppConfig, PathsConfig, resolve_path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _compose() -> dict:
    return yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def _docker_yaml() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / "docker.yaml").read_text(encoding="utf-8"))


# --- resolve_models_root: the actual runtime mechanism the compose file relies on ---


def test_backend_api_data_source_resolves_to_backend_api_artifacts():
    config = AppConfig(paths=PathsConfig(data_source="backend_api", models_dir="models"))
    assert resolve_models_root(config) == resolve_path("models") / "backend_api"


def test_sqlite_data_source_resolves_to_sqlite_baseline_artifacts_not_backend_api():
    config = AppConfig(paths=PathsConfig(data_source="sqlite", models_dir="models"))
    root = resolve_models_root(config)
    assert root == resolve_path("models") / "sqlite_baseline"
    assert root != resolve_path("models") / "backend_api"


# --- docker-compose.yml: the actual committed file, not just the mechanism ---


def test_api_service_is_configured_for_backend_api_serving():
    api_env = _compose()["services"]["api"]["environment"]
    assert api_env["RECS_DATA_SOURCE"] == "backend_api", (
        "the api service must serve from the real backend, not silently fall back to "
        "configs/docker.yaml's shared sqlite default"
    )


def test_api_service_has_no_hardcoded_secret_values():
    """Credentials must come from `env_file: .env` (gitignored) only -
    the committed compose file's own `environment:` block must contain
    nothing but the one non-secret data-source switch.
    """
    api_service = _compose()["services"]["api"]
    assert api_service.get("env_file"), "api service must load credentials via env_file, not inline"
    assert api_service.get("environment") == {"RECS_DATA_SOURCE": "backend_api"}


def test_train_service_uses_the_production_aligned_sqlite_db_explicitly():
    train_service = _compose()["services"]["train"]
    command = train_service["command"]
    assert "data/sqlite/production_aligned_training.db" in command
    assert not any("backend_shaped_synthetic.db" in str(part) for part in command), (
        "training must never default to the legacy/test-fixture SQLite database"
    )
    assert "scripts/train_backend_api_pipeline.py" in command, (
        "the current, sole production_safe_v2 training entrypoint"
    )


def test_train_service_never_sets_backend_api_as_its_data_source():
    train_service = _compose()["services"]["train"]
    train_env = train_service.get("environment", {})
    assert train_env.get("RECS_DATA_SOURCE") != "backend_api", (
        "training must never inherit or set the live-backend serving data source - "
        "it must not depend on live backend user-activity data"
    )


def test_train_service_is_profile_gated_not_started_by_default():
    train_service = _compose()["services"]["train"]
    assert "train" in train_service.get("profiles", []), (
        "training must be an explicit, deliberate action (`--profile train`), "
        "never started by a plain `docker compose up`"
    )


def test_dashboard_service_has_no_data_source_override():
    """The dashboard is a pure HTTP client of the api service - it has no
    data_source concept of its own and must not be given one.
    """
    dashboard_service = _compose()["services"]["dashboard"]
    assert "RECS_DATA_SOURCE" not in dashboard_service.get("environment", {})


# --- configs/docker.yaml: the shared file-level default must stay safe ---


def test_docker_yaml_shared_default_stays_sqlite_not_backend_api():
    """configs/docker.yaml is loaded (via RECS_CONFIG_PATH) by EVERY stage
    built from the `base` image, including `docker build --target test` -
    a "backend_api" file-level default here would make an ordinary test
    run silently depend on live backend credentials/network. The
    per-service `RECS_DATA_SOURCE=backend_api` override on the `api`
    service in docker-compose.yml is what switches serving instead.
    """
    assert _docker_yaml()["paths"]["data_source"] == "sqlite"


def test_docker_yaml_default_sqlite_db_is_not_the_production_aligned_one():
    """The shared default stays the test-fixture database
    (data/sqlite/backend_shaped_synthetic.db) - the production-aligned
    dataset is selected explicitly by the train service's `--db` flag,
    never implicitly by this shared config file.
    """
    assert _docker_yaml()["paths"]["data_sqlite"] == "data/sqlite/backend_shaped_synthetic.db"
