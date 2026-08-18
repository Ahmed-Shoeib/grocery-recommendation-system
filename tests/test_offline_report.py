"""Unit tests for `evaluation.offline_report` - the persisted
offline-evaluation-report (de)serialization and provenance-validation
helpers behind the STEP 9 `GET /v1/metrics/offline` fix (see that
module's docstring for the full before/after).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from recommendation.evaluation.offline_report import (
    REPORT_SCHEMA_VERSION,
    OfflineEvalSplitReport,
    OfflineEvaluationReport,
    OfflineReportMalformedError,
    OfflineReportMissingError,
    OfflineReportProvenanceMismatchError,
    load_offline_report,
    save_offline_report,
    validate_report_provenance,
)


def _split(split_name: str) -> OfflineEvalSplitReport:
    return OfflineEvalSplitReport(
        split_name=split_name, num_cases=3, precision_at_k={10: 0.4}, recall_at_k={10: 0.3}, hit_rate_at_k={10: 0.6},
        ndcg_at_k={10: 0.5}, mrr=0.45, mean_distinct_categories=2.5, catalog_coverage=0.9, mean_fill_rate=1.0,
    )


def _report(**overrides) -> OfflineEvaluationReport:
    fields = dict(
        schema_version=REPORT_SCHEMA_VERSION,
        generated_at=datetime(2026, 8, 18, 3, 0, 0, tzinfo=timezone.utc),
        run_id="20260818T030000",
        ranker_model_version="sqlite_baseline_ranker_v1",
        two_tower_model_version="sqlite_baseline_two_tower_v1",
        data_source="data/sqlite/backend_shaped_synthetic.db",
        dataset_fingerprint_sha256_16="112c3ab4a0a0cb4b",
        recency_enabled=True,
        recency_half_life_days=21.0,
        include_price_features=True,
        price_tier_boundaries=[4.0, 6.71],
        k_values=[5, 10, 20],
        top_n=10,
        val_report=_split("val"),
        test_report=_split("test"),
    )
    fields.update(overrides)
    return OfflineEvaluationReport(**fields)


# --- save / load round trip -------------------------------------------------

def test_save_then_load_round_trips_all_fields(tmp_path):
    report = _report()
    path = tmp_path / "sqlite_baseline" / "offline_report.json"
    save_offline_report(path, report)

    loaded = load_offline_report(path)

    assert loaded.run_id == report.run_id
    assert loaded.dataset_fingerprint_sha256_16 == report.dataset_fingerprint_sha256_16
    assert loaded.ranker_model_version == report.ranker_model_version
    assert loaded.two_tower_model_version == report.two_tower_model_version
    assert loaded.k_values == report.k_values
    assert loaded.top_n == report.top_n
    assert loaded.val_report.num_cases == 3
    assert loaded.val_report.ndcg_at_k == {10: 0.5}
    assert loaded.test_report.split_name == "test"
    assert loaded.generated_at == report.generated_at


def test_save_creates_parent_directories(tmp_path):
    path = tmp_path / "does" / "not" / "exist" / "offline_report.json"
    save_offline_report(path, _report())
    assert path.exists()


# --- missing report ----------------------------------------------------------

def test_load_missing_report_raises_missing_error(tmp_path):
    with pytest.raises(OfflineReportMissingError):
        load_offline_report(tmp_path / "nope" / "offline_report.json")


# --- malformed report ---------------------------------------------------------

def test_load_invalid_json_raises_malformed_error(tmp_path):
    path = tmp_path / "offline_report.json"
    path.write_text("{not valid json at all", encoding="utf-8")
    with pytest.raises(OfflineReportMalformedError):
        load_offline_report(path)


def test_load_missing_required_field_raises_malformed_error(tmp_path):
    path = tmp_path / "offline_report.json"
    path.write_text('{"schema_version": 1, "run_id": "x"}', encoding="utf-8")
    with pytest.raises(OfflineReportMalformedError):
        load_offline_report(path)


def test_load_non_object_json_raises_malformed_error(tmp_path):
    path = tmp_path / "offline_report.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(OfflineReportMalformedError):
        load_offline_report(path)


def test_load_wrong_schema_version_raises_malformed_error(tmp_path):
    report = _report()
    path = tmp_path / "offline_report.json"
    save_offline_report(path, report)
    # Simulate a future/incompatible schema version written by a newer generator.
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["schema_version"] = 999
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(OfflineReportMalformedError):
        load_offline_report(path)


# --- provenance validation -----------------------------------------------------

def _matching_kwargs(report: OfflineEvaluationReport) -> dict:
    return dict(
        ranker_model_version=report.ranker_model_version,
        two_tower_model_version=report.two_tower_model_version,
        ranker_run_id=report.run_id,
        ranker_dataset_fingerprint=report.dataset_fingerprint_sha256_16,
    )


def test_validate_provenance_passes_when_everything_matches():
    report = _report()
    validate_report_provenance(report, **_matching_kwargs(report))  # must not raise


def test_validate_provenance_fails_on_run_id_mismatch():
    report = _report()
    kwargs = _matching_kwargs(report)
    kwargs["ranker_run_id"] = "a_different_run"
    with pytest.raises(OfflineReportProvenanceMismatchError, match="run_id"):
        validate_report_provenance(report, **kwargs)


def test_validate_provenance_fails_on_dataset_fingerprint_mismatch():
    report = _report()
    kwargs = _matching_kwargs(report)
    kwargs["ranker_dataset_fingerprint"] = "deadbeefdeadbeef"
    with pytest.raises(OfflineReportProvenanceMismatchError, match="dataset_fingerprint"):
        validate_report_provenance(report, **kwargs)


def test_validate_provenance_fails_on_ranker_model_version_mismatch():
    report = _report()
    kwargs = _matching_kwargs(report)
    kwargs["ranker_model_version"] = "some_other_ranker_v2"
    with pytest.raises(OfflineReportProvenanceMismatchError, match="ranker_model_version"):
        validate_report_provenance(report, **kwargs)


def test_validate_provenance_fails_on_two_tower_model_version_mismatch():
    report = _report()
    kwargs = _matching_kwargs(report)
    kwargs["two_tower_model_version"] = "some_other_two_tower_v2"
    with pytest.raises(OfflineReportProvenanceMismatchError, match="two_tower_model_version"):
        validate_report_provenance(report, **kwargs)


def test_validate_provenance_fails_when_loaded_run_id_empty():
    """An empty currently-loaded run_id/fingerprint (e.g. a service that
    never populated ranker_metadata) must fail closed, not vacuously pass.
    """
    report = _report()
    kwargs = _matching_kwargs(report)
    kwargs["ranker_run_id"] = ""
    with pytest.raises(OfflineReportProvenanceMismatchError):
        validate_report_provenance(report, **kwargs)
