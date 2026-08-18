"""STEP 9 fix (docs/data-mapping.md section 18 follow-up): the PERSISTED
offline evaluation report artifact that `GET /v1/metrics/offline` reads,
plus the (de)serialization and provenance-validation helpers around it.

Why this exists: the endpoint used to call `ui.metrics.compute_offline_metrics`
synchronously on every request, which re-ran a FULL evaluation pass
(`generate_recommendations` once per evaluation-population user) inline in
the HTTP request - ~88s against the SQLite baseline, tripping the
dashboard's 10s client timeout. Worse, that evaluation used the OLD,
non-temporal leave-one-out protocol (`retrieval.two_tower.splitting`/
`examples`), not the STEP 5/7/8 APPROVED temporal future-purchase
protocol the currently-served `models/sqlite_baseline/` artifacts were
actually trained and selected against - so even ignoring latency, the
numbers it showed did not describe the model actually being served.

This module does NOT change what a metric means or how the evaluation
population is built - it reuses `evaluation.temporal_training
.evaluate_primary_pipeline` / `TemporalEvalCase` (STEP 5/7/8) UNCHANGED.
It only adds: a typed report shape carrying enough provenance to detect
staleness, JSON (de)serialization, and a compatibility check against the
currently-loaded `RecommendationService`.

The report is PRODUCED offline by `scripts/generate_offline_report.py`
(loads the already-trained `models/sqlite_baseline/` artifacts, never
retrains) and written next to those artifacts
(`{models_root}/offline_report.json` - the same directory convention as
`two_tower/metadata.json`/`ranker/metadata.json`). The API only ever
READS it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPORT_SCHEMA_VERSION = 1


class OfflineReportError(Exception):
    """Base class for every reason `GET /v1/metrics/offline` cannot return
    a trustworthy persisted report. Route handlers catch this (not the
    subclasses individually) and fail the request clearly - by design,
    NEVER falling back to running a live evaluation pass.
    """


class OfflineReportMissingError(OfflineReportError):
    """No report file exists at the expected path yet."""


class OfflineReportMalformedError(OfflineReportError):
    """The file exists but is not valid JSON / does not match the expected shape."""


class OfflineReportProvenanceMismatchError(OfflineReportError):
    """The report is well-formed but was generated against a different
    model/run/dataset than the one currently loaded by this process.
    """


@dataclass
class OfflineEvalSplitReport:
    split_name: str
    num_cases: int
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    hit_rate_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    mrr: float
    mean_distinct_categories: float
    catalog_coverage: float
    mean_fill_rate: float


@dataclass
class OfflineEvaluationReport:
    """The full persisted artifact. Provenance fields are exactly what
    `GET /v1/metrics/offline` needs to (a) prove which model/data/config
    produced these numbers and (b) detect a stale/mismatched report
    against whatever model this process actually loaded - requirement
    driving every field here, not a generic "metadata bag."
    """

    schema_version: int
    generated_at: datetime
    run_id: str
    ranker_model_version: str
    two_tower_model_version: str
    data_source: str
    dataset_fingerprint_sha256_16: str
    recency_enabled: bool
    recency_half_life_days: float | None
    include_price_features: bool
    price_tier_boundaries: list[float]
    k_values: list[int]
    top_n: int
    val_report: OfflineEvalSplitReport
    test_report: OfflineEvalSplitReport


def _report_to_jsonable(report: OfflineEvaluationReport) -> dict:
    payload = asdict(report)
    payload["generated_at"] = report.generated_at.isoformat()
    return payload


def save_offline_report(path: str | Path, report: OfflineEvaluationReport) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_report_to_jsonable(report), indent=2), encoding="utf-8")


def _split_report_from_dict(raw: dict, split_name: str) -> OfflineEvalSplitReport:
    try:
        return OfflineEvalSplitReport(
            split_name=raw["split_name"],
            num_cases=raw["num_cases"],
            precision_at_k={int(k): v for k, v in raw["precision_at_k"].items()},
            recall_at_k={int(k): v for k, v in raw["recall_at_k"].items()},
            hit_rate_at_k={int(k): v for k, v in raw["hit_rate_at_k"].items()},
            ndcg_at_k={int(k): v for k, v in raw["ndcg_at_k"].items()},
            mrr=raw["mrr"],
            mean_distinct_categories=raw["mean_distinct_categories"],
            catalog_coverage=raw["catalog_coverage"],
            mean_fill_rate=raw["mean_fill_rate"],
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise OfflineReportMalformedError(f"offline report {split_name} section is malformed: {exc}") from exc


def load_offline_report(path: str | Path) -> OfflineEvaluationReport:
    """Reads and validates the persisted artifact. Raises
    `OfflineReportMissingError`/`OfflineReportMalformedError` - never
    silently returns a partial/best-effort report and never triggers a
    live evaluation pass as a fallback.
    """
    path = Path(path)
    if not path.exists():
        raise OfflineReportMissingError(
            f"no offline evaluation report at {path} - run scripts/generate_offline_report.py "
            "against the currently-trained models/sqlite_baseline artifacts to produce one"
        )

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OfflineReportMalformedError(f"offline evaluation report at {path} is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise OfflineReportMalformedError(f"offline evaluation report at {path} must be a JSON object, got {type(raw).__name__}")

    if raw.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise OfflineReportMalformedError(
            f"offline evaluation report at {path} has schema_version={raw.get('schema_version')!r}, "
            f"expected {REPORT_SCHEMA_VERSION} - regenerate it with the current scripts/generate_offline_report.py"
        )

    try:
        generated_at = datetime.fromisoformat(raw["generated_at"])
        return OfflineEvaluationReport(
            schema_version=raw["schema_version"],
            generated_at=generated_at,
            run_id=raw["run_id"],
            ranker_model_version=raw["ranker_model_version"],
            two_tower_model_version=raw["two_tower_model_version"],
            data_source=raw["data_source"],
            dataset_fingerprint_sha256_16=raw["dataset_fingerprint_sha256_16"],
            recency_enabled=raw["recency_enabled"],
            recency_half_life_days=raw.get("recency_half_life_days"),
            include_price_features=raw["include_price_features"],
            price_tier_boundaries=list(raw["price_tier_boundaries"]),
            k_values=list(raw["k_values"]),
            top_n=raw["top_n"],
            val_report=_split_report_from_dict(raw["val_report"], "val"),
            test_report=_split_report_from_dict(raw["test_report"], "test"),
        )
    except OfflineReportMalformedError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise OfflineReportMalformedError(f"offline evaluation report at {path} is missing/has invalid fields: {exc}") from exc


def validate_report_provenance(
    report: OfflineEvaluationReport,
    *,
    ranker_model_version: str,
    two_tower_model_version: str,
    ranker_run_id: str,
    ranker_dataset_fingerprint: str,
) -> None:
    """Confirms the persisted report describes the SAME model/run/dataset
    this process actually has loaded - not just "a" report that happens to
    parse. Mismatches here mean the report predates a retrain (or was
    generated against a different dataset instantiation) and MUST NOT be
    served as if it were current (requirement: fail clearly, never fall
    back to a live evaluation run).
    """
    mismatches: list[str] = []
    if report.ranker_model_version != ranker_model_version:
        mismatches.append(f"ranker_model_version: report={report.ranker_model_version!r} loaded={ranker_model_version!r}")
    if report.two_tower_model_version != two_tower_model_version:
        mismatches.append(f"two_tower_model_version: report={report.two_tower_model_version!r} loaded={two_tower_model_version!r}")
    if not ranker_run_id or report.run_id != ranker_run_id:
        mismatches.append(f"run_id: report={report.run_id!r} loaded={ranker_run_id!r}")
    if not ranker_dataset_fingerprint or report.dataset_fingerprint_sha256_16 != ranker_dataset_fingerprint:
        mismatches.append(
            f"dataset_fingerprint_sha256_16: report={report.dataset_fingerprint_sha256_16!r} loaded={ranker_dataset_fingerprint!r}"
        )

    if mismatches:
        raise OfflineReportProvenanceMismatchError(
            "persisted offline evaluation report does not match the currently loaded model/data - "
            "regenerate it with scripts/generate_offline_report.py: " + "; ".join(mismatches)
        )
