import time

import pytest

from recommendation.evaluation.latency import measure_latency


def test_measure_latency_runs_requested_number_of_times():
    calls = []
    measure_latency(lambda: calls.append(1), num_runs=20, warmup_runs=3)
    assert len(calls) == 23  # warmup + timed runs


def test_measure_latency_report_has_expected_shape():
    report = measure_latency(lambda: None, num_runs=10, warmup_runs=0)
    assert report.num_runs == 10
    assert report.min_ms <= report.p50_ms <= report.p95_ms <= report.p99_ms <= report.max_ms
    assert report.mean_ms >= 0.0


def test_measure_latency_reflects_actual_sleep_duration():
    report = measure_latency(lambda: time.sleep(0.005), num_runs=5, warmup_runs=0)
    assert report.mean_ms >= 4.0  # allow scheduler slack below the 5ms target


def test_measure_latency_rejects_non_positive_num_runs():
    with pytest.raises(ValueError):
        measure_latency(lambda: None, num_runs=0)
