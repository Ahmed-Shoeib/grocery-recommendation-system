"""Generic latency measurement: times repeated calls to any zero-arg
callable and reports summary statistics in milliseconds.

Deliberately generic (like `retrieval_metrics.py`) so Phase 5 can use it
to measure `VectorIndex.search` latency and Phase 6/8 can reuse it
unchanged for ranking and end-to-end recommendation latency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class LatencyReport:
    num_runs: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


def measure_latency(fn: Callable[[], object], num_runs: int = 100, warmup_runs: int = 5) -> LatencyReport:
    """Run `fn` `warmup_runs` times (discarded, to exclude one-off setup
    cost like lazy imports or cache warming) then `num_runs` timed times.
    """
    if num_runs <= 0:
        raise ValueError(f"num_runs must be positive, got {num_runs}")

    for _ in range(warmup_runs):
        fn()

    durations_ms = np.empty(num_runs, dtype=np.float64)
    for i in range(num_runs):
        start = time.perf_counter()
        fn()
        durations_ms[i] = (time.perf_counter() - start) * 1000.0

    return LatencyReport(
        num_runs=num_runs,
        mean_ms=float(np.mean(durations_ms)),
        p50_ms=float(np.percentile(durations_ms, 50)),
        p95_ms=float(np.percentile(durations_ms, 95)),
        p99_ms=float(np.percentile(durations_ms, 99)),
        min_ms=float(np.min(durations_ms)),
        max_ms=float(np.max(durations_ms)),
    )
