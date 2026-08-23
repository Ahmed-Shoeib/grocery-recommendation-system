"""Build the configured `VectorIndex` backend (ScaNN inside Docker via
configs/docker.yaml, FAISS natively on Windows via configs/base.yaml)
from the trained item embeddings, sanity-check its recall against
brute-force cosine similarity, and report build time, on-disk size, and
retrieval latency.

Both backends now do genuine approximate nearest-neighbor search (see
retrieval/index/faiss_index.py, scann_index.py), so their top-k results
are no longer expected to match a brute-force reference exactly - this
script checks RECALL against that reference (see also
scripts/evaluate_ann_recall.py for a more rigorous recall/latency
evaluation over the real trained catalog).

Usage:
    python scripts/build_vector_index.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from recommendation.evaluation.latency import measure_latency
from recommendation.retrieval.index.embeddings_io import load_item_embeddings
from recommendation.retrieval.index.factory import build_vector_index
from recommendation.utils.config import get_config, resolve_path
from recommendation.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def brute_force_top_k(query: np.ndarray, item_ids: list[int], embeddings: np.ndarray, k: int) -> tuple[list[int], list[float]]:
    """Reference implementation: exact cosine similarity via a plain numpy
    dot product (embeddings are already L2-normalized), used to verify
    the configured VectorIndex backend returns identical results.
    """
    scores = embeddings @ query
    order = np.argsort(-scores)[:k]
    return [item_ids[i] for i in order], [float(scores[i]) for i in order]


def path_size_bytes(path: Path) -> int:
    """Total on-disk size: `path` may be a single file (FAISS) or a
    directory of multiple files (ScaNN's native serialization format).
    """
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return path.stat().st_size


def main() -> None:
    config = get_config()
    setup_logging(config.log_level)

    two_tower_dir = resolve_path(config.paths.models_dir) / "two_tower"
    item_ids, embeddings = load_item_embeddings(two_tower_dir)
    logger.info("Loaded %d item embeddings (dim=%d) from %s", len(item_ids), embeddings.shape[1], two_tower_dir)

    index = build_vector_index(config.retrieval)
    build_start = time.perf_counter()
    index.build(item_ids, embeddings)
    build_time_ms = (time.perf_counter() - build_start) * 1000.0

    out_dir = resolve_path(config.paths.models_dir) / "vector_index"
    out_path = out_dir / "faiss_index.bin" if config.retrieval.backend == "faiss" else out_dir / "scann_index"
    index.save(out_path)
    size_kb = path_size_bytes(out_path) / 1024.0
    print(f"\nBuilt {config.retrieval.backend} VectorIndex over {index.size} items in {build_time_ms:.2f}ms -> saved to {out_path} ({size_kb:.1f} KB on disk)")
    print("For a rigorous ANN-vs-exact recall/latency evaluation over the real trained catalog,")
    print("see scripts/evaluate_ann_recall.py - the check below is a quick sanity check only.")

    # --- Recall sanity check: configured (approximate) backend's top-k vs.
    # brute-force numpy cosine similarity. Both backends are now genuinely
    # approximate, so exact ID/score equality is no longer the right bar -
    # this checks that the ANN backend's top-k overlaps heavily with the
    # exact top-k, not that it reproduces it exactly.
    k = min(10, index.size)
    rng = np.random.default_rng(config.random_seed)
    sample_rows = rng.choice(len(item_ids), size=min(10, len(item_ids)), replace=False)

    print(f"\n=== Recall sanity check: {config.retrieval.backend} vs. brute-force cosine (top-{k}) ===")
    recalls = []
    for row in sample_rows:
        query = embeddings[row]
        [result] = index.search(query.reshape(1, -1), k)
        expected_ids, _ = brute_force_top_k(query, item_ids, embeddings, k)
        overlap = len(set(result.item_ids) & set(expected_ids))
        recall = overlap / k
        recalls.append(recall)
        if recall < 1.0:
            print(f"  item {item_ids[row]}: recall={recall:.2f} ({config.retrieval.backend} top-{k} overlaps {overlap}/{k} with exact)")
    mean_recall = float(np.mean(recalls))
    print(f"  Mean top-{k} recall over {len(sample_rows)} sampled queries: {mean_recall:.3f}")

    top1_scores = [index.search(embeddings[row].reshape(1, -1), 1)[0].scores[0] for row in sample_rows]
    print(f"  Self-match top-1 score range: [{min(top1_scores):.6f}, {max(top1_scores):.6f}] (expect ~1.0, exact cosine of a vector with itself)")

    example_row = int(sample_rows[0])
    [example_result] = index.search(embeddings[example_row].reshape(1, -1), k=5)
    print(f"\n=== Top-K retrieval example (query item {item_ids[example_row]}, top-5) ===")
    for rank, (item_id, score) in enumerate(zip(example_result.item_ids, example_result.scores), start=1):
        marker = "  <- query item itself" if item_id == item_ids[example_row] else ""
        print(f"  {rank}. item {item_id:>4}  score={score:.6f}{marker}")

    # --- Latency ---
    single_query = embeddings[0].reshape(1, -1)
    batch_query = embeddings[: min(32, len(item_ids))]

    single_report = measure_latency(lambda: index.search(single_query, k), num_runs=200)
    batch_report = measure_latency(lambda: index.search(batch_query, k), num_runs=200)

    print(f"\n=== Retrieval latency (top-{k}, {single_report.num_runs} runs) ===")
    print(f"  Single query : mean={single_report.mean_ms:.4f}ms  p50={single_report.p50_ms:.4f}ms  p95={single_report.p95_ms:.4f}ms  p99={single_report.p99_ms:.4f}ms")
    print(f"  Batch of {batch_query.shape[0]:<3}: mean={batch_report.mean_ms:.4f}ms  p50={batch_report.p50_ms:.4f}ms  p95={batch_report.p95_ms:.4f}ms  p99={batch_report.p99_ms:.4f}ms")


if __name__ == "__main__":
    main()
