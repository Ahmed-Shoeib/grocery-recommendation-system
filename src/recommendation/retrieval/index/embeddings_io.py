"""Loads only the precomputed `item_embeddings.npz` artifact written by
Phase 4's `save_two_tower_artifacts` (`retrieval.two_tower.serialization`).

Deliberately does not import TensorFlow, unlike
`retrieval.two_tower.serialization` (which also loads the Keras tower
models) - this is the only thing Phase 5's VectorIndex needs, so
environments that only build/serve retrieval (notably the ScaNN Docker
image) never need the ML stack (torch/tensorflow/sentence-transformers)
installed at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_item_embeddings(two_tower_dir: str | Path) -> tuple[list[int], np.ndarray]:
    data = np.load(Path(two_tower_dir) / "item_embeddings.npz")
    return [int(x) for x in data["product_ids"]], data["embeddings"]
