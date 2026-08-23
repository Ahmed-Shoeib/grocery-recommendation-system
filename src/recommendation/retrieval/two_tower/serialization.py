"""Model artifact serialization: everything needed to build a VectorIndex
and serve Two-Tower retrieval without retraining.

Written under `models/two_tower/` (gitignored - regenerable, not source):

  item_tower.keras        - Keras 3 native format
  user_tower.keras
  feature_encoder.json    - vocabularies + normalization stats (serving
                             MUST use this exact encoder, not a freshly
                             fit one, or item/user ids stop matching
                             the trained embedding space)
  item_embeddings.npz     - precomputed, L2-normalized 128-D embeddings
                             for the full current catalog (product_ids +
                             embeddings) - what gets loaded into the
                             VectorIndex
  metadata.json           - model_version, config snapshot, dimensions
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from recommendation.retrieval.two_tower.feature_encoding import TwoTowerFeatureEncoder
from recommendation.retrieval.two_tower.model import L2Normalize


@dataclass
class TwoTowerArtifacts:
    user_tower: tf.keras.Model
    item_tower: tf.keras.Model
    encoder: TwoTowerFeatureEncoder
    item_ids: list[int]
    item_embeddings: np.ndarray  # (len(item_ids), output_dim), L2-normalized
    metadata: dict


def save_two_tower_artifacts(
    out_dir: str | Path,
    user_tower: tf.keras.Model,
    item_tower: tf.keras.Model,
    encoder: TwoTowerFeatureEncoder,
    item_ids: list[int],
    item_embeddings: np.ndarray,
    metadata: dict,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    user_tower.save(out_dir / "user_tower.keras")
    item_tower.save(out_dir / "item_tower.keras")
    encoder.save(out_dir / "feature_encoder.json")
    np.savez_compressed(
        out_dir / "item_embeddings.npz",
        product_ids=np.array(item_ids, dtype=np.int64),
        embeddings=item_embeddings.astype(np.float32),
    )
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_two_tower_artifacts(in_dir: str | Path) -> TwoTowerArtifacts:
    in_dir = Path(in_dir)
    custom_objects = {"L2Normalize": L2Normalize}
    user_tower = tf.keras.models.load_model(in_dir / "user_tower.keras", custom_objects=custom_objects)
    item_tower = tf.keras.models.load_model(in_dir / "item_tower.keras", custom_objects=custom_objects)
    encoder = TwoTowerFeatureEncoder.load(in_dir / "feature_encoder.json")

    data = np.load(in_dir / "item_embeddings.npz")
    metadata = json.loads((in_dir / "metadata.json").read_text(encoding="utf-8"))

    return TwoTowerArtifacts(
        user_tower=user_tower,
        item_tower=item_tower,
        encoder=encoder,
        item_ids=[int(x) for x in data["product_ids"]],
        item_embeddings=data["embeddings"],
        metadata=metadata,
    )
