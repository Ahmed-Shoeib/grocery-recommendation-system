"""Ranker artifact serialization, mirroring `retrieval.two_tower
.serialization`'s conventions. Written under `models/ranker/` (gitignored
- regenerable from `scripts/train_ranker.py`, not source):

  ranker.keras       - Keras 3 native format
  feature_names.json - RANKING_FEATURE_NAMES snapshot (column order the
                        saved model expects; Phase 7/8 must build feature
                        vectors in this exact order)
  metadata.json       - model_version, config snapshot, metrics summary
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf


@dataclass
class RankerArtifacts:
    model: tf.keras.Model
    feature_names: list[str]
    metadata: dict


def save_ranker_artifacts(out_dir: str | Path, model: tf.keras.Model, feature_names: list[str], metadata: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "ranker.keras")
    (out_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_ranker_artifacts(in_dir: str | Path) -> RankerArtifacts:
    in_dir = Path(in_dir)
    model = tf.keras.models.load_model(in_dir / "ranker.keras")
    feature_names = json.loads((in_dir / "feature_names.json").read_text(encoding="utf-8"))
    metadata = json.loads((in_dir / "metadata.json").read_text(encoding="utf-8"))
    return RankerArtifacts(model=model, feature_names=feature_names, metadata=metadata)
