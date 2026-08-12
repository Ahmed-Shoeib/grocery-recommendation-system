"""The neural ranker: a plain feedforward MLP over the concatenated
`ranking.features.RANKING_FEATURE_NAMES` vector, ending in a sigmoid -
pointwise binary relevance scoring (`ranking.examples` labels a
candidate 1 if the user purchased it, 0 if it's a sampled non-purchased
candidate). Deliberately simpler than the Two-Tower's dual-embedding-
tower architecture (see `ranking.features` module docstring) - no
learned category/brand embedding lookups here, since the explicit cross
features already carry that signal in a form an MLP can use directly.

`config.ranking.hidden_units` are the hidden layer widths, config-driven
per docs/data-mapping.md convention (never hard-coded).
"""

from __future__ import annotations

import tensorflow as tf

from recommendation.utils.config import RankingConfig


def build_ranker_model(input_dim: int, config: RankingConfig) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = inputs
    for i, units in enumerate(config.hidden_units):
        x = tf.keras.layers.Dense(units, activation="relu", name=f"ranker_hidden_{i}")(x)
        x = tf.keras.layers.Dropout(config.dropout_rate, name=f"ranker_dropout_{i}")(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="ranker_output")(x)
    return tf.keras.Model(inputs=inputs, outputs=output, name="ranker")
