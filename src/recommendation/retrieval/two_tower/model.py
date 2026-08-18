"""User Tower, Item Tower, and the in-batch-softmax TwoTowerModel.

Both towers project their (semantic embedding + categorical embeddings +
structured numeric features) input onto a shared 128-D space, L2-normalized
so retrieval similarity is a plain dot product (== cosine similarity).

Architecture per tower: Concatenate -> Dense(hidden, relu) -> Dropout ->
Dense(output_dim) -> L2Normalize. `config.two_tower.projection_dims` are
the hidden layer widths (default [256] after dropping the trailing 128,
which is `output_dim`); both are config-driven, not hard-coded.

Training loss: in-batch softmax over cosine similarities (temperature-
scaled), implemented directly in Keras 3 rather than via
`tfrs.tasks.Retrieval` - see pyproject.toml's `ml` extra comment for why
(TFRS requires legacy Keras 2 under the installed TF 2.21/Keras 3 stack).
This is the same mechanism TFRS's Retrieval task implements internally:
for a batch of B (user, positive_item) pairs, logits[i, j] =
cosine(user_i, item_j) / temperature, and the label for row i is column i
- every other item in the batch acts as an implicit negative for user i.
"""

from __future__ import annotations

import tensorflow as tf

from recommendation.retrieval.two_tower.feature_encoding import TwoTowerFeatureEncoder
from recommendation.utils.config import TwoTowerConfig


@tf.keras.utils.register_keras_serializable(package="recommendation")
class L2Normalize(tf.keras.layers.Layer):
    """A real Layer subclass (not `Lambda`) so `.keras` save/load works
    without Keras 3's unsafe-deserialization warnings/`safe_mode=False`
    workaround that arbitrary-code Lambda layers require.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.math.l2_normalize(inputs, axis=-1)


def _projection_mlp(x: tf.Tensor, config: TwoTowerConfig, name_prefix: str) -> tf.Tensor:
    hidden_dims = config.projection_dims[:-1] or [config.projection_dims[0]]
    for i, dim in enumerate(hidden_dims):
        x = tf.keras.layers.Dense(dim, activation="relu", name=f"{name_prefix}_hidden_{i}")(x)
        x = tf.keras.layers.Dropout(config.dropout_rate, name=f"{name_prefix}_dropout_{i}")(x)
    x = tf.keras.layers.Dense(config.output_dim, activation=None, name=f"{name_prefix}_output")(x)
    return L2Normalize(name=f"{name_prefix}_l2norm")(x)


def build_item_tower(encoder: TwoTowerFeatureEncoder, config: TwoTowerConfig) -> tf.keras.Model:
    semantic_in = tf.keras.Input(shape=(encoder.embedding_dim,), name="semantic_embedding")
    category_in = tf.keras.Input(shape=(), dtype="int32", name="category_id")
    brand_in = tf.keras.Input(shape=(), dtype="int32", name="brand_id")
    numeric_in = tf.keras.Input(shape=(encoder.item_numeric_dim,), name="numeric")

    category_emb = tf.keras.layers.Embedding(
        encoder.category_vocab.size, config.category_embedding_dim, name="item_category_embedding"
    )(category_in)
    brand_emb = tf.keras.layers.Embedding(
        encoder.brand_vocab.size, config.brand_embedding_dim, name="item_brand_embedding"
    )(brand_in)

    inputs = {"semantic_embedding": semantic_in, "category_id": category_in, "brand_id": brand_in, "numeric": numeric_in}
    concat_parts = [semantic_in, category_emb, brand_emb]

    # STEP 6 (docs/data-mapping.md section 15): BUDGET/MID/PREMIUM + an
    # "unknown" bucket as a LEARNED embedding, not an ordinal 0/1/2 number
    # - a categorical tier has no inherent numeric distance the model
    # should be forced to assume. STEP 8 (section 17): omitted entirely
    # when `encoder.include_price_features` is False, so the controlled
    # ablation's BASE condition faithfully reproduces the pre-STEP-6
    # architecture (no price input at all), not a price input fed zeros.
    if encoder.include_price_features:
        price_tier_in = tf.keras.Input(shape=(), dtype="int32", name="price_tier_id")
        price_tier_emb = tf.keras.layers.Embedding(
            encoder.price_tier_vocab.size, config.price_tier_embedding_dim, name="item_price_tier_embedding"
        )(price_tier_in)
        inputs["price_tier_id"] = price_tier_in
        concat_parts.append(price_tier_emb)

    concat_parts.append(numeric_in)
    concat = tf.keras.layers.Concatenate(name="item_concat")(concat_parts)
    output = _projection_mlp(concat, config, "item")

    return tf.keras.Model(inputs=inputs, outputs=output, name="item_tower")


def build_user_tower(encoder: TwoTowerFeatureEncoder, config: TwoTowerConfig) -> tf.keras.Model:
    semantic_in = tf.keras.Input(shape=(encoder.embedding_dim,), name="semantic_embedding")
    preferred_category_in = tf.keras.Input(shape=(), dtype="int32", name="preferred_category_id")
    age_group_in = tf.keras.Input(shape=(), dtype="int32", name="age_group_id")
    category_affinity_in = tf.keras.Input(shape=(encoder.category_affinity_dim,), name="category_affinity")
    brand_affinity_in = tf.keras.Input(shape=(encoder.brand_affinity_dim,), name="brand_affinity")
    numeric_in = tf.keras.Input(shape=(encoder.user_numeric_dim,), name="numeric")

    preferred_category_emb = tf.keras.layers.Embedding(
        encoder.category_vocab.size, config.category_embedding_dim, name="user_preferred_category_embedding"
    )(preferred_category_in)
    age_group_emb = tf.keras.layers.Embedding(
        encoder.age_group_vocab.size, config.age_group_embedding_dim, name="user_age_group_embedding"
    )(age_group_in)

    inputs = {
        "semantic_embedding": semantic_in,
        "preferred_category_id": preferred_category_in,
        "age_group_id": age_group_in,
        "category_affinity": category_affinity_in,
        "brand_affinity": brand_affinity_in,
        "numeric": numeric_in,
    }
    concat_parts = [semantic_in, preferred_category_emb, age_group_emb]

    # Same fixed BUDGET/MID/PREMIUM/unknown vocabulary as the item tower
    # (`encoder.price_tier_vocab` is shared) - the user's DERIVED price
    # tier (`UserFeatures.price_profile.price_tier`), not the age_group-
    # style opaque label - see `features.price.build_user_price_profile`.
    # STEP 8 (section 17): omitted entirely when
    # `encoder.include_price_features` is False - see `build_item_tower`.
    if encoder.include_price_features:
        price_tier_in = tf.keras.Input(shape=(), dtype="int32", name="price_tier_id")
        price_tier_emb = tf.keras.layers.Embedding(
            encoder.price_tier_vocab.size, config.price_tier_embedding_dim, name="user_price_tier_embedding"
        )(price_tier_in)
        inputs["price_tier_id"] = price_tier_in
        concat_parts.append(price_tier_emb)

    concat_parts += [category_affinity_in, brand_affinity_in, numeric_in]
    concat = tf.keras.layers.Concatenate(name="user_concat")(concat_parts)
    output = _projection_mlp(concat, config, "user")

    return tf.keras.Model(inputs=inputs, outputs=output, name="user_tower")


class TwoTowerModel(tf.keras.Model):
    """Wraps the two towers and implements the in-batch softmax retrieval
    loss in `train_step`/`test_step`. `call()` returns (user_emb, item_emb)
    for a batch of aligned (user, positive_item) pairs; it is NOT how
    inference against the full catalog happens - that's plain
    `item_tower(...)` / `user_tower(...)` calls (see serialization.py and
    scripts/train_two_tower.py's evaluation step).
    """

    def __init__(self, user_tower: tf.keras.Model, item_tower: tf.keras.Model, temperature: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.temperature = temperature
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="in_batch_accuracy")

    def call(self, inputs, training=False):
        user_features, item_features = inputs
        user_emb = self.user_tower(user_features, training=training)
        item_emb = self.item_tower(item_features, training=training)
        return user_emb, item_emb

    def _compute_loss_and_logits(self, user_features, item_features, training: bool):
        user_emb = self.user_tower(user_features, training=training)
        item_emb = self.item_tower(item_features, training=training)
        # Both are already L2-normalized -> dot product == cosine similarity.
        logits = tf.matmul(user_emb, item_emb, transpose_b=True) / self.temperature
        batch_size = tf.shape(logits)[0]
        labels = tf.range(batch_size)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        return tf.reduce_mean(loss), logits, labels

    def train_step(self, data):
        user_features, item_features = data
        with tf.GradientTape() as tape:
            loss, logits, labels = self._compute_loss_and_logits(user_features, item_features, training=True)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(labels, logits)
        return {"loss": self.loss_tracker.result(), "in_batch_accuracy": self.accuracy_tracker.result()}

    def test_step(self, data):
        user_features, item_features = data
        loss, logits, labels = self._compute_loss_and_logits(user_features, item_features, training=False)
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(labels, logits)
        return {"loss": self.loss_tracker.result(), "in_batch_accuracy": self.accuracy_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy_tracker]
