"""End-to-end Two-Tower training orchestrator.

`train_two_tower(bundle, feature_result, config)` is the single entrypoint
used by both `scripts/train_two_tower.py` and tests: builds the leave-one-
out split, fits the feature encoder, constructs leakage-safe training
examples and eval cases, trains the model with early stopping on
validation loss, evaluates Recall@K/HitRate@K on val and test, and
precomputes L2-normalized embeddings for the full catalog (what Phase 5
will index). Nothing here touches FAISS/ScaNN - that starts in Phase 5.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from recommendation.data.adapters.base import AdapterBundle
from recommendation.data.schemas.engagement import EngagementProfile
from recommendation.embeddings.encoder import SentenceTransformerEncoder
from recommendation.features.pipeline import FeaturePipelineResult
from recommendation.features.price import build_price_catalog_context
from recommendation.features.product_features import ProductFeatures
from recommendation.features.user_features import build_user_text_embeddings
from recommendation.retrieval.two_tower.evaluation import RetrievalEvalReport, evaluate_split
from recommendation.retrieval.two_tower.examples import TrainingExample, build_eval_cases, build_training_examples
from recommendation.retrieval.two_tower.feature_encoding import TwoTowerFeatureEncoder
from recommendation.retrieval.two_tower.model import TwoTowerModel, build_item_tower, build_user_tower
from recommendation.retrieval.two_tower.splitting import UserSplit, build_user_splits
from recommendation.utils.config import AppConfig
from recommendation.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TwoTowerTrainingResult:
    user_tower: tf.keras.Model
    item_tower: tf.keras.Model
    encoder: TwoTowerFeatureEncoder
    history: dict
    val_report: RetrievalEvalReport
    test_report: RetrievalEvalReport
    item_ids: list[int]
    item_embeddings: np.ndarray
    splits: dict[int, UserSplit]
    num_training_examples: int
    num_train_users: int
    num_eval_users: int


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _fit_encoder(
    bundle: AdapterBundle, engagement_profiles: dict[int, EngagementProfile], config: AppConfig
) -> TwoTowerFeatureEncoder:
    """Vocabularies are fit from OBSERVED data only - the product catalog
    for category/brand (static item metadata, not user behavior) and the
    actual `age_group` values present on user profiles (whatever taxonomy
    the backend uses) - never from a hard-coded synthetic-specific list,
    so this works unchanged against a real backend's own age-group buckets.
    """
    products = bundle.products.list_products()
    observed_age_groups = sorted({p.profile.age_group for p in engagement_profiles.values() if p.profile.age_group})
    return TwoTowerFeatureEncoder.fit(
        category_names=[p.category_name for p in products if p.category_name],
        brand_names=[p.brand for p in products if p.brand],
        age_groups=observed_age_groups,
        prices=[p.price for p in products],
        embedding_dim=config.embedding.embedding_dim,
    )


def _examples_to_arrays(
    examples: list[TrainingExample],
    encoder: TwoTowerFeatureEncoder,
    product_features: dict[int, ProductFeatures],
    product_embeddings: dict[int, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    user_batch = encoder.encode_user_batch([e.user_features for e in examples])
    item_batch = encoder.encode_item_batch([e.product_id for e in examples], product_features, product_embeddings)
    return user_batch, item_batch


def train_two_tower(
    bundle: AdapterBundle,
    feature_result: FeaturePipelineResult,
    config: AppConfig,
    st_encoder: SentenceTransformerEncoder | None = None,
) -> TwoTowerTrainingResult:
    tt_config = config.two_tower
    _set_seeds(tt_config.random_seed)

    products = bundle.products.list_products()
    product_lookup = {p.id: p for p in products}
    product_embeddings = feature_result.product_embeddings.as_dict()
    product_features = feature_result.product_features
    engagement_profiles = feature_result.engagement_profiles

    encoder = _fit_encoder(bundle, engagement_profiles, config)

    splits = build_user_splits(
        engagement_profiles,
        min_distinct_products_for_holdout=tt_config.min_distinct_products_for_holdout,
        random_seed=tt_config.random_seed,
    )
    num_eval_users = sum(1 for s in splits.values() if s.is_evaluable)

    # Phase 3's feature pipeline computed UserFeatures WITHOUT any
    # exclusion (one static vector per user). Two-Tower training needs a
    # DIFFERENT, per-example leave-one-out vector for every (user,
    # product) pair, so user features are rebuilt here via
    # `build_user_features(..., exclude_product_ids=...)` per example -
    # cheap numpy arithmetic reusing the already-cached product/text
    # embeddings, not a re-encode. The free-text embedding lookup itself
    # (search terms without a matched product, chatbot summaries) IS a
    # real Sentence Transformer call, but only once, over the small set of
    # unique strings - consistent with Phase 3's "precompute once" cache
    # philosophy. Callers that already loaded an encoder for the Phase 3
    # pipeline should pass it in via `st_encoder` to avoid loading the
    # model twice.
    st_encoder = st_encoder or SentenceTransformerEncoder(
        config.embedding.sentence_transformer_model, device=config.embedding.device, batch_size=config.embedding.encode_batch_size
    )
    text_embeddings = build_user_text_embeddings(list(engagement_profiles.values()), st_encoder)
    price_context = build_price_catalog_context(products)

    train_examples = build_training_examples(
        engagement_profiles, splits, product_lookup, product_embeddings, config.features, text_embeddings, price_context
    )
    eval_cases = build_eval_cases(
        engagement_profiles, splits, product_lookup, product_embeddings, config.features, text_embeddings, price_context
    )
    if not train_examples:
        raise ValueError("No training examples were constructed - check that the synthetic dataset has purchases.")

    val_loss_examples = [
        TrainingExample(user_id=c.user_id, product_id=c.val_product_id, user_features=c.user_features)
        for c in eval_cases
    ]

    train_user_batch, train_item_batch = _examples_to_arrays(train_examples, encoder, product_features, product_embeddings)
    val_user_batch, val_item_batch = _examples_to_arrays(val_loss_examples, encoder, product_features, product_embeddings)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_user_batch, train_item_batch))
        .shuffle(buffer_size=len(train_examples), seed=tt_config.random_seed)
        .batch(tt_config.batch_size, drop_remainder=True)
    )
    val_ds = None
    if val_loss_examples:
        val_batch_size = min(tt_config.batch_size, len(val_loss_examples))
        val_ds = tf.data.Dataset.from_tensor_slices((val_user_batch, val_item_batch)).batch(
            val_batch_size, drop_remainder=True
        )

    user_tower = build_user_tower(encoder, tt_config)
    item_tower = build_item_tower(encoder, tt_config)
    model = TwoTowerModel(user_tower=user_tower, item_tower=item_tower, temperature=tt_config.temperature)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=tt_config.learning_rate))

    callbacks = []
    if val_ds is not None:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=tt_config.early_stopping_patience, restore_best_weights=True
            )
        )

    fit_kwargs = {"epochs": tt_config.epochs, "verbose": 0, "callbacks": callbacks}
    if val_ds is not None:
        fit_kwargs["validation_data"] = val_ds
    history = model.fit(train_ds, **fit_kwargs)

    # Precompute L2-normalized embeddings for the FULL current catalog -
    # what Phase 5 loads into FAISS, and what evaluation ranks against.
    all_item_ids = [p.id for p in products]
    all_item_batch = encoder.encode_item_batch(all_item_ids, product_features, product_embeddings)
    all_item_embeddings = item_tower.predict(all_item_batch, verbose=0)

    val_report = evaluate_split(
        eval_cases, "val_product_id", "val", user_tower, all_item_embeddings, all_item_ids, encoder, tt_config.eval_k_values
    )
    test_report = evaluate_split(
        eval_cases, "test_product_id", "test", user_tower, all_item_embeddings, all_item_ids, encoder, tt_config.eval_k_values
    )

    logger.info(
        "Two-Tower training complete: %d train examples, %d eval users, val HitRate@10=%.3f test HitRate@10=%.3f",
        len(train_examples),
        num_eval_users,
        val_report.hit_rate_at_k.get(10, float("nan")),
        test_report.hit_rate_at_k.get(10, float("nan")),
    )

    return TwoTowerTrainingResult(
        user_tower=user_tower,
        item_tower=item_tower,
        encoder=encoder,
        history=history.history,
        val_report=val_report,
        test_report=test_report,
        item_ids=all_item_ids,
        item_embeddings=all_item_embeddings,
        splits=splits,
        num_training_examples=len(train_examples),
        num_train_users=sum(1 for s in splits.values() if s.train_product_ids),
        num_eval_users=num_eval_users,
    )
