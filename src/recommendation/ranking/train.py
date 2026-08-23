"""End-to-end ranker training orchestrator.

`train_ranker(bundle, feature_result, two_tower_artifacts, config)` loads
an ALREADY-TRAINED Two-Tower model (unmodified - this does not retrain or
otherwise change the Two-Tower/VectorIndex architecture), builds a
`VectorIndex` over its item embeddings, constructs leakage-safe ranking
training/validation examples and evaluation cases (`ranking.examples`),
trains the MLP ranker with early
stopping on validation loss, and evaluates ranking quality (NDCG@K,
Precision@K, Recall@K/HitRate@K, MRR) against both the ranker's own
scores and the raw-retrieval-score baseline, on val and test splits.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from recommendation.data.adapters.base import AdapterBundle
from recommendation.embeddings.encoder import SentenceTransformerEncoder
from recommendation.features.pipeline import FeaturePipelineResult
from recommendation.features.price import build_price_catalog_context
from recommendation.features.user_features import build_user_text_embeddings
from recommendation.ranking.evaluation import RankingEvalReport, baseline_scores, evaluate_ranking
from recommendation.ranking.examples import RankingExample, build_ranking_dataset
from recommendation.ranking.features import RANKING_FEATURE_NAMES
from recommendation.ranking.model import build_ranker_model
from recommendation.retrieval.index.factory import build_vector_index, candidate_pool_size
from recommendation.retrieval.two_tower.serialization import TwoTowerArtifacts
from recommendation.retrieval.two_tower.splitting import build_user_splits
from recommendation.utils.config import AppConfig
from recommendation.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RankerTrainingResult:
    model: tf.keras.Model
    history: dict
    pool_size: int
    val_report_baseline: RankingEvalReport
    val_report_ranker: RankingEvalReport
    test_report_baseline: RankingEvalReport
    test_report_ranker: RankingEvalReport
    num_train_examples: int
    num_train_positive: int
    num_train_negative: int
    num_val_loss_examples: int
    num_eval_users: int


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _examples_to_arrays(examples: list[RankingExample]) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([e.features for e in examples])
    y = np.array([e.label for e in examples], dtype=np.float32)
    return X, y


def train_ranker(
    bundle: AdapterBundle,
    feature_result: FeaturePipelineResult,
    two_tower_artifacts: TwoTowerArtifacts,
    config: AppConfig,
    st_encoder: SentenceTransformerEncoder | None = None,
) -> RankerTrainingResult:
    rk_config = config.ranking
    _set_seeds(rk_config.random_seed)

    products = bundle.products.list_products()
    product_lookup = {p.id: p for p in products}
    product_embeddings = feature_result.product_embeddings.as_dict()
    product_features = feature_result.product_features
    engagement_profiles = feature_result.engagement_profiles

    # Reuses Two-Tower's OWN split parameters (seed + holdout threshold),
    # not a ranking-specific pair, so held-out val/test targets are
    # IDENTICAL to what Two-Tower training/evaluation used - required for
    # the ranker-vs-raw-retrieval-score baseline comparison to be
    # apples-to-apples rather than comparing against a different task.
    splits = build_user_splits(
        engagement_profiles,
        min_distinct_products_for_holdout=config.two_tower.min_distinct_products_for_holdout,
        random_seed=config.two_tower.random_seed,
    )
    num_eval_users = sum(1 for s in splits.values() if s.is_evaluable)

    st_encoder = st_encoder or SentenceTransformerEncoder(
        config.embedding.sentence_transformer_model, device=config.embedding.device, batch_size=config.embedding.encode_batch_size
    )
    text_embeddings = build_user_text_embeddings(list(engagement_profiles.values()), st_encoder)
    price_context = build_price_catalog_context(products)

    vector_index = build_vector_index(config.retrieval)
    vector_index.build(two_tower_artifacts.item_ids, two_tower_artifacts.item_embeddings)
    pool_size = candidate_pool_size(
        config.retrieval, limit=config.api.default_recommendation_count, catalog_size=len(two_tower_artifacts.item_ids)
    )

    train_examples, val_loss_examples, eval_cases = build_ranking_dataset(
        engagement_profiles,
        splits,
        product_lookup,
        product_features,
        product_embeddings,
        text_embeddings,
        config.features,
        two_tower_artifacts.encoder,
        two_tower_artifacts.user_tower,
        vector_index,
        pool_size,
        rk_config,
        price_context,
    )
    if not train_examples:
        raise ValueError("No ranking training examples were constructed - check that the synthetic dataset has purchases.")

    num_train_positive = sum(1 for e in train_examples if e.label == 1)
    num_train_negative = len(train_examples) - num_train_positive

    X_train, y_train = _examples_to_arrays(train_examples)
    X_val, y_val = (_examples_to_arrays(val_loss_examples) if val_loss_examples else (None, None))

    model = build_ranker_model(input_dim=X_train.shape[1], config=rk_config)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=rk_config.learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")],
    )

    callbacks = []
    fit_kwargs = {
        "epochs": rk_config.epochs,
        "batch_size": min(rk_config.batch_size, len(X_train)),
        "verbose": 0,
    }
    if X_val is not None and len(X_val) > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=rk_config.early_stopping_patience, restore_best_weights=True)
        )
        fit_kwargs["validation_data"] = (X_val, y_val)
    fit_kwargs["callbacks"] = callbacks

    history = model.fit(X_train, y_train, **fit_kwargs)

    def ranker_scores(case) -> np.ndarray:
        return model.predict(case.features, verbose=0).reshape(-1)

    val_report_baseline = evaluate_ranking(eval_cases, "val_product_id", "val_baseline", baseline_scores, rk_config.eval_k_values)
    val_report_ranker = evaluate_ranking(eval_cases, "val_product_id", "val_ranker", ranker_scores, rk_config.eval_k_values)
    test_report_baseline = evaluate_ranking(eval_cases, "test_product_id", "test_baseline", baseline_scores, rk_config.eval_k_values)
    test_report_ranker = evaluate_ranking(eval_cases, "test_product_id", "test_ranker", ranker_scores, rk_config.eval_k_values)

    logger.info(
        "Ranker training complete: %d train examples (%d pos / %d neg), %d eval users, "
        "test NDCG@10 ranker=%.3f baseline=%.3f",
        len(train_examples),
        num_train_positive,
        num_train_negative,
        num_eval_users,
        test_report_ranker.ndcg_at_k.get(10, float("nan")),
        test_report_baseline.ndcg_at_k.get(10, float("nan")),
    )

    return RankerTrainingResult(
        model=model,
        history=history.history,
        pool_size=pool_size,
        val_report_baseline=val_report_baseline,
        val_report_ranker=val_report_ranker,
        test_report_baseline=test_report_baseline,
        test_report_ranker=test_report_ranker,
        num_train_examples=len(train_examples),
        num_train_positive=num_train_positive,
        num_train_negative=num_train_negative,
        num_val_loss_examples=len(val_loss_examples),
        num_eval_users=num_eval_users,
    )


__all__ = ["RankerTrainingResult", "train_ranker", "RANKING_FEATURE_NAMES"]
