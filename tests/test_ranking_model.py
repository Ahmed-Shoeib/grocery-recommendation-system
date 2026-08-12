import numpy as np

from recommendation.ranking.model import build_ranker_model
from recommendation.utils.config import RankingConfig


def test_ranker_output_shape_and_range():
    config = RankingConfig(hidden_units=[8, 4])
    model = build_ranker_model(input_dim=6, config=config)
    x = np.random.default_rng(0).normal(size=(5, 6)).astype(np.float32)
    output = model.predict(x, verbose=0)
    assert output.shape == (5, 1)
    assert np.all(output >= 0.0) and np.all(output <= 1.0)


def test_ranker_hidden_layer_widths_match_config():
    config = RankingConfig(hidden_units=[16, 8, 4])
    model = build_ranker_model(input_dim=10, config=config)
    dense_layers = [layer for layer in model.layers if layer.name.startswith("ranker_hidden_")]
    assert [layer.units for layer in dense_layers] == [16, 8, 4]


def test_ranker_trains_and_reduces_loss():
    config = RankingConfig(hidden_units=[8], epochs=1, learning_rate=0.01)
    model = build_ranker_model(input_dim=4, config=config)
    model.compile(optimizer="adam", loss="binary_crossentropy")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)  # learnable signal in feature 0

    history = model.fit(X, y, epochs=20, batch_size=16, verbose=0)
    assert history.history["loss"][-1] < history.history["loss"][0]
