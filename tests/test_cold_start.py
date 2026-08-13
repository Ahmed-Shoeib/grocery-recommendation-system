from recommendation.serving.cold_start import HistoryTier, determine_history_tier
from recommendation.utils.config import ColdStartConfig


def _config() -> ColdStartConfig:
    return ColdStartConfig(strong_history_min_signals=5, sparse_history_min_signals=1)


def test_zero_events_is_no_history():
    assert determine_history_tier(0, _config()) == HistoryTier.NO_HISTORY


def test_below_sparse_threshold_is_no_history():
    config = ColdStartConfig(strong_history_min_signals=5, sparse_history_min_signals=2)
    assert determine_history_tier(1, config) == HistoryTier.NO_HISTORY


def test_at_sparse_threshold_is_sparse():
    assert determine_history_tier(1, _config()) == HistoryTier.SPARSE


def test_below_strong_threshold_is_sparse():
    assert determine_history_tier(4, _config()) == HistoryTier.SPARSE


def test_at_strong_threshold_is_strong():
    assert determine_history_tier(5, _config()) == HistoryTier.STRONG


def test_well_above_strong_threshold_is_strong():
    assert determine_history_tier(50, _config()) == HistoryTier.STRONG
