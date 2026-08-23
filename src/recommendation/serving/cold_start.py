"""Three-level personalization/cold-start tiering (docs/data-mapping.md
section 3). History strength = `UserFeatures.total_engagement_events`
(count across the five engagement signals: clicks, purchases, cart adds,
searches, chatbot-context presence) - thresholded via `configs/base.yaml:
cold_start.*`, never hard-coded.
"""

from __future__ import annotations

from enum import Enum

from recommendation.utils.config import ColdStartConfig


class HistoryTier(str, Enum):
    STRONG = "strong"
    SPARSE = "sparse"
    NO_HISTORY = "no_history"


def determine_history_tier(total_engagement_events: int, config: ColdStartConfig) -> HistoryTier:
    if total_engagement_events >= config.strong_history_min_signals:
        return HistoryTier.STRONG
    if total_engagement_events >= config.sparse_history_min_signals:
        return HistoryTier.SPARSE
    return HistoryTier.NO_HISTORY
