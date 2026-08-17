"""ClickAdapter: synthetic-only in V1 (no backend table exists yet
distinct from the future `User_events` activity log).

`SyntheticClickAdapter` indexes pre-generated `ClickRecord`s by user, the
same pattern as `SyntheticSearchAdapter`. A future real implementation
(`adapters.user_events_adapter.UserEventsAdapter`, filtering `User_events`
rows to `action_type == CLICK`) implements `ClickAdapter` the same way;
nothing downstream imports this class directly.
"""

from __future__ import annotations

from collections import defaultdict

from recommendation.data.adapters.base import ClickAdapter
from recommendation.data.schemas.engagement import ClickRecord


class SyntheticClickAdapter(ClickAdapter):
    def __init__(self, click_records: list[ClickRecord]) -> None:
        self._records_by_user: dict[int, list[ClickRecord]] = defaultdict(list)
        for record in click_records:
            self._records_by_user[record.user_id].append(record)

    def get_clicks(self, user_id: int) -> list[ClickRecord]:
        return list(self._records_by_user.get(user_id, []))

    def list_all_clicks(self) -> list[ClickRecord]:
        return [record for records in self._records_by_user.values() for record in records]
