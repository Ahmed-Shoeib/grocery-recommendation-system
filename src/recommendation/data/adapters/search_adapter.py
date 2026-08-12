"""SearchAdapter: synthetic-only in V1 (no backend table exists yet).

`SyntheticSearchAdapter` simply indexes pre-generated `SearchRecord`s by
user. A future real implementation (API-backed, DB-table-backed, or
event-pipeline-backed) implements `SearchAdapter` the same way and is a
drop-in replacement - nothing downstream imports this class directly.
"""

from __future__ import annotations

from collections import defaultdict

from recommendation.data.adapters.base import SearchAdapter
from recommendation.data.schemas.engagement import SearchRecord


class SyntheticSearchAdapter(SearchAdapter):
    def __init__(self, search_records: list[SearchRecord]) -> None:
        self._records_by_user: dict[int, list[SearchRecord]] = defaultdict(list)
        for record in search_records:
            self._records_by_user[record.user_id].append(record)

    def get_search_history(self, user_id: int) -> list[SearchRecord]:
        return list(self._records_by_user.get(user_id, []))
