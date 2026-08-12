"""ChatbotContextAdapter: synthetic-only in V1 (no backend entity exists yet).

`SyntheticChatbotAdapter` indexes pre-generated `ChatbotContextRecord`s by
user (at most one per user in V1). A future real implementation (API/DB
entity/event-summary-backed) implements `ChatbotContextAdapter` the same
way; nothing downstream imports this class directly.
"""

from __future__ import annotations

from recommendation.data.adapters.base import ChatbotContextAdapter
from recommendation.data.schemas.engagement import ChatbotContextRecord


class SyntheticChatbotAdapter(ChatbotContextAdapter):
    def __init__(self, chatbot_records: list[ChatbotContextRecord]) -> None:
        self._record_by_user: dict[int, ChatbotContextRecord] = {r.user_id: r for r in chatbot_records}

    def get_chatbot_context(self, user_id: int) -> ChatbotContextRecord | None:
        return self._record_by_user.get(user_id)
