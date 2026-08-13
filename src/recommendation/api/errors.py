"""API-layer exceptions - distinct from pipeline exceptions, so route
handlers can map them to specific, clean HTTP responses instead of
letting them fall through to the generic 500 handler.
"""

from __future__ import annotations


class UnknownUserError(Exception):
    """Raised when a user_id has no profile in the backend at all - a
    genuinely unknown user, distinct from a known user with an empty
    engagement history (which is NO_HISTORY tier, a normal 200 response
    with fallback recommendations - see `serving.cold_start`).
    """

    def __init__(self, user_id: int) -> None:
        self.user_id = user_id
        super().__init__(f"unknown user_id: {user_id}")
