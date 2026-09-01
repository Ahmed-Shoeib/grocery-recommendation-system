"""Structured backend-integration errors.

Every failure mode is a distinct type so a caller (and a log reader) can
tell WHERE integration broke - the network, the HTTP layer, the response
contract, identity resolution, or downstream canonical validation - rather
than a single opaque `Exception`. Nothing in this package ever catches
`Exception` broadly and continues with partial/corrupt recommendation
data.
"""

from __future__ import annotations


class BackendApiError(RuntimeError):
    """Base class for every backend-integration failure."""


class BackendUnavailableError(BackendApiError):
    """The backend could not be reached at all: DNS failure, connection
    refused, TLS handshake failure, or a timeout after all retries.
    """


class BackendResponseError(BackendApiError):
    """The backend answered with a non-success HTTP status (after retries
    for retryable statuses). Carries the status code and a short body
    excerpt for diagnosis.
    """

    def __init__(self, message: str, *, status_code: int, body_excerpt: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body_excerpt = body_excerpt


class BackendAuthError(BackendResponseError):
    """The backend rejected the request with 401/403. Raised only for
    endpoints the recommender genuinely needs; endpoints that are
    best-effort enrichment (e.g. per-user profile) log and degrade instead
    of raising. See `client.BackendApiClient`.
    """


class BackendContractError(BackendApiError):
    """The transport succeeded but the payload did not match the expected
    contract: invalid JSON, a missing/!=true `success` envelope flag, a
    missing `data` key, or a list page missing its `pagination` block.
    """


class BackendPaginationError(BackendContractError):
    """Pagination did not terminate within the configured page budget, or
    a `nextCursor` was advertised (`hasNext=true`) but absent/unusable.
    """
