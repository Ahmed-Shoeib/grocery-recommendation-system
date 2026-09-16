"""`BackendApiClient.iter_activity_pages` (bounded-by-design, never raises
on running out of budget) and `.list_users` (the `GET /api/users` full
roster, page-NUMBER paginated) - both added for the activity-loading
architecture fix (docs/data-mapping.md 19.13, the 1.5M-row
`/api/ai/user-activities` blocker).
"""

from __future__ import annotations

from recommendation.backend.client import BackendApiClient
from recommendation.config import BackendApiConfig
from tests._backend_fakes import FakeResponse, FakeSession


class _StubProvider:
    def __init__(self, token="tok"):
        self._token = token

    def token(self):
        return self._token

    def invalidate(self):
        pass

    def has_credentials(self):
        return True


def _client(responses, **cfg):
    config = BackendApiConfig(base_url="https://backend.test", max_retries=cfg.pop("max_retries", 1), **cfg)
    session = FakeSession(responses)
    return BackendApiClient(config, session=session, token_provider=_StubProvider()), session


def _activity_envelope(rows, *, has_next=False, next_cursor=None):
    return {"success": True, "data": {"data": rows, "pagination": {"hasNext": has_next, "nextCursor": next_cursor}}}


def _act(guid="g1", product_id=1, ts="2026-08-01T10:00:00"):
    return {"userId": guid, "actionType": "AddToCart", "productId": product_id, "timestamp": ts}


# --- iter_activity_pages -------------------------------------------------


def test_iter_activity_pages_stops_at_max_pages_without_raising():
    """The core safety property: unlike `list_activities`, running out of
    `max_pages` budget must never raise `BackendPaginationError` - the
    real table has 1.5M+ rows and effectively never reports hasNext=False
    within any bounded budget a caller would choose.
    """
    responses = [
        FakeResponse(json_body=_activity_envelope([_act()], has_next=True, next_cursor="c1")),
        FakeResponse(json_body=_activity_envelope([_act()], has_next=True, next_cursor="c2")),
        FakeResponse(json_body=_activity_envelope([_act()], has_next=True, next_cursor="c3")),
    ]
    client, session = _client(responses)
    pages = list(client.iter_activity_pages(max_pages=3))
    assert len(pages) == 3
    assert len(session.calls) == 3


def test_iter_activity_pages_stops_early_when_hasnext_is_false():
    responses = [FakeResponse(json_body=_activity_envelope([_act()], has_next=False))]
    client, session = _client(responses)
    pages = list(client.iter_activity_pages(max_pages=100))
    assert len(pages) == 1
    assert len(session.calls) == 1


def test_iter_activity_pages_passes_user_guid_as_userid_param():
    responses = [FakeResponse(json_body=_activity_envelope([_act(guid="g7")], has_next=False))]
    client, session = _client(responses)
    pages = list(client.iter_activity_pages(user_guid="g7"))
    assert session.calls[0]["params"]["userId"] == "g7"
    assert [a.user_id for a in pages[0]] == ["g7"]


def test_iter_activity_pages_is_lazy_generator_not_eager():
    """Consuming only the first page must not issue requests for later
    pages - required for the delta-sync early-stop optimization to be
    genuinely cheap.
    """
    responses = [
        FakeResponse(json_body=_activity_envelope([_act()], has_next=True, next_cursor="c1")),
        FakeResponse(json_body=_activity_envelope([_act()], has_next=True, next_cursor="c2")),
    ]
    client, session = _client(responses)
    gen = client.iter_activity_pages(max_pages=10)
    next(gen)
    assert len(session.calls) == 1


def test_iter_activity_pages_sends_bearer_auth():
    responses = [FakeResponse(json_body=_activity_envelope([], has_next=False))]
    client, session = _client(responses)
    list(client.iter_activity_pages())
    assert session.calls[0]["headers"]["Authorization"] == "Bearer tok"


# --- fetch_activity_window --------------------------------------------------


def test_fetch_activity_window_reports_exhausted_true_on_genuine_completion():
    responses = [FakeResponse(json_body=_activity_envelope([_act()], has_next=False))]
    client, session = _client(responses)
    rows, exhausted = client.fetch_activity_window(max_pages=100)
    assert len(rows) == 1
    assert exhausted is True
    assert len(session.calls) == 1


def test_fetch_activity_window_reports_exhausted_false_when_the_cap_is_hit():
    responses = [
        FakeResponse(json_body=_activity_envelope([_act()], has_next=True, next_cursor="c1")),
        FakeResponse(json_body=_activity_envelope([_act()], has_next=True, next_cursor="c2")),
    ]
    client, session = _client(responses)
    rows, exhausted = client.fetch_activity_window(max_pages=2)
    assert len(rows) == 2
    assert exhausted is False, "hitting the page cap without hasNext=false must never be reported as complete"


def test_fetch_activity_window_passes_user_guid():
    responses = [FakeResponse(json_body=_activity_envelope([_act(guid="g7")], has_next=False))]
    client, session = _client(responses)
    rows, exhausted = client.fetch_activity_window(user_guid="g7")
    assert session.calls[0]["params"]["userId"] == "g7"
    assert exhausted is True


# --- list_users ------------------------------------------------------------


def _user_envelope(rows, *, has_next=False):
    return {"success": True, "data": {"data": rows, "pagination": {"hasNext": has_next}}}


def test_list_users_paginates_by_page_number():
    responses = [
        FakeResponse(json_body=_user_envelope([{"guid": "u1"}], has_next=True)),
        FakeResponse(json_body=_user_envelope([{"guid": "u2"}], has_next=False)),
    ]
    client, session = _client(responses)
    users = client.list_users()
    assert [u.guid for u in users] == ["u1", "u2"]
    assert session.calls[0]["params"] == {"PageNumber": 1, "PageSize": 100}
    assert session.calls[1]["params"] == {"PageNumber": 2, "PageSize": 100}


def test_list_users_page_size_is_capped_at_100():
    client, session = _client([FakeResponse(json_body=_user_envelope([]))], page_size=500)
    client.list_users()
    assert session.calls[0]["params"]["PageSize"] == 100


def test_list_users_stops_on_empty_page():
    client, session = _client([FakeResponse(json_body=_user_envelope([], has_next=True))])
    assert client.list_users() == []
    assert len(session.calls) == 1


def test_list_users_skips_one_malformed_row_without_failing_the_rest():
    responses = [FakeResponse(json_body=_user_envelope([{"guid": "u1"}, "not-an-object"]))]
    client, _ = _client(responses)
    users = client.list_users()
    assert [u.guid for u in users] == ["u1"]


def test_list_users_sends_bearer_auth():
    client, session = _client([FakeResponse(json_body=_user_envelope([]))])
    client.list_users()
    assert session.calls[0]["headers"]["Authorization"] == "Bearer tok"
