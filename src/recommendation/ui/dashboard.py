"""Internal Streamlit recommendation dashboard (docs/data-mapping.md
section 18).

Streamlit → HTTP → FastAPI → `RecommendationService` → Pipeline is the
ONE serving path: every recommendation and every piece of user/catalog
data shown here comes from a `ui.api_client.RecommendationApiClient` HTTP
call (via the cached client from `ui.service_loader.load_api_client`).
This file no longer imports `RecommendationService`, any model/training/
retrieval module, or `ui.data_access` - it is a pure rendering layer over
API responses. It must never fall back to building a service or calling
the pipeline directly if the API is unreachable - see `ui.api_client`'s
module docstring for why.

Run FastAPI first (`uvicorn recommendation.api.app:app` or
`scripts/run_api.py`, if present), then run this dashboard with
`streamlit run src/recommendation/ui/dashboard.py` (or
`python scripts/run_dashboard.py`). The API base URL is configured via
`configs/*.yaml`'s `dashboard.api_base_url` (or the `RECS_DASHBOARD_API_BASE_URL`
env var) - `http://localhost:8000` for native dev,
`http://api:8000` inside Docker Compose (service-name networking).
"""

from __future__ import annotations

from collections import Counter

import pandas as pd
import streamlit as st

from recommendation.ui.api_client import (
    ApiClientError,
    ApiResponseError,
    ApiTimeoutError,
    ApiUnavailableError,
    MalformedResponseError,
    UnknownUserError,
)
from recommendation.ui.service_loader import load_api_client
from recommendation.utils.logging import get_logger

logger = get_logger(__name__)
st.set_page_config(page_title="Grocery Recommendation Dashboard", layout="wide")

TIER_LABELS = {"strong": "STRONG", "sparse": "SPARSE", "no_history": "NO_HISTORY"}
DEFAULT_TOP_N = 10
MAX_TOP_N = 25  # a friendly UI cap only - the API's own `ApiConfig.max_recommendation_count` is the authoritative limit


def _show_api_error(exc: ApiClientError, context: str) -> None:
    """Renders every `ui.api_client` failure mode as a clean message -
    never a raw traceback, never a silent fallback to direct model access
    (see `ui.api_client`'s module docstring).
    """
    if isinstance(exc, ApiUnavailableError):
        st.error(f"Cannot reach the recommendation API while {context}. Is FastAPI running? ({exc})")
    elif isinstance(exc, ApiTimeoutError):
        st.error(f"The recommendation API timed out while {context}. ({exc})")
    elif isinstance(exc, UnknownUserError):
        st.error(f"User {exc.user_id} was not found by the recommendation API.")
    elif isinstance(exc, ApiResponseError):
        st.error(f"The recommendation API returned an error while {context}: HTTP {exc.status_code} - {exc.message}")
    elif isinstance(exc, MalformedResponseError):
        st.error(f"The recommendation API returned an unexpected response while {context}: {exc}")
    else:
        st.error(f"An unexpected error occurred while {context}: {exc}")


def _render_user_table(users: list, selected_id: int) -> None:
    st.subheader("1. Users")
    df = pd.DataFrame(
        [{"user_id": u.user_id, "preferred_category": u.preferred_category or "—", "age_group": u.age_group or "—"} for u in users]
    )
    st.dataframe(df, height=220, use_container_width=True, hide_index=True)
    st.caption(f"{len(users)} users known to the recommendation API. Selected: **{selected_id}**")


def _render_engagement_signals(profile) -> None:
    st.subheader("2. Engagement signals")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Clicked products**")
        if profile.clicks:
            st.dataframe(pd.DataFrame(profile.clicks), hide_index=True, use_container_width=True)
        else:
            st.info("No click history for this user.")

        st.markdown("**Previous purchases**")
        if profile.purchases:
            st.dataframe(pd.DataFrame(profile.purchases), hide_index=True, use_container_width=True)
        else:
            st.info("No purchase history for this user.")

        st.markdown("**Add-to-cart behavior**")
        if profile.cart_items:
            st.dataframe(pd.DataFrame(profile.cart_items), hide_index=True, use_container_width=True)
        else:
            st.info("No items currently in cart.")

    with col2:
        st.markdown("**Searched items / terms**")
        if profile.searches:
            st.dataframe(pd.DataFrame(profile.searches), hide_index=True, use_container_width=True)
        else:
            st.info("No search history for this user.")

        st.markdown("**Chatbot context**")
        if profile.chatbot is None:
            st.info("No chatbot interaction on record for this user.")
        else:
            if profile.chatbot.get("summary"):
                st.write(profile.chatbot["summary"])
            details = {k: v for k, v in profile.chatbot.items() if k != "summary" and v}
            if details:
                st.json(details, expanded=False)


def _render_recommendation_context(profile) -> None:
    st.subheader("3. Recommendation context")
    col1, col2, col3 = st.columns(3)
    col1.metric("Cold-start tier", TIER_LABELS[profile.tier])
    col2.metric("Total engagement events", profile.total_engagement_events)
    col3.metric("Distinct products purchased", profile.distinct_products_purchased)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Category affinity**")
        if profile.category_affinity:
            st.bar_chart(pd.Series(profile.category_affinity, name="affinity"))
        else:
            st.info("No category affinity signal for this user.")
    with col2:
        st.markdown("**Brand affinity**")
        if profile.brand_affinity:
            st.bar_chart(pd.Series(profile.brand_affinity, name="affinity"))
        else:
            st.info("No brand affinity signal for this user.")


def _render_recommendations(items: list) -> None:
    st.subheader("4. Final recommendations")
    if not items:
        st.warning("No eligible recommendations were produced for this user/request.")
        return
    rows = [
        {
            "rank": item.rank,
            "product_id": item.product_id,
            "score": round(item.score, 4),
            "source": item.source,
            "name": item.product_name,
            "category": item.category,
            "brand": item.brand,
            "price": item.price,
            "is_active": item.is_active,
            "stock_quantity": item.stock_quantity,
        }
        for item in items
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        "`name` / `category` / `brand` / `price` / `is_active` / `stock_quantity` are application-level display "
        "fields returned by the API (joined from the catalog server-side) - never an internal model tensor/feature."
    )


def _render_pipeline_debug(meta, items: list) -> None:
    st.subheader("5. Pipeline / debug visibility")
    st.caption(
        "User signals → hard pre-retrieval eligibility (isActive/stock) → Two-Tower → VectorIndex retrieval "
        "(eligible products only) → Neural Ranker → Re-ranking → remaining business rules → final lightweight "
        "eligibility validation → Final recommendations (all server-side, behind the single `/v1/users/{id}"
        "/recommendations` endpoint)"
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Candidate pool size", meta.pool_size)
    col2.metric("Excluded pre-retrieval", meta.num_excluded_pre_retrieval)
    col3.metric("Excluded by final validation", meta.num_excluded_by_eligibility)
    col4.metric("Fill rate", f"{meta.fill_rate:.0%}")
    col5.metric("Server-side latency", f"{meta.latency_ms:.0f} ms")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Recommendation source breakdown**")
        sources = Counter(item.source for item in items)
        st.bar_chart(pd.Series(dict(sources), name="count")) if sources else st.info("No candidates to show.")
    with col2:
        st.markdown("**Final list category distribution**")
        categories = Counter(item.category or "(unknown)" for item in items)
        st.bar_chart(pd.Series(dict(categories), name="count")) if categories else st.info("No candidates to show.")


def _render_metrics_section(client) -> None:
    st.subheader("6. Metrics / debug")
    st.caption(
        "⚠️ These are OFFLINE metrics from the approved temporal future-purchase evaluation "
        "protocol, PERSISTED by `scripts/generate_offline_report.py` and served as a cheap read by "
        "`GET /v1/metrics/offline` (no evaluation pass runs on this request). They are NOT real "
        "production/online metrics (CTR, conversion, real engagement) - there is no event-tracking pipeline to "
        "compute those from."
    )
    try:
        response = client.get_offline_metrics()
    except ApiClientError as exc:
        _show_api_error(exc, "loading offline metrics")
        return

    provenance = response.provenance
    st.caption(
        f"run_id=`{provenance.run_id}` · ranker=`{provenance.ranker_model_version}` · "
        f"two_tower=`{provenance.two_tower_model_version}` · dataset_fingerprint=`{provenance.dataset_fingerprint_sha256_16}` · "
        f"recency={'on' if provenance.recency_enabled else 'off'}"
        + (f" ({provenance.recency_half_life_days}d half-life)" if provenance.recency_half_life_days else "")
        + f" · price_features={'on' if provenance.include_price_features else 'off'} · top_n={provenance.top_n} · "
        f"generated_at={provenance.generated_at.isoformat()}"
    )
    for label, report in (("Validation split", response.val_report), ("Test split", response.test_report)):
        st.markdown(f"**{label}** ({report.num_cases} evaluation cases)")
        k_values = sorted(report.ndcg_at_k)
        table = pd.DataFrame(
            {
                "K": k_values,
                "NDCG@K": [round(report.ndcg_at_k[k], 4) for k in k_values],
                "Precision@K": [round(report.precision_at_k[k], 4) for k in k_values],
                "Recall@K": [round(report.recall_at_k[k], 4) for k in k_values],
                "HitRate@K": [round(report.hit_rate_at_k[k], 4) for k in k_values],
            }
        )
        st.dataframe(table, hide_index=True, use_container_width=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MRR", f"{report.mrr:.4f}")
        col2.metric("Catalog coverage", f"{report.catalog_coverage:.0%}")
        col3.metric("Avg. distinct categories", f"{report.mean_distinct_categories:.1f}")
        col4.metric("Mean fill rate", f"{report.mean_fill_rate:.0%}")


def main() -> None:
    st.title("Grocery Recommendation System — Internal Dashboard")
    st.caption("Demonstration/debug tool for the recommendation engine. Not the grocery storefront.")

    client = load_api_client()

    try:
        users = client.list_users().users
    except ApiClientError as exc:
        _show_api_error(exc, "loading the user list")
        st.stop()
        return

    if not users:
        st.warning("No users found via the recommendation API.")
        st.stop()
        return

    with st.sidebar:
        st.header("Controls")
        user_ids = [u.user_id for u in users]
        selected_id = st.selectbox("User ID", user_ids, index=0)
        top_n = st.slider("Requested Top-N", min_value=1, max_value=MAX_TOP_N, value=DEFAULT_TOP_N)
        st.divider()
        model_version = st.session_state.get("_last_model_version")
        if model_version:
            st.caption(f"Model version: `{model_version}`")

    try:
        profile = client.get_user_profile(selected_id)
    except ApiClientError as exc:
        _show_api_error(exc, f"loading the profile for user {selected_id}")
        st.stop()
        return

    _render_user_table(users, selected_id)
    _render_engagement_signals(profile)
    _render_recommendation_context(profile)

    try:
        with st.spinner("Calling the recommendation API..."):
            response = client.get_recommendations(selected_id, limit=top_n)
    except ApiClientError as exc:
        _show_api_error(exc, f"generating recommendations for user {selected_id}")
        st.stop()
        return

    st.session_state["_last_model_version"] = response.meta.model_version

    _render_recommendations(response.items)
    _render_pipeline_debug(response.meta, response.items)
    _render_metrics_section(client)


main()
