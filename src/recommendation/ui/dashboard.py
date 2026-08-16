"""Internal Streamlit recommendation dashboard (Phase 9).

A demonstration/debug tool for inspecting the recommendation engine -
NOT a replacement for the grocery storefront. Every recommendation shown
here comes from the exact same `serving.pipeline.generate_recommendations`
call the Phase 8 API makes (via `ui.data_access.run_recommendations`,
using the shared `RecommendationService` from `ui.service_loader`) - no
recommendation logic is reimplemented in this file. This file is a thin
rendering layer only: all data access/formatting lives in `ui
.data_access`/`ui.metrics` (plain functions, no Streamlit import, unit-
tested independently); this script just calls them and renders `st.*`.

Run with `streamlit run src/recommendation/ui/dashboard.py` (or
`python scripts/run_dashboard.py`) after `scripts/train_two_tower.py`
and `scripts/train_ranker.py` have produced the artifacts under `models/`.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from recommendation.ui.data_access import (
    UserDetail,
    category_distribution,
    format_chatbot_context,
    format_cart_items,
    format_purchases,
    format_recommendation_table,
    format_searches,
    list_users,
    load_user_detail,
    run_recommendations,
    source_distribution,
)
from recommendation.ui.metrics import compute_offline_metrics
from recommendation.ui.service_loader import load_service
from recommendation.utils.config import get_config
from recommendation.utils.logging import setup_logging

setup_logging(get_config().log_level)
st.set_page_config(page_title="Grocery Recommendation Dashboard", layout="wide")

TIER_LABELS = {"strong": "STRONG", "sparse": "SPARSE", "no_history": "NO_HISTORY"}


def _render_user_table(users, selected_id: int) -> None:
    st.subheader("1. Users")
    df = pd.DataFrame(
        [{"user_id": u.user_id, "preferred_category": u.preferred_category or "—", "age_group": u.age_group or "—"} for u in users]
    )
    st.dataframe(df, height=220, use_container_width=True, hide_index=True)
    st.caption(f"{len(users)} users in the current synthetic dataset. Selected: **{selected_id}**")


def _render_engagement_signals(detail: UserDetail, product_lookup) -> None:
    st.subheader("2. Engagement signals")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Previous purchases**")
        purchases = format_purchases(detail.engagement, product_lookup)
        if purchases:
            st.dataframe(pd.DataFrame(purchases), hide_index=True, use_container_width=True)
        else:
            st.info("No purchase history for this user.")

        st.markdown("**Add-to-cart behavior**")
        cart_items = format_cart_items(detail.engagement, product_lookup)
        if cart_items:
            st.dataframe(pd.DataFrame(cart_items), hide_index=True, use_container_width=True)
        else:
            st.info("No items currently in cart.")

    with col2:
        st.markdown("**Searched items / terms**")
        searches = format_searches(detail.engagement, product_lookup)
        if searches:
            st.dataframe(pd.DataFrame(searches), hide_index=True, use_container_width=True)
        else:
            st.info("No search history for this user.")

        st.markdown("**Chatbot context**")
        chatbot = format_chatbot_context(detail.engagement, product_lookup)
        if chatbot is None:
            st.info("No chatbot interaction on record for this user.")
        else:
            if chatbot["summary"]:
                st.write(chatbot["summary"])
            details = {k: v for k, v in chatbot.items() if k != "summary" and v}
            if details:
                st.json(details, expanded=False)


def _render_recommendation_context(detail: UserDetail) -> None:
    st.subheader("3. Recommendation context")
    col1, col2, col3 = st.columns(3)
    col1.metric("Cold-start tier", TIER_LABELS[detail.tier.value])
    col2.metric("Total engagement events", detail.features.total_engagement_events)
    col3.metric("Distinct products purchased", detail.features.distinct_products_purchased)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Category affinity**")
        if detail.features.category_affinity:
            st.bar_chart(pd.Series(detail.features.category_affinity, name="affinity"))
        else:
            st.info("No category affinity signal for this user.")
    with col2:
        st.markdown("**Brand affinity**")
        if detail.features.brand_affinity:
            st.bar_chart(pd.Series(detail.features.brand_affinity, name="affinity"))
        else:
            st.info("No brand affinity signal for this user.")


def _render_recommendations(result, product_lookup) -> None:
    st.subheader("4. Final recommendations")
    rows = format_recommendation_table(result, product_lookup)
    if not rows:
        st.warning("No eligible recommendations were produced for this user/request.")
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        "`name` / `category` / `brand` / `price` / `is_active` / `stock_quantity` come from the canonical "
        "product catalog (looked up by product_id for display only) - the recommendation model itself never stores this data."
    )


def _render_pipeline_debug(result, product_lookup, latency_ms: float) -> None:
    st.subheader("5. Pipeline / debug visibility")
    st.caption(
        "User signals → hard pre-retrieval eligibility (isActive/stock) → Two-Tower → VectorIndex retrieval "
        "(eligible products only) → Neural Ranker → Re-ranking → remaining business rules → final lightweight "
        "eligibility validation → Final recommendations"
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Candidate pool size", result.pool_size)
    col2.metric("Excluded pre-retrieval", result.num_excluded_pre_retrieval)
    col3.metric("Excluded by final validation", result.num_excluded_by_eligibility)
    col4.metric("Fill rate", f"{result.fill_rate:.0%}")
    col5.metric("This request's latency", f"{latency_ms:.0f} ms")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Recommendation source breakdown**")
        sources = source_distribution(result)
        st.bar_chart(pd.Series(sources, name="count")) if sources else st.info("No candidates to show.")
    with col2:
        st.markdown("**Final list category distribution**")
        categories = category_distribution(result.product_ids, product_lookup)
        st.bar_chart(pd.Series(categories, name="count")) if categories else st.info("No candidates to show.")

    if result.excluded_reasons:
        with st.expander(f"{len(result.excluded_reasons)} candidate(s) excluded by business rules"):
            excluded_rows = [
                {"product_id": pid, "name": product_lookup[pid].name if pid in product_lookup else "?", "reasons": ", ".join(reasons)}
                for pid, reasons in result.excluded_reasons.items()
            ]
            st.dataframe(pd.DataFrame(excluded_rows), hide_index=True, use_container_width=True)


def _render_metrics_section(service, top_n: int) -> None:
    st.subheader("6. Metrics / debug")
    st.caption(
        "⚠️ These are OFFLINE metrics computed via leave-one-out evaluation on the synthetic V1 dataset "
        "(same protocol as scripts/run_pipeline.py). They are NOT real production/online metrics (CTR, "
        "conversion, real engagement) - V1 has no event-tracking pipeline to compute those from."
    )
    if st.button("Compute offline metrics (may take a moment)"):
        with st.spinner("Re-running the leave-one-out evaluation..."):
            summary = compute_offline_metrics(service, top_n)
        st.session_state["_offline_metrics"] = summary

    summary = st.session_state.get("_offline_metrics")
    if summary is None:
        st.info("Click the button above to compute offline metrics for the current Top-N.")
        return

    st.caption(f"{summary.num_eval_users} held-out evaluation users (leave-one-out val/test targets).")
    for label, report in (("Validation split", summary.val_report), ("Test split", summary.test_report)):
        st.markdown(f"**{label}**")
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
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("MRR", f"{report.mrr:.4f}")
        col2.metric("Catalog coverage", f"{report.catalog_coverage:.0%}")
        col3.metric("Avg. distinct categories", f"{report.mean_distinct_categories:.1f}")
        col4.metric("Duplicate rate", f"{report.duplicate_rate:.0%}")
        col5.metric("Fill rate", f"{report.fill_rate:.0%}")


def main() -> None:
    st.title("Grocery Recommendation System — Internal Dashboard")
    st.caption("Demonstration/debug tool for the recommendation engine. Not the grocery storefront.")

    try:
        service = load_service()
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not re-raised
        st.error(f"Failed to load the recommendation service: {exc}")
        st.info("Make sure scripts/train_two_tower.py and scripts/train_ranker.py have been run first.")
        st.stop()

    users = list_users(service)
    if not users:
        st.warning("No users found in the current dataset.")
        st.stop()

    with st.sidebar:
        st.header("Controls")
        user_ids = [u.user_id for u in users]
        selected_id = st.selectbox("User ID", user_ids, index=0)
        top_n = st.slider(
            "Requested Top-N",
            min_value=1,
            max_value=service.config.api.max_recommendation_count,
            value=service.config.api.default_recommendation_count,
        )
        st.divider()
        st.caption(f"Model version: `{service.ranker_model_version}`")
        st.caption(f"Retrieval backend: `{service.config.retrieval.backend}`")

    detail = load_user_detail(service, selected_id)
    if detail is None:
        st.error(f"User {selected_id} not found.")
        st.stop()

    _render_user_table(users, selected_id)
    _render_engagement_signals(detail, service.product_lookup)
    _render_recommendation_context(detail)

    try:
        with st.spinner("Running pre-retrieval eligibility → Two-Tower → VectorIndex → Ranker → Re-ranking → final eligibility validation..."):
            start = time.perf_counter()
            result = run_recommendations(service, detail, top_n)
            latency_ms = (time.perf_counter() - start) * 1000
    except Exception as exc:  # noqa: BLE001 - a pipeline-stage failure must not crash the whole page
        st.error(f"Failed to generate recommendations for user {selected_id}: {exc}")
        st.info("This may indicate a VectorIndex, ranker, or feature-encoding issue - check the application logs.")
        st.stop()

    _render_recommendations(result, service.product_lookup)
    _render_pipeline_debug(result, service.product_lookup, latency_ms)
    _render_metrics_section(service, top_n)


main()
