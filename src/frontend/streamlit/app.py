"""
LLM Benchmark Dashboard - Main Entry Point

A Streamlit-based web dashboard for visualizing and analyzing LLM benchmark data.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.frontend.streamlit.utils.data_loader import (
    check_database_exists,
    load_database_stats,
    load_latest_rankings,
    load_previous_rankings,
    load_new_models,
    clear_all_caches,
    get_model_type,
    get_model_types,
    get_model_type_counts,
    MODEL_TYPE_TEXT,
    MODEL_TYPE_ALL,
    is_main_vendor,
    get_vendor_options,
    get_vendor_counts,
    VENDOR_ALL,
    VENDOR_MAIN,
)
from src.frontend.streamlit.components.charts import (
    create_top_models_bar_chart,
    create_ranking_change_chart,
)

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="LLM Benchmark Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
with st.sidebar:
    st.title("📊 LLM Benchmarks")
    st.markdown("---")

    # System status
    if check_database_exists():
        st.success("Database connected")
    else:
        st.warning("Using cached data")

    st.markdown("---")

    # Refresh button
    if st.button("🔄 Refresh Data"):
        clear_all_caches()
        st.rerun()

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "Track and compare language model "
        "performance across benchmarks."
    )

# Main content
st.title("📊 LLM Benchmark Dashboard")
st.markdown("Track and compare language model performance across benchmarks")

# Stats cards
stats = load_database_stats()

if stats:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Models", stats["total_models"])

    with col2:
        st.metric("Benchmarks", stats["total_benchmarks"])

    with col3:
        st.metric("Data Points", f"{stats['total_scores']:,}")

    with col4:
        if stats["date_range_days"]:
            st.metric("Tracking Days", stats["date_range_days"])
        else:
            st.metric("Runs", stats["total_runs"])

    st.markdown("---")

    # Load rankings data - use LMSYS Arena ELO specifically to avoid
    # averaging issues with incompatible benchmark scales
    all_rankings = load_latest_rankings(benchmark="LMSYS Arena ELO")
    all_previous_rankings = load_previous_rankings(benchmark="LMSYS Arena ELO")

    # FILTERS - prominent at top
    if all_rankings:
        model_names = [r[0] for r in all_rankings]

        # Initialize session state - defaults
        if "home_model_type" not in st.session_state:
            st.session_state.home_model_type = MODEL_TYPE_TEXT
        if "home_vendor" not in st.session_state:
            st.session_state.home_vendor = VENDOR_MAIN

        # MODEL TYPE FILTER
        st.markdown("### Filter by Model Type")
        type_counts = get_model_type_counts(model_names)

        type_options_list = []
        for model_type in get_model_types():
            if model_type == MODEL_TYPE_ALL:
                count = len(all_rankings)
            else:
                count = type_counts.get(model_type, 0)
            if count > 0 or model_type == MODEL_TYPE_ALL:
                type_options_list.append((model_type, count))

        type_cols = st.columns(len(type_options_list))

        for i, (model_type, count) in enumerate(type_options_list):
            with type_cols[i]:
                is_selected = st.session_state.home_model_type == model_type
                button_type = "primary" if is_selected else "secondary"
                if st.button(
                    f"{model_type} ({count})",
                    key=f"home_type_btn_{model_type}",
                    use_container_width=True,
                    type=button_type,
                ):
                    st.session_state.home_model_type = model_type
                    st.rerun()

        selected_type = st.session_state.home_model_type

        # VENDOR FILTER
        st.markdown("### Filter by Vendor")
        vendor_counts = get_vendor_counts(model_names)

        vendor_options = []
        for vendor in get_vendor_options():
            if vendor == VENDOR_ALL:
                count = len(all_rankings)
            else:
                count = vendor_counts.get(vendor, 0)
            vendor_options.append((vendor, count))

        vendor_cols = st.columns(len(vendor_options))

        for i, (vendor, count) in enumerate(vendor_options):
            with vendor_cols[i]:
                is_selected = st.session_state.home_vendor == vendor
                button_type = "primary" if is_selected else "secondary"
                if st.button(
                    f"{vendor} ({count})",
                    key=f"home_vendor_btn_{vendor}",
                    use_container_width=True,
                    type=button_type,
                ):
                    st.session_state.home_vendor = vendor
                    st.rerun()

        selected_vendor = st.session_state.home_vendor

        # Apply filters
        rankings = all_rankings

        # Filter by type
        if selected_type != MODEL_TYPE_ALL:
            rankings = [r for r in rankings if get_model_type(r[0]) == selected_type]

        # Filter by vendor
        if selected_vendor == VENDOR_MAIN:
            rankings = [r for r in rankings if is_main_vendor(r[0])]
        elif selected_vendor != VENDOR_ALL:
            rankings = [r for r in rankings if not is_main_vendor(r[0])]

        # Filter previous_rankings dict to match
        if all_previous_rankings:
            filtered_model_names = {r[0] for r in rankings}
            previous_rankings = {k: v for k, v in all_previous_rankings.items() if k in filtered_model_names}
        else:
            previous_rankings = None

        st.markdown("---")

    # Top models section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Top 10 Models by Average Score")

        if rankings:
            # Use ranking change chart if we have previous data
            if previous_rankings:
                fig = create_ranking_change_chart(
                    rankings, previous_rankings, limit=10
                )
            else:
                fig = create_top_models_bar_chart(rankings, limit=10)

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No ranking data available.")

    with col2:
        st.subheader("Recently Added Models")

        new_models = load_new_models(limit=5)

        if new_models:
            for model in new_models:
                with st.container():
                    st.markdown(f"**{model.model_name}**")
                    cols = st.columns(2)
                    with cols[0]:
                        st.caption(model.first_seen.strftime("%Y-%m-%d"))
                    with cols[1]:
                        if model.initial_score:
                            st.caption(f"Score: {model.initial_score:.1f}")
                    st.markdown("---")
        else:
            st.info("No new models detected recently.")

    # Quick stats row
    st.subheader("Quick Stats")

    if rankings:
        col1, col2, col3 = st.columns(3)

        scores = [r[1] for r in rankings if r[1] is not None]

        with col1:
            if scores:
                avg_score = sum(scores) / len(scores)
                st.metric("Average Score", f"{avg_score:.1f}")

        with col2:
            if scores:
                st.metric("Highest Score", f"{max(scores):.1f}")

        with col3:
            st.metric("Total Models Ranked", len(rankings))

else:
    # No data state
    st.warning("No benchmark data available.")

    st.markdown("""
    ### Get Started

    To populate the dashboard with data:

    1. **Run the scraper** to collect benchmark data:
       ```bash
       llm-bench scrape
       ```

    2. **Refresh this page** after scraping completes.

    The dashboard will automatically display data once available.
    """)

    # Check for cache as fallback
    st.markdown("---")
    st.info("💡 If you have a cache file (data/processed/cache.json), it will be used as a fallback.")

# Footer
st.markdown("---")
st.caption(
    "LLM Benchmark Dashboard | "
    "Built with Streamlit | "
    f"Data source: {'Database' if check_database_exists() else 'Cache'}"
)
