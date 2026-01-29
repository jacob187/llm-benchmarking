"""
Historical Trends Page

View and compare model performance trends over time.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path for imports
project_root = Path(__file__).parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.frontend.streamlit.utils.data_loader import (
    load_latest_rankings,
    load_score_history,
    compare_models_over_time,
    get_available_benchmarks,
    get_model_types,
    filter_models_by_type,
    get_model_type_counts,
    MODEL_TYPE_ALL,
    MODEL_TYPE_TEXT,
)
from src.frontend.streamlit.components.charts import (
    create_trend_line_chart,
    create_multi_model_trend_chart,
)

# Page configuration
st.set_page_config(
    page_title="Historical Trends | LLM Benchmarks",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Historical Trends")
st.markdown("Track and compare model performance over time")

# Load model list - use LMSYS Arena ELO to avoid scale issues
rankings = load_latest_rankings(benchmark="LMSYS Arena ELO")

if not rankings:
    st.warning("No model data available. Run `llm-bench scrape` to collect data.")
    st.stop()

all_model_names = [r[0] for r in rankings]

# Get type counts for filter
type_counts = get_model_type_counts(all_model_names)

# Build type options with counts
type_options_list = []
for model_type in get_model_types():
    if model_type == MODEL_TYPE_ALL:
        count = len(all_model_names)
    else:
        count = type_counts.get(model_type, 0)
    if count > 0 or model_type == MODEL_TYPE_ALL:
        type_options_list.append((model_type, count))

# PROMINENT MODEL TYPE FILTER - at top of page
st.markdown("### Filter by Model Type")
type_cols = st.columns(len(type_options_list))

# Initialize session state for selected type
if "trends_model_type" not in st.session_state:
    st.session_state.trends_model_type = MODEL_TYPE_TEXT  # Default to Text

for i, (model_type, count) in enumerate(type_options_list):
    with type_cols[i]:
        is_selected = st.session_state.trends_model_type == model_type
        button_type = "primary" if is_selected else "secondary"
        if st.button(
            f"{model_type} ({count})",
            key=f"trends_type_btn_{model_type}",
            use_container_width=True,
            type=button_type,
        ):
            st.session_state.trends_model_type = model_type
            st.rerun()

selected_type = st.session_state.trends_model_type

st.markdown("---")

# Filter models by type
model_names = filter_models_by_type(all_model_names, selected_type)

if not model_names:
    st.warning(f"No {selected_type} models available.")
    st.stop()

# Show current filter status
st.success(f"Showing **{selected_type}** models ({len(model_names)} available)")

# Sidebar controls
with st.sidebar:
    st.header("Chart Configuration")

    # Model selection
    default_models = model_names[:3] if len(model_names) >= 3 else model_names
    selected_models = st.multiselect(
        "Select models to compare",
        model_names,
        default=default_models,
        max_selections=5,
        help="Select up to 5 models to compare",
    )

    # Benchmark selection
    benchmarks = get_available_benchmarks()
    selected_benchmark = st.selectbox(
        "Benchmark",
        benchmarks,
        index=0,
        help="Select a benchmark to compare on",
    )

    st.markdown("---")

    # Chart options
    st.subheader("Display Options")

    show_markers = st.checkbox(
        "Show data points",
        value=True,
        help="Display markers at each data point",
    )

    limit = st.slider(
        "Data points limit",
        min_value=10,
        max_value=100,
        value=50,
        step=10,
        help="Maximum number of historical data points",
    )

# Main content
if not selected_models:
    st.info("Select at least one model from the sidebar to view trends.")
    st.stop()

# Comparison mode vs single model mode
if len(selected_models) == 1:
    # Single model view
    model_name = selected_models[0]

    st.subheader(f"Score History: {model_name}")

    # Get history data
    benchmark_name = None if selected_benchmark == "Average Score" else selected_benchmark
    history = load_score_history(
        model_name,
        benchmark_name=benchmark_name,
        limit=limit,
    )

    if history:
        # Create chart
        fig = create_trend_line_chart(
            history,
            model_name,
            show_markers=show_markers,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        st.markdown("---")
        st.subheader("Statistics")

        scores = [h.score for h in history]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Latest Score", f"{scores[0]:.1f}")

        with col2:
            st.metric("Average", f"{sum(scores)/len(scores):.1f}")

        with col3:
            st.metric("Highest", f"{max(scores):.1f}")

        with col4:
            st.metric("Lowest", f"{min(scores):.1f}")

        # Change analysis
        if len(scores) >= 2:
            change = scores[0] - scores[-1]
            pct_change = (change / scores[-1]) * 100 if scores[-1] != 0 else 0

            st.markdown("---")
            st.subheader("Change Analysis")

            change_cols = st.columns(2)
            with change_cols[0]:
                st.metric(
                    "Change (first to last)",
                    f"{change:+.2f}",
                    delta=f"{pct_change:+.1f}%",
                )
            with change_cols[1]:
                st.metric(
                    "Data Points",
                    len(scores),
                )

        # Raw data
        with st.expander("View Raw Data"):
            history_df = pd.DataFrame([
                {
                    "Date": h.date.strftime("%Y-%m-%d"),
                    "Score": h.score,
                    "Change": f"{h.change:+.2f}" if h.change else "-",
                    "Source": h.source if hasattr(h, "source") else "-",
                }
                for h in history
            ])
            st.dataframe(history_df, use_container_width=True, hide_index=True)

    else:
        st.warning(f"No history data available for {model_name}.")

else:
    # Multi-model comparison view
    st.subheader(f"Comparing {len(selected_models)} Models")

    benchmark_name = None if selected_benchmark == "Average Score" else selected_benchmark

    # For comparison, we need to get data for each model
    # Using the compare function if benchmark is specified, otherwise individual queries
    if benchmark_name:
        comparison_data = compare_models_over_time(
            selected_models,
            benchmark_name,
            limit=limit,
        )
    else:
        # Get individual score histories for average
        comparison_data = {}
        for model in selected_models:
            history = load_score_history(model, limit=limit)
            if history:
                comparison_data[model] = history

    if comparison_data:
        # Create multi-line chart
        fig = create_multi_model_trend_chart(
            comparison_data,
            selected_benchmark,
            show_markers=show_markers,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        st.markdown("---")
        st.subheader("Comparison Summary")

        summary_data = []
        for model_name in selected_models:
            history = comparison_data.get(model_name, [])
            if history:
                scores = [h.score for h in history]
                summary_data.append({
                    "Model": model_name,
                    "Latest": f"{scores[0]:.1f}",
                    "Average": f"{sum(scores)/len(scores):.1f}",
                    "High": f"{max(scores):.1f}",
                    "Low": f"{min(scores):.1f}",
                    "Points": len(scores),
                })
            else:
                summary_data.append({
                    "Model": model_name,
                    "Latest": "-",
                    "Average": "-",
                    "High": "-",
                    "Low": "-",
                    "Points": 0,
                })

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
        )

        # Winner analysis
        st.markdown("---")
        st.subheader("Performance Analysis")

        models_with_data = [
            m for m in selected_models
            if comparison_data.get(m)
        ]

        if models_with_data:
            latest_scores = {
                m: comparison_data[m][0].score
                for m in models_with_data
            }

            winner = max(latest_scores, key=latest_scores.get)
            winner_score = latest_scores[winner]

            st.success(f"**Current Leader:** {winner} with score {winner_score:.1f}")

            # Show rankings
            sorted_models = sorted(
                latest_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )

            st.markdown("**Current Standings:**")
            for i, (model, score) in enumerate(sorted_models, 1):
                diff = winner_score - score
                st.write(
                    f"{i}. **{model}**: {score:.1f}"
                    + (f" (-{diff:.1f})" if diff > 0 else "")
                )

    else:
        st.warning("No comparison data available for selected models.")

# Export section
st.markdown("---")

with st.expander("Export Options"):
    col1, col2 = st.columns(2)

    with col1:
        if len(selected_models) == 1 and 'history' in dir() and history:
            csv_data = pd.DataFrame([
                {"Date": h.date, "Score": h.score}
                for h in history
            ]).to_csv(index=False)

            st.download_button(
                label="📥 Download Trend Data (CSV)",
                data=csv_data,
                file_name=f"{selected_models[0]}_trends.csv",
                mime="text/csv",
            )

    with col2:
        if 'comparison_data' in dir() and comparison_data:
            # Combine all model data
            all_data = []
            for model, history in comparison_data.items():
                for h in history:
                    all_data.append({
                        "Model": model,
                        "Date": h.date,
                        "Score": h.score,
                    })

            if all_data:
                csv_data = pd.DataFrame(all_data).to_csv(index=False)
                st.download_button(
                    label="📥 Download Comparison Data (CSV)",
                    data=csv_data,
                    file_name="model_comparison.csv",
                    mime="text/csv",
                )
