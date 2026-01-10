"""
Model Comparison Page

Side-by-side comparison of multiple models with radar charts and heatmaps.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.frontend.streamlit.utils.data_loader import (
    load_latest_rankings,
    load_aggregator_cache,
    get_model_types,
    get_model_type,
    filter_models_by_type,
    get_model_type_counts,
    MODEL_TYPE_ALL,
    MODEL_TYPE_TEXT,
)
from src.frontend.streamlit.components.charts import (
    create_radar_chart,
    create_heatmap_comparison,
)

# Page configuration
st.set_page_config(
    page_title="Model Comparison | LLM Benchmarks",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Model Comparison")
st.markdown("Compare multiple models side-by-side across all benchmarks")

# Load model list
rankings = load_latest_rankings()

if not rankings:
    st.warning("No model data available. Run `llm-bench scrape` to collect data.")
    st.stop()

all_model_names = [r[0] for r in rankings]

# Sidebar filters
with st.sidebar:
    st.header("Filters")

    # Model type filter
    type_counts = get_model_type_counts(all_model_names)

    # Format options with counts
    type_options = []
    for model_type in get_model_types():
        if model_type == MODEL_TYPE_ALL:
            type_options.append(f"{model_type} ({len(all_model_names)})")
        else:
            count = type_counts.get(model_type, 0)
            if count > 0:
                type_options.append(f"{model_type} ({count})")

    selected_type_option = st.selectbox(
        "Model Type",
        type_options,
        index=0,
        help="Filter models by type to compare similar models",
    )

    # Extract type from selection (remove count)
    selected_type = selected_type_option.split(" (")[0]

    st.markdown("---")
    st.caption(
        "Filter by type to compare similar models "
        "(e.g., only Text models or only Image models)"
    )

# Filter models by type
model_names = filter_models_by_type(all_model_names, selected_type)

if not model_names:
    st.warning(f"No {selected_type} models available.")
    st.stop()

# Show current filter
if selected_type != MODEL_TYPE_ALL:
    st.info(f"Showing **{selected_type}** models only ({len(model_names)} models)")

# Model selection
selected_models = st.multiselect(
    "Select 2-5 models to compare",
    model_names,
    default=model_names[:2] if len(model_names) >= 2 else model_names,
    max_selections=5,
    help="Choose between 2 and 5 models for comparison",
)

# Validation
if len(selected_models) < 2:
    st.warning("Please select at least 2 models to compare.")
    st.stop()

# Load benchmark data from aggregator cache
aggregator = load_aggregator_cache()

if not aggregator:
    st.warning("Benchmark details not available. Using ranking data only.")

    # Fallback: use rankings data
    st.subheader("Score Comparison")

    ranking_dict = {r[0]: r[1] for r in rankings}

    comparison_df = pd.DataFrame([
        {"Model": model, "Average Score": ranking_dict.get(model, 0)}
        for model in selected_models
    ])

    st.dataframe(
        comparison_df.style.background_gradient(
            subset=["Average Score"],
            cmap="RdYlGn",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.stop()

# Get benchmark data for selected models
model_benchmarks = {}
all_benchmarks = set()

for model_name in selected_models:
    model = aggregator.get_model(model_name)
    if model and model.benchmarks:
        model_benchmarks[model_name] = model.benchmarks
        all_benchmarks.update(model.benchmarks.keys())

if not model_benchmarks:
    st.warning("No benchmark data available for selected models.")
    st.stop()

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["📊 Score Table", "🎯 Radar Chart", "🗺️ Heatmap"])

with tab1:
    st.subheader("Benchmark Scores")

    # Create comparison DataFrame
    benchmark_list = sorted(list(all_benchmarks))

    data = []
    for benchmark in benchmark_list:
        row = {"Benchmark": benchmark}
        for model in selected_models:
            benchmarks = model_benchmarks.get(model, {})
            row[model] = benchmarks.get(benchmark, None)
        data.append(row)

    comparison_df = pd.DataFrame(data)

    # Add average row
    avg_row = {"Benchmark": "**Average**"}
    for model in selected_models:
        benchmarks = model_benchmarks.get(model, {})
        scores = [v for v in benchmarks.values() if v is not None]
        avg_row[model] = sum(scores) / len(scores) if scores else None
    data.append(avg_row)

    comparison_df = pd.DataFrame(data)

    # Style the dataframe
    model_columns = selected_models

    def highlight_best(row):
        """Highlight the best score in each row."""
        if row["Benchmark"] == "**Average**":
            return [""] * len(row)

        values = [row[m] for m in model_columns if pd.notna(row.get(m))]
        if not values:
            return [""] * len(row)

        max_val = max(values)
        styles = []
        for col in row.index:
            if col in model_columns and row[col] == max_val:
                styles.append("background-color: #d4edda")
            else:
                styles.append("")
        return styles

    styled_df = comparison_df.style.apply(highlight_best, axis=1)
    styled_df = styled_df.format(
        {model: "{:.1f}" for model in model_columns},
        na_rep="-",
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
    )

    # Winner summary
    st.markdown("---")
    st.subheader("Quick Summary")

    wins = {model: 0 for model in selected_models}

    for benchmark in benchmark_list:
        scores = {
            m: model_benchmarks.get(m, {}).get(benchmark)
            for m in selected_models
        }
        valid_scores = {k: v for k, v in scores.items() if v is not None}
        if valid_scores:
            winner = max(valid_scores, key=valid_scores.get)
            wins[winner] += 1

    cols = st.columns(len(selected_models))
    for i, model in enumerate(selected_models):
        with cols[i]:
            benchmarks = model_benchmarks.get(model, {})
            scores = list(benchmarks.values())
            avg = sum(scores) / len(scores) if scores else 0

            st.metric(
                model,
                f"{avg:.1f}",
                f"{wins[model]} wins",
            )

with tab2:
    st.subheader("Radar Comparison")

    fig = create_radar_chart(model_benchmarks)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "The radar chart shows normalized performance across all benchmarks. "
        "Larger area indicates better overall performance."
    )

with tab3:
    st.subheader("Performance Heatmap")

    fig = create_heatmap_comparison(model_benchmarks)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Green indicates higher scores, red indicates lower scores. "
        "Use this to identify each model's strengths and weaknesses."
    )

# Statistical analysis
st.markdown("---")
st.subheader("Statistical Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Per-Model Statistics**")

    stats_data = []
    for model in selected_models:
        benchmarks = model_benchmarks.get(model, {})
        scores = list(benchmarks.values())

        if scores:
            stats_data.append({
                "Model": model,
                "Mean": f"{sum(scores)/len(scores):.1f}",
                "Max": f"{max(scores):.1f}",
                "Min": f"{min(scores):.1f}",
                "Range": f"{max(scores) - min(scores):.1f}",
                "Benchmarks": len(scores),
            })

    if stats_data:
        st.dataframe(
            pd.DataFrame(stats_data),
            use_container_width=True,
            hide_index=True,
        )

with col2:
    st.markdown("**Head-to-Head Comparison**")

    if len(selected_models) == 2:
        m1, m2 = selected_models

        b1 = model_benchmarks.get(m1, {})
        b2 = model_benchmarks.get(m2, {})

        common = set(b1.keys()) & set(b2.keys())

        m1_wins = sum(1 for b in common if b1[b] > b2[b])
        m2_wins = sum(1 for b in common if b2[b] > b1[b])
        ties = len(common) - m1_wins - m2_wins

        st.metric(f"{m1} wins", m1_wins)
        st.metric(f"{m2} wins", m2_wins)
        st.metric("Ties", ties)
    else:
        st.info("Select exactly 2 models for head-to-head comparison.")

# Export
st.markdown("---")

with st.expander("Export Comparison"):
    # CSV export
    export_data = []
    for benchmark in sorted(all_benchmarks):
        row = {"Benchmark": benchmark}
        for model in selected_models:
            row[model] = model_benchmarks.get(model, {}).get(benchmark)
        export_data.append(row)

    export_df = pd.DataFrame(export_data)
    csv = export_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Comparison (CSV)",
        data=csv,
        file_name="model_comparison.csv",
        mime="text/csv",
    )
