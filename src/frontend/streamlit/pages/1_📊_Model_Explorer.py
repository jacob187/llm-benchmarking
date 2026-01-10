"""
Model Explorer Page

Browse, search, and filter LLM models with detailed benchmark information.
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
    load_score_history,
    load_model_trends,
)
from src.frontend.streamlit.components.charts import create_trend_line_chart

# Page configuration
st.set_page_config(
    page_title="Model Explorer | LLM Benchmarks",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Model Explorer")
st.markdown("Browse and search all tracked language models")

# Load data
rankings = load_latest_rankings()

if not rankings:
    st.warning("No model data available. Run `llm-bench scrape` to collect data.")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame([
    {"Model": r[0], "Score": r[1], "Rank": r[2]}
    for r in rankings
])

# Sidebar filters
with st.sidebar:
    st.header("Filters")

    # Search
    search = st.text_input(
        "🔍 Search models",
        "",
        placeholder="Type to search...",
    )

    # Sort options
    sort_by = st.selectbox(
        "Sort by",
        ["Rank", "Score", "Model Name"],
    )

    sort_order = st.radio(
        "Order",
        ["Ascending", "Descending"],
        horizontal=True,
    )

    # Score range filter
    if df["Score"].notna().any():
        min_score = float(df["Score"].min())
        max_score = float(df["Score"].max())

        score_range = st.slider(
            "Score range",
            min_value=min_score,
            max_value=max_score,
            value=(min_score, max_score),
        )
    else:
        score_range = None

# Apply filters
filtered_df = df.copy()

# Search filter
if search:
    filtered_df = filtered_df[
        filtered_df["Model"].str.contains(search, case=False, na=False)
    ]

# Score range filter
if score_range:
    filtered_df = filtered_df[
        (filtered_df["Score"] >= score_range[0]) &
        (filtered_df["Score"] <= score_range[1])
    ]

# Sorting
sort_column = {
    "Rank": "Rank",
    "Score": "Score",
    "Model Name": "Model",
}[sort_by]

ascending = sort_order == "Ascending"
filtered_df = filtered_df.sort_values(sort_column, ascending=ascending)

# Display stats
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Models Shown", len(filtered_df))
with col2:
    st.metric("Total Models", len(df))
with col3:
    if filtered_df["Score"].notna().any():
        avg = filtered_df["Score"].mean()
        st.metric("Avg Score (filtered)", f"{avg:.1f}")

st.markdown("---")

# Main content: Table and Details
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Models")

    # Display table
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Model": st.column_config.TextColumn(
                "Model Name",
                width="large",
            ),
            "Score": st.column_config.NumberColumn(
                "Avg Score",
                format="%.2f",
            ),
            "Rank": st.column_config.NumberColumn(
                "Rank",
                format="%d",
            ),
        },
        height=500,
    )

with col2:
    st.subheader("Model Details")

    # Model selector
    model_list = filtered_df["Model"].tolist()

    if model_list:
        selected_model = st.selectbox(
            "Select a model to view details",
            model_list,
            index=0,
        )

        if selected_model:
            # Get model info from rankings
            model_row = filtered_df[filtered_df["Model"] == selected_model].iloc[0]

            # Display basic info
            st.markdown(f"### {selected_model}")

            info_cols = st.columns(2)
            with info_cols[0]:
                st.metric("Current Rank", f"#{int(model_row['Rank'])}")
            with info_cols[1]:
                st.metric("Average Score", f"{model_row['Score']:.2f}")

            # Score history
            st.markdown("#### Score History")

            history = load_score_history(selected_model, limit=20)

            if history:
                # Create trend chart
                fig = create_trend_line_chart(history, selected_model)
                st.plotly_chart(fig, use_container_width=True)

                # Show recent history as table
                with st.expander("View History Data"):
                    history_data = []
                    for h in history[:10]:
                        row = {
                            "Date": h.date.strftime("%Y-%m-%d"),
                            "Score": f"{h.score:.2f}",
                        }
                        if h.change is not None:
                            row["Change"] = f"{h.change:+.2f}"
                        else:
                            row["Change"] = "-"
                        history_data.append(row)

                    st.dataframe(
                        pd.DataFrame(history_data),
                        use_container_width=True,
                        hide_index=True,
                    )
            else:
                st.info("No history data available for this model.")

            # Model trends
            trends = load_model_trends(selected_model)

            if trends:
                st.markdown("#### Trend Summary")

                trend_cols = st.columns(3)

                with trend_cols[0]:
                    trend_icon = {
                        "up": "📈",
                        "down": "📉",
                        "stable": "➡️",
                    }.get(trends.rank_trend, "❓")
                    st.metric(
                        "Trend",
                        f"{trend_icon} {trends.rank_trend.capitalize() if trends.rank_trend else 'Unknown'}",
                    )

                with trend_cols[1]:
                    if trends.first_seen:
                        st.metric(
                            "First Seen",
                            trends.first_seen.strftime("%Y-%m-%d"),
                        )

                with trend_cols[2]:
                    if trends.last_seen:
                        st.metric(
                            "Last Seen",
                            trends.last_seen.strftime("%Y-%m-%d"),
                        )
    else:
        st.info("No models match your search criteria.")

# Export section
st.markdown("---")
st.subheader("Export Data")

col1, col2 = st.columns(2)

with col1:
    # Export to CSV
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name="llm_benchmarks.csv",
        mime="text/csv",
    )

with col2:
    # Export to JSON
    json_data = filtered_df.to_json(orient="records", indent=2)
    st.download_button(
        label="📥 Download as JSON",
        data=json_data,
        file_name="llm_benchmarks.json",
        mime="application/json",
    )
