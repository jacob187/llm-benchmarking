"""Reusable chart components for Streamlit dashboard using Plotly."""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Optional


def create_top_models_bar_chart(
    rankings: list[tuple],
    benchmark_name: Optional[str] = None,
    limit: int = 10,
) -> go.Figure:
    """
    Create horizontal bar chart of top models.

    Args:
        rankings: List of (model_name, score, rank) tuples
        benchmark_name: Optional benchmark name for title
        limit: Maximum number of models to show

    Returns:
        Plotly Figure object
    """
    if not rankings:
        return _create_empty_chart("No data available")

    # Take top N models
    data = rankings[:limit]

    df = pd.DataFrame(
        [{"model": r[0], "score": r[1], "rank": r[2]} for r in data]
    )

    # Sort by score descending for display
    df = df.sort_values("score", ascending=True)

    fig = px.bar(
        df,
        x="score",
        y="model",
        orientation="h",
        color="score",
        color_continuous_scale="Blues",
        title=f"Top {len(df)} Models by {benchmark_name or 'Average Score'}",
    )

    fig.update_layout(
        showlegend=False,
        height=400,
        xaxis_title="Score",
        yaxis_title="",
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    # Add score labels on bars
    fig.update_traces(
        texttemplate="%{x:.1f}",
        textposition="inside",
        textfont_size=12,
    )

    return fig


def create_trend_line_chart(
    history_data: list,
    model_name: str,
    show_markers: bool = True,
) -> go.Figure:
    """
    Create time-series line chart for score evolution.

    Args:
        history_data: List of ScoreHistory records
        model_name: Model name for title
        show_markers: Whether to show data point markers

    Returns:
        Plotly Figure object
    """
    if not history_data:
        return _create_empty_chart("No history data available")

    # Convert to dataframe
    df = pd.DataFrame([
        {
            "date": h.date,
            "score": h.score,
            "change": h.change if hasattr(h, "change") else None,
        }
        for h in history_data
    ])

    # Sort by date ascending for proper line chart
    df = df.sort_values("date")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["score"],
        mode="lines+markers" if show_markers else "lines",
        name=model_name,
        line=dict(width=2, color="#2563eb"),
        marker=dict(size=8),
        hovertemplate="<b>%{y:.1f}</b><br>%{x|%Y-%m-%d}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Score History: {model_name}",
        xaxis_title="Date",
        yaxis_title="Score",
        height=350,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


def create_multi_model_trend_chart(
    comparison_data: dict,
    benchmark_name: str,
    show_markers: bool = True,
) -> go.Figure:
    """
    Create multi-line chart comparing multiple models.

    Args:
        comparison_data: Dict mapping model_name -> list of ScoreHistory
        benchmark_name: Benchmark name for title
        show_markers: Whether to show data point markers

    Returns:
        Plotly Figure object
    """
    if not comparison_data:
        return _create_empty_chart("No comparison data available")

    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for i, (model_name, history) in enumerate(comparison_data.items()):
        if not history:
            continue

        df = pd.DataFrame([
            {"date": h.date, "score": h.score}
            for h in history
        ])
        df = df.sort_values("date")

        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["score"],
            mode="lines+markers" if show_markers else "lines",
            name=model_name,
            line=dict(width=2, color=color),
            marker=dict(size=6, color=color),
            hovertemplate=f"<b>{model_name}</b><br>Score: %{{y:.1f}}<br>%{{x|%Y-%m-%d}}<extra></extra>",
        ))

    fig.update_layout(
        title=f"Score Trends: {benchmark_name}",
        xaxis_title="Date",
        yaxis_title="Score",
        height=500,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=10, r=10, t=60, b=10),
    )

    return fig


def create_radar_chart(
    model_benchmarks: dict[str, dict[str, float]],
) -> go.Figure:
    """
    Create radar/spider chart for multi-model comparison.

    Args:
        model_benchmarks: Dict mapping model_name -> {benchmark_name: score}

    Returns:
        Plotly Figure object
    """
    if not model_benchmarks:
        return _create_empty_chart("No benchmark data available")

    # Find all unique benchmarks across all models
    all_benchmarks = set()
    for benchmarks in model_benchmarks.values():
        all_benchmarks.update(benchmarks.keys())

    benchmark_list = sorted(list(all_benchmarks))

    if not benchmark_list:
        return _create_empty_chart("No benchmarks found")

    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for i, (model_name, benchmarks) in enumerate(model_benchmarks.items()):
        # Get scores in benchmark order, using 0 for missing
        scores = [benchmarks.get(b, 0) for b in benchmark_list]

        # Close the radar by repeating first value
        scores_closed = scores + [scores[0]]
        benchmarks_closed = benchmark_list + [benchmark_list[0]]

        color = colors[i % len(colors)]

        fig.add_trace(go.Scatterpolar(
            r=scores_closed,
            theta=benchmarks_closed,
            fill="toself",
            name=model_name,
            line=dict(color=color),
            fillcolor=color.replace("rgb", "rgba").replace(")", ", 0.2)") if "rgb" in color else color,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
            ),
        ),
        showlegend=True,
        title="Model Comparison (Radar)",
        height=500,
        margin=dict(l=60, r=60, t=60, b=60),
    )

    return fig


def create_ranking_change_chart(
    rankings: list[tuple],
    previous_rankings: dict,
    limit: int = 10,
) -> go.Figure:
    """
    Create bar chart with ranking changes indicated.

    Args:
        rankings: Current rankings as (model_name, score, rank) tuples
        previous_rankings: Previous rankings as {model_name: rank}
        limit: Maximum number of models to show

    Returns:
        Plotly Figure object
    """
    if not rankings:
        return _create_empty_chart("No rankings available")

    data = []
    for model_name, score, rank in rankings[:limit]:
        prev_rank = previous_rankings.get(model_name)
        if prev_rank is not None:
            change = prev_rank - rank  # Positive = improved
        else:
            change = None  # New model

        data.append({
            "model": model_name,
            "score": score,
            "rank": rank,
            "change": change,
        })

    df = pd.DataFrame(data)

    # Create indicator symbols
    def get_indicator(change):
        if change is None:
            return "NEW"
        elif change > 0:
            return f"+{change}"
        elif change < 0:
            return str(change)
        else:
            return "="

    df["indicator"] = df["change"].apply(get_indicator)

    # Color based on change
    def get_color(change):
        if change is None:
            return "#9333ea"  # Purple for new
        elif change > 0:
            return "#22c55e"  # Green for up
        elif change < 0:
            return "#ef4444"  # Red for down
        else:
            return "#6b7280"  # Gray for no change

    df["color"] = df["change"].apply(get_color)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["score"],
        y=df["model"],
        orientation="h",
        marker_color=df["color"],
        text=df.apply(lambda row: f"{row['score']:.1f} ({row['indicator']})", axis=1),
        textposition="inside",
        hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>",
    ))

    fig.update_layout(
        title="Model Rankings (with Changes)",
        xaxis_title="Score",
        yaxis_title="",
        height=400,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


def create_heatmap_comparison(
    model_benchmarks: dict[str, dict[str, float]],
) -> go.Figure:
    """
    Create heatmap showing model vs benchmark performance.

    Args:
        model_benchmarks: Dict mapping model_name -> {benchmark_name: score}

    Returns:
        Plotly Figure object
    """
    if not model_benchmarks:
        return _create_empty_chart("No data available")

    # Get all unique benchmarks
    all_benchmarks = set()
    for benchmarks in model_benchmarks.values():
        all_benchmarks.update(benchmarks.keys())

    benchmark_list = sorted(list(all_benchmarks))
    model_list = list(model_benchmarks.keys())

    # Create 2D array for heatmap
    z = []
    for model in model_list:
        row = [model_benchmarks[model].get(b, None) for b in benchmark_list]
        z.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=benchmark_list,
        y=model_list,
        colorscale="RdYlGn",
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}<extra></extra>",
    ))

    fig.update_layout(
        title="Model Performance Heatmap",
        xaxis_title="Benchmark",
        yaxis_title="Model",
        height=max(300, len(model_list) * 30 + 100),
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


def _create_empty_chart(message: str) -> go.Figure:
    """
    Create an empty chart with a message.

    Args:
        message: Message to display

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray"),
    )

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=300,
    )

    return fig
