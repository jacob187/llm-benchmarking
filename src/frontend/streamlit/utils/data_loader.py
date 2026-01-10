"""Data loading utilities with caching for Streamlit dashboard."""

import streamlit as st
from pathlib import Path
from datetime import datetime
from typing import Optional

# Import from llm_benchmarks module
from llm_benchmarks.database.repository import HistoryRepository
from llm_benchmarks.database.models import DatabaseStats
from llm_benchmarks.data_aggregator import BenchmarkAggregator


@st.cache_resource
def get_repository() -> HistoryRepository:
    """
    Get cached repository instance.

    Returns:
        HistoryRepository: Database repository instance
    """
    return HistoryRepository()


def check_database_exists() -> bool:
    """
    Check if database file exists.

    Returns:
        bool: True if database exists, False otherwise
    """
    db_path = Path("data/history.db")
    return db_path.exists()


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_database_stats() -> Optional[dict]:
    """
    Load database statistics with caching.

    Returns:
        dict: Database statistics or None if unavailable
    """
    try:
        repo = get_repository()
        stats = repo.get_database_stats()
        return {
            "total_models": stats.total_models,
            "total_benchmarks": stats.total_benchmarks,
            "total_scores": stats.total_scores,
            "total_runs": stats.total_runs,
            "successful_runs": stats.successful_runs,
            "date_range_days": stats.date_range_days,
            "first_run_date": stats.first_run_date,
            "last_run_date": stats.last_run_date,
        }
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return None


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_latest_rankings(benchmark: Optional[str] = None) -> list:
    """
    Load latest rankings with caching.

    Args:
        benchmark: Optional benchmark name filter

    Returns:
        list: List of (model_name, score, rank) tuples
    """
    try:
        repo = get_repository()
        rankings = repo.get_latest_run_rankings(benchmark_name=benchmark)
        return rankings
    except Exception as e:
        st.error(f"Error loading rankings: {str(e)}")
        return []


@st.cache_data(ttl=300)
def load_previous_rankings(benchmark: Optional[str] = None) -> dict:
    """
    Load previous run rankings for comparison.

    Args:
        benchmark: Optional benchmark name filter

    Returns:
        dict: Mapping of model_name -> rank
    """
    try:
        repo = get_repository()
        rankings = repo.get_previous_run_rankings(benchmark_name=benchmark)
        return rankings
    except Exception as e:
        return {}


@st.cache_data(ttl=300)
def load_new_models(since_date: Optional[datetime] = None, limit: int = 20) -> list:
    """
    Load recently discovered models.

    Args:
        since_date: Optional date filter
        limit: Maximum number of models to return

    Returns:
        list: List of NewModel records
    """
    try:
        repo = get_repository()
        new_models = repo.get_new_models(since_date=since_date, limit=limit)
        return new_models
    except Exception as e:
        return []


@st.cache_data(ttl=300)
def load_score_history(
    model_name: str,
    benchmark_name: Optional[str] = None,
    limit: int = 10,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
) -> list:
    """
    Load score history for a model.

    Args:
        model_name: Model name
        benchmark_name: Optional benchmark filter (None = all benchmarks)
        limit: Maximum number of history entries
        from_date: Optional start date filter
        to_date: Optional end date filter

    Returns:
        list: List of ScoreHistory records
    """
    try:
        repo = get_repository()
        history = repo.get_score_history(
            model_name=model_name,
            benchmark_name=benchmark_name,
            limit=limit,
            from_date=from_date,
            to_date=to_date,
        )
        return history
    except Exception as e:
        st.error(f"Error loading score history: {str(e)}")
        return []


@st.cache_data(ttl=300)
def load_ranking_history(
    model_name: str,
    benchmark_name: Optional[str] = None,
    limit: int = 10,
) -> list:
    """
    Load ranking history for a model.

    Args:
        model_name: Model name
        benchmark_name: Optional benchmark filter (None = average)
        limit: Maximum number of history entries

    Returns:
        list: List of RankingHistory records
    """
    try:
        repo = get_repository()
        history = repo.get_ranking_history(
            model_name=model_name,
            benchmark_name=benchmark_name,
            limit=limit,
        )
        return history
    except Exception as e:
        return []


@st.cache_data(ttl=300)
def compare_models_over_time(
    model_names: list[str],
    benchmark_name: str,
    limit: int = 50,
) -> dict:
    """
    Compare score evolution for multiple models.

    Args:
        model_names: List of model names to compare
        benchmark_name: Benchmark to compare on
        limit: Maximum number of data points per model

    Returns:
        dict: Mapping of model_name -> list of ScoreHistory
    """
    try:
        repo = get_repository()
        comparison = repo.compare_scores_over_time(
            model_names=model_names,
            benchmark_name=benchmark_name,
            limit=limit,
        )
        return comparison
    except Exception as e:
        st.error(f"Error comparing models: {str(e)}")
        return {}


@st.cache_data(ttl=300)
def load_model_trends(model_name: str, days: int = 30) -> Optional[dict]:
    """
    Load trend information for a model.

    Args:
        model_name: Model name
        days: Number of days to look back

    Returns:
        dict: ModelSummary data or None
    """
    try:
        repo = get_repository()
        trends = repo.get_model_trends(model_name=model_name, days=days)
        # Returns dict with key "model" -> ModelSummary
        return trends.get("model") if trends else None
    except Exception as e:
        return None


@st.cache_resource
def load_aggregator_cache() -> Optional[BenchmarkAggregator]:
    """
    Load aggregator from cache file as fallback.

    Returns:
        BenchmarkAggregator: Aggregator instance or None if cache unavailable
    """
    try:
        aggregator = BenchmarkAggregator()
        cache_path = Path("data/processed/cache.json")

        if cache_path.exists() and aggregator.load_cache():
            return aggregator
        return None
    except Exception as e:
        st.error(f"Error loading cache: {str(e)}")
        return None


def get_all_model_names() -> list[str]:
    """
    Get list of all model names.

    Returns:
        list: List of model names
    """
    try:
        rankings = load_latest_rankings()
        return [r[0] for r in rankings]
    except Exception as e:
        return []


def get_available_benchmarks() -> list[str]:
    """
    Get list of available benchmarks.

    Returns:
        list: List of benchmark names
    """
    # This could be enhanced to query from database
    # For now, return common benchmarks
    return [
        "Average Score",
        "MMLU",
        "HumanEval",
        "GSM8K",
        "HellaSwag",
        "TruthfulQA",
        "ARC",
    ]


def clear_all_caches():
    """Clear all Streamlit caches."""
    st.cache_data.clear()
    st.cache_resource.clear()
