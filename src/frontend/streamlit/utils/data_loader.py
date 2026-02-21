"""Data loading utilities with caching for Streamlit dashboard."""

import streamlit as st
from pathlib import Path
from datetime import datetime
from typing import Optional

# Import from llm_benchmarks module
from llm_benchmarks.database.repository import HistoryRepository
from llm_benchmarks.database.models import DatabaseStats
from llm_benchmarks.data_aggregator import BenchmarkAggregator
from llm_benchmarks.pipeline import BenchmarkPipeline


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
        # Use LMSYS Arena ELO to avoid scale issues with mixed benchmarks
        rankings = load_latest_rankings(benchmark="LMSYS Arena ELO")
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


def run_scraper() -> dict:
    """
    Run the benchmark scraper pipeline.

    Executes the same scrape logic as `llm-bench scrape` — fetches fresh
    data from all registered sources (LMSYS Arena, LM Arena, blogs),
    aggregates models, saves to cache and database.

    Returns:
        dict: Results with keys 'models_count', 'errors', and 'cache_path'
    """
    pipeline = BenchmarkPipeline()
    result = pipeline.run(
        skip_scrape=False,
        analyses=[],
        save_cache=True,
    )
    return {
        "models_count": len(result.models),
        "errors": result.errors,
        "cache_path": str(result.cache_path) if result.cache_path else None,
    }


def clear_all_caches():
    """Clear all Streamlit caches."""
    st.cache_data.clear()
    st.cache_resource.clear()


# Model type constants
MODEL_TYPE_TEXT = "Text"
MODEL_TYPE_IMAGE = "Image"
MODEL_TYPE_VIDEO = "Video"
MODEL_TYPE_AUDIO = "Audio"
MODEL_TYPE_SEARCH = "Search/Grounding"
MODEL_TYPE_ALL = "All Types"


def get_model_type(model_name: str) -> str:
    """
    Classify a model by its type based on name patterns.

    Args:
        model_name: Model name to classify

    Returns:
        str: Model type (Text, Image, Video, Audio, Search/Grounding)
    """
    name_lower = model_name.lower()

    # Audio models (check first as some video models have audio variants)
    if "audio" in name_lower or "voice" in name_lower:
        return MODEL_TYPE_AUDIO

    # Image models
    if "image" in name_lower or "seedream" in name_lower:
        return MODEL_TYPE_IMAGE

    # Video models
    video_keywords = [
        "video", "i2v", "t2v", "sora", "veo", "wan",
        "kling", "seedance", "reve", "pika", "runway"
    ]
    if any(kw in name_lower for kw in video_keywords):
        return MODEL_TYPE_VIDEO

    # Search/Grounding models
    if "search" in name_lower or "grounding" in name_lower:
        return MODEL_TYPE_SEARCH

    # Default to text
    return MODEL_TYPE_TEXT


def get_model_types() -> list[str]:
    """
    Get list of available model types for filtering.

    Returns:
        list: List of model type strings
    """
    return [
        MODEL_TYPE_ALL,
        MODEL_TYPE_TEXT,
        MODEL_TYPE_IMAGE,
        MODEL_TYPE_VIDEO,
        MODEL_TYPE_AUDIO,
        MODEL_TYPE_SEARCH,
    ]


def filter_models_by_type(
    model_names: list[str],
    model_type: str,
) -> list[str]:
    """
    Filter a list of model names by type.

    Args:
        model_names: List of model names to filter
        model_type: Type to filter by (or MODEL_TYPE_ALL for no filter)

    Returns:
        list: Filtered list of model names
    """
    if model_type == MODEL_TYPE_ALL:
        return model_names

    return [
        name for name in model_names
        if get_model_type(name) == model_type
    ]


def get_model_type_counts(model_names: list[str]) -> dict[str, int]:
    """
    Count models by type.

    Args:
        model_names: List of model names

    Returns:
        dict: Mapping of type -> count
    """
    counts = {
        MODEL_TYPE_TEXT: 0,
        MODEL_TYPE_IMAGE: 0,
        MODEL_TYPE_VIDEO: 0,
        MODEL_TYPE_AUDIO: 0,
        MODEL_TYPE_SEARCH: 0,
    }

    for name in model_names:
        model_type = get_model_type(name)
        if model_type in counts:
            counts[model_type] += 1

    return counts


# Vendor constants
VENDOR_ALL = "All Vendors"
VENDOR_MAIN = "Main Vendors"
VENDOR_OTHER = "Other"

# Main vendor patterns (case-insensitive matching)
MAIN_VENDORS = {
    "Anthropic": ["claude"],
    "OpenAI": ["gpt", "o1", "o3", "chatgpt"],
    "Google": ["gemini", "bard", "palm"],
    "Meta": ["llama", "meta"],
    "xAI": ["grok"],
    "Mistral": ["mistral", "mixtral"],
    "Cohere": ["command", "cohere"],
    "DeepSeek": ["deepseek"],
    "MoonshotAI": ["moonshot", "kimi"],
}


def get_model_vendor(model_name: str) -> str:
    """
    Identify the vendor of a model based on name patterns.

    Args:
        model_name: Model name to classify

    Returns:
        str: Vendor name or "Other"
    """
    name_lower = model_name.lower()

    for vendor, patterns in MAIN_VENDORS.items():
        if any(pattern in name_lower for pattern in patterns):
            return vendor

    return VENDOR_OTHER


def is_main_vendor(model_name: str) -> bool:
    """
    Check if a model is from a main vendor.

    Args:
        model_name: Model name to check

    Returns:
        bool: True if from a main vendor
    """
    return get_model_vendor(model_name) != VENDOR_OTHER


def get_vendor_options() -> list[str]:
    """
    Get list of vendor filter options.

    Returns:
        list: List of vendor filter strings
    """
    return [VENDOR_ALL, VENDOR_MAIN, VENDOR_OTHER]


def filter_models_by_vendor(
    model_names: list[str],
    vendor_filter: str,
) -> list[str]:
    """
    Filter models by vendor.

    Args:
        model_names: List of model names
        vendor_filter: Vendor filter option

    Returns:
        list: Filtered model names
    """
    if vendor_filter == VENDOR_ALL:
        return model_names
    elif vendor_filter == VENDOR_MAIN:
        return [name for name in model_names if is_main_vendor(name)]
    elif vendor_filter == VENDOR_OTHER:
        return [name for name in model_names if not is_main_vendor(name)]
    else:
        return model_names


def get_vendor_counts(model_names: list[str]) -> dict[str, int]:
    """
    Count models by vendor category.

    Args:
        model_names: List of model names

    Returns:
        dict: Mapping of vendor filter -> count
    """
    main_count = sum(1 for name in model_names if is_main_vendor(name))
    return {
        VENDOR_MAIN: main_count,
        VENDOR_OTHER: len(model_names) - main_count,
    }
