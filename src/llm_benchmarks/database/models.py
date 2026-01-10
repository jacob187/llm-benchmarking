"""Pydantic models for database entities."""

from datetime import datetime

from pydantic import BaseModel, Field


class RunRecord(BaseModel):
    """Database record for a pipeline run."""

    id: int | None = None
    timestamp: datetime
    status: str = Field(..., pattern="^(success|partial|failed|imported)$")
    sources_scraped: int = 0
    sources_failed: int = 0
    models_found: int = 0
    cache_path: str | None = None
    errors: str | None = None  # JSON string
    created_at: datetime = Field(default_factory=datetime.now)


class ModelRecord(BaseModel):
    """Database record for a model."""

    id: int | None = None
    normalized_name: str
    display_name: str
    first_seen_run_id: int
    first_seen_date: datetime
    last_seen_run_id: int
    last_seen_date: datetime


class BenchmarkRecord(BaseModel):
    """Database record for a benchmark."""

    id: int | None = None
    name: str
    first_seen_date: datetime


class ScoreRecord(BaseModel):
    """Database record for a score."""

    id: int | None = None
    run_id: int
    model_id: int
    benchmark_id: int
    score: float
    source: str
    timestamp: datetime
    metadata: str | None = None  # JSON string


class RankingRecord(BaseModel):
    """Database record for a ranking."""

    id: int | None = None
    run_id: int
    model_id: int
    benchmark_id: int | None = None  # NULL means average ranking
    rank: int
    score: float
    timestamp: datetime


class SourceRecord(BaseModel):
    """Database record for a source."""

    id: int | None = None
    name: str
    source_type: str
    url: str
    total_scrapes: int = 0
    successful_scrapes: int = 0
    last_success: datetime | None = None
    last_failure: datetime | None = None
    last_error: str | None = None


# Query result models (for historical queries)


class ScoreHistory(BaseModel):
    """Historical score data for a model/benchmark combination."""

    date: datetime
    score: float
    source: str
    change: float | None = None  # Change from previous run


class RankingHistory(BaseModel):
    """Historical ranking data for a model/benchmark combination."""

    date: datetime
    rank: int
    score: float
    change: int | None = None  # Rank change from previous run (positive = improved)


class ModelSummary(BaseModel):
    """Summary statistics for a model."""

    model_name: str
    normalized_name: str
    first_seen: datetime
    last_seen: datetime
    total_benchmarks: int
    average_score: float | None = None
    latest_rank: int | None = None
    rank_trend: str | None = None  # "up", "down", "stable"


class NewModel(BaseModel):
    """Information about a newly discovered model."""

    model_name: str
    normalized_name: str
    first_seen: datetime
    initial_score: float | None = None
    benchmark_name: str | None = None
    run_id: int


class DatabaseStats(BaseModel):
    """Overall database statistics."""

    total_runs: int
    successful_runs: int
    total_models: int
    total_benchmarks: int
    total_scores: int
    first_run_date: datetime | None = None
    last_run_date: datetime | None = None
    date_range_days: int | None = None
