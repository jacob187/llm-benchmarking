"""Database schema definitions for historical benchmark tracking."""

CURRENT_VERSION = 1

# Schema version table
SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);
"""

# Runs table - Track each pipeline execution
RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('success', 'partial', 'failed', 'imported')),
    sources_scraped INTEGER DEFAULT 0,
    sources_failed INTEGER DEFAULT 0,
    models_found INTEGER DEFAULT 0,
    cache_path TEXT,
    errors TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

RUNS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(timestamp);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
"""

# Models table - Unique models with first/last seen tracking
MODELS_TABLE = """
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    normalized_name TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    first_seen_run_id INTEGER NOT NULL,
    first_seen_date DATETIME NOT NULL,
    last_seen_run_id INTEGER NOT NULL,
    last_seen_date DATETIME NOT NULL,
    FOREIGN KEY (first_seen_run_id) REFERENCES runs(id),
    FOREIGN KEY (last_seen_run_id) REFERENCES runs(id)
);
"""

MODELS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_models_normalized_name ON models(normalized_name);
CREATE INDEX IF NOT EXISTS idx_models_first_seen ON models(first_seen_date);
"""

# Benchmarks table - Catalog of unique benchmarks
BENCHMARKS_TABLE = """
CREATE TABLE IF NOT EXISTS benchmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    first_seen_date DATETIME NOT NULL
);
"""

BENCHMARKS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_benchmarks_name ON benchmarks(name);
"""

# Scores table - Historical score records (time-series data)
SCORES_TABLE = """
CREATE TABLE IF NOT EXISTS scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    model_id INTEGER NOT NULL,
    benchmark_id INTEGER NOT NULL,
    score REAL NOT NULL,
    source TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    metadata TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(id),
    FOREIGN KEY (model_id) REFERENCES models(id),
    FOREIGN KEY (benchmark_id) REFERENCES benchmarks(id)
);
"""

SCORES_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_scores_model_benchmark ON scores(model_id, benchmark_id);
CREATE INDEX IF NOT EXISTS idx_scores_timestamp ON scores(timestamp);
CREATE INDEX IF NOT EXISTS idx_scores_run_id ON scores(run_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_scores_unique ON scores(run_id, model_id, benchmark_id);
"""

# Rankings table - Pre-calculated rankings per run
RANKINGS_TABLE = """
CREATE TABLE IF NOT EXISTS rankings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    model_id INTEGER NOT NULL,
    benchmark_id INTEGER,
    rank INTEGER NOT NULL,
    score REAL NOT NULL,
    timestamp DATETIME NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id),
    FOREIGN KEY (model_id) REFERENCES models(id),
    FOREIGN KEY (benchmark_id) REFERENCES benchmarks(id)
);
"""

RANKINGS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_rankings_run_benchmark ON rankings(run_id, benchmark_id);
CREATE INDEX IF NOT EXISTS idx_rankings_model ON rankings(model_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_rankings_unique ON rankings(run_id, model_id, benchmark_id);
"""

# Sources table - Track scraping source reliability
SOURCES_TABLE = """
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    source_type TEXT NOT NULL,
    url TEXT NOT NULL,
    total_scrapes INTEGER DEFAULT 0,
    successful_scrapes INTEGER DEFAULT 0,
    last_success DATETIME,
    last_failure DATETIME,
    last_error TEXT
);
"""

SOURCES_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_sources_name ON sources(name);
"""

# All tables in order
ALL_TABLES = [
    SCHEMA_VERSION_TABLE,
    RUNS_TABLE,
    MODELS_TABLE,
    BENCHMARKS_TABLE,
    SCORES_TABLE,
    RANKINGS_TABLE,
    SOURCES_TABLE,
]

# All indexes
ALL_INDEXES = [
    RUNS_INDEXES,
    MODELS_INDEXES,
    BENCHMARKS_INDEXES,
    SCORES_INDEXES,
    RANKINGS_INDEXES,
    SOURCES_INDEXES,
]


def get_initial_schema_sql() -> str:
    """
    Get the complete initial schema SQL.

    Returns:
        SQL string with all CREATE TABLE and CREATE INDEX statements
    """
    sql_parts = []

    # Add tables
    for table_sql in ALL_TABLES:
        sql_parts.append(table_sql.strip())

    # Add indexes
    for index_sql in ALL_INDEXES:
        sql_parts.append(index_sql.strip())

    return "\n\n".join(sql_parts)
