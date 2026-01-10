-- Initial database schema for LLM benchmark tracking
-- Version: 1
-- Description: Creates core tables for runs, models, benchmarks, scores, rankings, and sources

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

-- Pipeline runs
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

CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(timestamp);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);

-- Models with first/last seen tracking
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

CREATE INDEX IF NOT EXISTS idx_models_normalized_name ON models(normalized_name);
CREATE INDEX IF NOT EXISTS idx_models_first_seen ON models(first_seen_date);

-- Benchmark catalog
CREATE TABLE IF NOT EXISTS benchmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    first_seen_date DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_benchmarks_name ON benchmarks(name);

-- Historical scores (time-series data)
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

CREATE INDEX IF NOT EXISTS idx_scores_model_benchmark ON scores(model_id, benchmark_id);
CREATE INDEX IF NOT EXISTS idx_scores_timestamp ON scores(timestamp);
CREATE INDEX IF NOT EXISTS idx_scores_run_id ON scores(run_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_scores_unique ON scores(run_id, model_id, benchmark_id);

-- Pre-calculated rankings
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

CREATE INDEX IF NOT EXISTS idx_rankings_run_benchmark ON rankings(run_id, benchmark_id);
CREATE INDEX IF NOT EXISTS idx_rankings_model ON rankings(model_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_rankings_unique ON rankings(run_id, model_id, benchmark_id);

-- Source reliability tracking
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

CREATE INDEX IF NOT EXISTS idx_sources_name ON sources(name);

-- Record schema version
INSERT OR IGNORE INTO schema_version (version, description)
VALUES (1, 'Initial schema with core tables for historical benchmark tracking');
