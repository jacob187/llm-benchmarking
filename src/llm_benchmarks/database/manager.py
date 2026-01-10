"""Database manager for saving pipeline results."""

import json
from datetime import datetime
from pathlib import Path

from ..pipeline import PipelineResult
from .connection import DatabaseConnection
from .migrations import MigrationRunner


class DatabaseManager:
    """
    Manages database operations for saving pipeline results.

    Handles insertion of runs, models, benchmarks, scores, and rankings.
    """

    def __init__(self, db_path: str | Path = "data/history.db", auto_migrate: bool = True):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
            auto_migrate: Whether to automatically run migrations on initialization
        """
        self.db_path = Path(db_path)
        self.connection = DatabaseConnection(db_path)

        # Auto-migrate if needed
        if auto_migrate:
            runner = MigrationRunner(db_path)
            runner.initialize_if_needed()

    def record_run(self, result: PipelineResult) -> int:
        """
        Record a complete pipeline run with all associated data.

        Args:
            result: PipelineResult from pipeline execution

        Returns:
            run_id: ID of the created run record

        Raises:
            sqlite3.Error: If database operation fails
        """
        with self.connection as conn:
            # 1. Create run record
            run_id = self._create_run_record(conn, result)

            # 2. Save models and get their IDs
            model_ids = self._save_models(conn, result, run_id)

            # 3. Save benchmarks and get their IDs
            benchmark_ids = self._save_benchmarks(conn, result)

            # 4. Save scores
            self._save_scores(conn, result, run_id, model_ids, benchmark_ids)

            # 5. Calculate and save rankings
            self._calculate_rankings(conn, run_id, result.timestamp)

            return run_id

    def _create_run_record(self, conn: DatabaseConnection, result: PipelineResult) -> int:
        """Create run record and return its ID."""
        # Determine status
        if not result.models:
            status = "failed"
        elif result.errors:
            status = "partial"
        else:
            status = "success"

        # Convert errors list to JSON string
        errors_json = json.dumps(result.errors) if result.errors else None

        # Count sources (we don't have this info in PipelineResult, so we estimate)
        sources_scraped = len(set(source for model in result.models.values() for source in model.sources))
        sources_failed = len([e for e in result.errors if ":" in e])  # Rough estimate

        conn.execute(
            """
            INSERT INTO runs (timestamp, status, sources_scraped, sources_failed,
                            models_found, cache_path, errors)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.timestamp.isoformat(),
                status,
                sources_scraped,
                sources_failed,
                len(result.models),
                str(result.cache_path) if result.cache_path else None,
                errors_json,
            ),
        )

        return conn.lastrowid()

    def _save_models(
        self, conn: DatabaseConnection, result: PipelineResult, run_id: int
    ) -> dict[str, int]:
        """
        Save or update models and return mapping of normalized_name -> model_id.

        Handles first_seen vs last_seen logic.
        """
        model_ids = {}

        for normalized_name, model_data in result.models.items():
            # Check if model already exists
            conn.execute(
                "SELECT id, first_seen_run_id, first_seen_date FROM models WHERE normalized_name = ?",
                (normalized_name,),
            )
            existing = conn.fetchone()

            if existing:
                # Update last_seen
                model_id = existing["id"]
                conn.execute(
                    """
                    UPDATE models
                    SET last_seen_run_id = ?,
                        last_seen_date = ?,
                        display_name = ?
                    WHERE id = ?
                    """,
                    (run_id, result.timestamp.isoformat(), model_data.name, model_id),
                )
            else:
                # Insert new model
                conn.execute(
                    """
                    INSERT INTO models (normalized_name, display_name,
                                      first_seen_run_id, first_seen_date,
                                      last_seen_run_id, last_seen_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        normalized_name,
                        model_data.name,
                        run_id,
                        result.timestamp.isoformat(),
                        run_id,
                        result.timestamp.isoformat(),
                    ),
                )
                model_id = conn.lastrowid()

            model_ids[normalized_name] = model_id

        return model_ids

    def _save_benchmarks(
        self, conn: DatabaseConnection, result: PipelineResult
    ) -> dict[str, int]:
        """
        Save benchmarks and return mapping of benchmark_name -> benchmark_id.
        """
        benchmark_ids = {}
        benchmark_names = set()

        # Collect all unique benchmark names
        for model_data in result.models.values():
            benchmark_names.update(model_data.benchmarks.keys())

        # Save each benchmark
        for benchmark_name in benchmark_names:
            # Check if benchmark already exists
            conn.execute(
                "SELECT id FROM benchmarks WHERE name = ?",
                (benchmark_name,),
            )
            existing = conn.fetchone()

            if existing:
                benchmark_id = existing["id"]
            else:
                # Insert new benchmark
                conn.execute(
                    "INSERT INTO benchmarks (name, first_seen_date) VALUES (?, ?)",
                    (benchmark_name, result.timestamp.isoformat()),
                )
                benchmark_id = conn.lastrowid()

            benchmark_ids[benchmark_name] = benchmark_id

        return benchmark_ids

    def _save_scores(
        self,
        conn: DatabaseConnection,
        result: PipelineResult,
        run_id: int,
        model_ids: dict[str, int],
        benchmark_ids: dict[str, int],
    ):
        """Save all scores for this run."""
        scores_to_insert = []

        for normalized_name, model_data in result.models.items():
            model_id = model_ids[normalized_name]

            for benchmark_name, score_value in model_data.benchmarks.items():
                benchmark_id = benchmark_ids[benchmark_name]

                # Determine source (use first source if multiple, or "aggregated")
                source = list(model_data.sources)[0] if model_data.sources else "aggregated"

                scores_to_insert.append(
                    (
                        run_id,
                        model_id,
                        benchmark_id,
                        score_value,
                        source,
                        result.timestamp.isoformat(),
                        None,  # metadata (unused for now)
                    )
                )

        # Bulk insert scores
        if scores_to_insert:
            conn.executemany(
                """
                INSERT OR IGNORE INTO scores
                (run_id, model_id, benchmark_id, score, source, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                scores_to_insert,
            )

    def _calculate_rankings(
        self, conn: DatabaseConnection, run_id: int, timestamp: datetime
    ):
        """
        Calculate and save rankings for this run.

        Computes rankings per benchmark and overall average rankings.
        """
        # Get all benchmarks for this run
        conn.execute(
            """
            SELECT DISTINCT benchmark_id
            FROM scores
            WHERE run_id = ?
            """,
            (run_id,),
        )
        benchmark_ids = [row["benchmark_id"] for row in conn.fetchall()]

        rankings_to_insert = []

        # Calculate rankings for each benchmark
        for benchmark_id in benchmark_ids:
            conn.execute(
                """
                SELECT model_id, score
                FROM scores
                WHERE run_id = ? AND benchmark_id = ?
                ORDER BY score DESC
                """,
                (run_id, benchmark_id),
            )

            for rank, row in enumerate(conn.fetchall(), start=1):
                rankings_to_insert.append(
                    (
                        run_id,
                        row["model_id"],
                        benchmark_id,
                        rank,
                        row["score"],
                        timestamp.isoformat(),
                    )
                )

        # Calculate average rankings (benchmark_id = NULL)
        conn.execute(
            """
            SELECT model_id, AVG(score) as avg_score
            FROM scores
            WHERE run_id = ?
            GROUP BY model_id
            ORDER BY avg_score DESC
            """,
            (run_id,),
        )

        for rank, row in enumerate(conn.fetchall(), start=1):
            rankings_to_insert.append(
                (
                    run_id,
                    row["model_id"],
                    None,  # NULL benchmark_id means average ranking
                    rank,
                    row["avg_score"],
                    timestamp.isoformat(),
                )
            )

        # Bulk insert rankings
        if rankings_to_insert:
            conn.executemany(
                """
                INSERT OR IGNORE INTO rankings
                (run_id, model_id, benchmark_id, rank, score, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rankings_to_insert,
            )

    def update_source_stats(self, source_name: str, source_type: str, url: str, success: bool, error: str | None = None):
        """
        Update source scraping statistics.

        Args:
            source_name: Name of the source
            source_type: Type of source (leaderboard, blog, research)
            url: Source URL
            success: Whether scrape was successful
            error: Error message if failed
        """
        with self.connection as conn:
            # Check if source exists
            conn.execute("SELECT id FROM sources WHERE name = ?", (source_name,))
            existing = conn.fetchone()

            if existing:
                # Update existing source
                if success:
                    conn.execute(
                        """
                        UPDATE sources
                        SET total_scrapes = total_scrapes + 1,
                            successful_scrapes = successful_scrapes + 1,
                            last_success = ?
                        WHERE name = ?
                        """,
                        (datetime.now().isoformat(), source_name),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE sources
                        SET total_scrapes = total_scrapes + 1,
                            last_failure = ?,
                            last_error = ?
                        WHERE name = ?
                        """,
                        (datetime.now().isoformat(), error, source_name),
                    )
            else:
                # Insert new source
                conn.execute(
                    """
                    INSERT INTO sources (name, source_type, url, total_scrapes,
                                       successful_scrapes, last_success, last_failure, last_error)
                    VALUES (?, ?, ?, 1, ?, ?, ?, ?)
                    """,
                    (
                        source_name,
                        source_type,
                        url,
                        1 if success else 0,
                        datetime.now().isoformat() if success else None,
                        datetime.now().isoformat() if not success else None,
                        error if not success else None,
                    ),
                )
