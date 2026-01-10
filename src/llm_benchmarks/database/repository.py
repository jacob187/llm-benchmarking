"""Repository for historical data queries."""

from datetime import datetime, timedelta
from pathlib import Path

from .connection import DatabaseConnection
from .models import (
    DatabaseStats,
    ModelSummary,
    NewModel,
    RankingHistory,
    ScoreHistory,
)


class HistoryRepository:
    """
    Repository for querying historical benchmark data.

    Provides high-level query methods for CLI commands and analysis.
    """

    def __init__(self, db_path: str | Path = "data/history.db"):
        """
        Initialize repository.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.connection = DatabaseConnection(db_path)

        # Ensure database is initialized
        from .migrations import MigrationRunner

        runner = MigrationRunner(db_path)
        runner.initialize_if_needed()

    def get_score_history(
        self,
        model_name: str,
        benchmark_name: str | None = None,
        limit: int = 10,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> list[ScoreHistory]:
        """
        Get score history for a model.

        Args:
            model_name: Model name (display name or normalized)
            benchmark_name: Specific benchmark (None = all benchmarks)
            limit: Maximum number of results
            from_date: Start date filter
            to_date: End date filter

        Returns:
            List of ScoreHistory records ordered by date descending
        """
        with self.connection as conn:
            # Build query
            query = """
                SELECT
                    s.timestamp as date,
                    s.score,
                    s.source,
                    b.name as benchmark_name
                FROM scores s
                JOIN models m ON s.model_id = m.id
                JOIN benchmarks b ON s.benchmark_id = b.id
                WHERE (m.display_name = ? OR m.normalized_name = ?)
            """
            params = [model_name, model_name.lower().replace(" ", "_")]

            if benchmark_name:
                query += " AND b.name = ?"
                params.append(benchmark_name)

            if from_date:
                query += " AND s.timestamp >= ?"
                params.append(from_date.isoformat())

            if to_date:
                query += " AND s.timestamp <= ?"
                params.append(to_date.isoformat())

            query += " ORDER BY s.timestamp DESC LIMIT ?"
            params.append(limit)

            conn.execute(query, tuple(params))
            rows = conn.fetchall()

            # Calculate changes
            results = []
            prev_score = None

            for i, row in enumerate(reversed(rows)):  # Process oldest to newest for change calc
                change = None
                if prev_score is not None:
                    change = row["score"] - prev_score

                results.append(
                    ScoreHistory(
                        date=datetime.fromisoformat(row["date"]),
                        score=row["score"],
                        source=row["source"],
                        change=change,
                    )
                )
                prev_score = row["score"]

            return list(reversed(results))  # Return newest first

    def get_ranking_history(
        self,
        model_name: str,
        benchmark_name: str | None = None,
        limit: int = 10,
    ) -> list[RankingHistory]:
        """
        Get ranking history for a model.

        Args:
            model_name: Model name (display name or normalized)
            benchmark_name: Specific benchmark (None = average ranking)
            limit: Maximum number of results

        Returns:
            List of RankingHistory records ordered by date descending
        """
        with self.connection as conn:
            # Build query
            if benchmark_name:
                query = """
                    SELECT
                        r.timestamp as date,
                        r.rank,
                        r.score
                    FROM rankings r
                    JOIN models m ON r.model_id = m.id
                    JOIN benchmarks b ON r.benchmark_id = b.id
                    WHERE (m.display_name = ? OR m.normalized_name = ?)
                      AND b.name = ?
                    ORDER BY r.timestamp DESC
                    LIMIT ?
                """
                params = (model_name, model_name.lower().replace(" ", "_"), benchmark_name, limit)
            else:
                # Average rankings (benchmark_id IS NULL)
                query = """
                    SELECT
                        r.timestamp as date,
                        r.rank,
                        r.score
                    FROM rankings r
                    JOIN models m ON r.model_id = m.id
                    WHERE (m.display_name = ? OR m.normalized_name = ?)
                      AND r.benchmark_id IS NULL
                    ORDER BY r.timestamp DESC
                    LIMIT ?
                """
                params = (model_name, model_name.lower().replace(" ", "_"), limit)

            conn.execute(query, params)
            rows = conn.fetchall()

            # Calculate rank changes (positive = improved, negative = declined)
            results = []
            prev_rank = None

            for row in reversed(rows):  # Process oldest to newest for change calc
                change = None
                if prev_rank is not None:
                    change = prev_rank - row["rank"]  # Lower rank number = better

                results.append(
                    RankingHistory(
                        date=datetime.fromisoformat(row["date"]),
                        rank=row["rank"],
                        score=row["score"],
                        change=change,
                    )
                )
                prev_rank = row["rank"]

            return list(reversed(results))  # Return newest first

    def get_new_models(
        self, since_date: datetime | None = None, limit: int = 20
    ) -> list[NewModel]:
        """
        Get recently discovered models.

        Args:
            since_date: Only models first seen after this date
            limit: Maximum number of results

        Returns:
            List of NewModel records ordered by first_seen descending
        """
        with self.connection as conn:
            query = """
                SELECT
                    m.display_name as model_name,
                    m.normalized_name,
                    m.first_seen_date as first_seen,
                    m.first_seen_run_id as run_id,
                    s.score as initial_score,
                    b.name as benchmark_name
                FROM models m
                LEFT JOIN scores s ON m.first_seen_run_id = s.run_id AND m.id = s.model_id
                LEFT JOIN benchmarks b ON s.benchmark_id = b.id
                WHERE 1=1
            """
            params = []

            if since_date:
                query += " AND m.first_seen_date >= ?"
                params.append(since_date.isoformat())

            query += " ORDER BY m.first_seen_date DESC LIMIT ?"
            params.append(limit)

            conn.execute(query, tuple(params))
            rows = conn.fetchall()

            return [
                NewModel(
                    model_name=row["model_name"],
                    normalized_name=row["normalized_name"],
                    first_seen=datetime.fromisoformat(row["first_seen"]),
                    initial_score=row["initial_score"],
                    benchmark_name=row["benchmark_name"],
                    run_id=row["run_id"],
                )
                for row in rows
            ]

    def compare_scores_over_time(
        self,
        model_names: list[str],
        benchmark_name: str,
        limit: int = 10,
    ) -> dict[str, list[ScoreHistory]]:
        """
        Compare score evolution for multiple models on the same benchmark.

        Args:
            model_names: List of model names to compare
            benchmark_name: Benchmark to compare on
            limit: Maximum number of results per model

        Returns:
            Dict mapping model_name -> list of ScoreHistory
        """
        results = {}

        for model_name in model_names:
            history = self.get_score_history(
                model_name=model_name,
                benchmark_name=benchmark_name,
                limit=limit,
            )
            results[model_name] = history

        return results

    def get_model_trends(
        self, model_name: str, days: int = 30
    ) -> dict[str, ModelSummary]:
        """
        Get trend information for a model over the last N days.

        Args:
            model_name: Model name
            days: Number of days to analyze

        Returns:
            ModelSummary with trend indicators
        """
        since_date = datetime.now() - timedelta(days=days)

        with self.connection as conn:
            # Get model info and recent rankings
            query = """
                SELECT
                    m.display_name,
                    m.normalized_name,
                    m.first_seen_date,
                    m.last_seen_date,
                    COUNT(DISTINCT b.id) as total_benchmarks,
                    AVG(s.score) as average_score,
                    (SELECT rank FROM rankings r2
                     WHERE r2.model_id = m.id AND r2.benchmark_id IS NULL
                     ORDER BY r2.timestamp DESC LIMIT 1) as latest_rank
                FROM models m
                LEFT JOIN scores s ON m.id = s.model_id AND s.timestamp >= ?
                LEFT JOIN benchmarks b ON s.benchmark_id = b.id
                WHERE (m.display_name = ? OR m.normalized_name = ?)
                GROUP BY m.id
            """

            conn.execute(
                query,
                (since_date.isoformat(), model_name, model_name.lower().replace(" ", "_")),
            )
            row = conn.fetchone()

            if not row:
                return {}

            # Determine trend from recent rankings
            trend = self._calculate_trend(conn, row["normalized_name"], days)

            return {
                "model": ModelSummary(
                    model_name=row["display_name"],
                    normalized_name=row["normalized_name"],
                    first_seen=datetime.fromisoformat(row["first_seen_date"]),
                    last_seen=datetime.fromisoformat(row["last_seen_date"]),
                    total_benchmarks=row["total_benchmarks"],
                    average_score=row["average_score"],
                    latest_rank=row["latest_rank"],
                    rank_trend=trend,
                )
            }

    def _calculate_trend(
        self, conn: DatabaseConnection, normalized_name: str, days: int
    ) -> str:
        """Calculate rank trend: 'up', 'down', or 'stable'."""
        since_date = datetime.now() - timedelta(days=days)

        # Get recent average rankings
        conn.execute(
            """
            SELECT rank, timestamp
            FROM rankings r
            JOIN models m ON r.model_id = m.id
            WHERE m.normalized_name = ?
              AND r.benchmark_id IS NULL
              AND r.timestamp >= ?
            ORDER BY r.timestamp ASC
            """,
            (normalized_name, since_date.isoformat()),
        )

        ranks = [row["rank"] for row in conn.fetchall()]

        if len(ranks) < 2:
            return "stable"

        # Compare first and last ranks
        first_rank = ranks[0]
        last_rank = ranks[-1]

        if last_rank < first_rank:  # Lower rank number = better
            return "up"
        elif last_rank > first_rank:
            return "down"
        else:
            return "stable"

    def get_top_models_by_date(
        self, date: datetime, benchmark_name: str | None = None, limit: int = 10
    ) -> list[tuple[str, float, int]]:
        """
        Get top models at a specific date.

        Args:
            date: Date to query
            benchmark_name: Specific benchmark (None = average)
            limit: Number of models to return

        Returns:
            List of (model_name, score, rank) tuples
        """
        with self.connection as conn:
            if benchmark_name:
                query = """
                    SELECT
                        m.display_name,
                        r.score,
                        r.rank
                    FROM rankings r
                    JOIN models m ON r.model_id = m.id
                    JOIN benchmarks b ON r.benchmark_id = b.id
                    WHERE b.name = ?
                      AND DATE(r.timestamp) = DATE(?)
                    ORDER BY r.rank ASC
                    LIMIT ?
                """
                params = (benchmark_name, date.isoformat(), limit)
            else:
                query = """
                    SELECT
                        m.display_name,
                        r.score,
                        r.rank
                    FROM rankings r
                    JOIN models m ON r.model_id = m.id
                    WHERE r.benchmark_id IS NULL
                      AND DATE(r.timestamp) = DATE(?)
                    ORDER BY r.rank ASC
                    LIMIT ?
                """
                params = (date.isoformat(), limit)

            conn.execute(query, params)
            return [(row["display_name"], row["score"], row["rank"]) for row in conn.fetchall()]

    def get_database_stats(self) -> DatabaseStats:
        """
        Get overall database statistics.

        Returns:
            DatabaseStats with summary information
        """
        with self.connection as conn:
            # Total runs and success rate
            conn.execute(
                """
                SELECT
                    COUNT(*) as total_runs,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_runs,
                    MIN(timestamp) as first_run,
                    MAX(timestamp) as last_run
                FROM runs
                """
            )
            runs_row = conn.fetchone()

            # Total models, benchmarks, scores
            conn.execute("SELECT COUNT(*) as count FROM models")
            total_models = conn.fetchone()["count"]

            conn.execute("SELECT COUNT(*) as count FROM benchmarks")
            total_benchmarks = conn.fetchone()["count"]

            conn.execute("SELECT COUNT(*) as count FROM scores")
            total_scores = conn.fetchone()["count"]

            # Calculate date range
            first_run = (
                datetime.fromisoformat(runs_row["first_run"]) if runs_row["first_run"] else None
            )
            last_run = (
                datetime.fromisoformat(runs_row["last_run"]) if runs_row["last_run"] else None
            )
            date_range_days = None

            if first_run and last_run:
                date_range_days = (last_run - first_run).days

            return DatabaseStats(
                total_runs=runs_row["total_runs"],
                successful_runs=runs_row["successful_runs"] or 0,
                total_models=total_models,
                total_benchmarks=total_benchmarks,
                total_scores=total_scores,
                first_run_date=first_run,
                last_run_date=last_run,
                date_range_days=date_range_days,
            )

    def get_latest_run_rankings(
        self, benchmark_name: str | None = None
    ) -> list[tuple[str, float, int]]:
        """
        Get rankings from the most recent run.

        Args:
            benchmark_name: Specific benchmark (None = average)

        Returns:
            List of (model_name, score, rank) tuples
        """
        with self.connection as conn:
            # Get latest run ID
            conn.execute("SELECT id FROM runs ORDER BY timestamp DESC LIMIT 1")
            run_row = conn.fetchone()

            if not run_row:
                return []

            run_id = run_row["id"]

            # Get rankings for that run
            if benchmark_name:
                query = """
                    SELECT
                        m.display_name,
                        r.score,
                        r.rank
                    FROM rankings r
                    JOIN models m ON r.model_id = m.id
                    JOIN benchmarks b ON r.benchmark_id = b.id
                    WHERE r.run_id = ? AND b.name = ?
                    ORDER BY r.rank ASC
                """
                params = (run_id, benchmark_name)
            else:
                query = """
                    SELECT
                        m.display_name,
                        r.score,
                        r.rank
                    FROM rankings r
                    JOIN models m ON r.model_id = m.id
                    WHERE r.run_id = ? AND r.benchmark_id IS NULL
                    ORDER BY r.rank ASC
                """
                params = (run_id,)

            conn.execute(query, params)
            return [(row["display_name"], row["score"], row["rank"]) for row in conn.fetchall()]

    def get_previous_run_rankings(
        self, benchmark_name: str | None = None
    ) -> dict[str, int]:
        """
        Get rankings from the second-most recent run.

        Args:
            benchmark_name: Specific benchmark (None = average)

        Returns:
            Dict mapping model_name -> rank
        """
        with self.connection as conn:
            # Get second latest run ID
            conn.execute("SELECT id FROM runs ORDER BY timestamp DESC LIMIT 2")
            runs = conn.fetchall()

            if len(runs) < 2:
                return {}

            run_id = runs[1]["id"]

            # Get rankings for that run
            if benchmark_name:
                query = """
                    SELECT
                        m.display_name,
                        r.rank
                    FROM rankings r
                    JOIN models m ON r.model_id = m.id
                    JOIN benchmarks b ON r.benchmark_id = b.id
                    WHERE r.run_id = ? AND b.name = ?
                """
                params = (run_id, benchmark_name)
            else:
                query = """
                    SELECT
                        m.display_name,
                        r.rank
                    FROM rankings r
                    JOIN models m ON r.model_id = m.id
                    WHERE r.run_id = ? AND r.benchmark_id IS NULL
                """
                params = (run_id,)

            conn.execute(query, params)
            return {row["display_name"]: row["rank"] for row in conn.fetchall()}
