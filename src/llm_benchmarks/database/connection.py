"""Database connection management for SQLite."""

import sqlite3
from pathlib import Path
from typing import Any


class DatabaseConnection:
    """
    Manages SQLite database connection lifecycle.

    Provides context manager for transactions and proper connection handling.
    Uses WAL mode for better concurrency support.
    """

    def __init__(self, db_path: str | Path = "data/history.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.connection: sqlite3.Connection | None = None
        self.cursor: sqlite3.Cursor | None = None

    def connect(self) -> sqlite3.Connection:
        """
        Establish database connection.

        Returns:
            SQLite connection object

        Raises:
            sqlite3.Error: If connection fails
        """
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect with row factory for dict-like access
        self.connection = sqlite3.connect(
            self.db_path,
            check_same_thread=False,  # Allow multi-threaded access
            timeout=10.0,  # Wait up to 10 seconds for locks
        )

        # Enable foreign keys
        self.connection.execute("PRAGMA foreign_keys = ON")

        # Use WAL mode for better concurrency
        self.connection.execute("PRAGMA journal_mode = WAL")

        # Return rows as dict-like objects
        self.connection.row_factory = sqlite3.Row

        self.cursor = self.connection.cursor()

        return self.connection

    def close(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
            self.cursor = None

        if self.connection:
            self.connection.close()
            self.connection = None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit with transaction handling.

        Commits if no exception, rollback otherwise.
        """
        if self.connection:
            if exc_type is None:
                self.connection.commit()
            else:
                self.connection.rollback()

        self.close()

    def execute(self, query: str, params: tuple | dict | None = None) -> sqlite3.Cursor:
        """
        Execute a single query.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Cursor with query results

        Raises:
            RuntimeError: If not connected
            sqlite3.Error: If query fails
        """
        if not self.connection or not self.cursor:
            raise RuntimeError("Not connected to database. Use context manager or call connect().")

        if params:
            return self.cursor.execute(query, params)
        return self.cursor.execute(query)

    def executemany(
        self, query: str, params_list: list[tuple] | list[dict]
    ) -> sqlite3.Cursor:
        """
        Execute a query with multiple parameter sets (bulk insert/update).

        Args:
            query: SQL query string
            params_list: List of parameter tuples/dicts

        Returns:
            Cursor with query results

        Raises:
            RuntimeError: If not connected
            sqlite3.Error: If query fails
        """
        if not self.connection or not self.cursor:
            raise RuntimeError("Not connected to database. Use context manager or call connect().")

        return self.cursor.executemany(query, params_list)

    def fetchone(self) -> dict[str, Any] | None:
        """
        Fetch one row from last query.

        Returns:
            Row as dict, or None if no results
        """
        if not self.cursor:
            return None

        row = self.cursor.fetchone()
        return dict(row) if row else None

    def fetchall(self) -> list[dict[str, Any]]:
        """
        Fetch all rows from last query.

        Returns:
            List of rows as dicts
        """
        if not self.cursor:
            return []

        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]

    def lastrowid(self) -> int:
        """
        Get the last inserted row ID.

        Returns:
            Last row ID

        Raises:
            RuntimeError: If not connected
        """
        if not self.cursor:
            raise RuntimeError("No cursor available")

        return self.cursor.lastrowid

    def commit(self):
        """Commit current transaction."""
        if self.connection:
            self.connection.commit()

    def rollback(self):
        """Rollback current transaction."""
        if self.connection:
            self.connection.rollback()

    def get_schema_version(self) -> int:
        """
        Get current schema version from database.

        Returns:
            Schema version number, or 0 if not initialized
        """
        try:
            with self:
                self.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
                )
                if not self.fetchone():
                    return 0

                self.execute("SELECT MAX(version) as version FROM schema_version")
                result = self.fetchone()
                return result["version"] if result and result["version"] else 0

        except sqlite3.Error:
            return 0

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_name: Name of the table

        Returns:
            True if table exists, False otherwise
        """
        try:
            with self:
                self.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,),
                )
                return self.fetchone() is not None
        except sqlite3.Error:
            return False
