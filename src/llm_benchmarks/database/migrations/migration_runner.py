"""Database migration runner."""

from pathlib import Path

from ..connection import DatabaseConnection
from ..schema import CURRENT_VERSION


class MigrationRunner:
    """Handles database schema migrations."""

    def __init__(self, db_path: str | Path = "data/history.db"):
        """
        Initialize migration runner.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.migrations_dir = Path(__file__).parent

    def get_current_version(self) -> int:
        """
        Get current schema version from database.

        Returns:
            Schema version number, or 0 if database not initialized
        """
        conn = DatabaseConnection(self.db_path)
        return conn.get_schema_version()

    def needs_migration(self) -> bool:
        """
        Check if database needs migration.

        Returns:
            True if migration needed, False otherwise
        """
        current = self.get_current_version()
        return current < CURRENT_VERSION

    def get_pending_migrations(self) -> list[tuple[int, Path]]:
        """
        Get list of pending migration files.

        Returns:
            List of (version, file_path) tuples for migrations to apply
        """
        current = self.get_current_version()
        pending = []

        # Find all migration files
        for migration_file in sorted(self.migrations_dir.glob("v*.sql")):
            # Extract version number from filename (e.g., v001_initial.sql -> 1)
            version_str = migration_file.stem.split("_")[0][1:]  # Remove 'v' prefix
            try:
                version = int(version_str)
                if version > current and version <= CURRENT_VERSION:
                    pending.append((version, migration_file))
            except ValueError:
                # Skip files that don't match the naming pattern
                continue

        return sorted(pending)

    def run_migrations(self, target_version: int | None = None) -> list[int]:
        """
        Run all pending migrations up to target version.

        Args:
            target_version: Target schema version (None = latest)

        Returns:
            List of applied migration versions

        Raises:
            sqlite3.Error: If migration fails
        """
        if target_version is None:
            target_version = CURRENT_VERSION

        pending = self.get_pending_migrations()
        applied = []

        for version, migration_file in pending:
            if version > target_version:
                break

            print(f"Applying migration v{version:03d}: {migration_file.name}...")

            # Read migration SQL
            sql = migration_file.read_text()

            # Apply migration
            with DatabaseConnection(self.db_path) as conn:
                # Execute all statements in the migration file
                conn.connection.executescript(sql)

            applied.append(version)
            print(f"✓ Applied migration v{version:03d}")

        if not applied:
            print("✓ Database schema is up to date")

        return applied

    def create_database(self):
        """
        Create new database with initial schema.

        This is a convenience method that runs the v001_initial.sql migration.
        """
        print("Creating new database...")

        # Ensure database file doesn't exist or is empty
        current_version = self.get_current_version()

        if current_version > 0:
            print(f"✓ Database already exists (version {current_version})")
            return

        # Run migrations to create schema
        applied = self.run_migrations()

        if applied:
            print(f"✓ Database created successfully (version {max(applied)})")
        else:
            print("✗ Failed to create database")

    def initialize_if_needed(self):
        """
        Initialize database if it doesn't exist or needs migration.

        This is the main entry point for auto-initialization.
        """
        if not self.db_path.exists():
            print(f"Database not found at {self.db_path}, creating...")
            self.create_database()
        elif self.needs_migration():
            print("Database needs migration...")
            self.run_migrations()
