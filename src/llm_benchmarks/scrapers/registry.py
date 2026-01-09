"""Auto-discovery registry for scrapers."""

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Type

from .base import BaseScraper, SourceMetadata, SourceType


class ScraperRegistry:
    """Registry for auto-discovering and managing scrapers."""

    _scrapers: dict[str, Type[BaseScraper]] = {}
    _discovered: bool = False

    @classmethod
    def discover(cls) -> None:
        """
        Auto-discover all scraper modules in the scrapers package.

        Finds all .py files in the scrapers directory (except base.py and registry.py),
        imports them, and registers any BaseScraper subclasses with metadata.
        """
        if cls._discovered:
            return

        # Get the scrapers package directory
        scrapers_dir = Path(__file__).parent

        # Iterate through all .py files in the directory
        for module_info in pkgutil.iter_modules([str(scrapers_dir)]):
            module_name = module_info.name

            # Skip base and registry modules
            if module_name in ("base", "registry", "__init__"):
                continue

            # Import the module
            try:
                module = importlib.import_module(f"llm_benchmarks.scrapers.{module_name}")

                # Find all classes in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Check if it's a BaseScraper subclass (but not BaseScraper itself)
                    try:
                        if issubclass(obj, BaseScraper) and obj is not BaseScraper:
                            if hasattr(obj, "metadata"):
                                # Create a slug from the metadata name
                                slug = obj.metadata.name.lower().replace(" ", "_").replace("-", "_")
                                cls._scrapers[slug] = obj
                    except TypeError:
                        # issubclass() arg 1 must be a class
                        pass

            except Exception as e:
                print(f"Warning: Failed to import scraper module '{module_name}': {e}")

        cls._discovered = True

    @classmethod
    def all(cls) -> list[BaseScraper]:
        """
        Get all registered scrapers as instances.

        Returns:
            List of scraper instances
        """
        cls.discover()
        return [scraper_class() for scraper_class in cls._scrapers.values()]

    @classmethod
    def by_type(cls, source_type: SourceType) -> list[BaseScraper]:
        """
        Get scrapers filtered by source type.

        Args:
            source_type: Type of source to filter by

        Returns:
            List of scraper instances matching the type
        """
        cls.discover()
        return [
            scraper_class()
            for scraper_class in cls._scrapers.values()
            if scraper_class.metadata.source_type == source_type
        ]

    @classmethod
    def get(cls, name: str) -> BaseScraper | None:
        """
        Get a specific scraper by name (slug).

        Args:
            name: Scraper slug (e.g., "lmsys_arena")

        Returns:
            Scraper instance or None if not found
        """
        cls.discover()
        scraper_class = cls._scrapers.get(name)
        return scraper_class() if scraper_class else None

    @classmethod
    def list_sources(cls) -> list[SourceMetadata]:
        """
        Get metadata for all registered sources.

        Returns:
            List of SourceMetadata objects
        """
        cls.discover()
        return [scraper_class.metadata for scraper_class in cls._scrapers.values()]

    @classmethod
    def count(cls) -> int:
        """
        Get the number of registered scrapers.

        Returns:
            Number of scrapers
        """
        cls.discover()
        return len(cls._scrapers)

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (mainly for testing)."""
        cls._scrapers = {}
        cls._discovered = False
