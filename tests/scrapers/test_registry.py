"""Tests for scraper registry."""

import pytest
from llm_benchmarks.scrapers.registry import ScraperRegistry
from llm_benchmarks.scrapers.base import BaseScraper, SourceType


def test_registry_discovers_scrapers():
    """Test that registry auto-discovers scraper modules."""
    registry = ScraperRegistry()
    registry.reset()  # Reset for clean test
    registry.discover()

    assert registry.count() >= 2, "Should discover at least 2 scrapers"


def test_registry_lists_sources():
    """Test that registry can list all source metadata."""
    registry = ScraperRegistry()
    sources = registry.list_sources()

    assert len(sources) >= 2, "Should have at least 2 sources"
    assert all(hasattr(s, "name") for s in sources)
    assert all(hasattr(s, "url") for s in sources)
    assert all(hasattr(s, "source_type") for s in sources)


def test_registry_get_by_name():
    """Test getting a scraper by name."""
    registry = ScraperRegistry()
    scraper = registry.get("lmsys_arena")

    assert scraper is not None, "Should find LMSYS scraper"
    assert isinstance(scraper, BaseScraper)
    assert scraper.metadata.name == "LMSYS Arena"


def test_registry_filter_by_type():
    """Test filtering scrapers by source type."""
    registry = ScraperRegistry()
    leaderboard_scrapers = registry.by_type(SourceType.LEADERBOARD)
    blog_scrapers = registry.by_type(SourceType.BLOG)

    assert len(leaderboard_scrapers) >= 1, "Should have at least 1 leaderboard scraper"
    assert len(blog_scrapers) >= 1, "Should have at least 1 blog scraper"


def test_registry_all_returns_instances():
    """Test that all() returns scraper instances."""
    registry = ScraperRegistry()
    scrapers = registry.all()

    assert len(scrapers) >= 2
    assert all(isinstance(s, BaseScraper) for s in scrapers)


def test_registry_get_nonexistent_scraper():
    """Test that getting non-existent scraper returns None."""
    registry = ScraperRegistry()
    scraper = registry.get("nonexistent_scraper_xyz")

    assert scraper is None
