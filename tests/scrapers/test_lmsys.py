"""Tests for LMSYS Arena scraper."""

import pytest
from llm_benchmarks.scrapers.lmsys import LMSYSArenaScraper
from llm_benchmarks.scrapers.base import SourceType, ScrapedContent


def test_lmsys_scraper_metadata():
    """Test that LMSYS scraper has correct metadata."""
    scraper = LMSYSArenaScraper()

    assert scraper.metadata.name == "LMSYS Arena"
    assert scraper.metadata.source_type == SourceType.LEADERBOARD
    assert "lmarena.ai" in scraper.metadata.url
    assert scraper.metadata.description
    assert scraper.metadata.update_frequency


def test_lmsys_scraper_instantiation():
    """Test that LMSYS scraper can be instantiated."""
    scraper = LMSYSArenaScraper()

    assert scraper is not None
    assert scraper.timeout == 30000
    assert scraper.headless is True


def test_lmsys_scraper_custom_timeout():
    """Test creating scraper with custom timeout."""
    scraper = LMSYSArenaScraper(timeout=60000, headless=False)

    assert scraper.timeout == 60000
    assert scraper.headless is False


@pytest.mark.slow
def test_lmsys_scraper_live_scrape():
    """Test actually scraping LMSYS Arena (slow, network-dependent)."""
    scraper = LMSYSArenaScraper()
    result = scraper.safe_scrape()

    # Should not error
    assert result.error is None, f"Scraping failed: {result.error}"

    # Should have entries
    assert len(result.entries) > 0, "Should find benchmark entries"

    # Check structure of first entry
    if result.entries:
        entry = result.entries[0]
        assert entry.model_name, "Entry should have model name"
        assert entry.score > 0, "Entry should have positive score"
        assert entry.benchmark_name == "LMSYS Arena ELO"
        assert "source_url" in entry.metadata


@pytest.mark.slow
def test_lmsys_scraper_results_structure():
    """Test the structure of scraping results."""
    scraper = LMSYSArenaScraper()
    result = scraper.safe_scrape()

    # Check ScrapedContent structure
    assert isinstance(result, ScrapedContent)
    assert result.source == "LMSYS Arena"
    assert result.timestamp is not None
    assert isinstance(result.entries, list)

    # LEADERBOARD type should have entries, not raw text
    assert len(result.entries) > 0
    assert result.raw_text == ""
