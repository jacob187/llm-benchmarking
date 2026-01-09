"""Base classes for web scraping with headless browser support."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from bs4 import BeautifulSoup
from playwright.sync_api import Browser, Page, sync_playwright


class SourceType(Enum):
    """Types of benchmark sources."""
    LEADERBOARD = "leaderboard"  # Structured data with rankings
    BLOG = "blog"                # Text content for LLM analysis
    RESEARCH = "research"        # Academic papers and research


@dataclass
class SourceMetadata:
    """Metadata about a benchmark source."""
    name: str
    url: str
    source_type: SourceType
    description: str
    update_frequency: str  # e.g., "daily", "weekly", "on-release"


@dataclass
class BenchmarkEntry:
    """Single benchmark data point."""
    model_name: str
    benchmark_name: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScrapedContent:
    """Result from scraping a source."""
    source: str
    timestamp: datetime
    entries: list[BenchmarkEntry] = field(default_factory=list)
    raw_text: str = ""
    structured_data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class BaseScraper(ABC):
    """
    Base class for all scrapers with headless browser support.

    Each scraper subclass should:
    1. Define a `metadata` class attribute with SourceMetadata
    2. Implement the `scrape()` method
    """

    metadata: SourceMetadata

    def __init__(self, timeout: int = 30000, headless: bool = True):
        """
        Initialize scraper.

        Args:
            timeout: Page load timeout in milliseconds (default 30s)
            headless: Run browser in headless mode (default True)
        """
        self.timeout = timeout
        self.headless = headless

    def fetch_page(self, url: str, wait_for: str | None = None) -> Page:
        """
        Fetch a page using Playwright.

        Args:
            url: URL to fetch
            wait_for: CSS selector to wait for before returning (optional)

        Returns:
            Playwright Page object

        Raises:
            Exception: If page fails to load
        """
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=self.headless)
        page = browser.new_page()

        try:
            page.goto(url, timeout=self.timeout, wait_until="domcontentloaded")

            # Wait for specific element if requested
            if wait_for:
                page.wait_for_selector(wait_for, timeout=self.timeout)

            return page
        except Exception as e:
            browser.close()
            playwright.stop()
            raise Exception(f"Failed to fetch {url}: {str(e)}")

    def soup(self, html: str) -> BeautifulSoup:
        """
        Parse HTML with BeautifulSoup.

        Args:
            html: HTML content

        Returns:
            BeautifulSoup object
        """
        return BeautifulSoup(html, "lxml")

    def fetch_soup(self, url: str, wait_for: str | None = None) -> tuple[BeautifulSoup, Page]:
        """
        Fetch page and parse with BeautifulSoup.

        Args:
            url: URL to fetch
            wait_for: CSS selector to wait for (optional)

        Returns:
            Tuple of (BeautifulSoup object, Page object)

        Note:
            Caller is responsible for closing the page/browser
        """
        page = self.fetch_page(url, wait_for)
        html = page.content()
        return self.soup(html), page

    @abstractmethod
    def scrape(self) -> ScrapedContent:
        """
        Scrape the source and return structured data.

        Returns:
            ScrapedContent with results or error
        """
        pass

    def safe_scrape(self) -> ScrapedContent:
        """
        Safely scrape with error handling.

        Returns:
            ScrapedContent with results or error information
        """
        try:
            return self.scrape()
        except Exception as e:
            return ScrapedContent(
                source=self.metadata.name,
                timestamp=datetime.now(),
                error=f"Scraping failed: {str(e)}"
            )
