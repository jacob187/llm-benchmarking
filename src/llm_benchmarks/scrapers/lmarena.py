"""LM Arena leaderboard scraper."""

from datetime import datetime
import json
import re

from .base import (
    BaseScraper,
    BenchmarkEntry,
    ScrapedContent,
    SourceMetadata,
    SourceType,
)


class LMArenaScraper(BaseScraper):
    """Scraper for LM Arena leaderboard."""

    metadata = SourceMetadata(
        name="LM Arena",
        url="https://lmarena.ai/leaderboard",
        source_type=SourceType.LEADERBOARD,
        description="Human-voted rankings for AI models across multiple modalities",
        update_frequency="daily",
    )

    def scrape(self) -> ScrapedContent:
        """
        Scrape LM Arena leaderboard.

        Returns:
            ScrapedContent with benchmark entries
        """
        playwright = None
        browser = None
        page = None

        try:
            from playwright.sync_api import sync_playwright

            playwright = sync_playwright().start()
            browser = playwright.chromium.launch(headless=self.headless)
            page = browser.new_page()

            # Navigate to leaderboard
            page.goto(self.metadata.url, timeout=self.timeout, wait_until="domcontentloaded")

            # Wait for content to load - LM Arena uses dynamic rendering
            # Wait for either a table or data container to appear
            try:
                page.wait_for_selector("table, [data-testid*='leaderboard'], div[class*='leaderboard']",
                                      timeout=self.timeout)
            except:
                # If specific selectors fail, wait for general content
                page.wait_for_timeout(3000)

            # Get the page content
            html = page.content()
            soup = self.soup(html)

            entries = []

            # Try to extract data from tables first
            tables = soup.find_all("table")

            for table in tables:
                rows = table.find_all("tr")

                # Skip if table is too small
                if len(rows) < 2:
                    continue

                for row in rows[1:]:  # Skip header row
                    cells = row.find_all(["td", "th"])

                    if len(cells) < 2:
                        continue

                    model_name = None
                    score = None
                    rank = None

                    # Extract data from cells
                    for idx, cell in enumerate(cells):
                        text = cell.get_text(strip=True)

                        # Check if this cell contains a rank (usually first column)
                        if idx == 0 and text.isdigit():
                            rank = int(text)
                            continue

                        # Check if this looks like a model name
                        # Model names typically contain letters and may have versions
                        if model_name is None and text and len(text) > 2:
                            # Skip common header text
                            if text.lower() not in ['model', 'rank', 'score', 'elo', 'rating', 'votes']:
                                model_name = text
                                continue

                        # Check if this looks like a score
                        # Scores could be ELO ratings (800-2000), percentages, or other numbers
                        if score is None:
                            # Remove any non-numeric characters except decimal point and negative sign
                            clean_text = re.sub(r'[^\d.\-]', '', text)
                            if clean_text and clean_text.replace('.', '').replace('-', '').isdigit():
                                try:
                                    score_val = float(clean_text)
                                    # Valid score ranges: ELO (800-2000) or percentage (0-100) or normalized (0-1)
                                    if (800 <= score_val <= 2000) or (0 <= score_val <= 100) or (-1 <= score_val <= 1):
                                        score = score_val
                                except ValueError:
                                    continue

                    # Create entry if we have both model name and score
                    if model_name and score is not None:
                        metadata = {"source_url": self.metadata.url}
                        if rank is not None:
                            metadata["rank"] = rank

                        entry = BenchmarkEntry(
                            model_name=model_name,
                            benchmark_name="LM Arena Score",
                            score=score,
                            metadata=metadata,
                        )
                        entries.append(entry)

            # If no entries found from tables, try to extract from JSON-LD or script tags
            if not entries:
                # Look for embedded JSON data
                scripts = soup.find_all("script", type="application/json")
                for script in scripts:
                    try:
                        data = json.loads(script.string)
                        # Try to parse leaderboard data from JSON
                        # This is a fallback for when data is embedded in the page
                        entries.extend(self._parse_json_data(data))
                    except (json.JSONDecodeError, AttributeError, KeyError):
                        continue

            return ScrapedContent(
                source=self.metadata.name,
                timestamp=datetime.now(),
                entries=entries,
            )

        except Exception as e:
            return ScrapedContent(
                source=self.metadata.name,
                timestamp=datetime.now(),
                error=f"Failed to scrape LM Arena: {str(e)}",
            )

        finally:
            # Clean up browser resources
            if page:
                page.close()
            if browser:
                browser.close()
            if playwright:
                playwright.stop()

    def _parse_json_data(self, data: dict) -> list[BenchmarkEntry]:
        """
        Parse embedded JSON data to extract benchmark entries.

        Args:
            data: JSON data from script tag

        Returns:
            List of BenchmarkEntry objects
        """
        entries = []

        # Try different possible data structures
        # LM Arena might embed data in various formats
        possible_keys = ["leaderboard", "models", "data", "rankings", "results"]

        for key in possible_keys:
            if key in data and isinstance(data[key], list):
                for item in data[key]:
                    if isinstance(item, dict):
                        # Extract model name and score with flexible key names
                        model_name = (item.get("model") or
                                     item.get("model_name") or
                                     item.get("name"))

                        score = (item.get("score") or
                                item.get("elo") or
                                item.get("rating") or
                                item.get("points"))

                        if model_name and score is not None:
                            try:
                                entries.append(BenchmarkEntry(
                                    model_name=str(model_name),
                                    benchmark_name="LM Arena Score",
                                    score=float(score),
                                    metadata={"source_url": self.metadata.url},
                                ))
                            except (ValueError, TypeError):
                                continue

        return entries
