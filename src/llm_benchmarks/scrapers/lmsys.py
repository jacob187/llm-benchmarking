"""LMSYS Chatbot Arena leaderboard scraper."""

from datetime import datetime

from .base import (
    BaseScraper,
    BenchmarkEntry,
    ScrapedContent,
    SourceMetadata,
    SourceType,
)


class LMSYSArenaScraper(BaseScraper):
    """Scraper for LMSYS Chatbot Arena leaderboard."""

    metadata = SourceMetadata(
        name="LMSYS Arena",
        url="https://lmarena.ai/leaderboard",
        source_type=SourceType.LEADERBOARD,
        description="Community-driven chatbot arena with ELO ratings",
        update_frequency="daily",
    )

    def scrape(self) -> ScrapedContent:
        """
        Scrape LMSYS Arena leaderboard.

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

            # Wait for the leaderboard table to load
            # LMSYS uses dynamic content, so we need to wait for it
            page.wait_for_selector("table", timeout=self.timeout)

            # Give extra time for JavaScript to populate the table
            page.wait_for_timeout(2000)

            # Get the page content
            html = page.content()
            soup = self.soup(html)

            entries = []

            # Find the leaderboard table
            # LMSYS typically uses a table or data grid for rankings
            tables = soup.find_all("table")

            for table in tables:
                rows = table.find_all("tr")

                for row in rows[1:]:  # Skip header row
                    cells = row.find_all(["td", "th"])

                    if len(cells) < 3:
                        continue

                    # Try to find rank, model name, and ELO score
                    # Common pattern: Rank | Model | ELO | Other columns
                    rank_text = cells[0].get_text(strip=True)
                    model_name = None
                    elo_score = None

                    # Check if first cell is a rank number
                    if rank_text.isdigit():
                        # Model name should be in second column
                        if len(cells) > 1:
                            model_name = cells[1].get_text(strip=True)

                        # ELO score typically in third or fourth column
                        for idx in range(2, min(len(cells), 5)):
                            text = cells[idx].get_text(strip=True)
                            # Look for ELO score (typically a 4-digit number)
                            if text.replace(".", "").replace(",", "").replace("-", "").isdigit():
                                try:
                                    score_val = float(text.replace(",", ""))
                                    if 800 <= score_val <= 2000:  # Reasonable ELO range
                                        elo_score = score_val
                                        break
                                except ValueError:
                                    continue
                    else:
                        # If no rank, assume model is first column
                        model_name = cells[0].get_text(strip=True)
                        for cell in cells[1:]:
                            text = cell.get_text(strip=True)
                            if text.replace(".", "").replace(",", "").isdigit():
                                try:
                                    score_val = float(text.replace(",", ""))
                                    if 800 <= score_val <= 2000:
                                        elo_score = score_val
                                        break
                                except ValueError:
                                    continue

                    # Only add if we have both model name and score
                    if model_name and elo_score and model_name not in ["Model", "Rank", "#"]:
                        entry = BenchmarkEntry(
                            model_name=model_name,
                            benchmark_name="LMSYS Arena ELO",
                            score=elo_score,
                            metadata={"source_url": self.metadata.url},
                        )
                        entries.append(entry)

            return ScrapedContent(
                source=self.metadata.name,
                timestamp=datetime.now(),
                entries=entries,
            )

        except Exception as e:
            return ScrapedContent(
                source=self.metadata.name,
                timestamp=datetime.now(),
                error=f"Failed to scrape LMSYS Arena: {str(e)}",
            )

        finally:
            # Clean up browser resources
            if page:
                page.close()
            if browser:
                browser.close()
            if playwright:
                playwright.stop()
