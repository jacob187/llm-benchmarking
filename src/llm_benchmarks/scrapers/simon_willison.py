"""Simon Willison's blog scraper for LLM-related content."""

from datetime import datetime

from .base import (
    BaseScraper,
    ScrapedContent,
    SourceMetadata,
    SourceType,
)


class SimonWillisonBlogScraper(BaseScraper):
    """Scraper for Simon Willison's LLM-related blog posts."""

    metadata = SourceMetadata(
        name="Simon Willison Blog",
        url="https://simonwillison.net/tags/llms/",
        source_type=SourceType.BLOG,
        description="Expert commentary on LLM developments and benchmarks",
        update_frequency="weekly",
    )

    def scrape(self) -> ScrapedContent:
        """
        Scrape recent LLM-related blog posts.

        Returns:
            ScrapedContent with raw text for LLM analysis
        """
        playwright = None
        browser = None
        page = None

        try:
            from playwright.sync_api import sync_playwright

            playwright = sync_playwright().start()
            browser = playwright.chromium.launch(headless=self.headless)
            page = browser.new_page()

            # Navigate to LLM tag page
            page.goto(self.metadata.url, timeout=self.timeout, wait_until="domcontentloaded")

            # Wait for content to load
            page.wait_for_selector("article, .entry, .post", timeout=self.timeout)

            html = page.content()
            soup = self.soup(html)

            # Extract blog posts
            posts = []
            post_elements = soup.find_all(["article", "div"], class_=lambda x: x and ("entry" in x or "post" in x))

            # If no posts found with class, try finding by article tag
            if not post_elements:
                post_elements = soup.find_all("article")

            for post in post_elements[:5]:  # Get last 5 posts
                # Extract title
                title_elem = post.find(["h1", "h2", "h3"])
                title = title_elem.get_text(strip=True) if title_elem else "Untitled"

                # Extract content/excerpt
                content_elem = post.find(["p", "div"], class_=lambda x: x and "content" in x.lower() if x else False)
                if not content_elem:
                    # Fallback: get all paragraphs
                    paragraphs = post.find_all("p")
                    content = " ".join([p.get_text(strip=True) for p in paragraphs[:3]])
                else:
                    content = content_elem.get_text(strip=True)

                # Extract date if available
                date_elem = post.find("time")
                date_str = date_elem.get_text(strip=True) if date_elem else "Recent"

                posts.append(f"## {title}\n**Date:** {date_str}\n\n{content}\n")

            # Combine into markdown
            raw_text = f"# Recent LLM Posts from Simon Willison\n\n"
            raw_text += "\n\n".join(posts)

            return ScrapedContent(
                source=self.metadata.name,
                timestamp=datetime.now(),
                raw_text=raw_text,
            )

        except Exception as e:
            return ScrapedContent(
                source=self.metadata.name,
                timestamp=datetime.now(),
                error=f"Failed to scrape Simon Willison blog: {str(e)}",
            )

        finally:
            if page:
                page.close()
            if browser:
                browser.close()
            if playwright:
                playwright.stop()
