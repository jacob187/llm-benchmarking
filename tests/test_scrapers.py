#!/usr/bin/env python3
"""Quick test script to verify scrapers work."""

from pathlib import Path
from llm_benchmarks.scrapers.registry import ScraperRegistry

def test_registry():
    """Test that registry discovers scrapers."""
    print("Testing registry auto-discovery...")

    # Debug: check if scraper files exist
    scrapers_dir = Path("src/llm_benchmarks/scrapers")
    print(f"  Scrapers directory: {scrapers_dir.absolute()}")
    print(f"  Files in directory:")
    for f in scrapers_dir.glob("*.py"):
        print(f"    - {f.name}")

    registry = ScraperRegistry()
    registry.discover()

    count = registry.count()
    print(f"✓ Found {count} scrapers")

    sources = registry.list_sources()
    for source in sources:
        print(f"  - {source.name} ({source.source_type.value}): {source.url}")

    return count > 0

def test_lmsys_scraper():
    """Test LMSYS scraper."""
    print("\nTesting LMSYS Arena scraper...")
    registry = ScraperRegistry()
    scraper = registry.get("lmsys_arena")

    if not scraper:
        print("✗ LMSYS scraper not found in registry")
        return False

    print(f"  Scraping {scraper.metadata.url}...")
    result = scraper.safe_scrape()

    if result.error:
        print(f"✗ Error: {result.error}")
        return False

    print(f"✓ Found {len(result.entries)} benchmark entries")

    if result.entries:
        print("  Sample entries:")
        for entry in result.entries[:3]:
            print(f"    - {entry.model_name}: {entry.score}")

    return len(result.entries) > 0

def test_blog_scraper():
    """Test blog scraper."""
    print("\nTesting Simon Willison blog scraper...")
    registry = ScraperRegistry()
    scraper = registry.get("simon_willison_blog")

    if not scraper:
        print("✗ Blog scraper not found in registry")
        return False

    print(f"  Scraping {scraper.metadata.url}...")
    result = scraper.safe_scrape()

    if result.error:
        print(f"✗ Error: {result.error}")
        return False

    print(f"✓ Scraped {len(result.raw_text)} characters of text")

    if result.raw_text:
        preview = result.raw_text[:200]
        print(f"  Preview: {preview}...")

    return len(result.raw_text) > 0

if __name__ == "__main__":
    print("=" * 60)
    print("SCRAPER TEST SUITE")
    print("=" * 60)

    tests = [
        ("Registry Discovery", test_registry),
        ("LMSYS Arena Scraper", test_lmsys_scraper),
        ("Blog Scraper", test_blog_scraper),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
