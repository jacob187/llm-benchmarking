#!/usr/bin/env python3
"""Integration test for data aggregator with real scrapers."""

from llm_benchmarks.scrapers.registry import ScraperRegistry
from llm_benchmarks.data_aggregator import BenchmarkAggregator


def main():
    print("=" * 60)
    print("DATA AGGREGATOR INTEGRATION TEST")
    print("=" * 60)

    # Get all scrapers
    print("\n1. Discovering scrapers...")
    registry = ScraperRegistry()
    scrapers = registry.all()
    print(f"   Found {len(scrapers)} scrapers")

    # Scrape all sources
    print("\n2. Scraping all sources...")
    scraped_contents = []
    for scraper in scrapers:
        print(f"   Scraping {scraper.metadata.name}...")
        result = scraper.safe_scrape()
        if result.error:
            print(f"      ✗ Error: {result.error}")
        else:
            if result.entries:
                print(f"      ✓ Found {len(result.entries)} entries")
            if result.raw_text:
                print(f"      ✓ Extracted {len(result.raw_text)} chars of text")
        scraped_contents.append(result)

    # Aggregate data
    print("\n3. Aggregating data...")
    aggregator = BenchmarkAggregator()
    models = aggregator.aggregate(scraped_contents)
    print(f"   ✓ Aggregated {len(models)} unique models")
    print(f"   ✓ Found {len(aggregator.blog_content)} blog sources")

    # Show top models
    if models:
        print("\n4. Top 5 models by average score:")
        top_models = aggregator.get_top_models(n=5)
        for i, model in enumerate(top_models, 1):
            sources_str = ", ".join(model.sources)
            benchmarks_str = ", ".join(f"{k}: {v}" for k, v in model.benchmarks.items())
            print(f"   {i}. {model.name}")
            print(f"      Average: {model.average_score:.1f}")
            print(f"      Benchmarks: {benchmarks_str}")
            print(f"      Sources: {sources_str}")

    # Test caching
    print("\n5. Testing cache...")
    cache_path = aggregator.save_cache("integration_test_cache.json")
    print(f"   ✓ Saved cache to {cache_path}")

    # Load cache
    aggregator2 = BenchmarkAggregator()
    success = aggregator2.load_cache("integration_test_cache.json")
    if success:
        print(f"   ✓ Loaded cache successfully")
        print(f"   ✓ Verified {len(aggregator2.models)} models in cache")
    else:
        print("   ✗ Failed to load cache")

    # Test model lookup
    print("\n6. Testing model lookup...")
    if models:
        first_model_name = list(models.values())[0].name
        found_model = aggregator.get_model(first_model_name)
        if found_model:
            print(f"   ✓ Successfully looked up '{first_model_name}'")
        else:
            print(f"   ✗ Failed to find '{first_model_name}'")

    print("\n" + "=" * 60)
    print("✓ INTEGRATION TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
