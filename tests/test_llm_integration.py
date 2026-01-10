#!/usr/bin/env python3
"""Integration test for LLM analyzer."""

from llm_benchmarks.scrapers.registry import ScraperRegistry
from llm_benchmarks.data_aggregator import BenchmarkAggregator
from llm_benchmarks.llm import BenchmarkAnalyzer, check_ollama_available


def main():
    print("=" * 60)
    print("LLM ANALYZER INTEGRATION TEST")
    print("=" * 60)

    # Check if Ollama is available
    print("\n1. Checking Ollama availability...")
    if not check_ollama_available():
        print("   ✗ Ollama is not running!")
        print("   Please start Ollama: ollama serve")
        print("   And ensure gemma2:9b is available: ollama pull gemma2:9b")
        return

    print("   ✓ Ollama is running")

    # Scrape data
    print("\n2. Scraping benchmark sources...")
    registry = ScraperRegistry()
    scrapers = registry.all()
    scraped_contents = []

    for scraper in scrapers:
        print(f"   Scraping {scraper.metadata.name}...")
        result = scraper.safe_scrape()
        if not result.error:
            print(f"      ✓ Success")
        else:
            print(f"      ✗ Error: {result.error}")
        scraped_contents.append(result)

    # Aggregate data
    print("\n3. Aggregating data...")
    aggregator = BenchmarkAggregator()
    models = aggregator.aggregate(scraped_contents)
    print(f"   ✓ Aggregated {len(models)} models")
    print(f"   ✓ Found {len(aggregator.blog_content)} blog sources")

    # Initialize analyzer
    print("\n4. Initializing LLM analyzer...")
    analyzer = BenchmarkAnalyzer()
    print("   ✓ Analyzer ready")

    # Test summary
    print("\n5. Generating executive summary...")
    print("   (This may take 10-30 seconds...)")
    try:
        result = analyzer.summarize(models)
        print(f"   ✓ Summary generated ({len(result.content)} chars)")
        print("\n   --- SUMMARY ---")
        print(result.content)
        print("   " + "-" * 57)
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return

    # Test blog analysis
    if aggregator.blog_content:
        print("\n6. Analyzing blog content...")
        try:
            result = analyzer.analyze_blogs(aggregator.blog_content)
            print(f"   ✓ Blog analysis complete ({len(result.content)} chars)")
            print("\n   --- BLOG INSIGHTS ---")
            print(result.content)
            print("   " + "-" * 57)
        except Exception as e:
            print(f"   ✗ Error: {e}")
    else:
        print("\n6. No blog content to analyze")

    # Test streaming
    print("\n7. Testing streaming output...")
    print("   Question: What is the top performing model?")
    print("\n   Answer (streaming):")
    print("   ", end="")
    try:
        for chunk in analyzer.answer_question_stream("What is the top performing model?", models):
            print(chunk, end="", flush=True)
        print("\n")
        print("   ✓ Streaming works")
    except Exception as e:
        print(f"\n   ✗ Error: {e}")

    print("\n" + "=" * 60)
    print("✓ LLM INTEGRATION TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
