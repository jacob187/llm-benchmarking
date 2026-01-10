#!/usr/bin/env python3
"""Integration test for the full pipeline."""

from llm_benchmarks.pipeline import BenchmarkPipeline
from llm_benchmarks.llm import check_ollama_available


def main():
    print("=" * 60)
    print("PIPELINE INTEGRATION TEST")
    print("=" * 60)

    # Check Ollama
    ollama_available = check_ollama_available()
    if ollama_available:
        print("✓ Ollama is available")
    else:
        print("✗ Ollama not available - will skip analyses")

    # Test 1: Full pipeline run
    print("\n" + "=" * 60)
    print("TEST 1: Full Pipeline Run (with scraping)")
    print("=" * 60)

    pipeline = BenchmarkPipeline()

    analyses = ["summary", "coding"] if ollama_available else []
    result = pipeline.run(
        skip_scrape=False,
        analyses=analyses,
        include_blogs=True,
        save_cache=True
    )

    print(f"\n✓ Pipeline completed")
    print(f"  Models found: {len(result.models)}")
    print(f"  Analyses run: {len(result.analyses)}")
    print(f"  Errors: {len(result.errors)}")

    if result.errors:
        print("\n  Errors encountered:")
        for error in result.errors:
            print(f"    - {error}")

    if result.cache_path:
        print(f"  Cache saved to: {result.cache_path}")

    # Show top models
    print("\n  Top 5 Models:")
    top_models = pipeline.get_top_models(5)
    for i, model in enumerate(top_models, 1):
        print(f"    {i}. {model.name}: {model.average_score:.1f}")

    # Show analyses
    if result.analyses:
        print("\n  Analyses:")
        for analysis_type, analysis in result.analyses.items():
            print(f"\n  --- {analysis_type.upper()} ---")
            print(f"  {analysis.content[:300]}...")
            print(f"  ({len(analysis.content)} total chars)")

    # Test 2: Using cache
    print("\n" + "=" * 60)
    print("TEST 2: Pipeline with Cache (no scraping)")
    print("=" * 60)

    pipeline2 = BenchmarkPipeline()
    result2 = pipeline2.run(
        skip_scrape=True,
        analyses=[],
        save_cache=False
    )

    print(f"✓ Loaded from cache")
    print(f"  Models: {len(result2.models)}")

    # Test 3: Model comparison
    if ollama_available and len(result.models) >= 2:
        print("\n" + "=" * 60)
        print("TEST 3: Model Comparison")
        print("=" * 60)

        # Get first two model names
        model_names = [m.name for m in list(result.models.values())[:2]]
        print(f"Comparing: {model_names[0]} vs {model_names[1]}")

        try:
            comparison = pipeline2.compare_models(model_names)
            print(f"\n{comparison.content[:400]}...")
        except Exception as e:
            print(f"✗ Error: {e}")

    # Test 4: Question answering
    if ollama_available and len(result.models) > 0:
        print("\n" + "=" * 60)
        print("TEST 4: Question Answering (Streaming)")
        print("=" * 60)

        question = "What are the key trends in LLM performance?"
        print(f"Question: {question}\n")
        print("Answer: ", end="", flush=True)

        try:
            for chunk in pipeline2.answer_question(question, stream=True):
                print(chunk, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"\n✗ Error: {e}")

    # Test 5: Top models by specific benchmark
    print("\n" + "=" * 60)
    print("TEST 5: Top Models by Benchmark")
    print("=" * 60)

    # Try to find a common benchmark
    if result.models:
        all_benchmarks = set()
        for model in result.models.values():
            all_benchmarks.update(model.benchmarks.keys())

        if all_benchmarks:
            benchmark = list(all_benchmarks)[0]
            print(f"Top 5 models for '{benchmark}':")

            top = pipeline2.get_top_models(5, benchmark=benchmark)
            for i, model in enumerate(top, 1):
                score = model.benchmarks.get(benchmark, 0)
                print(f"  {i}. {model.name}: {score}")

    print("\n" + "=" * 60)
    print("✓ ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
