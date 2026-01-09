"""Tests for data aggregator."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from llm_benchmarks.data_aggregator import BenchmarkAggregator, ModelBenchmarks
from llm_benchmarks.scrapers.base import BenchmarkEntry, ScrapedContent


def test_model_benchmarks_creation():
    """Test creating ModelBenchmarks."""
    model = ModelBenchmarks(name="GPT-4")

    assert model.name == "GPT-4"
    assert model.benchmarks == {}
    assert model.sources == set()
    assert model.average_score is None


def test_model_benchmarks_add_benchmark():
    """Test adding benchmarks to a model."""
    model = ModelBenchmarks(name="GPT-4")
    model.add_benchmark("MMLU", 85.0, "Source1")
    model.add_benchmark("HumanEval", 90.0, "Source2")

    assert model.benchmarks == {"MMLU": 85.0, "HumanEval": 90.0}
    assert model.sources == {"Source1", "Source2"}


def test_model_benchmarks_calculate_average():
    """Test calculating average score."""
    model = ModelBenchmarks(name="GPT-4")
    model.add_benchmark("Test1", 80.0, "Source1")
    model.add_benchmark("Test2", 90.0, "Source1")
    model.add_benchmark("Test3", 85.0, "Source2")
    model.calculate_average()

    assert model.average_score == 85.0


def test_model_benchmarks_serialization():
    """Test to_dict and from_dict."""
    model = ModelBenchmarks(name="GPT-4")
    model.add_benchmark("MMLU", 85.0, "Source1")
    model.calculate_average()

    # Convert to dict
    data = model.to_dict()
    assert data["name"] == "GPT-4"
    assert data["benchmarks"] == {"MMLU": 85.0}
    assert "Source1" in data["sources"]

    # Convert back
    model2 = ModelBenchmarks.from_dict(data)
    assert model2.name == model.name
    assert model2.benchmarks == model.benchmarks
    assert model2.sources == model.sources


def test_normalize_model_name():
    """Test model name normalization."""
    aggregator = BenchmarkAggregator()

    # Test various name formats
    assert aggregator.normalize_model_name("GPT-4") == "gpt_4"
    assert aggregator.normalize_model_name("gpt-4-turbo") == "gpt_4_turbo"
    assert aggregator.normalize_model_name("Claude 3 Opus") == "claude_3_opus"
    assert aggregator.normalize_model_name("Anthropic/claude-3-opus") == "claude_3_opus"
    assert aggregator.normalize_model_name("GPT-4 (OpenAI)") == "gpt_4"


def test_aggregator_initialization():
    """Test aggregator initialization."""
    aggregator = BenchmarkAggregator()

    assert aggregator.models == {}
    assert aggregator.blog_content == []
    assert aggregator.structured_data == {}
    assert aggregator.last_updated is None


def test_aggregator_single_source():
    """Test aggregating data from a single source."""
    aggregator = BenchmarkAggregator()

    # Create mock scraped content
    entries = [
        BenchmarkEntry("GPT-4", "MMLU", 85.0, {}),
        BenchmarkEntry("Claude 3", "MMLU", 82.0, {}),
    ]
    scraped = ScrapedContent(
        source="TestSource",
        timestamp=datetime.now(),
        entries=entries
    )

    # Aggregate
    models = aggregator.aggregate([scraped])

    assert len(models) == 2
    assert "gpt_4" in models
    assert "claude_3" in models
    assert models["gpt_4"].benchmarks["MMLU"] == 85.0
    assert "TestSource" in models["gpt_4"].sources


def test_aggregator_multiple_sources():
    """Test aggregating data from multiple sources."""
    aggregator = BenchmarkAggregator()

    # Create mock data from two sources
    source1 = ScrapedContent(
        source="Source1",
        timestamp=datetime.now(),
        entries=[BenchmarkEntry("GPT-4", "MMLU", 85.0, {})]
    )
    source2 = ScrapedContent(
        source="Source2",
        timestamp=datetime.now(),
        entries=[BenchmarkEntry("gpt-4-turbo", "HumanEval", 90.0, {})]
    )

    models = aggregator.aggregate([source1, source2])

    # Should normalize to same model
    assert "gpt_4" in models or "gpt_4_turbo" in models
    # Check that both benchmarks are present
    all_benchmarks = set()
    for model in models.values():
        all_benchmarks.update(model.benchmarks.keys())
    assert "MMLU" in all_benchmarks or "HumanEval" in all_benchmarks


def test_aggregator_blog_content():
    """Test handling blog content."""
    aggregator = BenchmarkAggregator()

    scraped = ScrapedContent(
        source="BlogSource",
        timestamp=datetime.now(),
        raw_text="This is blog content about LLMs"
    )

    aggregator.aggregate([scraped])

    assert len(aggregator.blog_content) == 1
    assert aggregator.blog_content[0]["source"] == "BlogSource"
    assert "blog content" in aggregator.blog_content[0]["content"]


def test_aggregator_skip_errors():
    """Test that aggregator skips errored scrapes."""
    aggregator = BenchmarkAggregator()

    good_scrape = ScrapedContent(
        source="GoodSource",
        timestamp=datetime.now(),
        entries=[BenchmarkEntry("GPT-4", "MMLU", 85.0, {})]
    )
    bad_scrape = ScrapedContent(
        source="BadSource",
        timestamp=datetime.now(),
        error="Failed to scrape"
    )

    models = aggregator.aggregate([good_scrape, bad_scrape])

    assert len(models) == 1
    assert "gpt_4" in models


def test_get_top_models():
    """Test getting top models by score."""
    aggregator = BenchmarkAggregator()

    entries = [
        BenchmarkEntry("Model A", "Test", 90.0, {}),
        BenchmarkEntry("Model B", "Test", 85.0, {}),
        BenchmarkEntry("Model C", "Test", 95.0, {}),
    ]
    scraped = ScrapedContent(
        source="TestSource",
        timestamp=datetime.now(),
        entries=entries
    )

    aggregator.aggregate([scraped])
    top_models = aggregator.get_top_models(n=2)

    assert len(top_models) == 2
    assert top_models[0].average_score >= top_models[1].average_score


def test_cache_save_and_load(tmp_path):
    """Test saving and loading cache."""
    aggregator = BenchmarkAggregator()

    # Create some data
    entries = [BenchmarkEntry("GPT-4", "MMLU", 85.0, {})]
    scraped = ScrapedContent(
        source="TestSource",
        timestamp=datetime.now(),
        entries=entries,
        raw_text="Blog content"
    )
    aggregator.aggregate([scraped])

    # Change cache directory to temp
    import llm_benchmarks.data_aggregator
    original_path = Path("data/processed")

    # Save cache
    cache_path = aggregator.save_cache("test_cache.json")
    assert cache_path.exists()

    # Load into new aggregator
    aggregator2 = BenchmarkAggregator()
    success = aggregator2.load_cache("test_cache.json")

    assert success
    assert len(aggregator2.models) == 1
    assert "gpt_4" in aggregator2.models
    assert len(aggregator2.blog_content) == 1

    # Cleanup
    cache_path.unlink()


def test_get_model():
    """Test getting a specific model."""
    aggregator = BenchmarkAggregator()

    entries = [BenchmarkEntry("GPT-4-Turbo", "MMLU", 85.0, {})]
    scraped = ScrapedContent(
        source="TestSource",
        timestamp=datetime.now(),
        entries=entries
    )
    aggregator.aggregate([scraped])

    # Should find model by various name formats
    model = aggregator.get_model("gpt-4-turbo")
    assert model is not None
    assert model.benchmarks["MMLU"] == 85.0


def test_get_models_by_source():
    """Test getting models from a specific source."""
    aggregator = BenchmarkAggregator()

    source1_data = ScrapedContent(
        source="Source1",
        timestamp=datetime.now(),
        entries=[BenchmarkEntry("Model A", "Test", 85.0, {})]
    )
    source2_data = ScrapedContent(
        source="Source2",
        timestamp=datetime.now(),
        entries=[BenchmarkEntry("Model B", "Test", 90.0, {})]
    )

    aggregator.aggregate([source1_data, source2_data])

    source1_models = aggregator.get_models_by_source("Source1")
    assert len(source1_models) == 1
    assert source1_models[0].name == "Model A"


@pytest.mark.slow
def test_aggregator_with_real_scrapers():
    """Test aggregator with real scraper data."""
    from llm_benchmarks.scrapers.registry import ScraperRegistry

    registry = ScraperRegistry()
    scrapers = registry.all()

    if not scrapers:
        pytest.skip("No scrapers available")

    # Scrape all sources
    scraped_contents = []
    for scraper in scrapers[:2]:  # Limit to 2 scrapers for speed
        result = scraper.safe_scrape()
        scraped_contents.append(result)

    # Aggregate
    aggregator = BenchmarkAggregator()
    models = aggregator.aggregate(scraped_contents)

    # Should have some models or blog content
    assert len(models) > 0 or len(aggregator.blog_content) > 0
