"""Tests for pipeline orchestration."""

import pytest
from pathlib import Path

from llm_benchmarks.pipeline import BenchmarkPipeline, PipelineResult
from llm_benchmarks.data_aggregator import ModelBenchmarks


def test_pipeline_initialization():
    """Test creating a pipeline."""
    pipeline = BenchmarkPipeline()

    assert pipeline is not None
    assert pipeline.registry is not None
    assert pipeline.aggregator is not None
    assert pipeline.analyzer is None  # Lazy-loaded


def test_pipeline_initialization_with_model():
    """Test creating pipeline with specific model."""
    pipeline = BenchmarkPipeline(ollama_model="llama3.2:3b")

    assert pipeline.ollama_model == "llama3.2:3b"


@pytest.mark.slow
def test_scrape_all():
    """Test scraping all sources."""
    pipeline = BenchmarkPipeline()
    scraped_contents = pipeline.scrape_all()

    assert len(scraped_contents) >= 2
    assert all(hasattr(s, 'source') for s in scraped_contents)
    assert all(hasattr(s, 'timestamp') for s in scraped_contents)


@pytest.mark.slow
def test_run_pipeline_basic():
    """Test running pipeline without analysis."""
    pipeline = BenchmarkPipeline()

    # Run without analyses (no Ollama required)
    result = pipeline.run(analyses=[], save_cache=False)

    assert isinstance(result, PipelineResult)
    assert len(result.models) > 0
    assert result.timestamp is not None


@pytest.mark.slow
def test_run_pipeline_with_cache():
    """Test running pipeline and using cache."""
    pipeline = BenchmarkPipeline()

    # First run - scrape and cache
    result1 = pipeline.run(analyses=[], save_cache=True)
    assert result1.cache_path is not None
    assert result1.cache_path.exists()

    # Second run - load from cache
    pipeline2 = BenchmarkPipeline()
    result2 = pipeline2.run(skip_scrape=True, analyses=[])

    assert len(result2.models) == len(result1.models)
    assert len(result2.errors) == 0


@pytest.mark.slow
def test_run_pipeline_with_analysis():
    """Test running pipeline with LLM analysis."""
    try:
        pipeline = BenchmarkPipeline()
        result = pipeline.run(analyses=["summary"], save_cache=False)

        assert isinstance(result, PipelineResult)
        assert len(result.models) > 0

        if "summary" in result.analyses:
            # Ollama was available
            assert result.analyses["summary"].content
            assert result.analyses["summary"].analysis_type == "summary"
        else:
            # Ollama not available, should have error
            assert any("Ollama" in e or "Cannot run" in e for e in result.errors)

    except ConnectionError:
        pytest.skip("Ollama not available")


@pytest.mark.slow
def test_run_pipeline_skip_blogs():
    """Test running pipeline without blog analysis."""
    try:
        pipeline = BenchmarkPipeline()
        result = pipeline.run(analyses=["blog"], include_blogs=False, save_cache=False)

        # Should not have blog analysis even if requested
        assert "blog" not in result.analyses or result.analyses["blog"] is None

    except ConnectionError:
        pytest.skip("Ollama not available")


def test_get_top_models_no_data():
    """Test getting top models when no data loaded."""
    pipeline = BenchmarkPipeline()
    top = pipeline.get_top_models(5)

    # Should return empty list or load from cache if available
    assert isinstance(top, list)


@pytest.mark.slow
def test_get_top_models_with_data():
    """Test getting top models after running pipeline."""
    pipeline = BenchmarkPipeline()
    result = pipeline.run(analyses=[], save_cache=False)

    top = pipeline.get_top_models(5)

    assert len(top) <= 5
    assert all(isinstance(m, ModelBenchmarks) for m in top)

    # Should be sorted by score (descending)
    if len(top) > 1:
        for i in range(len(top) - 1):
            assert top[i].average_score >= top[i + 1].average_score


@pytest.mark.slow
def test_compare_models_with_cache():
    """Test comparing models using cache."""
    # First, create a cache
    pipeline = BenchmarkPipeline()
    result = pipeline.run(analyses=[], save_cache=True)

    if len(result.models) < 2:
        pytest.skip("Not enough models to compare")

    # Get first two model names
    model_names = [m.name for m in list(result.models.values())[:2]]

    try:
        # Compare using cache
        pipeline2 = BenchmarkPipeline()
        comparison = pipeline2.compare_models(model_names, use_cache=True)

        assert comparison is not None
        assert comparison.analysis_type == "comparison"
        assert len(comparison.content) > 0

    except ConnectionError:
        pytest.skip("Ollama not available")


@pytest.mark.slow
def test_answer_question_with_cache():
    """Test answering question using cache."""
    # First, create a cache
    pipeline = BenchmarkPipeline()
    result = pipeline.run(analyses=[], save_cache=True)

    if len(result.models) == 0:
        pytest.skip("No models to query about")

    try:
        # Answer question using cache
        pipeline2 = BenchmarkPipeline()
        answer = pipeline2.answer_question("What is the best model?", use_cache=True)

        assert answer is not None
        assert answer.analysis_type == "qa"
        assert len(answer.content) > 0

    except ConnectionError:
        pytest.skip("Ollama not available")


@pytest.mark.slow
def test_answer_question_stream():
    """Test streaming question answering."""
    # Create a cache first
    pipeline = BenchmarkPipeline()
    result = pipeline.run(analyses=[], save_cache=True)

    if len(result.models) == 0:
        pytest.skip("No models to query about")

    try:
        # Answer with streaming
        pipeline2 = BenchmarkPipeline()
        chunks = list(pipeline2.answer_question(
            "What is the top model?",
            use_cache=True,
            stream=True
        ))

        assert len(chunks) > 0
        combined = "".join(chunks)
        assert len(combined) > 0

    except ConnectionError:
        pytest.skip("Ollama not available")


def test_compare_models_no_cache():
    """Test comparing models without cache or data fails appropriately."""
    pipeline = BenchmarkPipeline()

    # Clear any existing data
    pipeline.aggregator.models = {}

    # Should raise ValueError when no data available
    with pytest.raises(ValueError, match="No models loaded"):
        pipeline.compare_models(["Model A"], use_cache=False)


def test_answer_question_no_cache():
    """Test answering question without cache or data fails appropriately."""
    pipeline = BenchmarkPipeline()

    # Clear any existing data
    pipeline.aggregator.models = {}

    # Should raise ValueError when no data available
    with pytest.raises(ValueError, match="No models loaded"):
        pipeline.answer_question("What?", use_cache=False)


@pytest.mark.slow
def test_pipeline_handles_scraper_errors():
    """Test that pipeline handles scraper errors gracefully."""
    pipeline = BenchmarkPipeline()
    result = pipeline.run(analyses=[], save_cache=False)

    # Pipeline should complete even if some scrapers fail
    assert isinstance(result, PipelineResult)
    # May have errors but should still have some results
    if result.errors:
        print(f"Errors encountered: {result.errors}")
