"""Tests for benchmark analyzer."""

import pytest

from llm_benchmarks.llm.analyzer import BenchmarkAnalyzer, AnalysisResult
from llm_benchmarks.data_aggregator import ModelBenchmarks


@pytest.fixture
def sample_models():
    """Create sample model data for testing."""
    model1 = ModelBenchmarks(name="GPT-4")
    model1.add_benchmark("MMLU", 85.0, "TestSource")
    model1.add_benchmark("HumanEval", 90.0, "TestSource")
    model1.calculate_average()

    model2 = ModelBenchmarks(name="Claude 3")
    model2.add_benchmark("MMLU", 82.0, "TestSource")
    model2.add_benchmark("HumanEval", 88.0, "TestSource")
    model2.calculate_average()

    return {
        "gpt_4": model1,
        "claude_3": model2
    }


@pytest.fixture
def sample_blog_content():
    """Create sample blog content for testing."""
    return [
        {
            "source": "Test Blog",
            "timestamp": "2024-01-01T00:00:00",
            "content": "This is a test blog post about LLMs and their capabilities."
        }
    ]


def test_analyzer_initialization():
    """Test creating analyzer without Ollama running."""
    # Should not fail on initialization
    try:
        analyzer = BenchmarkAnalyzer()
        # If Ollama is running, this should work
        assert analyzer is not None
    except ConnectionError:
        # If Ollama is not running, that's expected
        pass


def test_format_benchmark_data(sample_models):
    """Test formatting benchmark data."""
    try:
        analyzer = BenchmarkAnalyzer()
        formatted = analyzer._format_benchmark_data(sample_models)

        assert "GPT-4" in formatted
        assert "Claude 3" in formatted
        assert "MMLU" in formatted
        assert "85.0" in formatted or "85" in formatted
    except ConnectionError:
        pytest.skip("Ollama not running")


def test_format_blog_content(sample_blog_content):
    """Test formatting blog content."""
    try:
        analyzer = BenchmarkAnalyzer()
        formatted = analyzer._format_blog_content(sample_blog_content)

        assert "Test Blog" in formatted
        assert "test blog post" in formatted.lower()
    except ConnectionError:
        pytest.skip("Ollama not running")


def test_format_empty_data():
    """Test formatting empty data."""
    try:
        analyzer = BenchmarkAnalyzer()

        formatted = analyzer._format_benchmark_data({})
        assert "No benchmark data" in formatted

        formatted = analyzer._format_blog_content([])
        assert "No blog content" in formatted
    except ConnectionError:
        pytest.skip("Ollama not running")


@pytest.mark.slow
def test_summarize(sample_models):
    """Test generating summary (requires Ollama)."""
    try:
        analyzer = BenchmarkAnalyzer()
        result = analyzer.summarize(sample_models)

        assert isinstance(result, AnalysisResult)
        assert result.analysis_type == "summary"
        assert len(result.content) > 0
        assert result.model_used
    except ConnectionError:
        pytest.skip("Ollama not running")


@pytest.mark.slow
def test_compare_models(sample_models):
    """Test comparing models (requires Ollama)."""
    try:
        analyzer = BenchmarkAnalyzer()
        result = analyzer.compare_models(["GPT-4", "Claude 3"], sample_models)

        assert isinstance(result, AnalysisResult)
        assert result.analysis_type == "comparison"
        assert len(result.content) > 0
    except ConnectionError:
        pytest.skip("Ollama not running")


@pytest.mark.slow
def test_analyze_coding(sample_models):
    """Test coding analysis (requires Ollama)."""
    try:
        analyzer = BenchmarkAnalyzer()
        result = analyzer.analyze_coding(sample_models)

        assert isinstance(result, AnalysisResult)
        assert result.analysis_type == "coding"
        assert len(result.content) > 0
    except ConnectionError:
        pytest.skip("Ollama not running")


@pytest.mark.slow
def test_analyze_blogs(sample_blog_content):
    """Test blog analysis (requires Ollama)."""
    try:
        analyzer = BenchmarkAnalyzer()
        result = analyzer.analyze_blogs(sample_blog_content)

        assert isinstance(result, AnalysisResult)
        assert result.analysis_type == "blog"
        assert len(result.content) > 0
    except ConnectionError:
        pytest.skip("Ollama not running")


@pytest.mark.slow
def test_answer_question(sample_models):
    """Test question answering (requires Ollama)."""
    try:
        analyzer = BenchmarkAnalyzer()
        result = analyzer.answer_question("Which model is better for coding?", sample_models)

        assert isinstance(result, AnalysisResult)
        assert result.analysis_type == "qa"
        assert len(result.content) > 0
    except ConnectionError:
        pytest.skip("Ollama not running")


@pytest.mark.slow
def test_summarize_stream(sample_models):
    """Test streaming summary (requires Ollama)."""
    try:
        analyzer = BenchmarkAnalyzer()
        chunks = list(analyzer.summarize_stream(sample_models))

        assert len(chunks) > 0
        combined = "".join(chunks)
        assert len(combined) > 0
    except ConnectionError:
        pytest.skip("Ollama not running")


@pytest.mark.slow
def test_answer_question_stream(sample_models):
    """Test streaming question answering (requires Ollama)."""
    try:
        analyzer = BenchmarkAnalyzer()
        chunks = list(analyzer.answer_question_stream("What is the best model?", sample_models))

        assert len(chunks) > 0
        combined = "".join(chunks)
        assert len(combined) > 0
    except ConnectionError:
        pytest.skip("Ollama not running")


def test_compare_models_not_found(sample_models):
    """Test comparing models that don't exist."""
    try:
        analyzer = BenchmarkAnalyzer()
        result = analyzer.compare_models(["NonExistent"], sample_models)

        assert "Could not find models" in result.content
    except ConnectionError:
        pytest.skip("Ollama not running")


def test_analyze_blogs_empty():
    """Test analyzing empty blog content."""
    try:
        analyzer = BenchmarkAnalyzer()
        result = analyzer.analyze_blogs([])

        assert "No blog content available" in result.content
    except ConnectionError:
        pytest.skip("Ollama not running")
