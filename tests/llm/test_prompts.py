"""Tests for prompt templates."""

import pytest
from langchain_core.prompts import PromptTemplate

from llm_benchmarks.llm.prompts import (
    get_template,
    SUMMARY_TEMPLATE,
    COMPARISON_TEMPLATE,
    CODING_TEMPLATE,
    BLOG_ANALYSIS_TEMPLATE,
    QA_TEMPLATE,
)


def test_summary_template_exists():
    """Test that summary template is defined."""
    assert SUMMARY_TEMPLATE is not None
    assert isinstance(SUMMARY_TEMPLATE, PromptTemplate)


def test_comparison_template_exists():
    """Test that comparison template is defined."""
    assert COMPARISON_TEMPLATE is not None
    assert isinstance(COMPARISON_TEMPLATE, PromptTemplate)


def test_coding_template_exists():
    """Test that coding template is defined."""
    assert CODING_TEMPLATE is not None
    assert isinstance(CODING_TEMPLATE, PromptTemplate)


def test_blog_analysis_template_exists():
    """Test that blog analysis template is defined."""
    assert BLOG_ANALYSIS_TEMPLATE is not None
    assert isinstance(BLOG_ANALYSIS_TEMPLATE, PromptTemplate)


def test_qa_template_exists():
    """Test that QA template is defined."""
    assert QA_TEMPLATE is not None
    assert isinstance(QA_TEMPLATE, PromptTemplate)


def test_get_template_summary():
    """Test getting summary template by name."""
    template = get_template("summary")
    assert template == SUMMARY_TEMPLATE


def test_get_template_comparison():
    """Test getting comparison template by name."""
    template = get_template("comparison")
    assert template == COMPARISON_TEMPLATE


def test_get_template_coding():
    """Test getting coding template by name."""
    template = get_template("coding")
    assert template == CODING_TEMPLATE


def test_get_template_blog():
    """Test getting blog template by name."""
    template = get_template("blog")
    assert template == BLOG_ANALYSIS_TEMPLATE


def test_get_template_qa():
    """Test getting QA template by name."""
    template = get_template("qa")
    assert template == QA_TEMPLATE


def test_get_template_invalid():
    """Test error when requesting invalid template."""
    with pytest.raises(ValueError, match="Unknown template"):
        get_template("nonexistent")


def test_summary_template_formatting():
    """Test that summary template can be formatted."""
    prompt = SUMMARY_TEMPLATE.format(benchmark_data="Test data")
    assert "Test data" in prompt
    assert "executive summary" in prompt.lower()


def test_comparison_template_formatting():
    """Test that comparison template can be formatted."""
    prompt = COMPARISON_TEMPLATE.format(
        model_names="Model A, Model B",
        benchmark_data="Test data"
    )
    assert "Model A, Model B" in prompt
    assert "Test data" in prompt
    assert "compare" in prompt.lower()


def test_qa_template_formatting():
    """Test that QA template can be formatted."""
    prompt = QA_TEMPLATE.format(
        question="What is the best model?",
        benchmark_data="Test data"
    )
    assert "What is the best model?" in prompt
    assert "Test data" in prompt
