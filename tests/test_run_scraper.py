"""Tests for the run_scraper data loader function."""

from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest

from src.frontend.streamlit.utils.data_loader import run_scraper


# Patch where BenchmarkPipeline is imported (data_loader module), not where
# it's defined (pipeline module). This ensures each test gets its own mock.
@patch("src.frontend.streamlit.utils.data_loader.BenchmarkPipeline")
def test_run_scraper_returns_result(mock_pipeline_cls):
    """Test that run_scraper calls the pipeline and returns a summary dict."""
    mock_result = MagicMock()
    mock_result.models = {"model_a": object(), "model_b": object()}
    mock_result.errors = []
    mock_result.cache_path = Path("data/processed/cache.json")

    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = mock_result
    mock_pipeline_cls.return_value = mock_pipeline

    result = run_scraper()

    # Pipeline was called with scrape-only args (no analyses)
    mock_pipeline.run.assert_called_once_with(
        skip_scrape=False,
        analyses=[],
        save_cache=True,
    )
    assert result["models_count"] == 2
    assert result["errors"] == []
    assert result["cache_path"] == "data/processed/cache.json"


@patch("src.frontend.streamlit.utils.data_loader.BenchmarkPipeline")
def test_run_scraper_with_errors(mock_pipeline_cls):
    """Test that scraper errors are passed through in the result."""
    mock_result = MagicMock()
    mock_result.models = {"model_a": object()}
    mock_result.errors = ["Source X timed out", "Source Y returned 403"]
    mock_result.cache_path = None

    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = mock_result
    mock_pipeline_cls.return_value = mock_pipeline

    result = run_scraper()

    assert result["models_count"] == 1
    assert len(result["errors"]) == 2
    assert result["cache_path"] is None
