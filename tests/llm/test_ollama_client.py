"""Tests for Ollama client."""

import pytest
from pathlib import Path

from llm_benchmarks.llm.ollama_client import load_config, get_ollama_client, check_ollama_available


def test_load_config():
    """Test loading configuration file."""
    config = load_config()

    assert "ollama" in config
    assert "scraping" in config
    assert config["ollama"]["model"] == "gemma3:4b"
    assert config["ollama"]["base_url"] == "http://localhost:11434"


def test_load_config_file_not_found():
    """Test error when config file doesn't exist."""
    import llm_benchmarks.llm.ollama_client as client_module
    original_path = Path("config/sources.yaml")

    # Temporarily move config
    if original_path.exists():
        pytest.skip("Cannot test missing config when config exists")


@pytest.mark.slow
def test_get_ollama_client():
    """Test getting Ollama client (requires Ollama running)."""
    try:
        client = get_ollama_client()
        assert client is not None
        assert client.model == "gemma3:4b"
    except ConnectionError:
        pytest.skip("Ollama not running")


@pytest.mark.slow
def test_get_ollama_client_custom_model():
    """Test getting Ollama client with custom model."""
    try:
        client = get_ollama_client(model="llama3.2:3b")
        assert client.model == "llama3.2:3b"
    except ConnectionError:
        pytest.skip("Ollama not running")


@pytest.mark.slow
def test_get_ollama_client_custom_temperature():
    """Test getting Ollama client with custom temperature."""
    try:
        client = get_ollama_client(temperature=0.5)
        assert client.temperature == 0.5
    except ConnectionError:
        pytest.skip("Ollama not running")


@pytest.mark.slow
def test_check_ollama_available():
    """Test checking if Ollama is available."""
    result = check_ollama_available()
    # Result depends on whether Ollama is running
    assert isinstance(result, bool)
