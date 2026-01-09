"""Ollama client configuration and setup."""

import yaml
from pathlib import Path
from typing import Any

from langchain_ollama import ChatOllama


def load_config() -> dict[str, Any]:
    """
    Load configuration from sources.yaml.

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_path = Path("config/sources.yaml")

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please create config/sources.yaml with Ollama settings."
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_ollama_client(model: str | None = None, temperature: float | None = None) -> ChatOllama:
    """
    Get a configured ChatOllama client.

    Args:
        model: Model name to use (overrides config default)
        temperature: Temperature setting (overrides config default)

    Returns:
        Configured ChatOllama instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ConnectionError: If Ollama service is not running
    """
    config = load_config()
    ollama_config = config.get("ollama", {})

    # Use provided values or fall back to config
    model_name = model or ollama_config.get("model", "gemma3:4b")
    temp = temperature if temperature is not None else ollama_config.get("temperature", 0.7)
    base_url = ollama_config.get("base_url", "http://localhost:11434")

    try:
        client = ChatOllama(
            model=model_name,
            temperature=temp,
            base_url=base_url,
        )

        # Test connection by checking if model exists
        # This will raise an error if Ollama is not running
        return client

    except Exception as e:
        raise ConnectionError(
            f"Failed to connect to Ollama at {base_url}\n"
            f"Error: {str(e)}\n"
            f"Make sure Ollama is running: ollama serve\n"
            f"And the model is available: ollama pull {model_name}"
        )


def check_ollama_available() -> bool:
    """
    Check if Ollama service is available.

    Returns:
        True if Ollama is running and accessible, False otherwise
    """
    try:
        get_ollama_client()
        return True
    except (ConnectionError, FileNotFoundError):
        return False


def get_available_models() -> list[str]:
    """
    Get list of available Ollama models.

    Returns:
        List of model names

    Note:
        This requires the ollama Python package
    """
    try:
        import ollama
        models = ollama.list()
        return [model["name"] for model in models.get("models", [])]
    except Exception:
        return []
