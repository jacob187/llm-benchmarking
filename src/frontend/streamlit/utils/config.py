"""Configuration management for Streamlit dashboard."""

import streamlit as st
import yaml
from pathlib import Path
from typing import Any, Optional


@st.cache_resource
def load_app_config() -> dict[str, Any]:
    """
    Load configuration from config/sources.yaml.

    Returns:
        dict: Configuration dictionary with defaults
    """
    # Try to find config file relative to project root
    config_paths = [
        Path("config/sources.yaml"),
        Path(__file__).parents[5] / "config" / "sources.yaml",
        Path.cwd() / "config" / "sources.yaml",
    ]

    config = None
    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                    break
            except Exception as e:
                st.warning(f"Could not load config from {config_path}: {e}")

    # Return default config if not found
    if config is None:
        config = {
            "ollama": {
                "base_url": "http://localhost:11434",
                "model": "gemma3:4b",
                "temperature": 0.7,
                "max_tokens": 2000,
            },
            "database": {
                "enabled": True,
                "path": "data/history.db",
                "auto_migrate": True,
            },
            "scraping": {
                "timeout": 30000,
                "cache_ttl": 3600,
            },
            "analysis": {
                "top_models_count": 10,
            },
        }

    return config


def get_ollama_config() -> dict[str, Any]:
    """
    Get Ollama configuration.

    Returns:
        dict: Ollama settings
    """
    config = load_app_config()
    return config.get("ollama", {
        "base_url": "http://localhost:11434",
        "model": "gemma3:4b",
        "temperature": 0.7,
    })


def get_database_path() -> str:
    """
    Get database path from configuration.

    Returns:
        str: Path to database file
    """
    config = load_app_config()
    return config.get("database", {}).get("path", "data/history.db")


def get_top_models_count() -> int:
    """
    Get default number of top models to display.

    Returns:
        int: Number of models to show
    """
    config = load_app_config()
    return config.get("analysis", {}).get("top_models_count", 10)


def get_cache_ttl() -> int:
    """
    Get cache TTL in seconds.

    Returns:
        int: Cache TTL
    """
    config = load_app_config()
    return config.get("scraping", {}).get("cache_ttl", 3600)


def save_config(config: dict[str, Any], config_path: Optional[Path] = None):
    """
    Save configuration to file.

    Args:
        config: Configuration dictionary to save
        config_path: Optional custom config path
    """
    if config_path is None:
        config_path = Path("config/sources.yaml")

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        st.success(f"Configuration saved to {config_path}")
        # Clear cache to reload config
        st.cache_resource.clear()
    except Exception as e:
        st.error(f"Could not save configuration: {e}")
