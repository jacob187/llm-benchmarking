"""LLM integration module."""

from .analyzer import AnalysisResult, BenchmarkAnalyzer
from .ollama_client import (
    check_ollama_available,
    get_available_models,
    get_ollama_client,
    load_config,
)
from .prompts import get_template

__all__ = [
    "AnalysisResult",
    "BenchmarkAnalyzer",
    "check_ollama_available",
    "get_available_models",
    "get_ollama_client",
    "get_template",
    "load_config",
]
