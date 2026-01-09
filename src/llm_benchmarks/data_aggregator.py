"""Data aggregation and normalization for benchmark results."""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .scrapers.base import ScrapedContent


@dataclass
class ModelBenchmarks:
    """Aggregated benchmark data for a single model."""
    name: str
    benchmarks: dict[str, float] = field(default_factory=dict)
    sources: set[str] = field(default_factory=set)
    average_score: float | None = None

    def add_benchmark(self, benchmark_name: str, score: float, source: str) -> None:
        """Add a benchmark score."""
        self.benchmarks[benchmark_name] = score
        self.sources.add(source)

    def calculate_average(self) -> None:
        """Calculate average score across all benchmarks."""
        if self.benchmarks:
            self.average_score = sum(self.benchmarks.values()) / len(self.benchmarks)
        else:
            self.average_score = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "benchmarks": self.benchmarks,
            "sources": list(self.sources),
            "average_score": self.average_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelBenchmarks":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            benchmarks=data["benchmarks"],
            sources=set(data["sources"]),
            average_score=data.get("average_score"),
        )


class BenchmarkAggregator:
    """Aggregates and normalizes benchmark data from multiple sources."""

    def __init__(self):
        """Initialize aggregator."""
        self.models: dict[str, ModelBenchmarks] = {}
        self.blog_content: list[dict[str, Any]] = []
        self.structured_data: dict[str, Any] = {}
        self.last_updated: datetime | None = None

    @staticmethod
    def normalize_model_name(name: str) -> str:
        """
        Normalize model name for consistent aggregation.

        Args:
            name: Raw model name

        Returns:
            Normalized name (lowercase, alphanumeric only)
        """
        # Convert to lowercase
        normalized = name.lower()

        # Remove common prefixes/suffixes
        normalized = re.sub(r'^(anthropic|openai|google|meta|mistral)[-/]?', '', normalized)

        # Remove version indicators in parentheses
        normalized = re.sub(r'\([^)]*\)', '', normalized)

        # Replace hyphens and underscores with spaces
        normalized = normalized.replace('-', ' ').replace('_', ' ')

        # Remove extra whitespace
        normalized = ' '.join(normalized.split())

        # Create a slug-like identifier
        slug = re.sub(r'[^a-z0-9]+', '_', normalized).strip('_')

        return slug

    def aggregate(self, scraped_contents: list[ScrapedContent]) -> dict[str, ModelBenchmarks]:
        """
        Aggregate data from multiple scraped sources.

        Args:
            scraped_contents: List of ScrapedContent from various scrapers

        Returns:
            Dictionary mapping normalized model names to ModelBenchmarks
        """
        self.models = {}
        self.blog_content = []
        self.structured_data = {}
        self.last_updated = datetime.now()

        for content in scraped_contents:
            # Skip errored scrapes
            if content.error:
                print(f"Warning: Skipping {content.source} due to error: {content.error}")
                continue

            # Process benchmark entries (from LEADERBOARD sources)
            for entry in content.entries:
                normalized_name = self.normalize_model_name(entry.model_name)

                # Get or create ModelBenchmarks
                if normalized_name not in self.models:
                    self.models[normalized_name] = ModelBenchmarks(
                        name=entry.model_name  # Keep original name as display name
                    )

                # Add benchmark score
                self.models[normalized_name].add_benchmark(
                    benchmark_name=entry.benchmark_name,
                    score=entry.score,
                    source=content.source
                )

            # Process blog content (from BLOG sources)
            if content.raw_text:
                self.blog_content.append({
                    "source": content.source,
                    "timestamp": content.timestamp.isoformat(),
                    "content": content.raw_text
                })

            # Process structured data (from RESEARCH sources)
            if content.structured_data:
                self.structured_data[content.source] = content.structured_data

        # Calculate averages for all models
        for model in self.models.values():
            model.calculate_average()

        return self.models

    def get_top_models(self, n: int = 10, benchmark: str | None = None) -> list[ModelBenchmarks]:
        """
        Get top N models by score.

        Args:
            n: Number of models to return
            benchmark: Specific benchmark to sort by (if None, uses average)

        Returns:
            List of top models
        """
        if benchmark:
            # Sort by specific benchmark
            models_with_benchmark = [
                m for m in self.models.values()
                if benchmark in m.benchmarks
            ]
            sorted_models = sorted(
                models_with_benchmark,
                key=lambda m: m.benchmarks[benchmark],
                reverse=True
            )
        else:
            # Sort by average score
            models_with_average = [
                m for m in self.models.values()
                if m.average_score is not None
            ]
            sorted_models = sorted(
                models_with_average,
                key=lambda m: m.average_score or 0,
                reverse=True
            )

        return sorted_models[:n]

    def save_cache(self, filename: str = "cache.json") -> Path:
        """
        Save aggregated data to cache file.

        Args:
            filename: Cache filename (saved in data/processed/)

        Returns:
            Path to saved cache file
        """
        cache_dir = Path("data/processed")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / filename

        cache_data = {
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "models": {name: model.to_dict() for name, model in self.models.items()},
            "blog_content": self.blog_content,
            "structured_data": self.structured_data,
        }

        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)

        return cache_path

    def load_cache(self, filename: str = "cache.json") -> bool:
        """
        Load aggregated data from cache file.

        Args:
            filename: Cache filename (loaded from data/processed/)

        Returns:
            True if cache loaded successfully, False otherwise
        """
        cache_path = Path("data/processed") / filename

        if not cache_path.exists():
            return False

        try:
            with open(cache_path, "r") as f:
                cache_data = json.load(f)

            self.last_updated = (
                datetime.fromisoformat(cache_data["last_updated"])
                if cache_data.get("last_updated")
                else None
            )
            self.models = {
                name: ModelBenchmarks.from_dict(data)
                for name, data in cache_data.get("models", {}).items()
            }
            self.blog_content = cache_data.get("blog_content", [])
            self.structured_data = cache_data.get("structured_data", {})

            return True
        except Exception as e:
            print(f"Error loading cache: {e}")
            return False

    def get_model(self, name: str) -> ModelBenchmarks | None:
        """
        Get a model by normalized name.

        Args:
            name: Model name (will be normalized)

        Returns:
            ModelBenchmarks or None if not found
        """
        normalized = self.normalize_model_name(name)
        return self.models.get(normalized)

    def get_models_by_source(self, source: str) -> list[ModelBenchmarks]:
        """
        Get all models that have data from a specific source.

        Args:
            source: Source name

        Returns:
            List of models with data from that source
        """
        return [
            model for model in self.models.values()
            if source in model.sources
        ]
