"""Report generation from benchmark data."""

from datetime import datetime

from ..data_aggregator import ModelBenchmarks
from ..llm import AnalysisResult
from .models import Report


class ReportGenerator:
    """Generates structured report data from benchmark results."""

    def generate(
        self,
        models: dict[str, ModelBenchmarks],
        analyses: dict[str, AnalysisResult] | None = None,
        blog_insights: AnalysisResult | None = None,
        title: str | None = None,
    ) -> Report:
        """
        Generate a structured benchmark report.

        Args:
            models: Dictionary of ModelBenchmarks
            analyses: Dictionary of analysis results (summary, coding, etc.)
            blog_insights: Blog analysis result
            title: Report title (auto-generated if None)

        Returns:
            Report object with structured data
        """
        # Create report with title
        if title is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
            title = f"LLM Benchmark Report - {date_str}"

        report = Report(title=title)

        # Add metadata
        report.metadata = {
            "models_count": len(models),
            "sources": list(self._get_unique_sources(models)),
            "generated_at": report.generated_at.isoformat(),
        }

        # Add executive summary if available
        if analyses and "summary" in analyses:
            report.add_section(
                "Executive Summary",
                analyses["summary"].content,
                level=2
            )

        # Add top performers table
        top_models = self._get_top_models(models, n=10)
        if top_models:
            headers = ["Rank", "Model", "Average Score", "# Benchmarks", "Sources"]
            rows = []
            for i, model in enumerate(top_models, 1):
                rows.append([
                    str(i),
                    model.name,
                    f"{model.average_score:.1f}" if model.average_score else "N/A",
                    str(len(model.benchmarks)),
                    ", ".join(sorted(model.sources)),
                ])

            report.add_section("Top Performing Models", "", level=2)
            report.add_table(headers, rows, f"Top {len(rows)} Models by Average Score")

        # Add coding analysis if available
        if analyses and "coding" in analyses:
            report.add_section(
                "Coding Capabilities Analysis",
                analyses["coding"].content,
                level=2
            )

            # Add coding-focused table
            coding_data = self._get_coding_data(models)
            if coding_data:
                report.add_table(
                    coding_data["headers"],
                    coding_data["rows"],
                    "Coding Benchmark Performance"
                )

        # Add blog insights if available
        if blog_insights:
            report.add_section(
                "Industry Insights",
                blog_insights.content,
                level=2
            )

        # Add detailed benchmark data section
        report.add_section(
            "Detailed Benchmark Data",
            "Complete benchmark scores for all models (expandable)",
            level=2
        )

        # Store detailed model data for formatters to use
        report.metadata["detailed_models"] = {
            name: model.to_dict() for name, model in models.items()
        }

        # Add sources
        report.add_section(
            "Data Sources",
            f"Data collected from: {', '.join(sorted(self._get_unique_sources(models)))}",
            level=2
        )

        return report

    def _get_unique_sources(self, models: dict[str, ModelBenchmarks]) -> set[str]:
        """Get unique sources from models."""
        sources = set()
        for model in models.values():
            sources.update(model.sources)
        return sources

    def _get_top_models(self, models: dict[str, ModelBenchmarks], n: int = 10) -> list[ModelBenchmarks]:
        """Get top N models by average score."""
        sorted_models = sorted(
            [m for m in models.values() if m.average_score is not None],
            key=lambda m: m.average_score or 0,
            reverse=True
        )
        return sorted_models[:n]

    def _get_coding_data(self, models: dict[str, ModelBenchmarks]) -> dict | None:
        """Get coding benchmark data for table."""
        # Find coding-related benchmarks
        coding_keywords = ["code", "humaneval", "mbpp", "coding", "programming"]
        coding_benchmarks = set()

        for model in models.values():
            for bench_name in model.benchmarks.keys():
                if any(keyword in bench_name.lower() for keyword in coding_keywords):
                    coding_benchmarks.add(bench_name)

        if not coding_benchmarks:
            return None

        # Get models with coding benchmarks
        models_with_coding = [
            m for m in models.values()
            if any(b in m.benchmarks for b in coding_benchmarks)
        ]

        if not models_with_coding:
            return None

        # Sort by average of coding benchmarks
        def coding_avg(model: ModelBenchmarks) -> float:
            scores = [
                model.benchmarks[b]
                for b in coding_benchmarks
                if b in model.benchmarks
            ]
            return sum(scores) / len(scores) if scores else 0

        sorted_models = sorted(models_with_coding, key=coding_avg, reverse=True)[:10]

        headers = ["Model"] + list(sorted(coding_benchmarks))
        rows = []

        for model in sorted_models:
            row = [model.name]
            for bench in sorted(coding_benchmarks):
                score = model.benchmarks.get(bench)
                row.append(f"{score:.1f}" if score else "-")
            rows.append(row)

        return {"headers": headers, "rows": rows}
