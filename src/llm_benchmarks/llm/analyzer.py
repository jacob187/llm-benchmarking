"""LLM-powered analysis of benchmark data."""

from dataclasses import dataclass
from typing import Iterator

from langchain_core.output_parsers import StrOutputParser

from .ollama_client import get_ollama_client
from .prompts import get_template


@dataclass
class AnalysisResult:
    """Result of an LLM analysis."""
    analysis_type: str
    content: str
    model_used: str


class BenchmarkAnalyzer:
    """Analyzes benchmark data using LLM."""

    def __init__(self, model: str | None = None):
        """
        Initialize analyzer.

        Args:
            model: Ollama model to use (None uses config default)
        """
        self.llm = get_ollama_client(model=model)
        self.model_name = model or "gemma3:4b"

    def _format_benchmark_data(self, models: dict) -> str:
        """
        Format model benchmark data as markdown for LLM consumption.

        Args:
            models: Dictionary of ModelBenchmarks

        Returns:
            Formatted markdown string
        """
        if not models:
            return "No benchmark data available."

        lines = []
        lines.append("# LLM Benchmark Data\n")

        # Sort models by average score
        sorted_models = sorted(
            models.values(),
            key=lambda m: m.average_score or 0,
            reverse=True
        )

        for model in sorted_models[:50]:  # Limit to top 50 to avoid context overflow
            lines.append(f"## {model.name}")
            if model.average_score:
                lines.append(f"**Average Score:** {model.average_score:.1f}")

            lines.append("\n**Benchmarks:**")
            for bench_name, score in model.benchmarks.items():
                lines.append(f"- {bench_name}: {score}")

            lines.append(f"\n**Sources:** {', '.join(model.sources)}")
            lines.append("")

        return "\n".join(lines)

    def _format_blog_content(self, blog_content: list[dict]) -> str:
        """
        Format blog content for LLM consumption.

        Args:
            blog_content: List of blog content dictionaries

        Returns:
            Formatted string
        """
        if not blog_content:
            return "No blog content available."

        lines = []
        for blog in blog_content:
            lines.append(f"# {blog['source']}")
            lines.append(f"**Date:** {blog['timestamp']}\n")
            lines.append(blog['content'])
            lines.append("\n---\n")

        return "\n".join(lines)

    def summarize(self, models: dict) -> AnalysisResult:
        """
        Generate executive summary of benchmark data.

        Args:
            models: Dictionary of ModelBenchmarks

        Returns:
            AnalysisResult with summary
        """
        template = get_template("summary")
        chain = template | self.llm | StrOutputParser()

        benchmark_data = self._format_benchmark_data(models)
        result = chain.invoke({"benchmark_data": benchmark_data})

        return AnalysisResult(
            analysis_type="summary",
            content=result,
            model_used=self.model_name
        )

    def summarize_stream(self, models: dict) -> Iterator[str]:
        """
        Generate executive summary with streaming output.

        Args:
            models: Dictionary of ModelBenchmarks

        Yields:
            Chunks of the summary as they're generated
        """
        template = get_template("summary")
        chain = template | self.llm | StrOutputParser()

        benchmark_data = self._format_benchmark_data(models)

        for chunk in chain.stream({"benchmark_data": benchmark_data}):
            yield chunk

    def compare_models(self, model_names: list[str], all_models: dict) -> AnalysisResult:
        """
        Compare specific models.

        Args:
            model_names: List of model names to compare
            all_models: Dictionary of all ModelBenchmarks

        Returns:
            AnalysisResult with comparison
        """
        # Filter to only requested models
        models_to_compare = {
            name: model for name, model in all_models.items()
            if any(mn.lower() in model.name.lower() for mn in model_names)
        }

        if not models_to_compare:
            return AnalysisResult(
                analysis_type="comparison",
                content=f"Could not find models matching: {', '.join(model_names)}",
                model_used=self.model_name
            )

        template = get_template("comparison")
        chain = template | self.llm | StrOutputParser()

        benchmark_data = self._format_benchmark_data(models_to_compare)
        result = chain.invoke({
            "model_names": ", ".join(model_names),
            "benchmark_data": benchmark_data
        })

        return AnalysisResult(
            analysis_type="comparison",
            content=result,
            model_used=self.model_name
        )

    def analyze_coding(self, models: dict) -> AnalysisResult:
        """
        Analyze coding capabilities across models.

        Args:
            models: Dictionary of ModelBenchmarks

        Returns:
            AnalysisResult with coding analysis
        """
        template = get_template("coding")
        chain = template | self.llm | StrOutputParser()

        benchmark_data = self._format_benchmark_data(models)
        result = chain.invoke({"benchmark_data": benchmark_data})

        return AnalysisResult(
            analysis_type="coding",
            content=result,
            model_used=self.model_name
        )

    def analyze_blogs(self, blog_content: list[dict]) -> AnalysisResult:
        """
        Analyze blog posts about LLMs.

        Args:
            blog_content: List of blog content dictionaries

        Returns:
            AnalysisResult with blog analysis
        """
        if not blog_content:
            return AnalysisResult(
                analysis_type="blog",
                content="No blog content available to analyze.",
                model_used=self.model_name
            )

        template = get_template("blog")
        chain = template | self.llm | StrOutputParser()

        formatted_content = self._format_blog_content(blog_content)
        result = chain.invoke({"blog_content": formatted_content})

        return AnalysisResult(
            analysis_type="blog",
            content=result,
            model_used=self.model_name
        )

    def answer_question(self, question: str, models: dict) -> AnalysisResult:
        """
        Answer a question about the benchmark data.

        Args:
            question: User's question
            models: Dictionary of ModelBenchmarks

        Returns:
            AnalysisResult with answer
        """
        template = get_template("qa")
        chain = template | self.llm | StrOutputParser()

        benchmark_data = self._format_benchmark_data(models)
        result = chain.invoke({
            "question": question,
            "benchmark_data": benchmark_data
        })

        return AnalysisResult(
            analysis_type="qa",
            content=result,
            model_used=self.model_name
        )

    def answer_question_stream(self, question: str, models: dict) -> Iterator[str]:
        """
        Answer a question with streaming output.

        Args:
            question: User's question
            models: Dictionary of ModelBenchmarks

        Yields:
            Chunks of the answer as they're generated
        """
        template = get_template("qa")
        chain = template | self.llm | StrOutputParser()

        benchmark_data = self._format_benchmark_data(models)

        for chunk in chain.stream({
            "question": question,
            "benchmark_data": benchmark_data
        }):
            yield chunk
