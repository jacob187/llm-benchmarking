"""Pipeline for orchestrating benchmark scraping, aggregation, and analysis."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .data_aggregator import BenchmarkAggregator, ModelBenchmarks
from .llm import AnalysisResult, BenchmarkAnalyzer, check_ollama_available
from .scrapers.base import ScrapedContent
from .scrapers.registry import ScraperRegistry


@dataclass
class PipelineResult:
    """Result from running the benchmark pipeline."""

    models: dict[str, ModelBenchmarks]
    analyses: dict[str, AnalysisResult] = field(default_factory=dict)
    blog_content: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    cache_path: Path | None = None


class BenchmarkPipeline:
    """
    Orchestrates the full benchmark workflow:
    1. Scrape data from sources
    2. Aggregate and normalize
    3. Run LLM analyses
    4. Cache results
    """

    def __init__(self, ollama_model: str | None = None):
        """
        Initialize pipeline.

        Args:
            ollama_model: Ollama model to use for analysis (None uses config default)
        """
        self.ollama_model = ollama_model
        self.registry = ScraperRegistry()
        self.aggregator = BenchmarkAggregator()
        self.analyzer = None  # Lazy-loaded when needed

    def _ensure_analyzer(self) -> BenchmarkAnalyzer:
        """
        Ensure analyzer is initialized (lazy loading).

        Returns:
            BenchmarkAnalyzer instance

        Raises:
            ConnectionError: If Ollama is not available
        """
        if self.analyzer is None:
            if not check_ollama_available():
                raise ConnectionError(
                    "Ollama is not available. Please start Ollama:\n"
                    "  ollama serve\n"
                    f"And ensure the model is installed:\n"
                    f"  ollama pull {self.ollama_model or 'gemma3:4b'}"
                )
            self.analyzer = BenchmarkAnalyzer(model=self.ollama_model)
        return self.analyzer

    def scrape_all(self) -> list[ScrapedContent]:
        """
        Scrape all registered sources.

        Returns:
            List of ScrapedContent from all sources
        """
        scrapers = self.registry.all()
        scraped_contents = []

        for scraper in scrapers:
            print(f"Scraping {scraper.metadata.name}...", end=" ", flush=True)
            result = scraper.safe_scrape()

            if result.error:
                print(f"✗ Error: {result.error}")
            else:
                if result.entries:
                    print(f"✓ Found {len(result.entries)} entries")
                elif result.raw_text:
                    print(f"✓ Extracted {len(result.raw_text)} chars")
                else:
                    print("✓ Success")

            scraped_contents.append(result)

        return scraped_contents

    def run(
        self,
        skip_scrape: bool = False,
        analyses: list[str] | None = None,
        include_blogs: bool = True,
        save_cache: bool = True,
    ) -> PipelineResult:
        """
        Run the full benchmark pipeline.

        Args:
            skip_scrape: If True, load from cache instead of scraping
            analyses: List of analysis types to run (summary, coding, comparison, blog)
                     If None, uses default from config
            include_blogs: Whether to analyze blog content
            save_cache: Whether to save aggregated data to cache

        Returns:
            PipelineResult with all data and analyses
        """
        errors = []
        result = PipelineResult(models={})

        # Step 1: Get data (scrape or load from cache)
        if skip_scrape:
            print("Loading data from cache...")
            if self.aggregator.load_cache():
                print(f"✓ Loaded {len(self.aggregator.models)} models from cache")
                result.models = self.aggregator.models
                result.blog_content = self.aggregator.blog_content
            else:
                print("✗ No cache found, scraping instead...")
                skip_scrape = False

        if not skip_scrape:
            print("\n=== Scraping Sources ===")
            scraped_contents = self.scrape_all()

            # Count successes and failures
            successful = sum(1 for s in scraped_contents if not s.error)
            failed = sum(1 for s in scraped_contents if s.error)
            print(f"\nScraped {successful} sources successfully, {failed} failed")

            # Collect errors
            for content in scraped_contents:
                if content.error:
                    errors.append(f"{content.source}: {content.error}")

            # Step 2: Aggregate data
            print("\n=== Aggregating Data ===")
            models = self.aggregator.aggregate(scraped_contents)
            print(f"✓ Aggregated {len(models)} unique models")

            result.models = models
            result.blog_content = self.aggregator.blog_content

            # Save cache if requested
            if save_cache:
                cache_path = self.aggregator.save_cache()
                result.cache_path = cache_path
                print(f"✓ Saved cache to {cache_path}")

        # Step 3: Run analyses (if any models found)
        if result.models:
            # Determine which analyses to run
            if analyses is None:
                # Load default from config
                try:
                    from .llm.ollama_client import load_config
                    config = load_config()
                    analyses = config.get("analysis", {}).get("default_analyses", ["summary"])
                except Exception:
                    analyses = ["summary"]

            if analyses:
                print(f"\n=== Running Analyses ({', '.join(analyses)}) ===")

                try:
                    analyzer = self._ensure_analyzer()

                    for analysis_type in analyses:
                        print(f"Running {analysis_type} analysis...", end=" ", flush=True)

                        try:
                            if analysis_type == "summary":
                                analysis_result = analyzer.summarize(result.models)
                            elif analysis_type == "coding":
                                analysis_result = analyzer.analyze_coding(result.models)
                            elif analysis_type == "blog" and include_blogs and result.blog_content:
                                analysis_result = analyzer.analyze_blogs(result.blog_content)
                            else:
                                print(f"✗ Unknown or skipped")
                                continue

                            result.analyses[analysis_type] = analysis_result
                            print("✓ Complete")

                        except Exception as e:
                            error_msg = f"Failed to run {analysis_type} analysis: {str(e)}"
                            errors.append(error_msg)
                            print(f"✗ Error: {str(e)}")

                except ConnectionError as e:
                    error_msg = f"Cannot run analyses: {str(e)}"
                    errors.append(error_msg)
                    print(f"✗ {error_msg}")

        else:
            print("\n✗ No models found, skipping analyses")
            errors.append("No models found after aggregation")

        result.errors = errors
        return result

    def compare_models(self, model_names: list[str], use_cache: bool = True) -> AnalysisResult:
        """
        Compare specific models.

        Args:
            model_names: List of model names to compare
            use_cache: Whether to load from cache

        Returns:
            AnalysisResult with comparison

        Raises:
            ValueError: If no models found
            ConnectionError: If Ollama not available
        """
        # Load data
        if use_cache:
            if not self.aggregator.load_cache():
                raise ValueError("No cache found. Run full pipeline first.")
        elif not self.aggregator.models:
            raise ValueError("No models loaded. Run pipeline or use cache.")

        if not self.aggregator.models:
            raise ValueError("No models available for comparison")

        # Run comparison
        analyzer = self._ensure_analyzer()
        return analyzer.compare_models(model_names, self.aggregator.models)

    def answer_question(self, question: str, use_cache: bool = True, stream: bool = False):
        """
        Answer a question about benchmark data.

        Args:
            question: User's question
            use_cache: Whether to load from cache
            stream: Whether to stream the response

        Returns:
            AnalysisResult if not streaming, or iterator of chunks if streaming

        Raises:
            ValueError: If no models found
            ConnectionError: If Ollama not available
        """
        # Load data
        if use_cache:
            if not self.aggregator.load_cache():
                raise ValueError("No cache found. Run full pipeline first.")
        elif not self.aggregator.models:
            raise ValueError("No models loaded. Run pipeline or use cache.")

        if not self.aggregator.models:
            raise ValueError("No models available to answer questions about")

        # Answer question
        analyzer = self._ensure_analyzer()

        if stream:
            return analyzer.answer_question_stream(question, self.aggregator.models)
        else:
            return analyzer.answer_question(question, self.aggregator.models)

    def get_top_models(self, n: int = 10, benchmark: str | None = None) -> list[ModelBenchmarks]:
        """
        Get top N models.

        Args:
            n: Number of models to return
            benchmark: Specific benchmark to sort by (None uses average)

        Returns:
            List of top ModelBenchmarks
        """
        if not self.aggregator.models:
            # Try loading from cache
            if not self.aggregator.load_cache():
                return []

        return self.aggregator.get_top_models(n=n, benchmark=benchmark)
