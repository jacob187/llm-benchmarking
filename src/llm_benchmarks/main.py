"""CLI for LLM Benchmark Tracker."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .pipeline import BenchmarkPipeline
from .reports import JSONFormatter, MarkdownFormatter, ReportGenerator
from .scrapers.base import SourceType
from .scrapers.registry import ScraperRegistry

app = typer.Typer(
    name="llm-bench",
    help="LLM Benchmark Tracker - Track and analyze LLM performance across sources",
    add_completion=False,
)
console = Console()


@app.command()
def report(
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output filename (auto-generated if not provided)",
    ),
    cached: bool = typer.Option(
        False,
        "--cached",
        "-c",
        help="Use cached data instead of scraping",
    ),
    format: str = typer.Option(
        "markdown",
        "--format",
        "-f",
        help="Output format: markdown, json, html",
    ),
    analyses: str = typer.Option(
        "summary,coding",
        "--analyses",
        "-a",
        help="Comma-separated list of analyses to run: summary,coding,blog",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Ollama model to use for analysis",
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        "-p",
        help="Preview report in terminal instead of saving",
    ),
):
    """
    Generate a comprehensive benchmark report.

    This command scrapes data from all sources, runs LLM analyses,
    and generates a formatted report.
    """
    console.print("\n[bold cyan]LLM Benchmark Report Generator[/bold cyan]\n")

    # Parse analyses
    analysis_list = [a.strip() for a in analyses.split(",") if a.strip()]

    # Run pipeline
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running pipeline...", total=None)

        pipeline = BenchmarkPipeline(ollama_model=model)
        result = pipeline.run(
            skip_scrape=cached,
            analyses=analysis_list,
            include_blogs=True,
        )

        progress.update(task, description="Pipeline complete!")

    # Show summary
    console.print(f"\n[green]✓[/green] Pipeline completed")
    console.print(f"  Models: {len(result.models)}")
    console.print(f"  Analyses: {len(result.analyses)}")
    if result.errors:
        console.print(f"  [yellow]Errors: {len(result.errors)}[/yellow]")

    # Generate report
    console.print("\n[cyan]Generating report...[/cyan]")
    generator = ReportGenerator()
    report = generator.generate(
        models=result.models,
        analyses=result.analyses,
    )

    # Preview or save
    if preview:
        # Show in terminal
        console.print("\n" + "=" * 60)
        md = Markdown(report.to_markdown())
        console.print(md)
        console.print("=" * 60 + "\n")
    else:
        # Save to file
        if format == "markdown":
            formatter = MarkdownFormatter()
        elif format == "json":
            formatter = JSONFormatter()
        elif format == "html":
            from .reports import HTMLFormatter
            formatter = HTMLFormatter()
        else:
            console.print(f"[red]Error: Unknown format '{format}'[/red]")
            raise typer.Exit(1)

        output_path = formatter.save(report, filename=output)
        console.print(f"\n[green]✓[/green] Report saved to: [bold]{output_path}[/bold]\n")


@app.command()
def compare(
    models: str = typer.Argument(..., help="Comma-separated model names to compare"),
    cached: bool = typer.Option(
        True,
        "--cached/--no-cached",
        help="Use cached data",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Ollama model to use for analysis",
    ),
):
    """
    Compare specific models.

    Example: llm-bench compare "GPT-4,Claude 3"
    """
    console.print("\n[bold cyan]Model Comparison[/bold cyan]\n")

    model_list = [m.strip() for m in models.split(",")]

    if len(model_list) < 2:
        console.print("[red]Error: Need at least 2 models to compare[/red]")
        raise typer.Exit(1)

    console.print(f"Comparing: {', '.join(model_list)}\n")

    # Run comparison
    pipeline = BenchmarkPipeline(ollama_model=model)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing...", total=None)
            result = pipeline.compare_models(model_list, use_cache=cached)
            progress.update(task, description="Analysis complete!")

        console.print("\n" + "=" * 60)
        md = Markdown(result.content)
        console.print(md)
        console.print("=" * 60 + "\n")

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except ConnectionError as e:
        console.print(f"[red]Ollama Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def sources(
    type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by source type: leaderboard, blog, research",
    ),
):
    """List all available benchmark sources."""
    console.print("\n[bold cyan]Available Sources[/bold cyan]\n")

    registry = ScraperRegistry()
    registry.discover()

    source_list = registry.list_sources()

    # Filter by type if specified
    if type:
        try:
            source_type = SourceType(type.lower())
            source_list = [s for s in source_list if s.source_type == source_type]
        except ValueError:
            console.print(f"[red]Error: Unknown type '{type}'[/red]")
            console.print("Valid types: leaderboard, blog, research")
            raise typer.Exit(1)

    if not source_list:
        console.print("[yellow]No sources found[/yellow]")
        return

    # Create table
    table = Table(title=f"Found {len(source_list)} sources")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("URL", style="blue")
    table.add_column("Update Frequency", style="green")

    for source in source_list:
        table.add_row(
            source.name,
            source.source_type.value,
            source.url,
            source.update_frequency,
        )

    console.print(table)
    console.print()


@app.command()
def scrape(
    save_raw: bool = typer.Option(
        False,
        "--save-raw",
        help="Save raw HTML to data/raw/",
    ),
):
    """
    Scrape all sources without running analysis.

    Useful for collecting data without using Ollama.
    """
    console.print("\n[bold cyan]Scraping Benchmark Sources[/bold cyan]\n")

    pipeline = BenchmarkPipeline()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Scraping...", total=None)
        result = pipeline.run(
            skip_scrape=False,
            analyses=[],  # No analyses
            save_cache=True,
        )
        progress.update(task, description="Scraping complete!")

    # Show results
    console.print(f"\n[green]✓[/green] Scraping completed")
    console.print(f"  Models collected: {len(result.models)}")
    console.print(f"  Blog sources: {len(result.blog_content)}")

    if result.cache_path:
        console.print(f"  Cache saved: {result.cache_path}")

    if result.errors:
        console.print(f"\n[yellow]Errors encountered:[/yellow]")
        for error in result.errors:
            console.print(f"  - {error}")

    console.print()


@app.command()
def ask(
    question: Optional[str] = typer.Argument(
        None,
        help="Question to ask about the benchmarks",
    ),
    cached: bool = typer.Option(
        True,
        "--cached/--no-cached",
        help="Use cached data",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Ollama model to use",
    ),
):
    """
    Ask questions about benchmark data.

    Interactive mode if no question provided.
    Streams responses in real-time.
    """
    console.print("\n[bold cyan]Benchmark Q&A[/bold cyan]\n")

    pipeline = BenchmarkPipeline(ollama_model=model)

    # Interactive mode
    if question is None:
        console.print("[dim]Ask questions about LLM benchmarks. Type 'exit' to quit.[/dim]\n")

        while True:
            try:
                question = typer.prompt("\n[cyan]Question[/cyan]")

                if question.lower() in ["exit", "quit", "q"]:
                    console.print("\n[dim]Goodbye![/dim]\n")
                    break

                console.print("\n[green]Answer:[/green] ", end="")

                try:
                    for chunk in pipeline.answer_question(question, use_cache=cached, stream=True):
                        console.print(chunk, end="")

                    console.print("\n")

                except Exception as e:
                    console.print(f"\n[red]Error: {e}[/red]\n")

            except (KeyboardInterrupt, EOFError):
                console.print("\n\n[dim]Goodbye![/dim]\n")
                break

    else:
        # Single question mode
        console.print(f"[cyan]Question:[/cyan] {question}\n")
        console.print("[green]Answer:[/green] ", end="")

        try:
            for chunk in pipeline.answer_question(question, use_cache=cached, stream=True):
                console.print(chunk, end="")

            console.print("\n")

        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")
            raise typer.Exit(1)


@app.command()
def top(
    n: int = typer.Option(
        10,
        "--number",
        "-n",
        help="Number of top models to show",
    ),
    benchmark: Optional[str] = typer.Option(
        None,
        "--benchmark",
        "-b",
        help="Specific benchmark to rank by (default: average)",
    ),
):
    """Show top performing models."""
    console.print(f"\n[bold cyan]Top {n} Models[/bold cyan]\n")

    pipeline = BenchmarkPipeline()
    top_models = pipeline.get_top_models(n=n, benchmark=benchmark)

    if not top_models:
        console.print("[yellow]No models found. Run 'llm-bench scrape' first.[/yellow]\n")
        return

    # Create table
    title = f"Top {len(top_models)} Models"
    if benchmark:
        title += f" by {benchmark}"
    else:
        title += " by Average Score"

    table = Table(title=title)
    table.add_column("Rank", style="cyan", justify="right")
    table.add_column("Model", style="bold")
    table.add_column("Score", style="green", justify="right")
    table.add_column("# Benchmarks", justify="right")
    table.add_column("Sources", style="dim")

    for i, model in enumerate(top_models, 1):
        if benchmark and benchmark in model.benchmarks:
            score = f"{model.benchmarks[benchmark]:.1f}"
        else:
            score = f"{model.average_score:.1f}" if model.average_score else "N/A"

        table.add_row(
            str(i),
            model.name,
            score,
            str(len(model.benchmarks)),
            ", ".join(sorted(model.sources)),
        )

    console.print(table)
    console.print()


@app.callback()
def main():
    """LLM Benchmark Tracker - Track and analyze LLM performance."""
    pass


if __name__ == "__main__":
    app()
