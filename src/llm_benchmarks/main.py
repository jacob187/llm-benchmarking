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
    show_history: bool = typer.Option(
        False,
        "--show-history",
        help="Show historical score trends after comparison",
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

        # Show historical context if requested
        if show_history:
            try:
                from .database import HistoryRepository

                repo = HistoryRepository()

                console.print("[bold cyan]Historical Context (Last 5 Runs)[/bold cyan]\n")

                # Show average score trends for each model
                for model_name in model_list:
                    history = repo.get_score_history(
                        model_name=model_name, benchmark_name=None, limit=5
                    )

                    if history:
                        console.print(f"[bold]{model_name}:[/bold]")
                        table = Table(show_header=True, box=None, padding=(0, 2))
                        table.add_column("Date", style="dim")
                        table.add_column("Score", justify="right")
                        table.add_column("Change", justify="right")

                        for entry in history:
                            date_str = entry.date.strftime("%Y-%m-%d")
                            score_str = f"{entry.score:.1f}"

                            if entry.change is not None:
                                if entry.change > 0:
                                    change_str = f"[green]+{entry.change:.1f}[/green]"
                                elif entry.change < 0:
                                    change_str = f"[red]{entry.change:.1f}[/red]"
                                else:
                                    change_str = "[dim]0.0[/dim]"
                            else:
                                change_str = "[dim]-[/dim]"

                            table.add_row(date_str, score_str, change_str)

                        console.print(table)
                        console.print()
                    else:
                        console.print(f"[bold]{model_name}:[/bold] [dim]No historical data[/dim]\n")

            except Exception as e:
                console.print(f"[yellow]⚠ Could not load history: {str(e)}[/yellow]\n")

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
    history: bool = typer.Option(
        False,
        "--history",
        help="Show trend indicators from previous run",
    ),
):
    """Show top performing models."""
    console.print(f"\n[bold cyan]Top {n} Models[/bold cyan]\n")

    pipeline = BenchmarkPipeline()
    top_models = pipeline.get_top_models(n=n, benchmark=benchmark)

    if not top_models:
        console.print("[yellow]No models found. Run 'llm-bench scrape' first.[/yellow]\n")
        return

    # Get previous rankings if history flag is set
    previous_rankings = {}
    if history:
        try:
            from .database import HistoryRepository

            repo = HistoryRepository()
            previous_rankings = repo.get_previous_run_rankings(benchmark_name=benchmark)
        except Exception as e:
            console.print(f"[yellow]⚠ Could not load history: {str(e)}[/yellow]\n")

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

    if history:
        table.add_column("Change", justify="center")

    table.add_column("# Benchmarks", justify="right")
    table.add_column("Sources", style="dim")

    for i, model in enumerate(top_models, 1):
        if benchmark and benchmark in model.benchmarks:
            score = f"{model.benchmarks[benchmark]:.1f}"
        else:
            score = f"{model.average_score:.1f}" if model.average_score else "N/A"

        # Calculate rank change if history enabled
        change_str = ""
        if history and previous_rankings:
            prev_rank = previous_rankings.get(model.name)
            if prev_rank is not None:
                rank_change = prev_rank - i  # Positive = improved
                if rank_change > 0:
                    change_str = f"[green]↑ +{rank_change}[/green]"
                elif rank_change < 0:
                    change_str = f"[red]↓ {rank_change}[/red]"
                else:
                    change_str = "[dim]→[/dim]"
            else:
                change_str = "[cyan]NEW[/cyan]"

        row_data = [
            str(i),
            model.name,
            score,
        ]

        if history:
            row_data.append(change_str if change_str else "[dim]-[/dim]")

        row_data.extend([
            str(len(model.benchmarks)),
            ", ".join(sorted(model.sources)),
        ])

        table.add_row(*row_data)

    console.print(table)
    console.print()


# ====== History Command Group ======

history_app = typer.Typer(
    name="history",
    help="Query historical benchmark data and trends",
)
app.add_typer(history_app, name="history")


@history_app.command()
def trends(
    model_name: str = typer.Argument(..., help="Model name to query"),
    benchmark: Optional[str] = typer.Option(
        None, "--benchmark", "-b", help="Specific benchmark to show"
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of results to show"),
):
    """Show score trends over time for a model."""
    from .database import HistoryRepository

    console.print(f"\n[bold cyan]Score Trends: {model_name}[/bold cyan]\n")

    try:
        repo = HistoryRepository()
        history = repo.get_score_history(
            model_name=model_name, benchmark_name=benchmark, limit=limit
        )

        if not history:
            console.print(f"[yellow]No historical data found for '{model_name}'[/yellow]")
            return

        # Create table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Date", style="cyan")
        table.add_column("Benchmark" if not benchmark else "Score", justify="right")
        if not benchmark:
            table.add_column("Score", justify="right")
        table.add_column("Change", justify="right")

        for entry in history:
            date_str = entry.date.strftime("%Y-%m-%d")
            score_str = f"{entry.score:.1f}"

            # Format change with color
            if entry.change is not None:
                if entry.change > 0:
                    change_str = f"[green]+{entry.change:.1f}[/green]"
                elif entry.change < 0:
                    change_str = f"[red]{entry.change:.1f}[/red]"
                else:
                    change_str = "[dim]0.0[/dim]"
            else:
                change_str = "[dim]-[/dim]"

            if benchmark:
                table.add_row(date_str, score_str, change_str)
            else:
                # When showing all benchmarks, we'd need to query differently
                # For now, just show the score and source
                table.add_row(date_str, entry.source, score_str, change_str)

        console.print(table)
        console.print()

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@history_app.command()
def compare(
    models: str = typer.Argument(..., help="Comma-separated list of model names"),
    benchmark: str = typer.Option(..., "--benchmark", "-b", help="Benchmark to compare on"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of results per model"),
):
    """Compare score evolution for multiple models."""
    from .database import HistoryRepository

    model_list = [m.strip() for m in models.split(",")]

    console.print(f"\n[bold cyan]Score Comparison: {benchmark}[/bold cyan]\n")

    try:
        repo = HistoryRepository()
        comparison = repo.compare_scores_over_time(
            model_names=model_list, benchmark_name=benchmark, limit=limit
        )

        if not any(comparison.values()):
            console.print(f"[yellow]No historical data found for comparison[/yellow]")
            return

        # Create table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Date", style="cyan")

        for model_name in model_list:
            table.add_column(model_name, justify="right")

        # Collect all unique dates
        all_dates = set()
        for histories in comparison.values():
            all_dates.update(h.date for h in histories)

        # Sort dates
        sorted_dates = sorted(all_dates, reverse=True)[:limit]

        # Build rows
        for date in sorted_dates:
            row = [date.strftime("%Y-%m-%d")]

            for model_name in model_list:
                histories = comparison.get(model_name, [])
                # Find score for this date
                score_entry = next((h for h in histories if h.date == date), None)

                if score_entry:
                    row.append(f"{score_entry.score:.1f}")
                else:
                    row.append("[dim]-[/dim]")

            table.add_row(*row)

        console.print(table)
        console.print()

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@history_app.command()
def rankings(
    model_name: str = typer.Argument(..., help="Model name to query"),
    benchmark: Optional[str] = typer.Option(
        None, "--benchmark", "-b", help="Specific benchmark (default: average ranking)"
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of results to show"),
):
    """Show ranking changes over time for a model."""
    from .database import HistoryRepository

    console.print(f"\n[bold cyan]Ranking History: {model_name}[/bold cyan]\n")
    if benchmark:
        console.print(f"Benchmark: {benchmark}\n")
    else:
        console.print("Showing average rankings across all benchmarks\n")

    try:
        repo = HistoryRepository()
        history = repo.get_ranking_history(
            model_name=model_name, benchmark_name=benchmark, limit=limit
        )

        if not history:
            console.print(f"[yellow]No ranking history found for '{model_name}'[/yellow]")
            return

        # Create table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Date", style="cyan")
        table.add_column("Rank", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Change", justify="right")

        for entry in history:
            date_str = entry.date.strftime("%Y-%m-%d")
            rank_str = f"#{entry.rank}"
            score_str = f"{entry.score:.1f}"

            # Format change with arrows and color
            if entry.change is not None:
                if entry.change > 0:
                    change_str = f"[green]↑ +{entry.change}[/green]"
                elif entry.change < 0:
                    change_str = f"[red]↓ {entry.change}[/red]"
                else:
                    change_str = "[dim]→ same[/dim]"
            else:
                change_str = "[dim]-[/dim]"

            table.add_row(date_str, rank_str, score_str, change_str)

        console.print(table)
        console.print()

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@history_app.command(name="new-models")
def new_models(
    since: Optional[str] = typer.Option(
        None, "--since", "-s", help="Show models first seen after this date (YYYY-MM-DD)"
    ),
    limit: int = typer.Option(20, "--limit", "-l", help="Number of results to show"),
):
    """List recently discovered models."""
    from datetime import datetime

    from .database import HistoryRepository

    console.print("\n[bold cyan]Recently Discovered Models[/bold cyan]\n")

    try:
        repo = HistoryRepository()

        # Parse since date if provided
        since_date = None
        if since:
            try:
                since_date = datetime.fromisoformat(since)
            except ValueError:
                console.print(f"[red]Invalid date format: {since}. Use YYYY-MM-DD[/red]")
                raise typer.Exit(1)

        new_models_list = repo.get_new_models(since_date=since_date, limit=limit)

        if not new_models_list:
            console.print("[yellow]No new models found[/yellow]")
            return

        # Create table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Model Name", style="cyan")
        table.add_column("First Seen", justify="center")
        table.add_column("Initial Score", justify="right")
        table.add_column("Benchmark", style="dim")

        for model in new_models_list:
            first_seen_str = model.first_seen.strftime("%Y-%m-%d")
            score_str = f"{model.initial_score:.1f}" if model.initial_score else "[dim]-[/dim]"
            benchmark_str = model.benchmark_name or "[dim]-[/dim]"

            table.add_row(model.model_name, first_seen_str, score_str, benchmark_str)

        console.print(table)
        console.print()

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@history_app.command()
def stats():
    """Show database statistics."""
    from .database import HistoryRepository

    console.print("\n[bold cyan]Database Statistics[/bold cyan]\n")

    try:
        repo = HistoryRepository()
        db_stats = repo.get_database_stats()

        # Create info table
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="bold")
        table.add_column("Value", style="cyan")

        table.add_row("Total Runs", str(db_stats.total_runs))
        table.add_row(
            "Successful Runs",
            f"{db_stats.successful_runs} ({db_stats.successful_runs / max(db_stats.total_runs, 1) * 100:.1f}%)",
        )
        table.add_row("Total Models Tracked", str(db_stats.total_models))
        table.add_row("Total Benchmarks", str(db_stats.total_benchmarks))
        table.add_row("Total Score Records", str(db_stats.total_scores))

        if db_stats.first_run_date and db_stats.last_run_date:
            table.add_row(
                "First Run",
                db_stats.first_run_date.strftime("%Y-%m-%d %H:%M"),
            )
            table.add_row(
                "Last Run",
                db_stats.last_run_date.strftime("%Y-%m-%d %H:%M"),
            )
            table.add_row("Date Range", f"{db_stats.date_range_days} days")

        console.print(table)
        console.print()

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command(name="import-cache")
def import_cache(
    cache_file: str = typer.Option(
        "data/processed/cache.json", "--cache-file", "-f", help="Cache file to import"
    ),
    run_date: Optional[str] = typer.Option(
        None, "--run-date", "-d", help="Override run date (YYYY-MM-DD HH:MM)"
    ),
):
    """Import existing cache.json as historical baseline."""
    import json
    from datetime import datetime
    from pathlib import Path

    from .data_aggregator import ModelBenchmarks
    from .database import DatabaseManager
    from .pipeline import PipelineResult

    console.print("\n[bold cyan]Importing Cache to Database[/bold cyan]\n")

    # Load cache file
    cache_path = Path(cache_file)
    if not cache_path.exists():
        console.print(f"[red]Cache file not found: {cache_file}[/red]")
        raise typer.Exit(1)

    try:
        with open(cache_path) as f:
            cache_data = json.load(f)

        # Parse run date
        if run_date:
            try:
                timestamp = datetime.fromisoformat(run_date)
            except ValueError:
                console.print(f"[red]Invalid date format: {run_date}. Use YYYY-MM-DD HH:MM[/red]")
                raise typer.Exit(1)
        else:
            # Use last_updated from cache, or file modification time
            if "last_updated" in cache_data:
                timestamp = datetime.fromisoformat(cache_data["last_updated"])
            else:
                timestamp = datetime.fromtimestamp(cache_path.stat().st_mtime)

        console.print(f"Import timestamp: {timestamp.strftime('%Y-%m-%d %H:%M')}")

        # Convert cache data to ModelBenchmarks
        models_dict = {}
        for normalized_name, model_data in cache_data.get("models", {}).items():
            models_dict[normalized_name] = ModelBenchmarks(
                name=model_data["name"],
                benchmarks=model_data["benchmarks"],
                sources=set(model_data["sources"]),
                average_score=model_data.get("average_score"),
            )

        console.print(f"Found {len(models_dict)} models in cache")

        # Create PipelineResult
        result = PipelineResult(
            models=models_dict,
            analyses={},
            blog_content=cache_data.get("blog_content", []),
            errors=[],
            timestamp=timestamp,
            cache_path=cache_path,
        )

        # Save to database
        db_manager = DatabaseManager()

        # Check if already imported
        from .database.connection import DatabaseConnection
        with DatabaseConnection() as conn:
            conn.execute(
                """
                SELECT COUNT(*) as count FROM runs
                WHERE status = 'imported' AND timestamp = ?
                """,
                (timestamp.isoformat(),),
            )
            existing = conn.fetchone()

            if existing and existing["count"] > 0:
                console.print(f"[yellow]⚠ Cache from {timestamp.strftime('%Y-%m-%d %H:%M')} already imported[/yellow]")
                console.print("Skipping import to avoid duplicates")
                return

        # Modify status to 'imported' in the run record
        run_id = db_manager.record_run(result)

        # Update run status to 'imported'
        with DatabaseConnection() as conn:
            conn.execute(
                "UPDATE runs SET status = 'imported' WHERE id = ?",
                (run_id,),
            )

        console.print(f"\n[green]✓ Successfully imported cache to database (run_id: {run_id})[/green]")
        console.print(f"  Models imported: {len(models_dict)}")
        console.print(f"  Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M')}\n")

    except json.JSONDecodeError as e:
        console.print(f"[red]Error parsing cache file: {str(e)}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error importing cache: {str(e)}[/red]")
        raise typer.Exit(1)


@app.callback()
def main():
    """LLM Benchmark Tracker - Track and analyze LLM performance."""
    pass


if __name__ == "__main__":
    app()
