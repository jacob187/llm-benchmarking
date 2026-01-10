# LLM Benchmark Tracker

A modular, extensible tool for tracking and analyzing Large Language Model (LLM) benchmarks across multiple sources. Automatically scrapes benchmark data, aggregates results, and uses local LLMs (via Ollama) to generate insights and comparisons.

## Features

- **Multi-Source Scraping**: Automatically collects benchmark data from various sources (LMSYS, blogs, research papers)
- **Plugin Architecture**: Easy-to-add scrapers with automatic discovery
- **LLM-Powered Analysis**: Uses Ollama to generate insights, summaries, and comparisons
- **Multiple Output Formats**: Generate reports in Markdown, JSON, or HTML
- **Interactive CLI**: Beautiful terminal UI powered by Typer and Rich
- **Caching**: Smart data caching to avoid unnecessary re-scraping
- **Extensible**: Add new sources by simply creating a new scraper file

## Quick Start

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- [Ollama](https://ollama.ai/) for LLM analysis (optional, but recommended)
- Playwright browsers (for web scraping)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd llm-benchmarking

# Install dependencies using uv
uv sync

# Install Playwright browsers
uv run playwright install chromium

# (Optional) Start Ollama and pull a model
ollama serve
ollama pull gemma3:4b
```

### Basic Usage

```bash
# Generate a full benchmark report
uv run llm-bench report

# Generate report with cached data (no re-scraping)
uv run llm-bench report --cached

# Preview report in terminal without saving
uv run llm-bench report --preview

# Generate JSON report
uv run llm-bench report --format json

# Scrape data without analysis
uv run llm-bench scrape

# Compare specific models
uv run llm-bench compare "GPT-4,Claude 3.5"

# Ask questions about the data
uv run llm-bench ask "Which model is best for coding?"

# Interactive Q&A mode
uv run llm-bench ask

# Show top 10 models
uv run llm-bench top

# List available data sources
uv run llm-bench sources
```

## CLI Commands

### `llm-bench report`

Generate a comprehensive benchmark report.

**Options:**
- `--output, -o FILE`: Output filename (auto-generated if not provided)
- `--cached, -c`: Use cached data instead of scraping
- `--format, -f FORMAT`: Output format: `markdown`, `json`, `html` (default: `markdown`)
- `--analyses, -a LIST`: Comma-separated analyses to run: `summary`, `coding`, `blog` (default: `summary,coding`)
- `--model, -m MODEL`: Ollama model to use for analysis
- `--preview, -p`: Preview report in terminal instead of saving

**Example:**
```bash
uv run llm-bench report --format markdown --analyses summary,coding --preview
```

### `llm-bench compare`

Compare specific models side-by-side.

**Arguments:**
- `MODELS`: Comma-separated model names to compare

**Options:**
- `--cached/--no-cached`: Use cached data (default: True)
- `--model, -m MODEL`: Ollama model to use
- `--show-history`: Show historical score trends after comparison

**Example:**
```bash
uv run llm-bench compare "GPT-4o,Claude 3.5 Sonnet,Gemini Pro" --cached
uv run llm-bench compare "GPT-4o,Claude 3.5" --show-history  # With historical context
```

### `llm-bench ask`

Ask questions about benchmark data using natural language.

**Arguments:**
- `QUESTION`: Optional question (starts interactive mode if omitted)

**Options:**
- `--cached/--no-cached`: Use cached data (default: True)
- `--model, -m MODEL`: Ollama model to use

**Example:**
```bash
# Single question
uv run llm-bench ask "What are the top 3 models for coding tasks?"

# Interactive mode
uv run llm-bench ask
```

### `llm-bench top`

Show top performing models.

**Options:**
- `--number, -n N`: Number of top models to show (default: 10)
- `--benchmark, -b NAME`: Specific benchmark to rank by (default: average)
- `--history`: Show trend indicators from previous run

**Example:**
```bash
uv run llm-bench top --number 5 --benchmark "MMLU"
uv run llm-bench top --history  # Show with trend indicators (↑↓→)
```

### `llm-bench sources`

List all available benchmark sources.

**Options:**
- `--type, -t TYPE`: Filter by source type: `leaderboard`, `blog`, `research`

**Example:**
```bash
uv run llm-bench sources --type leaderboard
```

### `llm-bench scrape`

Scrape all sources without running analysis.

**Options:**
- `--save-raw`: Save raw HTML to `data/raw/`

**Example:**
```bash
uv run llm-bench scrape --save-raw
```

## Historical Data Tracking

The system automatically tracks all benchmark data in a SQLite database for historical trend analysis.

### `llm-bench history trends`

Show score trends over time for a model.

**Arguments:**
- `MODEL_NAME`: Model name to query

**Options:**
- `--benchmark, -b NAME`: Specific benchmark to show
- `--limit, -l N`: Number of results to show (default: 10)

**Example:**
```bash
uv run llm-bench history trends "Claude Opus 4.5"
uv run llm-bench history trends "GPT-4o" --benchmark "MMLU" --limit 5
```

### `llm-bench history compare`

Compare score evolution for multiple models on the same benchmark.

**Arguments:**
- `MODELS`: Comma-separated list of model names

**Options:**
- `--benchmark, -b NAME`: Benchmark to compare on (required)
- `--limit, -l N`: Number of results per model (default: 10)

**Example:**
```bash
uv run llm-bench history compare "GPT-4o,Claude Opus 4.5" --benchmark "LMSYS Arena ELO"
```

### `llm-bench history rankings`

Show ranking changes over time for a model.

**Arguments:**
- `MODEL_NAME`: Model name to query

**Options:**
- `--benchmark, -b NAME`: Specific benchmark (default: average ranking)
- `--limit, -l N`: Number of results to show (default: 10)

**Example:**
```bash
uv run llm-bench history rankings "Claude Opus 4.5"
uv run llm-bench history rankings "GPT-4o" --benchmark "MMLU"
```

### `llm-bench history new-models`

List recently discovered models.

**Options:**
- `--since, -s DATE`: Show models first seen after this date (YYYY-MM-DD)
- `--limit, -l N`: Number of results to show (default: 20)

**Example:**
```bash
uv run llm-bench history new-models --since 2025-01-01
uv run llm-bench history new-models --limit 10
```

### `llm-bench history stats`

Show database statistics.

**Example:**
```bash
uv run llm-bench history stats
```

### `llm-bench import-cache`

Import existing cache.json as historical baseline (one-time operation).

**Options:**
- `--cache-file, -f PATH`: Cache file to import (default: data/processed/cache.json)
- `--run-date, -d DATE`: Override run date (YYYY-MM-DD HH:MM)

**Example:**
```bash
uv run llm-bench import-cache
uv run llm-bench import-cache --cache-file old_cache.json --run-date "2025-01-01 12:00"
```

## Project Structure

```
llm-benchmarking/
├── config/
│   └── sources.yaml         # Configuration for Ollama, scraping, analysis, and database
├── data/
│   ├── raw/                 # Cached HTML/raw data
│   ├── processed/           # Generated reports and aggregated data
│   └── history.db           # SQLite database for historical tracking
├── docs/                    # Detailed documentation
│   ├── 01_project_structure_uv.md
│   ├── 02_backend_data_collection.md
│   ├── 03_llm_invocation.md
│   └── 04_report_generation.md
├── src/llm_benchmarks/
│   ├── main.py              # CLI entry point (Typer)
│   ├── pipeline.py          # Orchestrates scraping → analysis → reporting
│   ├── data_aggregator.py   # Combines data from multiple sources
│   ├── scrapers/            # Auto-discovered scraper plugins
│   │   ├── base.py          # BaseScraper abstract class
│   │   ├── registry.py      # Auto-discovery system
│   │   ├── lmsys.py         # LMSYS leaderboard scraper
│   │   ├── simon_willison.py # Blog scraper
│   │   └── [add new here].py # Just create a file, it's auto-registered!
│   ├── llm/                 # LLM analysis components
│   │   ├── ollama_client.py # Ollama connection and configuration
│   │   ├── analyzer.py      # Analysis orchestration
│   │   └── prompts.py       # Prompt templates
│   ├── database/            # Historical data tracking
│   │   ├── connection.py    # SQLite connection management
│   │   ├── schema.py        # Database schema definitions
│   │   ├── manager.py       # Save pipeline results to database
│   │   ├── repository.py    # Query historical data
│   │   ├── models.py        # Pydantic models for database entities
│   │   └── migrations/      # Schema migrations
│   │       ├── migration_runner.py
│   │       └── v001_initial.sql
│   └── reports/             # Report generation
│       ├── models.py        # Data models for reports
│       ├── generator.py     # Report generator
│       └── formatters.py    # Output formatters (Markdown, JSON, HTML)
├── tests/                   # Unit and integration tests
├── pyproject.toml           # Project metadata and dependencies
└── README.md
```

## Architecture

### Data Flow

1. **Scraping**: Scrapers collect data from various sources (leaderboards, blogs, research papers)
2. **Aggregation**: Data is normalized and combined across sources
3. **Analysis**: Ollama LLM analyzes the data to generate insights
4. **Reporting**: Results are formatted and output to desired format

### Plugin System

The scraper system uses automatic plugin discovery. To add a new source:

1. Create a new file in `src/llm_benchmarks/scrapers/`
2. Inherit from `BaseScraper`
3. Define `metadata` and implement `scrape()`
4. That's it! The scraper is automatically discovered and registered

**Example scraper:**

```python
from datetime import datetime
from .base import BaseScraper, SourceMetadata, SourceType, ScrapedContent, BenchmarkEntry

class MyNewScraper(BaseScraper):
    metadata = SourceMetadata(
        name="My Benchmark",
        url="https://example.com/benchmark",
        source_type=SourceType.LEADERBOARD,
        description="Description of the benchmark",
        update_frequency="daily",
    )

    def scrape(self) -> ScrapedContent:
        soup, page = self.fetch_soup(self.metadata.url)

        # Extract data...
        entries = [
            BenchmarkEntry(
                model_name="GPT-4",
                benchmark_name="MMLU",
                score=86.4,
            ),
            # ...
        ]

        page.close()

        return ScrapedContent(
            source=self.metadata.name,
            timestamp=datetime.now(),
            entries=entries,
        )
```

### Source Types

- **LEADERBOARD**: Structured benchmark data with scores (e.g., LMSYS, HuggingFace)
- **BLOG**: Text content analyzed by LLM (e.g., Simon Willison's blog)
- **RESEARCH**: Academic papers and research findings

## Configuration

Edit `config/sources.yaml` to customize behavior:

```yaml
# Ollama Settings
ollama:
  base_url: "http://localhost:11434"
  model: "gemma3:4b"
  temperature: 0.7
  max_tokens: 2000

# Scraping Settings
scraping:
  timeout: 30000
  cache_ttl: 3600
  headless: true

# Analysis Settings
analysis:
  default_analyses:
    - summary
    - coding
  max_blog_length: 5000
  top_models_count: 10
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/llm_benchmarks

# Run specific test
uv run pytest tests/test_pipeline.py

# Run integration tests
uv run pytest test_pipeline_integration.py
```

### Adding Dependencies

```bash
uv add <package-name>
```

### Code Quality

The project follows modern Python best practices:

- Type hints throughout
- SOLID principles
- Modular, testable design
- Comprehensive error handling

## Troubleshooting

### Ollama Connection Issues

If you see "Ollama is not available" errors:

```bash
# Start Ollama service
ollama serve

# Pull the required model
ollama pull gemma3:4b

# Verify it's working
ollama list
```

### Playwright Browser Issues

If scraping fails with browser errors:

```bash
# Reinstall Playwright browsers
uv run playwright install chromium
```

### Cache Issues

If you're seeing stale data:

```bash
# Delete cache and re-scrape
rm -rf data/processed/*.json
uv run llm-bench scrape
```

## Contributing

Contributions are welcome! To add a new benchmark source:

1. Create a scraper in `src/llm_benchmarks/scrapers/`
2. Inherit from `BaseScraper`
3. Implement the `scrape()` method
4. Add tests
5. Submit a pull request

See existing scrapers for examples.

## License

[Add your license here]

## Acknowledgments

- Built with [uv](https://github.com/astral-sh/uv) for fast dependency management
- Powered by [Ollama](https://ollama.ai/) for local LLM analysis
- Uses [Playwright](https://playwright.dev/) for reliable web scraping
- CLI built with [Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/)

## Roadmap

- [ ] Add more benchmark sources
- [ ] Support for historical trend analysis
- [ ] Web dashboard for visualization
- [ ] Export to additional formats (CSV, Excel)
- [ ] API server mode
- [ ] Scheduled automatic updates
