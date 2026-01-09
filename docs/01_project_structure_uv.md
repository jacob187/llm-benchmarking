# Roadmap 1: Project Structure with uv

## Goal
Set up a modern Python project using `uv` with a plugin-style architecture for easy source addition.

---

## Quick Start

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project
mkdir llm-benchmark-tracker && cd llm-benchmark-tracker
uv init --name llm-benchmark-tracker --python 3.12

# Create structure
mkdir -p src/llm_benchmarks/{scrapers,llm,reports} config data/{raw,processed} tests
```

---

## Project Structure

```
llm-benchmark-tracker/
├── pyproject.toml          # Dependencies + CLI entry point
├── uv.lock                  # Lock file (commit this!)
├── config/
│   └── sources.yaml         # Optional config overrides
├── data/
│   ├── raw/                 # Cached HTML
│   └── processed/           # Reports + aggregated JSON
├── src/llm_benchmarks/
│   ├── main.py              # CLI commands (typer)
│   ├── pipeline.py          # Orchestrates scrape → analyze → report
│   ├── data_aggregator.py   # Combines data from multiple sources
│   ├── scrapers/            # AUTO-DISCOVERED plugins
│   │   ├── base.py          # BaseScraper + SourceMetadata + SourceType enum
│   │   ├── registry.py      # Auto-discovery via pkgutil
│   │   ├── lmsys.py         # Each scraper = 1 file, ~30 lines
│   │   ├── huggingface.py
│   │   └── [add_new_here.py] ← Just create file, auto-registered!
│   ├── llm/
│   │   ├── ollama_client.py # LangChain + Ollama connection
│   │   ├── prompts.py       # Prompt templates
│   │   └── analyzer.py      # Runs analysis chains
│   └── reports/
│       ├── models.py        # Report/Table/Section dataclasses
│       └── generator.py     # Renders to Markdown
└── tests/
```

---

## Key Dependencies (pyproject.toml)

```toml
dependencies = [
    "beautifulsoup4",    # HTML parsing
    "httpx",             # HTTP client
    "lxml",              # Fast HTML parser
    "langchain",         # LLM orchestration
    "langchain-ollama",  # Ollama integration
    "pydantic",          # Data validation
    "pyyaml",            # Config files
    "rich",              # Pretty output
    "typer",             # CLI framework
]
```

---

## Source Types

| Type | Output | Example |
|------|--------|---------|
| `LEADERBOARD` | `BenchmarkEntry` list with scores | LMSYS, HuggingFace |
| `BLOG` | `raw_text` for LLM analysis | Simon Willison |
| `RESEARCH` | `structured_data` with paper info | ArXiv |

---

## uv Workflow

```bash
uv sync                    # Install deps
uv run llm-bench --help    # Run CLI
uv run pytest              # Run tests
uv add requests            # Add dependency
```

---

## Next: Roadmap 2 → Build the scraper system
