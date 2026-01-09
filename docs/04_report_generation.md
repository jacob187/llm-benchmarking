# Roadmap 4: Report Generation - Markdown Output

## Goal
Generate polished Markdown reports from benchmark data + LLM analysis.

---

## Sample Output

```markdown
# LLM Benchmark Report
*Generated: January 7, 2026*

## Executive Summary
[LLM-generated summary]

## Top Performers
| Rank | Model | Score | Sources |
|------|-------|-------|---------|
| 1 | GPT-4o | 92.5 | LMSYS, AA |
| 2 | Claude 3.5 | 91.8 | LMSYS, AA |

## Coding Analysis
[LLM analysis of coding benchmarks]

## Blog Insights
[LLM synthesis of recent blog posts]

<details>
<summary>Raw Data</summary>
[Collapsible detailed data]
</details>

---
*Sources: LMSYS Arena, HuggingFace, Artificial Analysis, Simon Willison Blog*
```

---

## File: `reports/models.py`

**Purpose**: Data structures for report content.

```python
@dataclass
class ReportSection:
    title: str
    content: str
    level: int = 2  # ## heading
    
    def to_markdown(self) -> str:
        prefix = "#" * self.level
        return f"{prefix} {self.title}\n\n{self.content}\n"

@dataclass
class TableData:
    headers: list[str]
    rows: list[list[str]]
    caption: str | None = None
    
    def to_markdown(self) -> str:
        """
        | Header1 | Header2 |
        |---------|---------|
        | val1    | val2    |
        """

@dataclass
class Report:
    title: str
    generated_at: datetime
    sections: list[ReportSection]
    tables: dict[str, TableData]
    metadata: dict  # sources, model used, etc.
    
    def add_section(title, content, level=2): ...
    def add_table(name, table): ...
    
    def to_markdown(self) -> str:
        """Render complete report as Markdown string"""
```

---

## File: `reports/generator.py`

**Purpose**: Build reports from pipeline results.

```python
class ReportGenerator:
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
    
    def generate(
        self,
        models: dict[str, ModelBenchmarks],
        analyses: dict[str, AnalysisResult],
        blog_insights: AnalysisResult | None = None,
    ) -> Report:
        """
        Build report from:
        - models: Aggregated benchmark data
        - analyses: LLM analysis results (summary, coding, etc.)
        - blog_insights: Optional blog analysis
        """
        report = Report(title="LLM Benchmark Report", ...)
        
        # Add executive summary (from LLM)
        if "summary" in analyses:
            report.add_section("Executive Summary", analyses["summary"].content)
        
        # Add top performers table
        report.add_table("top_performers", self._create_top_models_table(models))
        report.add_section("Top Performers", "Models ranked by average score.")
        
        # Add coding analysis (from LLM)
        if "coding" in analyses:
            report.add_section("Coding Performance", analyses["coding"].content)
            report.add_table("coding", self._create_coding_table(models))
        
        # Add blog insights (from LLM analyzing BLOG sources)
        if blog_insights:
            report.add_section("Industry Insights", blog_insights.content)
        
        # Add collapsible raw data
        report.add_section("Detailed Data", self._create_details_section(models))
        
        return report
    
    def _create_top_models_table(self, models: dict) -> TableData:
        """
        Sort models by average score
        Return top 15 as table
        """
    
    def _create_coding_table(self, models: dict) -> TableData:
        """
        Filter benchmarks containing 'code', 'humaneval', 'swe'
        Return as table
        """
    
    def _create_details_section(self, models: dict) -> str:
        """
        Create <details> collapsible with all model data
        """
        return """
<details>
<summary>Click to expand</summary>

### Model 1
- Score: X
- Benchmarks: ...

</details>
"""
    
    def save(self, report: Report, filename: str | None = None) -> Path:
        """Write report.to_markdown() to file"""
        filepath = self.output_dir / (filename or f"report_{date}.md")
        filepath.write_text(report.to_markdown())
        return filepath
```

---

## File: `main.py` (CLI)

**Purpose**: User-facing commands.

```python
app = typer.Typer()

@app.command()
def report(
    output: Path = None,
    cached: bool = False,
    analyses: str = "summary,coding",
    model: str = None,
    preview: bool = False,
):
    """Generate a benchmark report."""
    pipeline = BenchmarkPipeline(ollama_model=model)
    result = pipeline.run(
        skip_scrape=cached,
        analyses=analyses.split(","),
    )
    
    generator = ReportGenerator()
    report = generator.generate(result.models, result.analyses)
    
    if preview:
        console.print(Markdown(report.to_markdown()))
    else:
        filepath = generator.save(report, output)
        console.print(f"Saved to: {filepath}")

@app.command()
def compare(models: str):
    """Compare specific models."""
    # Parse "GPT-4,Claude,Llama"
    # Load cached data
    # Run comparison analysis
    # Print result

@app.command()
def sources(type: str = None):
    """List available benchmark sources."""
    for source in ScraperRegistry.list_sources():
        # Print name, type, url, description

@app.command()
def add_source():
    """Interactive: create new scraper file from template."""
    name = prompt("Source name")
    url = prompt("URL")
    type = prompt("Type (leaderboard/blog/research)")
    
    # Generate template file
    # Write to scrapers/{slug}.py

@app.command()
def scrape(save_raw: bool = False):
    """Scrape all sources without analysis."""
    results = scrape_all()
    aggregator.aggregate(results)
    aggregator.save_cache()

@app.command()
def ask(question: str):
    """Ask a question about benchmark data."""
    # Load cache
    # Stream LLM response
    for chunk in analyzer.interactive_query(question, models):
        print(chunk, end="")
```

---

## Usage Examples

```bash
# Generate full report
uv run llm-bench report

# Use cached data, preview in terminal
uv run llm-bench report --cached --preview

# Include blog analysis
uv run llm-bench report --analyses "summary,coding,blogs"

# Compare specific models
uv run llm-bench compare "GPT-4o,Claude 3.5 Sonnet,Llama 3.1 70B"

# List sources
uv run llm-bench sources
uv run llm-bench sources --type blog

# Add new source interactively
uv run llm-bench add-source

# Quick question
uv run llm-bench ask "Best model for code review under $10/month?"
```

---

## Complete Data Flow

```
1. SCRAPE
   ├── Leaderboards → BenchmarkEntry list
   ├── Blogs → raw_text
   └── Research → structured_data

2. AGGREGATE
   ├── Group entries by model name
   ├── Store blog/research content separately
   └── Save to cache

3. ANALYZE (Ollama)
   ├── Summary: All model data → executive summary
   ├── Coding: Filtered data → coding analysis
   ├── Blogs: raw_text → industry insights
   └── Custom: User question → targeted answer

4. REPORT
   ├── Combine LLM outputs into sections
   ├── Generate tables from model data
   ├── Render to Markdown
   └── Save to file
```

---

## Adding a New Source: Complete Flow

```bash
# 1. Create scraper (30 seconds)
uv run llm-bench add-source
# → Enter: "Vellum Leaderboard", "https://vellum.ai/llm-leaderboard", "leaderboard"
# → Creates: scrapers/vellum_leaderboard.py

# 2. Edit scrape() method (5 minutes)
# → Inspect page structure
# → Write BeautifulSoup selectors
# → Test: uv run python -c "from llm_benchmarks.scrapers import ScraperRegistry; print(ScraperRegistry.get('vellum_leaderboard').scrape())"

# 3. Run full pipeline
uv run llm-bench report
# → New source automatically included!
```

---

## Summary

| File | Purpose |
|------|---------|
| `scrapers/base.py` | BaseScraper, SourceMetadata, SourceType |
| `scrapers/registry.py` | Auto-discovery via pkgutil |
| `scrapers/*.py` | One file per source (~30 lines each) |
| `data_aggregator.py` | Combine sources, normalize names |
| `llm/ollama_client.py` | LangChain + Ollama connection |
| `llm/prompts.py` | Reusable prompt templates |
| `llm/analyzer.py` | Run analysis chains |
| `reports/models.py` | Report/Table/Section dataclasses |
| `reports/generator.py` | Build Markdown from results |
| `pipeline.py` | Orchestrate everything |
| `main.py` | CLI commands |

**Total: ~800 lines of code for a fully modular, extensible benchmark tracker.**
