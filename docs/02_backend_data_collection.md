# Roadmap 2: Backend - Modular Scraper System & Data Collection

## Goal
Build a plugin-style scraper system where **adding a new source = creating one file**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   ScraperRegistry                        │
│         (auto-discovers all scrapers via pkgutil)        │
└─────────────────────────┬───────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   ┌─────────┐      ┌──────────┐      ┌──────────┐
   │ lmsys.py│      │huggingface│     │ blog.py  │
   │LEADERBOARD│     │LEADERBOARD│     │  BLOG    │
   └─────────┘      └──────────┘      └──────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
                  ┌───────────────┐
                  │ScrapedContent │
                  │(unified output)│
                  └───────────────┘
```

---

## File: `scrapers/base.py`

**Purpose**: Define the contract all scrapers must follow.

```python
class SourceType(Enum):
    LEADERBOARD = "leaderboard"  # Returns BenchmarkEntry list
    BLOG = "blog"                 # Returns raw_text for LLM
    RESEARCH = "research"         # Returns structured_data

@dataclass
class SourceMetadata:
    name: str                     # "LMSYS Arena"
    url: str                      # "https://lmarena.ai/"
    source_type: SourceType       # How to process output
    description: str              # What this source provides
    benchmarks_covered: list[str] # ["ELO", "MMLU"]
    requires_js: bool             # Needs Selenium?

@dataclass
class BenchmarkEntry:
    model_name: str
    score: float | None
    benchmark_name: str
    source: str

@dataclass
class ScrapedContent:
    source_name: str
    source_type: SourceType
    entries: list[BenchmarkEntry]  # For LEADERBOARD
    raw_text: str | None           # For BLOG
    structured_data: dict | None   # For RESEARCH
    error: str | None              # If failed

class BaseScraper(ABC):
    metadata: SourceMetadata  # MUST define this
    
    def fetch(url) -> str           # HTTP GET
    def soup(html) -> BeautifulSoup # Parse HTML
    def fetch_soup(url) -> BeautifulSoup  # Both in one
    
    @abstractmethod
    def scrape(self) -> ScrapedContent:
        """ONLY method you implement"""
        pass
```

---

## File: `scrapers/registry.py`

**Purpose**: Auto-discover scrapers at runtime.

```python
class ScraperRegistry:
    @classmethod
    def discover():
        """
        Uses pkgutil.iter_modules to find all .py files in scrapers/
        Imports each, finds BaseScraper subclasses with metadata attr
        Registers them by slugified name
        """
        for module in pkgutil.iter_modules([scrapers_dir]):
            # import module
            # find classes with `metadata` attribute
            # register: _scrapers[slug(metadata.name)] = cls
    
    @classmethod
    def all() -> list[BaseScraper]:
        """Return instances of all registered scrapers"""
    
    @classmethod
    def by_type(source_type) -> list[BaseScraper]:
        """Filter by LEADERBOARD/BLOG/RESEARCH"""
    
    @classmethod
    def get(name: str) -> BaseScraper:
        """Get specific scraper by key"""
```

---

## File: `scrapers/lmsys.py` (Example Leaderboard)

**Purpose**: Scrape LMSYS Arena ELO ratings.

```python
class LMSYSArenaScraper(BaseScraper):
    metadata = SourceMetadata(
        name="LMSYS Arena",
        url="https://lmarena.ai/",
        source_type=SourceType.LEADERBOARD,
        benchmarks_covered=["ELO Rating"],
    )
    
    def scrape(self) -> ScrapedContent:
        soup = self.fetch_soup()
        entries = []
        
        # Parse tables for model names + ELO scores
        # Parse embedded JSON in <script> tags
        # Deduplicate by model name
        
        return ScrapedContent(
            source_name=self.metadata.name,
            entries=entries,
        )
```

---

## File: `scrapers/simon_willison.py` (Example Blog)

**Purpose**: Scrape blog posts for LLM analysis.

```python
class SimonWillisonBlogScraper(BaseScraper):
    metadata = SourceMetadata(
        name="Simon Willison Blog",
        url="https://simonwillison.net/tags/llms/",
        source_type=SourceType.BLOG,  # Different type!
    )
    
    def scrape(self) -> ScrapedContent:
        soup = self.fetch_soup()
        
        # Extract last 5 post titles + content
        posts = [{"title": ..., "content": ...}]
        raw_text = format_posts_as_markdown(posts)
        
        return ScrapedContent(
            source_name=self.metadata.name,
            raw_text=raw_text,  # For LLM to analyze
            structured_data={"posts": posts},
        )
```

---

## File: `data_aggregator.py`

**Purpose**: Combine data from multiple sources into unified model view.

```python
class BenchmarkAggregator:
    def aggregate(scraped_contents: list[ScrapedContent]) -> dict[str, ModelBenchmarks]:
        """
        For each ScrapedContent:
          - LEADERBOARD: Extract entries, group by normalized model name
          - BLOG: Store raw_text for LLM analysis
          - RESEARCH: Store structured_data
        
        Returns: {
            "GPT-4o": ModelBenchmarks(benchmarks={...}, sources={...}),
            "Claude 3.5": ModelBenchmarks(...),
        }
        """
    
    def save_cache(filename) -> Path:
        """Save to JSON for reuse"""
    
    def load_cache(filename) -> bool:
        """Load previous scrape results"""
```

---

## File: `llm/ollama_client.py`

**Purpose**: Connect to local Ollama instance.

```python
def get_ollama_client() -> ChatOllama:
    """
    Load config from sources.yaml (model, temperature)
    Return LangChain ChatOllama instance
    """
    config = load_config()
    return ChatOllama(
        base_url=config.base_url,
        model=config.model,  # e.g., "llama3.2:3b"
        temperature=config.temperature,
    )
```

---

## Adding a New Source

```bash
# 1. Create file
touch src/llm_benchmarks/scrapers/my_new_source.py

# 2. Define scraper (~30 lines)
class MyNewSourceScraper(BaseScraper):
    metadata = SourceMetadata(name="My Source", url="...", source_type=...)
    def scrape(self) -> ScrapedContent: ...

# 3. Done! Auto-discovered
uv run llm-bench sources  # See it listed
```

---

## CLI Commands (Preview)

```bash
llm-bench sources              # List all registered sources
llm-bench sources --type blog  # Filter by type
llm-bench scrape               # Run all scrapers
llm-bench add-source           # Interactive: generate scraper template
```

---

## Next: Roadmap 3 → Feed data to Ollama for analysis
