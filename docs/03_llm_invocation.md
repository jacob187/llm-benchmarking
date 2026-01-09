# Roadmap 3: LLM Invocation - Analyzing Data with Ollama

## Goal
Use LangChain + Ollama to analyze benchmark data and generate insights.

---

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  Aggregated     │────▶│   Prompt     │────▶│   Ollama    │
│  Model Data     │     │  Templates   │     │ (LangChain) │
│  + Blog Text    │     └──────────────┘     └──────┬──────┘
└─────────────────┘                                 │
                                                    ▼
                              ┌──────────────────────────────┐
                              │     AnalysisResult           │
                              │  - summary                   │
                              │  - coding_analysis           │
                              │  - recommendations           │
                              │  - blog_insights             │
                              └──────────────────────────────┘
```

---

## File: `llm/prompts.py`

**Purpose**: Define reusable prompt templates.

```python
SYSTEM_CONTEXT = """You are an AI researcher analyzing LLM benchmarks.
Provide factual, data-driven insights. Be concise."""

SUMMARY_TEMPLATE = PromptTemplate.from_template("""
{system_context}

## Benchmark Data
{benchmark_data}

## Task
1. Identify top 3-5 performing models
2. Note surprising findings
3. Highlight trade-offs (quality vs speed, open vs closed)
Keep under 300 words.
""")

COMPARISON_TEMPLATE = PromptTemplate.from_template("""
{system_context}

## Models: {models}
## Data: {benchmark_data}

Compare these models. Recommend for:
- General chat
- Coding tasks  
- Cost-sensitive apps
""")

CODING_TEMPLATE = PromptTemplate.from_template("""
{system_context}

## Coding Benchmarks: {coding_data}

Rank models by coding ability. Best for:
- Code generation
- Debugging
- Multi-language support
""")

BLOG_ANALYSIS_TEMPLATE = PromptTemplate.from_template("""
{system_context}

## Recent Blog Posts:
{blog_content}

Summarize key insights about LLM trends from these posts.
What models are practitioners excited about?
""")

def get_template(name: str) -> PromptTemplate:
    """Get template by name: summary, comparison, coding, blog"""
```

---

## File: `llm/analyzer.py`

**Purpose**: Run analysis chains against Ollama.

```python
@dataclass
class AnalysisResult:
    analysis_type: str   # "summary", "comparison", etc.
    content: str         # LLM output
    model_used: str      # "llama3.2:3b"
    timestamp: datetime

class BenchmarkAnalyzer:
    def __init__(self, model: str | None = None):
        self.llm = get_ollama_client()
        if model:
            self.llm.model = model
    
    def _format_benchmark_data(self, models: dict) -> str:
        """
        Convert model dict to markdown for LLM context:
        
        ### GPT-4o
        - Average: 92.5
        - Sources: LMSYS, Artificial Analysis
        - Benchmarks: ELO: 1287, Quality: 92.5
        """
    
    def summarize(self, models: dict) -> AnalysisResult:
        """
        Generate executive summary of all benchmark data.
        Uses SUMMARY_TEMPLATE.
        """
        template = get_template("summary")
        chain = template | self.llm | StrOutputParser()
        result = chain.invoke({"benchmark_data": self._format_benchmark_data(models)})
        return AnalysisResult(analysis_type="summary", content=result, ...)
    
    def compare_models(self, model_names: list[str], all_models: dict) -> AnalysisResult:
        """
        Compare specific models (e.g., ["GPT-4", "Claude", "Llama"]).
        Uses COMPARISON_TEMPLATE.
        """
    
    def analyze_coding(self, models: dict) -> AnalysisResult:
        """
        Filter to coding benchmarks (HumanEval, SWE-bench, etc.)
        Uses CODING_TEMPLATE.
        """
    
    def analyze_blogs(self, blog_content: list[dict]) -> AnalysisResult:
        """
        Analyze scraped blog posts for qualitative insights.
        Uses BLOG_ANALYSIS_TEMPLATE.
        """
    
    def get_recommendations(self, user_context: str, models: dict) -> AnalysisResult:
        """
        Custom recommendations based on user's stated needs.
        E.g., "I need a model for code review on a budget"
        """
    
    # Streaming versions
    def summarize_stream(self, models: dict) -> Iterator[str]:
        """Yield tokens as they're generated"""
        for chunk in self.llm.stream(prompt):
            yield chunk.content
```

---

## File: `pipeline.py`

**Purpose**: Orchestrate the full flow.

```python
class BenchmarkPipeline:
    def __init__(self, ollama_model: str | None = None):
        self.aggregator = BenchmarkAggregator()
        self.analyzer = BenchmarkAnalyzer(model=ollama_model)
    
    def run(
        self, 
        skip_scrape: bool = False,
        analyses: list[str] = ["summary"],
        include_blogs: bool = True,
    ) -> PipelineResult:
        """
        Full pipeline:
        1. Scrape all sources (or load cache)
        2. Aggregate by model
        3. Run requested analyses
        4. Return structured results
        """
        
        # Step 1: Get data
        if skip_scrape:
            self.aggregator.load_cache()
        else:
            scraped = scrape_all()
            self.aggregator.aggregate(scraped)
            self.aggregator.save_cache()
        
        # Step 2: Run analyses
        results = {}
        
        if "summary" in analyses:
            results["summary"] = self.analyzer.summarize(self.aggregator.models)
        
        if "coding" in analyses:
            results["coding"] = self.analyzer.analyze_coding(self.aggregator.models)
        
        if "blogs" in analyses and self.aggregator.blog_content:
            results["blogs"] = self.analyzer.analyze_blogs(self.aggregator.blog_content)
        
        return PipelineResult(
            models=self.aggregator.models,
            analyses=results,
        )
```

---

## How Different Source Types Are Used

| Source Type | Data Flow |
|-------------|-----------|
| `LEADERBOARD` | entries → aggregate by model → format as markdown → LLM summarizes |
| `BLOG` | raw_text → passed directly to BLOG_ANALYSIS_TEMPLATE → LLM extracts insights |
| `RESEARCH` | structured_data → could feed paper titles to LLM for trend detection |

---

## Example: Analyzing a New Blog Source

```python
# 1. Blog scraper returns raw_text
content = scraper.scrape()
# content.raw_text = "# Post 1\nGPT-4o is great for...\n\n# Post 2\n..."

# 2. Aggregator stores it
aggregator.blog_content.append({
    "source": content.source_name,
    "text": content.raw_text,
})

# 3. Analyzer processes it
result = analyzer.analyze_blogs(aggregator.blog_content)
# result.content = "Key insights from recent posts: ..."
```

---

## CLI Commands

```bash
# Run with specific analyses
llm-bench report --analyses "summary,coding,blogs"

# Compare specific models
llm-bench compare "GPT-4o,Claude 3.5,Llama 3.1"

# Ask a question
llm-bench ask "Which model is best for code review?"

# Use a different Ollama model
llm-bench report --model llama3.1:8b
```

---

## Next: Roadmap 4 → Generate Markdown reports
