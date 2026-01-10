# LLM Benchmarking Dashboard

A Streamlit-based web interface for visualizing and analyzing LLM benchmark data.

## Quick Start

```bash
# Install dependencies
uv sync

# Run dashboard
./run_dashboard.sh

# Or run directly
streamlit run src/frontend/streamlit/app.py
```

Access the dashboard at: **http://localhost:8501**

## Features

### Home Dashboard
- Overview statistics (total models, benchmarks, data points)
- Top 10 models by average score with ranking changes
- Recently added models

### Model Explorer
- Searchable and filterable model table
- Sort by score, rank, or name
- Model detail view with score history
- Trend charts for individual models
- Export to CSV/JSON

### Historical Trends
- Compare up to 5 models over time
- Interactive line charts with Plotly
- Summary statistics and change analysis
- Data export capabilities

### Model Comparison
- Side-by-side benchmark comparison
- Radar chart visualization
- Performance heatmap
- Statistical analysis

### Q&A Chat (requires Ollama)
- Ask questions about benchmark data
- Streaming AI responses
- Suggested questions
- Export conversations

## Requirements

- Python 3.12+
- SQLite database (`data/history.db`) or cache file (`data/processed/cache.json`)
- Ollama (optional, for Q&A feature)

## Data Sources

The dashboard uses data from:

1. **Database** (`data/history.db`) - Primary source with historical tracking
2. **Cache** (`data/processed/cache.json`) - Fallback when database unavailable

To collect data, run:
```bash
llm-bench scrape
```

## Pages

| Page | File | Description |
|------|------|-------------|
| Home | `app.py` | Dashboard overview |
| Model Explorer | `pages/1_📊_Model_Explorer.py` | Browse and search models |
| Trends | `pages/2_📈_Historical_Trends.py` | Track performance over time |
| Comparison | `pages/3_🔍_Comparison.py` | Compare multiple models |
| Q&A Chat | `pages/4_💬_Ask_Questions.py` | AI-powered Q&A |

## Configuration

The dashboard reads configuration from `config/sources.yaml`:

```yaml
ollama:
  base_url: "http://localhost:11434"
  model: "gemma3:4b"

database:
  enabled: true
  path: "data/history.db"
```

## Development

### Project Structure

```
src/frontend/streamlit/
├── app.py                  # Main dashboard entry point
├── pages/                  # Multi-page app pages
│   ├── 1_📊_Model_Explorer.py
│   ├── 2_📈_Historical_Trends.py
│   ├── 3_🔍_Comparison.py
│   └── 4_💬_Ask_Questions.py
├── utils/                  # Utility functions
│   ├── data_loader.py     # Database access with caching
│   └── config.py          # Configuration management
├── components/            # Reusable UI components
│   └── charts.py         # Plotly chart components
└── README.md
```

### Caching

The dashboard uses Streamlit's caching for performance:

- `@st.cache_resource` - For database connections
- `@st.cache_data(ttl=300)` - For query results (5-minute TTL)

### Error Handling

The dashboard gracefully handles:

- Missing database (falls back to cache)
- No data available (shows helpful instructions)
- Ollama unavailable (disables Q&A feature)

## Troubleshooting

**Dashboard won't start:**
```bash
# Ensure dependencies are installed
uv sync

# Check Streamlit installation
streamlit version
```

**No data showing:**
```bash
# Run the scraper to collect data
llm-bench scrape
```

**Q&A not working:**
```bash
# Start Ollama
ollama serve

# Pull a model
ollama pull gemma3:4b
```

## License

Part of the LLM Benchmarking project.
