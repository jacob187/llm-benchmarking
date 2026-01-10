# Frontend Implementation Roadmap

**Date:** 2026-01-10
**Status:** Planning Phase
**Current State:** CLI-only application with Rich terminal UI

## Executive Summary

This document outlines three distinct approaches to adding a web frontend to the LLM Benchmarking application. Each approach balances different priorities: speed of development, user experience richness, and technical complexity.

The current application is a well-architected Python CLI tool with:
- SQLite database for historical benchmark tracking
- Pydantic models for type-safe data handling
- LLM integration with streaming capabilities
- Rich terminal UI with tables, charts, and progress indicators
- Report generation (Markdown, JSON, HTML)

---

## Current Application Analysis

### Technology Stack
- **Language:** Python 3.12+
- **CLI Framework:** Typer + Rich
- **Database:** SQLite
- **LLM Integration:** Langchain + Ollama
- **Web Scraping:** Playwright + BeautifulSoup4
- **Data Validation:** Pydantic

### Core Features Ready for Frontend
1. **Benchmark Data Exploration** - Browse scores, rankings, model metadata
2. **Historical Trend Analysis** - Time-series score evolution
3. **Model Comparison** - Side-by-side multi-model analysis
4. **Natural Language Q&A** - LLM-powered chat interface with streaming
5. **Report Generation** - Export in multiple formats
6. **Data Source Management** - View and manage scrapers

### Data Already Available
- ✅ SQLite database with historical records (`data/history.db`)
- ✅ JSON-serializable Pydantic models
- ✅ Caching system (`data/processed/cache.json`)
- ✅ Report generation infrastructure
- ✅ Complex aggregation and query logic

---

## Three Proposed Approaches

### Approach 1: Rapid Prototype (Streamlit)
**Timeline:** Fastest
**Target Audience:** Internal tool, data scientists, researchers
**Philosophy:** Maximum functionality with minimum code

### Approach 2: Lightweight Modern (Svelte + FastAPI)
**Timeline:** Moderate
**Target Audience:** Public-facing, broad user base
**Philosophy:** Modern UX without heavyweight frameworks

### Approach 3: Production-Grade (Next.js + FastAPI)
**Timeline:** Comprehensive
**Target Audience:** Enterprise, scalable product
**Philosophy:** Full-featured, production-ready SaaS

---

## Approach 1: Streamlit - Rapid Prototype

### Overview
Build a fully functional web dashboard using pure Python with Streamlit. No JavaScript required. Ideal for quick MVPs and internal tools.

### Technology Stack
- **Frontend Framework:** Streamlit
- **Visualization:** Plotly, Altair
- **Data Layer:** Direct access to existing SQLite database
- **Deployment:** Streamlit Cloud, Docker, or local

### Key Features

#### Phase 1: Core Dashboard (Foundation)
1. **Home Page**
   - Latest benchmark scores in interactive table
   - Top 10 models with score bars
   - Last updated timestamp
   - Quick stats (total models, benchmarks, data sources)

2. **Model Explorer**
   - Searchable/filterable model list
   - Model detail view with all benchmark scores
   - Radar chart for multi-dimensional performance
   - Source attribution

3. **Historical Trends**
   - Date range selector
   - Line chart showing score evolution for selected model
   - Multi-model comparison overlay
   - Ranking changes over time

#### Phase 2: Interactive Analysis (Enhancement)
4. **Comparison Tool**
   - Multi-select model picker
   - Side-by-side score tables
   - Visual comparison (bar charts, radar charts)
   - Statistical summary (mean, median, percentile)

5. **Q&A Interface**
   - Chat-style input box
   - Streaming LLM responses
   - Conversation history
   - Suggested questions

6. **Data Sources**
   - List all scrapers with status indicators
   - Last scrape time
   - Success/failure rates
   - Manual trigger button (optional)

#### Phase 3: Export & Customization (Polish)
7. **Report Builder**
   - Select models and benchmarks
   - Preview report
   - Download as Markdown/JSON/HTML/PDF

8. **Settings**
   - Configure Ollama endpoint
   - Set cache TTL
   - Choose default visualizations
   - Theme selection (light/dark)

### Implementation Roadmap

#### Step 1: Project Setup
```bash
# Install Streamlit
uv add streamlit plotly altair

# Create Streamlit app structure
mkdir -p src/frontend/streamlit
touch src/frontend/streamlit/app.py
touch src/frontend/streamlit/pages/1_📊_Explorer.py
touch src/frontend/streamlit/pages/2_📈_Trends.py
touch src/frontend/streamlit/pages/3_🔍_Compare.py
touch src/frontend/streamlit/pages/4_💬_Ask.py
```

#### Step 2: Create Main Dashboard
- Home page with overview statistics
- Reuse `BenchmarkRepository` for data queries
- Create Plotly charts for top models
- Add caching with `@st.cache_data`

#### Step 3: Build Model Explorer Page
- Sidebar filters (benchmark type, date range)
- Interactive table with sorting
- Model detail view on row click
- Radar chart using Plotly

#### Step 4: Implement Historical Trends
- Date picker for range selection
- Line chart with Plotly
- Multi-model selector with `st.multiselect`
- Query `BenchmarkRepository.get_score_trends()`

#### Step 5: Add Comparison Tool
- Model multi-select interface
- Create comparison data structure
- Side-by-side tables with highlighting
- Statistical summary cards

#### Step 6: Integrate Q&A Chat
- Chat input with `st.chat_input`
- Stream responses using `st.write_stream`
- Display chat history with `st.chat_message`
- Connect to existing `OllamaClient.ask_question()`

#### Step 7: Create Report Export
- Form to select report parameters
- Generate report using existing `ReportGenerator`
- Provide download buttons for each format
- Preview in collapsible section

#### Step 8: Add Settings & Configuration
- Form-based settings editor
- Store in `~/.llm-bench/config.json`
- Validate inputs
- Apply changes dynamically

### Code Structure
```
src/frontend/streamlit/
├── app.py                 # Main dashboard (home)
├── pages/
│   ├── 1_📊_Explorer.py   # Model explorer
│   ├── 2_📈_Trends.py     # Historical trends
│   ├── 3_🔍_Compare.py    # Model comparison
│   ├── 4_💬_Ask.py        # Q&A chat
│   └── 5_⚙️_Settings.py   # Configuration
├── components/
│   ├── charts.py          # Reusable chart components
│   ├── tables.py          # Data table utilities
│   └── filters.py         # Filter widgets
└── utils/
    ├── data_loader.py     # Database query helpers
    └── cache.py           # Caching utilities
```

### Example Code Snippets

**app.py (Home Dashboard):**
```python
import streamlit as st
from llm_benchmarks.database.repository import BenchmarkRepository
import plotly.express as px

st.set_page_config(page_title="LLM Benchmark Dashboard", layout="wide")

st.title("🏆 LLM Benchmark Dashboard")

@st.cache_resource
def get_repository():
    return BenchmarkRepository()

repo = get_repository()

# Stats cards
col1, col2, col3, col4 = st.columns(4)
stats = repo.get_database_stats()

with col1:
    st.metric("Total Models", stats['total_models'])
with col2:
    st.metric("Total Benchmarks", stats['total_benchmarks'])
with col3:
    st.metric("Data Points", stats['total_scores'])
with col4:
    st.metric("Sources", stats['total_sources'])

# Top models chart
st.subheader("Top 10 Models by Average Score")
top_models = repo.get_top_models(limit=10)
fig = px.bar(top_models, x='average_score', y='model_name', orientation='h')
st.plotly_chart(fig, use_container_width=True)

# Recent runs
st.subheader("Recent Benchmark Runs")
recent = repo.get_recent_runs(limit=5)
st.dataframe(recent, use_container_width=True)
```

**pages/4_💬_Ask.py (Q&A Chat):**
```python
import streamlit as st
from llm_benchmarks.llm.ollama_client import OllamaClient
from llm_benchmarks.data_aggregator import DataAggregator

st.title("💬 Ask Questions About Benchmarks")

@st.cache_resource
def get_client():
    return OllamaClient()

client = get_client()
aggregator = DataAggregator()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if question := st.chat_input("Ask about benchmark data..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get LLM response
    with st.chat_message("assistant"):
        data = aggregator.aggregate()
        response = st.write_stream(
            client.ask_question_stream(question, data)
        )

    st.session_state.messages.append({"role": "assistant", "content": response})
```

### Deployment Options

**Option 1: Streamlit Cloud (Easiest)**
```bash
# Push to GitHub
# Connect repository to Streamlit Cloud
# Auto-deploys on push
```

**Option 2: Docker (Self-Hosted)**
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install uv
RUN uv sync

EXPOSE 8501

CMD ["streamlit", "run", "src/frontend/streamlit/app.py"]
```

**Option 3: Local Development**
```bash
streamlit run src/frontend/streamlit/app.py
```

### Advantages
- ✅ **Fastest to implement** - Can build functional dashboard in days
- ✅ **Pure Python** - No JavaScript knowledge required
- ✅ **Built-in caching** - Performance optimization included
- ✅ **Reactive updates** - Automatic re-runs on input changes
- ✅ **Deployment simplicity** - Streamlit Cloud free tier available
- ✅ **Rich widget library** - Charts, tables, inputs all built-in
- ✅ **Code reuse** - Direct access to existing Python modules

### Limitations
- ❌ **Limited customization** - Opinionated layout and styling
- ❌ **Performance** - Full page reruns on interactions (mitigated with caching)
- ❌ **State management** - Can be tricky for complex apps
- ❌ **Mobile experience** - Not as polished as custom responsive designs
- ❌ **Advanced interactions** - Limited compared to React/Vue
- ❌ **Branding** - Harder to fully customize look and feel

### Best For
- Internal tools and dashboards
- Data science teams
- Rapid prototyping and MVPs
- Projects where speed trumps customization
- Python-only teams

---

## Approach 2: Svelte + FastAPI - Lightweight Modern

### Overview
Build a modern, lightweight web application with Svelte frontend and FastAPI backend. Svelte compiles to vanilla JavaScript with minimal runtime overhead, providing excellent performance and developer experience.

### Technology Stack
- **Frontend Framework:** SvelteKit (Svelte + routing + SSR)
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **Charts:** Chart.js or Lightweight-Charts
- **State Management:** Svelte stores (built-in)
- **Backend API:** FastAPI
- **Real-time:** WebSockets (FastAPI native)
- **Deployment:** Vercel/Netlify (frontend) + Railway/Fly.io (backend)

### Architecture

```
┌─────────────────────────────────────┐
│      SvelteKit Frontend             │
│  ┌─────────────────────────────┐   │
│  │  Pages (routes/)             │   │
│  │  - Dashboard                 │   │
│  │  - Model Explorer            │   │
│  │  - Trends                    │   │
│  │  - Compare                   │   │
│  │  - Chat                      │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │  Components (lib/)           │   │
│  │  - Charts, Tables, Filters   │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │  Stores (lib/stores/)        │   │
│  │  - benchmarks, models        │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
              ↓ HTTP/WebSocket
┌─────────────────────────────────────┐
│      FastAPI Backend                │
│  ┌─────────────────────────────┐   │
│  │  REST API Endpoints          │   │
│  │  - /api/models               │   │
│  │  - /api/benchmarks           │   │
│  │  - /api/trends               │   │
│  │  - /api/compare              │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │  WebSocket Endpoints         │   │
│  │  - /ws/chat                  │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │  Existing Python Modules     │   │
│  │  - Repository, Aggregator    │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

### Key Features

#### Phase 1: API Foundation
1. **FastAPI REST Endpoints**
   - `GET /api/models` - List all models with pagination
   - `GET /api/models/{name}` - Model details
   - `GET /api/benchmarks` - List benchmarks
   - `GET /api/trends/{model}` - Historical trends
   - `POST /api/compare` - Compare multiple models
   - `GET /api/sources` - Data source status
   - `GET /api/stats` - Dashboard statistics

2. **Type Safety**
   - Generate TypeScript types from Pydantic models
   - OpenAPI schema auto-generation
   - Request/response validation

3. **API Documentation**
   - Automatic Swagger UI at `/docs`
   - ReDoc at `/redoc`

#### Phase 2: SvelteKit Frontend
4. **Dashboard Page** (`/`)
   - Overview cards with key metrics
   - Top models bar chart
   - Recent updates timeline
   - Quick search

5. **Model Explorer** (`/models`)
   - Sortable, filterable table
   - Search with debouncing
   - Model detail modal/page
   - Pagination

6. **Trends Analysis** (`/trends`)
   - Model selector with autocomplete
   - Date range picker
   - Line chart with zoom/pan
   - Export chart as image

7. **Comparison Tool** (`/compare`)
   - Multi-select model picker
   - Side-by-side score grid
   - Radar chart overlay
   - Percentage difference highlighting

#### Phase 3: Real-Time Features
8. **Chat Interface** (`/chat`)
   - WebSocket connection for streaming
   - Message history
   - Typing indicators
   - Markdown rendering for responses

9. **Live Updates**
   - Server-sent events for new benchmark data
   - Toast notifications
   - Auto-refresh indicators

### Implementation Roadmap

#### Step 1: FastAPI Backend Setup
```bash
# Create backend directory
mkdir -p src/backend
cd src/backend

# Create FastAPI app
touch main.py
touch api.py
touch websockets.py
mkdir routers
touch routers/models.py
touch routers/benchmarks.py
touch routers/trends.py
```

**main.py:**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import models, benchmarks, trends

app = FastAPI(
    title="LLM Benchmark API",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # SvelteKit dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(benchmarks.router, prefix="/api/benchmarks", tags=["benchmarks"])
app.include_router(trends.router, prefix="/api/trends", tags=["trends"])

@app.get("/")
async def root():
    return {"message": "LLM Benchmark API"}
```

**routers/models.py:**
```python
from fastapi import APIRouter, Query
from typing import List
from pydantic import BaseModel
from llm_benchmarks.database.repository import BenchmarkRepository

router = APIRouter()
repo = BenchmarkRepository()

class ModelResponse(BaseModel):
    name: str
    average_score: float | None
    benchmark_count: int
    first_seen: str
    last_seen: str

class ModelDetail(BaseModel):
    name: str
    benchmarks: dict[str, float]
    sources: list[str]
    average_score: float | None

@router.get("/", response_model=List[ModelResponse])
async def list_models(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    search: str | None = None
):
    """List all models with pagination and search"""
    models = repo.get_all_models(skip=skip, limit=limit, search=search)
    return models

@router.get("/{model_name}", response_model=ModelDetail)
async def get_model(model_name: str):
    """Get detailed information for a specific model"""
    model = repo.get_model_details(model_name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model
```

#### Step 2: TypeScript Type Generation
```bash
# Install datamodel-code-generator
uv add datamodel-code-generator

# Generate TypeScript types from Pydantic models
datamodel-codegen \
  --input src/llm_benchmarks/models/ \
  --output src/frontend/src/lib/types/ \
  --output-model-type=typescript
```

#### Step 3: SvelteKit Frontend Setup
```bash
# Create SvelteKit project
npm create svelte@latest src/frontend
# Choose: Skeleton project, TypeScript, Prettier, Playwright

cd src/frontend
npm install

# Add dependencies
npm install -D tailwindcss autoprefixer postcss
npm install chart.js
npm install @sveltejs/adapter-auto
npx tailwindcss init -p
```

**svelte.config.js:**
```javascript
import adapter from '@sveltejs/adapter-auto';

export default {
  kit: {
    adapter: adapter(),
    alias: {
      $lib: 'src/lib'
    }
  }
};
```

#### Step 4: Create API Client
**src/frontend/src/lib/api.ts:**
```typescript
import type { ModelResponse, ModelDetail, TrendData } from './types';

const API_BASE = 'http://localhost:8000/api';

export async function getModels(
  skip = 0,
  limit = 100,
  search?: string
): Promise<ModelResponse[]> {
  const params = new URLSearchParams({
    skip: skip.toString(),
    limit: limit.toString(),
    ...(search && { search })
  });

  const response = await fetch(`${API_BASE}/models?${params}`);
  if (!response.ok) throw new Error('Failed to fetch models');
  return response.json();
}

export async function getModelDetails(name: string): Promise<ModelDetail> {
  const response = await fetch(`${API_BASE}/models/${encodeURIComponent(name)}`);
  if (!response.ok) throw new Error('Model not found');
  return response.json();
}

export async function getTrends(modelName: string): Promise<TrendData[]> {
  const response = await fetch(`${API_BASE}/trends/${encodeURIComponent(modelName)}`);
  if (!response.ok) throw new Error('Failed to fetch trends');
  return response.json();
}
```

#### Step 5: Create Svelte Stores
**src/frontend/src/lib/stores/models.ts:**
```typescript
import { writable, derived } from 'svelte/store';
import type { ModelResponse } from '$lib/types';
import { getModels } from '$lib/api';

function createModelsStore() {
  const { subscribe, set, update } = writable<ModelResponse[]>([]);

  return {
    subscribe,
    load: async (search?: string) => {
      const models = await getModels(0, 100, search);
      set(models);
    },
    search: writable('')
  };
}

export const models = createModelsStore();
export const searchQuery = writable('');

export const filteredModels = derived(
  [models, searchQuery],
  ([$models, $search]) => {
    if (!$search) return $models;
    return $models.filter(m =>
      m.name.toLowerCase().includes($search.toLowerCase())
    );
  }
);
```

#### Step 6: Build Dashboard Page
**src/frontend/src/routes/+page.svelte:**
```svelte
<script lang="ts">
  import { onMount } from 'svelte';
  import { getStats, getTopModels } from '$lib/api';
  import BarChart from '$lib/components/BarChart.svelte';

  let stats = { total_models: 0, total_benchmarks: 0, total_scores: 0 };
  let topModels = [];

  onMount(async () => {
    [stats, topModels] = await Promise.all([
      getStats(),
      getTopModels(10)
    ]);
  });
</script>

<div class="container mx-auto p-6">
  <h1 class="text-4xl font-bold mb-8">LLM Benchmark Dashboard</h1>

  <!-- Stats Cards -->
  <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
    <div class="bg-white rounded-lg shadow p-6">
      <p class="text-gray-500 text-sm">Total Models</p>
      <p class="text-3xl font-bold">{stats.total_models}</p>
    </div>
    <div class="bg-white rounded-lg shadow p-6">
      <p class="text-gray-500 text-sm">Benchmarks</p>
      <p class="text-3xl font-bold">{stats.total_benchmarks}</p>
    </div>
    <div class="bg-white rounded-lg shadow p-6">
      <p class="text-gray-500 text-sm">Data Points</p>
      <p class="text-3xl font-bold">{stats.total_scores}</p>
    </div>
  </div>

  <!-- Top Models Chart -->
  <div class="bg-white rounded-lg shadow p-6">
    <h2 class="text-2xl font-semibold mb-4">Top 10 Models</h2>
    <BarChart data={topModels} />
  </div>
</div>
```

#### Step 7: Build Model Explorer
**src/frontend/src/routes/models/+page.svelte:**
```svelte
<script lang="ts">
  import { models, searchQuery } from '$lib/stores/models';
  import { onMount } from 'svelte';

  onMount(() => models.load());

  function handleSearch(e: Event) {
    const target = e.target as HTMLInputElement;
    $searchQuery = target.value;
  }
</script>

<div class="container mx-auto p-6">
  <h1 class="text-4xl font-bold mb-8">Model Explorer</h1>

  <input
    type="text"
    placeholder="Search models..."
    on:input={handleSearch}
    class="w-full px-4 py-2 border rounded-lg mb-6"
  />

  <div class="bg-white rounded-lg shadow overflow-hidden">
    <table class="min-w-full">
      <thead class="bg-gray-50">
        <tr>
          <th class="px-6 py-3 text-left">Model Name</th>
          <th class="px-6 py-3 text-left">Avg Score</th>
          <th class="px-6 py-3 text-left">Benchmarks</th>
          <th class="px-6 py-3 text-left">Last Seen</th>
        </tr>
      </thead>
      <tbody>
        {#each $models as model}
          <tr class="border-t hover:bg-gray-50">
            <td class="px-6 py-4">
              <a href="/models/{model.name}" class="text-blue-600 hover:underline">
                {model.name}
              </a>
            </td>
            <td class="px-6 py-4">{model.average_score?.toFixed(2) ?? 'N/A'}</td>
            <td class="px-6 py-4">{model.benchmark_count}</td>
            <td class="px-6 py-4">{model.last_seen}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
</div>
```

#### Step 8: Implement WebSocket Chat
**Backend (src/backend/websockets.py):**
```python
from fastapi import WebSocket
from llm_benchmarks.llm.ollama_client import OllamaClient
from llm_benchmarks.data_aggregator import DataAggregator

@app.websocket("/ws/chat")
async def chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    client = OllamaClient()
    aggregator = DataAggregator()

    try:
        while True:
            question = await websocket.receive_text()
            data = aggregator.aggregate()

            async for chunk in client.ask_question_stream_async(question, data):
                await websocket.send_text(chunk)

            await websocket.send_text("[DONE]")
    except WebSocketDisconnect:
        pass
```

**Frontend (src/frontend/src/routes/chat/+page.svelte):**
```svelte
<script lang="ts">
  let messages: Array<{role: 'user' | 'assistant', content: string}> = [];
  let input = '';
  let ws: WebSocket | null = null;
  let currentResponse = '';

  function connect() {
    ws = new WebSocket('ws://localhost:8000/ws/chat');

    ws.onmessage = (event) => {
      if (event.data === '[DONE]') {
        messages = [...messages, { role: 'assistant', content: currentResponse }];
        currentResponse = '';
      } else {
        currentResponse += event.data;
      }
    };
  }

  function sendMessage() {
    if (!input.trim() || !ws) return;

    messages = [...messages, { role: 'user', content: input }];
    ws.send(input);
    input = '';
  }

  onMount(connect);
</script>

<div class="flex flex-col h-screen">
  <div class="flex-1 overflow-y-auto p-6 space-y-4">
    {#each messages as message}
      <div class="flex {message.role === 'user' ? 'justify-end' : 'justify-start'}">
        <div class="max-w-lg px-4 py-2 rounded-lg {
          message.role === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200'
        }">
          {message.content}
        </div>
      </div>
    {/each}

    {#if currentResponse}
      <div class="flex justify-start">
        <div class="max-w-lg px-4 py-2 rounded-lg bg-gray-200">
          {currentResponse}
        </div>
      </div>
    {/if}
  </div>

  <div class="border-t p-4">
    <div class="flex gap-2">
      <input
        type="text"
        bind:value={input}
        on:keydown={(e) => e.key === 'Enter' && sendMessage()}
        placeholder="Ask about benchmarks..."
        class="flex-1 px-4 py-2 border rounded-lg"
      />
      <button
        on:click={sendMessage}
        class="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
      >
        Send
      </button>
    </div>
  </div>
</div>
```

#### Step 9: Build Trends Chart
**src/frontend/src/lib/components/TrendChart.svelte:**
```svelte
<script lang="ts">
  import { onMount } from 'svelte';
  import { Chart, registerables } from 'chart.js';
  import type { TrendData } from '$lib/types';

  export let data: TrendData[];

  let canvas: HTMLCanvasElement;
  let chart: Chart;

  onMount(() => {
    Chart.register(...registerables);

    chart = new Chart(canvas, {
      type: 'line',
      data: {
        labels: data.map(d => d.date),
        datasets: [{
          label: 'Score',
          data: data.map(d => d.score),
          borderColor: 'rgb(59, 130, 246)',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          tension: 0.4
        }]
      },
      options: {
        responsive: true,
        interaction: {
          intersect: false,
          mode: 'index'
        },
        plugins: {
          legend: {
            display: false
          }
        }
      }
    });

    return () => chart.destroy();
  });
</script>

<canvas bind:this={canvas}></canvas>
```

#### Step 10: Deployment

**Backend (Railway/Fly.io):**
```bash
# Railway
railway init
railway up

# Or Fly.io
fly launch
fly deploy
```

**Frontend (Vercel):**
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
cd src/frontend
vercel
```

### Project Structure
```
src/
├── backend/
│   ├── main.py
│   ├── routers/
│   │   ├── models.py
│   │   ├── benchmarks.py
│   │   ├── trends.py
│   │   └── compare.py
│   ├── websockets.py
│   └── schemas.py
└── frontend/
    ├── src/
    │   ├── routes/
    │   │   ├── +page.svelte          # Dashboard
    │   │   ├── models/
    │   │   │   ├── +page.svelte      # Model list
    │   │   │   └── [name]/+page.svelte  # Model detail
    │   │   ├── trends/+page.svelte
    │   │   ├── compare/+page.svelte
    │   │   └── chat/+page.svelte
    │   ├── lib/
    │   │   ├── api.ts                # API client
    │   │   ├── types/                # TypeScript types
    │   │   ├── stores/               # Svelte stores
    │   │   └── components/
    │   │       ├── BarChart.svelte
    │   │       ├── TrendChart.svelte
    │   │       ├── RadarChart.svelte
    │   │       └── ModelTable.svelte
    │   └── app.css
    ├── package.json
    └── svelte.config.js
```

### Advantages
- ✅ **Lightweight** - Svelte compiles to vanilla JS, small bundle size
- ✅ **Excellent performance** - No virtual DOM overhead
- ✅ **Great DX** - Simple syntax, less boilerplate than React
- ✅ **Built-in reactivity** - No need for useState/useEffect
- ✅ **Modern** - SvelteKit provides SSR, routing, API routes
- ✅ **Type-safe** - TypeScript throughout, generated from Pydantic
- ✅ **FastAPI benefits** - Auto docs, validation, async support
- ✅ **WebSocket native** - Real-time features built-in

### Limitations
- ❌ **Smaller ecosystem** - Fewer libraries than React
- ❌ **Learning curve** - New framework for most developers
- ❌ **Job market** - Less Svelte expertise available
- ❌ **Component libraries** - Fewer pre-built UI components
- ❌ **Two deployments** - Frontend and backend separate

### Best For
- Modern web applications
- Performance-critical dashboards
- Teams open to learning new frameworks
- Projects valuing developer experience
- Public-facing applications

---

## Approach 3: Next.js + FastAPI - Production-Grade

### Overview
Build a full-featured, enterprise-ready web application with Next.js frontend and FastAPI backend. Leverages the React ecosystem for maximum flexibility and scalability.

### Technology Stack
- **Frontend Framework:** Next.js 15 (App Router)
- **Language:** TypeScript
- **UI Library:** React 19
- **Styling:** Tailwind CSS + shadcn/ui
- **Charts:** Recharts + TanStack Table
- **State Management:** TanStack Query (React Query) + Zustand
- **Real-time:** WebSocket + Server-Sent Events
- **Backend API:** FastAPI
- **API Client:** TanStack Query with auto-generated client
- **Testing:** Vitest + Playwright
- **Deployment:** Vercel (frontend) + Railway (backend)

### Architecture

```
┌─────────────────────────────────────────────────────┐
│           Next.js 15 (App Router)                   │
│                                                     │
│  ┌────────────────────────────────────────────┐   │
│  │  App Routes (app/)                          │   │
│  │  - page.tsx (Dashboard)                     │   │
│  │  - models/page.tsx                          │   │
│  │  - trends/page.tsx                          │   │
│  │  - compare/page.tsx                         │   │
│  │  - chat/page.tsx                            │   │
│  └────────────────────────────────────────────┘   │
│                                                     │
│  ┌────────────────────────────────────────────┐   │
│  │  Components (components/)                   │   │
│  │  - ui/ (shadcn/ui)                          │   │
│  │  - charts/ (Recharts)                       │   │
│  │  - tables/ (TanStack Table)                 │   │
│  └────────────────────────────────────────────┘   │
│                                                     │
│  ┌────────────────────────────────────────────┐   │
│  │  Data Layer (lib/)                          │   │
│  │  - TanStack Query hooks                     │   │
│  │  - Zustand stores                           │   │
│  │  - API client (auto-generated)              │   │
│  └────────────────────────────────────────────┘   │
│                                                     │
│  ┌────────────────────────────────────────────┐   │
│  │  Server Components (RSC)                    │   │
│  │  - Pre-fetch data on server                 │   │
│  │  - Streaming SSR                            │   │
│  └────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
                    ↓ HTTP/WebSocket
┌─────────────────────────────────────────────────────┐
│           FastAPI Backend (Production)              │
│                                                     │
│  ┌────────────────────────────────────────────┐   │
│  │  REST API v1 (/api/v1/)                     │   │
│  │  - Pagination, filtering, sorting           │   │
│  │  - Rate limiting                            │   │
│  │  - Authentication (JWT)                     │   │
│  │  - Caching (Redis)                          │   │
│  └────────────────────────────────────────────┘   │
│                                                     │
│  ┌────────────────────────────────────────────┐   │
│  │  WebSocket Gateway                          │   │
│  │  - Chat streaming                           │   │
│  │  - Live updates                             │   │
│  │  - Connection management                    │   │
│  └────────────────────────────────────────────┘   │
│                                                     │
│  ┌────────────────────────────────────────────┐   │
│  │  Background Tasks (Celery)                  │   │
│  │  - Scheduled scraping                       │   │
│  │  - Report generation                        │   │
│  │  - Data processing                          │   │
│  └────────────────────────────────────────────┘   │
│                                                     │
│  ┌────────────────────────────────────────────┘   │
│  │  Database Layer                             │   │
│  │  - SQLite/PostgreSQL                        │   │
│  │  - Migrations (Alembic)                     │   │
│  │  - Connection pooling                       │   │
│  └────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Key Features

#### Phase 1: Backend API Foundation
1. **RESTful API with Versioning**
   - `/api/v1/models` - CRUD operations with pagination
   - `/api/v1/benchmarks` - Benchmark management
   - `/api/v1/trends` - Historical data queries
   - `/api/v1/compare` - Model comparisons
   - `/api/v1/reports` - Report generation
   - `/api/v1/sources` - Data source management

2. **Advanced API Features**
   - Request/response validation with Pydantic
   - Rate limiting (per IP/user)
   - JWT authentication (optional)
   - API key management
   - Swagger/ReDoc documentation
   - Request logging and monitoring

3. **Performance Optimizations**
   - Redis caching layer
   - Database query optimization
   - Response compression
   - CDN integration for static assets

#### Phase 2: Next.js Frontend Foundation
4. **Modern React Setup**
   - Server Components for initial load
   - Client Components for interactivity
   - Streaming SSR for progressive rendering
   - Image optimization with next/image
   - Font optimization

5. **UI Component System**
   - shadcn/ui for beautiful, accessible components
   - Dark mode support
   - Responsive design (mobile-first)
   - Keyboard navigation
   - ARIA compliance

6. **Data Fetching Strategy**
   - TanStack Query for server state
   - Automatic background refetching
   - Optimistic updates
   - Infinite scroll pagination
   - Prefetching on hover

#### Phase 3: Core Features
7. **Dashboard** (`/`)
   - Real-time stats cards
   - Top models visualization
   - Recent updates feed
   - Activity timeline
   - Quick actions panel

8. **Model Explorer** (`/models`)
   - Advanced filtering (multi-select, ranges)
   - Column sorting and customization
   - Virtualized scrolling for large datasets
   - Export to CSV/JSON
   - Saved filter presets
   - Model comparison selection

9. **Trends Analysis** (`/trends`)
   - Interactive time-series charts
   - Multiple y-axis support
   - Zoom and pan
   - Brush selection for range
   - Annotation support
   - Export charts as PNG/SVG

10. **Comparison Tool** (`/compare`)
    - Drag-and-drop model selection
    - Multiple visualization modes:
      - Radar chart overlay
      - Bar chart side-by-side
      - Heatmap grid
      - Scatter plot
    - Statistical analysis
    - Shareable comparison URLs

#### Phase 4: Advanced Features
11. **Chat Interface** (`/chat`)
    - Real-time streaming responses
    - Conversation history persistence
    - Markdown rendering with syntax highlighting
    - Code block copy buttons
    - Suggested follow-up questions
    - Export conversations

12. **Report Builder** (`/reports`)
    - Visual drag-and-drop editor
    - Template library
    - Custom branding
    - Schedule automated reports
    - Email delivery
    - PDF generation

13. **Admin Panel** (`/admin`)
    - Data source configuration
    - Scraper scheduling (cron)
    - User management (if multi-user)
    - API key management
    - System health monitoring
    - Audit logs

14. **User Settings** (`/settings`)
    - Appearance (theme, density)
    - Notification preferences
    - Default filters
    - API integration settings
    - Export/import preferences

### Implementation Roadmap

#### Step 1: Backend API Setup

**Create FastAPI App with Production Structure:**
```bash
mkdir -p src/api
cd src/api

# Create production-grade structure
mkdir -p app/{api,core,db,models,schemas,services}
touch app/{__init__.py,main.py}
touch app/api/{__init__.py,deps.py}
mkdir app/api/v1
touch app/api/v1/{__init__.py,router.py}
```

**app/main.py:**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from app.api.v1.router import api_router
from app.core.config import settings

app = FastAPI(
    title="LLM Benchmark API",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

**app/api/v1/router.py:**
```python
from fastapi import APIRouter
from app.api.v1.endpoints import models, benchmarks, trends, compare

api_router = APIRouter()

api_router.include_router(
    models.router,
    prefix="/models",
    tags=["models"]
)
api_router.include_router(
    benchmarks.router,
    prefix="/benchmarks",
    tags=["benchmarks"]
)
api_router.include_router(
    trends.router,
    prefix="/trends",
    tags=["trends"]
)
api_router.include_router(
    compare.router,
    prefix="/compare",
    tags=["compare"]
)
```

**app/api/v1/endpoints/models.py:**
```python
from fastapi import APIRouter, Depends, Query
from typing import List
from app.schemas.model import ModelList, ModelDetail
from app.services.model_service import ModelService
from app.api.deps import get_model_service

router = APIRouter()

@router.get("/", response_model=ModelList)
async def list_models(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    search: str | None = None,
    sort_by: str = Query("average_score", regex="^(name|average_score|last_seen)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    service: ModelService = Depends(get_model_service)
):
    """List models with pagination, search, and sorting"""
    return await service.list_models(
        skip=skip,
        limit=limit,
        search=search,
        sort_by=sort_by,
        sort_order=sort_order
    )

@router.get("/{model_name}", response_model=ModelDetail)
async def get_model(
    model_name: str,
    service: ModelService = Depends(get_model_service)
):
    """Get detailed model information"""
    return await service.get_model_detail(model_name)
```

#### Step 2: Generate TypeScript Client

**Install OpenAPI Generator:**
```bash
npm install -g @openapitools/openapi-generator-cli
```

**Generate Client:**
```bash
# Start FastAPI to expose OpenAPI schema
cd src/api
uvicorn app.main:app --reload

# In another terminal
openapi-generator-cli generate \
  -i http://localhost:8000/api/v1/openapi.json \
  -g typescript-fetch \
  -o src/web/src/lib/api-client
```

#### Step 3: Next.js Setup

**Create Next.js App:**
```bash
npx create-next-app@latest src/web \
  --typescript \
  --tailwind \
  --app \
  --import-alias "@/*"

cd src/web
```

**Install Dependencies:**
```bash
npm install @tanstack/react-query
npm install zustand
npm install recharts
npm install @tanstack/react-table
npm install date-fns
npm install lucide-react
npm install react-markdown
npm install remark-gfm
npm install rehype-highlight
```

**Install shadcn/ui:**
```bash
npx shadcn@latest init
npx shadcn@latest add button card input table tabs dialog dropdown-menu
```

#### Step 4: Configure TanStack Query

**src/web/src/app/providers.tsx:**
```typescript
'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { useState } from 'react';

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(() => new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 60 * 1000, // 1 minute
        refetchOnWindowFocus: false,
      },
    },
  }));

  return (
    <QueryClientProvider client={queryClient}>
      {children}
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}
```

**src/web/src/app/layout.tsx:**
```typescript
import { Providers } from './providers';
import './globals.css';

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
```

#### Step 5: Create API Hooks

**src/web/src/lib/hooks/use-models.ts:**
```typescript
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { ModelsApi } from '@/lib/api-client';

const modelsApi = new ModelsApi();

export function useModels(params?: {
  skip?: number;
  limit?: number;
  search?: string;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}) {
  return useQuery({
    queryKey: ['models', params],
    queryFn: () => modelsApi.listModels(params),
  });
}

export function useModel(name: string) {
  return useQuery({
    queryKey: ['model', name],
    queryFn: () => modelsApi.getModel({ modelName: name }),
    enabled: !!name,
  });
}

export function useInfiniteModels(search?: string) {
  return useInfiniteQuery({
    queryKey: ['models', 'infinite', search],
    queryFn: ({ pageParam = 0 }) =>
      modelsApi.listModels({ skip: pageParam, limit: 50, search }),
    getNextPageParam: (lastPage, allPages) => {
      const nextSkip = allPages.length * 50;
      return lastPage.data.length === 50 ? nextSkip : undefined;
    },
  });
}
```

#### Step 6: Build Dashboard Page

**src/web/src/app/page.tsx:**
```typescript
import { Suspense } from 'react';
import { StatsCards } from '@/components/dashboard/stats-cards';
import { TopModelsChart } from '@/components/dashboard/top-models-chart';
import { RecentUpdates } from '@/components/dashboard/recent-updates';
import { QuickActions } from '@/components/dashboard/quick-actions';
import { Skeleton } from '@/components/ui/skeleton';

export default function DashboardPage() {
  return (
    <div className="container mx-auto p-6 space-y-8">
      <div>
        <h1 className="text-4xl font-bold tracking-tight">LLM Benchmark Dashboard</h1>
        <p className="text-muted-foreground mt-2">
          Track and compare language model performance across benchmarks
        </p>
      </div>

      <Suspense fallback={<StatsSkeleton />}>
        <StatsCards />
      </Suspense>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Suspense fallback={<ChartSkeleton />}>
            <TopModelsChart />
          </Suspense>
        </div>
        <div>
          <QuickActions />
        </div>
      </div>

      <Suspense fallback={<UpdatesSkeleton />}>
        <RecentUpdates />
      </Suspense>
    </div>
  );
}
```

**src/web/src/components/dashboard/stats-cards.tsx:**
```typescript
'use client';

import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Database, Target, TrendingUp, Globe } from 'lucide-react';

export function StatsCards() {
  const { data: stats } = useQuery({
    queryKey: ['stats'],
    queryFn: async () => {
      const res = await fetch('/api/v1/stats');
      return res.json();
    },
  });

  const cards = [
    {
      title: 'Total Models',
      value: stats?.total_models ?? 0,
      icon: Database,
      description: '+12 from last month',
    },
    {
      title: 'Benchmarks',
      value: stats?.total_benchmarks ?? 0,
      icon: Target,
      description: 'Across all models',
    },
    {
      title: 'Data Points',
      value: stats?.total_scores?.toLocaleString() ?? 0,
      icon: TrendingUp,
      description: 'Historical records',
    },
    {
      title: 'Data Sources',
      value: stats?.total_sources ?? 0,
      icon: Globe,
      description: 'Active scrapers',
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {cards.map((card) => (
        <Card key={card.title}>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              {card.title}
            </CardTitle>
            <card.icon className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{card.value}</div>
            <p className="text-xs text-muted-foreground mt-1">
              {card.description}
            </p>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
```

#### Step 7: Build Model Explorer with TanStack Table

**src/web/src/app/models/page.tsx:**
```typescript
'use client';

import { useState } from 'react';
import { useModels } from '@/lib/hooks/use-models';
import { ModelTable } from '@/components/models/model-table';
import { ModelFilters } from '@/components/models/model-filters';
import { Button } from '@/components/ui/button';
import { Download } from 'lucide-react';

export default function ModelsPage() {
  const [filters, setFilters] = useState({
    search: '',
    sortBy: 'average_score',
    sortOrder: 'desc' as const,
  });

  const { data, isLoading } = useModels(filters);

  const handleExport = () => {
    // Export to CSV logic
    const csv = generateCSV(data?.data);
    downloadFile(csv, 'models.csv');
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-4xl font-bold">Model Explorer</h1>
          <p className="text-muted-foreground mt-2">
            Browse and compare {data?.total ?? 0} language models
          </p>
        </div>
        <Button onClick={handleExport} variant="outline">
          <Download className="mr-2 h-4 w-4" />
          Export CSV
        </Button>
      </div>

      <ModelFilters filters={filters} onChange={setFilters} />

      <ModelTable data={data?.data ?? []} isLoading={isLoading} />
    </div>
  );
}
```

**src/web/src/components/models/model-table.tsx:**
```typescript
'use client';

import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  flexRender,
} from '@tanstack/react-table';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import Link from 'next/link';

export function ModelTable({ data, isLoading }) {
  const columns = [
    {
      accessorKey: 'name',
      header: 'Model Name',
      cell: ({ row }) => (
        <Link
          href={`/models/${row.original.name}`}
          className="font-medium text-blue-600 hover:underline"
        >
          {row.original.name}
        </Link>
      ),
    },
    {
      accessorKey: 'average_score',
      header: 'Avg Score',
      cell: ({ row }) => {
        const score = row.original.average_score;
        return score ? score.toFixed(2) : 'N/A';
      },
    },
    {
      accessorKey: 'benchmark_count',
      header: 'Benchmarks',
      cell: ({ row }) => (
        <Badge variant="secondary">{row.original.benchmark_count}</Badge>
      ),
    },
    {
      accessorKey: 'last_seen',
      header: 'Last Seen',
      cell: ({ row }) => {
        const date = new Date(row.original.last_seen);
        return date.toLocaleDateString();
      },
    },
  ];

  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
  });

  if (isLoading) {
    return <TableSkeleton />;
  }

  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow key={headerGroup.id}>
              {headerGroup.headers.map((header) => (
                <TableHead key={header.id}>
                  {flexRender(header.column.columnDef.header, header.getContext())}
                </TableHead>
              ))}
            </TableRow>
          ))}
        </TableHeader>
        <TableBody>
          {table.getRowModel().rows.map((row) => (
            <TableRow key={row.id}>
              {row.getVisibleCells().map((cell) => (
                <TableCell key={cell.id}>
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
```

#### Step 8: Build Trends Chart with Recharts

**src/web/src/app/trends/page.tsx:**
```typescript
'use client';

import { useState } from 'react';
import { useTrends } from '@/lib/hooks/use-trends';
import { TrendChart } from '@/components/trends/trend-chart';
import { ModelSelector } from '@/components/trends/model-selector';
import { DateRangePicker } from '@/components/trends/date-range-picker';

export default function TrendsPage() {
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [dateRange, setDateRange] = useState({ from: null, to: null });

  const { data, isLoading } = useTrends(selectedModel, dateRange);

  return (
    <div className="container mx-auto p-6 space-y-6">
      <h1 className="text-4xl font-bold">Historical Trends</h1>

      <div className="flex gap-4">
        <ModelSelector value={selectedModel} onChange={setSelectedModel} />
        <DateRangePicker value={dateRange} onChange={setDateRange} />
      </div>

      <TrendChart data={data} isLoading={isLoading} />
    </div>
  );
}
```

**src/web/src/components/trends/trend-chart.tsx:**
```typescript
'use client';

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function TrendChart({ data, isLoading }) {
  if (isLoading) return <ChartSkeleton />;
  if (!data || data.length === 0) return <EmptyState />;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Score Evolution</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="score"
              stroke="#2563eb"
              strokeWidth={2}
              dot={{ r: 4 }}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
```

#### Step 9: Build Chat Interface with WebSocket

**src/web/src/app/chat/page.tsx:**
```typescript
'use client';

import { useState, useEffect, useRef } from 'react';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Send } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

type Message = {
  role: 'user' | 'assistant';
  content: string;
};

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Connect to WebSocket
    wsRef.current = new WebSocket('ws://localhost:8000/ws/chat');

    let currentResponse = '';

    wsRef.current.onmessage = (event) => {
      if (event.data === '[DONE]') {
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: currentResponse },
        ]);
        currentResponse = '';
        setIsStreaming(false);
      } else {
        currentResponse += event.data;
        setMessages((prev) => {
          const updated = [...prev];
          const lastMsg = updated[updated.length - 1];
          if (lastMsg?.role === 'assistant' && isStreaming) {
            lastMsg.content = currentResponse;
          } else {
            updated.push({ role: 'assistant', content: currentResponse });
          }
          return updated;
        });
      }
    };

    return () => {
      wsRef.current?.close();
    };
  }, []);

  useEffect(() => {
    // Auto-scroll to bottom
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = () => {
    if (!input.trim() || !wsRef.current) return;

    setMessages((prev) => [...prev, { role: 'user', content: input }]);
    wsRef.current.send(input);
    setInput('');
    setIsStreaming(true);
  };

  return (
    <div className="container mx-auto p-6 h-[calc(100vh-8rem)]">
      <Card className="h-full flex flex-col">
        <div className="p-6 border-b">
          <h1 className="text-2xl font-bold">Ask About Benchmarks</h1>
          <p className="text-sm text-muted-foreground">
            Chat with AI about model performance data
          </p>
        </div>

        <ScrollArea className="flex-1 p-6">
          <div className="space-y-4">
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg px-4 py-2 ${
                    msg.role === 'user'
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-100 dark:bg-gray-800'
                  }`}
                >
                  {msg.role === 'assistant' ? (
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      className="prose dark:prose-invert max-w-none"
                    >
                      {msg.content}
                    </ReactMarkdown>
                  ) : (
                    <p>{msg.content}</p>
                  )}
                </div>
              </div>
            ))}
            <div ref={scrollRef} />
          </div>
        </ScrollArea>

        <div className="p-6 border-t">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
            }}
            className="flex gap-2"
          >
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about model performance..."
              disabled={isStreaming}
            />
            <Button type="submit" disabled={isStreaming || !input.trim()}>
              <Send className="h-4 w-4" />
            </Button>
          </form>
        </div>
      </Card>
    </div>
  );
}
```

#### Step 10: Add Advanced Features

**Infinite Scroll:**
```typescript
import { useInView } from 'react-intersection-observer';
import { useInfiniteModels } from '@/lib/hooks/use-models';

export function InfiniteModelList() {
  const { data, fetchNextPage, hasNextPage, isFetchingNextPage } = useInfiniteModels();
  const { ref, inView } = useInView();

  useEffect(() => {
    if (inView && hasNextPage) {
      fetchNextPage();
    }
  }, [inView, hasNextPage, fetchNextPage]);

  return (
    <div>
      {data?.pages.map((page, i) => (
        <div key={i}>
          {page.data.map((model) => (
            <ModelCard key={model.name} model={model} />
          ))}
        </div>
      ))}
      <div ref={ref}>{isFetchingNextPage && <Spinner />}</div>
    </div>
  );
}
```

**Optimistic Updates:**
```typescript
export function useUpdateModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data) => modelsApi.updateModel(data),
    onMutate: async (newData) => {
      await queryClient.cancelQueries({ queryKey: ['model', newData.name] });
      const previous = queryClient.getQueryData(['model', newData.name]);
      queryClient.setQueryData(['model', newData.name], newData);
      return { previous };
    },
    onError: (err, newData, context) => {
      queryClient.setQueryData(['model', newData.name], context.previous);
    },
    onSettled: (data, error, variables) => {
      queryClient.invalidateQueries({ queryKey: ['model', variables.name] });
    },
  });
}
```

#### Step 11: Testing

**Vitest Setup:**
```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
  },
});
```

**Component Test:**
```typescript
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { StatsCards } from '@/components/dashboard/stats-cards';

describe('StatsCards', () => {
  it('renders stats correctly', async () => {
    const queryClient = new QueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <StatsCards />
      </QueryClientProvider>
    );

    expect(await screen.findByText('Total Models')).toBeInTheDocument();
  });
});
```

#### Step 12: Deployment

**Backend (Railway):**
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login and initialize
railway login
railway init
railway up
```

**Frontend (Vercel):**
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
cd src/web
vercel --prod
```

**Environment Variables:**
```bash
# .env.production
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://localhost:6379
```

### Project Structure
```
src/
├── api/                          # FastAPI Backend
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   │   ├── deps.py
│   │   │   └── v1/
│   │   │       ├── router.py
│   │   │       └── endpoints/
│   │   │           ├── models.py
│   │   │           ├── benchmarks.py
│   │   │           ├── trends.py
│   │   │           └── compare.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── security.py
│   │   ├── db/
│   │   │   └── session.py
│   │   ├── models/               # SQLAlchemy models
│   │   ├── schemas/              # Pydantic schemas
│   │   └── services/             # Business logic
│   ├── tests/
│   └── requirements.txt
│
└── web/                          # Next.js Frontend
    ├── src/
    │   ├── app/
    │   │   ├── layout.tsx
    │   │   ├── page.tsx          # Dashboard
    │   │   ├── providers.tsx
    │   │   ├── models/
    │   │   │   ├── page.tsx
    │   │   │   └── [name]/
    │   │   │       └── page.tsx
    │   │   ├── trends/
    │   │   │   └── page.tsx
    │   │   ├── compare/
    │   │   │   └── page.tsx
    │   │   └── chat/
    │   │       └── page.tsx
    │   ├── components/
    │   │   ├── ui/               # shadcn/ui
    │   │   ├── dashboard/
    │   │   ├── models/
    │   │   ├── trends/
    │   │   └── charts/
    │   └── lib/
    │       ├── api-client/       # Auto-generated
    │       ├── hooks/
    │       ├── stores/
    │       └── utils/
    ├── public/
    ├── tests/
    ├── package.json
    └── next.config.js
```

### Advantages
- ✅ **Production-ready** - Battle-tested stack used by major companies
- ✅ **Rich ecosystem** - Massive library of React components
- ✅ **Excellent DX** - TypeScript, auto-generated types, dev tools
- ✅ **Performance** - Server Components, streaming, automatic optimization
- ✅ **Scalability** - Designed for large applications
- ✅ **SEO-friendly** - Server-side rendering
- ✅ **Advanced features** - Infinite scroll, optimistic updates, prefetching
- ✅ **Testing** - Mature testing ecosystem
- ✅ **Deployment** - Easy with Vercel, excellent DX

### Limitations
- ❌ **Complexity** - Steeper learning curve
- ❌ **Bundle size** - React has overhead compared to Svelte
- ❌ **Development time** - More code to write
- ❌ **Cost** - May require more infrastructure (Redis, PostgreSQL)
- ❌ **Overkill** - May be too much for simple use cases

### Best For
- Production SaaS applications
- Enterprise projects
- Teams with React expertise
- Projects requiring scalability
- Feature-rich dashboards
- Public-facing products

---

## Comparison Matrix

| Feature | Streamlit | Svelte + FastAPI | Next.js + FastAPI |
|---------|-----------|------------------|-------------------|
| **Development Speed** | ⭐⭐⭐⭐⭐ Fastest | ⭐⭐⭐⭐ Fast | ⭐⭐⭐ Moderate |
| **User Experience** | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐⭐ Best |
| **Customization** | ⭐⭐ Limited | ⭐⭐⭐⭐ High | ⭐⭐⭐⭐⭐ Maximum |
| **Performance** | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Very Good |
| **Mobile Support** | ⭐⭐ Basic | ⭐⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent |
| **Learning Curve** | ⭐⭐⭐⭐⭐ Easy | ⭐⭐⭐ Moderate | ⭐⭐ Steep |
| **Ecosystem** | ⭐⭐⭐ Good | ⭐⭐⭐ Growing | ⭐⭐⭐⭐⭐ Massive |
| **Scalability** | ⭐⭐ Limited | ⭐⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent |
| **Type Safety** | ⭐⭐ Python only | ⭐⭐⭐⭐⭐ Full stack | ⭐⭐⭐⭐⭐ Full stack |
| **Deployment** | ⭐⭐⭐⭐⭐ Easiest | ⭐⭐⭐⭐ Easy | ⭐⭐⭐⭐ Easy |
| **Code Reuse** | ⭐⭐⭐⭐⭐ Direct | ⭐⭐⭐ Via API | ⭐⭐⭐ Via API |
| **Real-time Features** | ⭐⭐⭐ Basic | ⭐⭐⭐⭐⭐ Native | ⭐⭐⭐⭐ Good |

---

## Decision Framework

### Choose **Streamlit** if:
- You need a working dashboard ASAP (days, not weeks)
- Your team is Python-only
- This is an internal tool or MVP
- You want to avoid JavaScript entirely
- You prioritize code simplicity over customization
- You're comfortable with opinionated layouts

### Choose **Svelte + FastAPI** if:
- You want modern, performant UX without heavyweight frameworks
- Your team is willing to learn Svelte
- You need excellent performance (lightweight bundle)
- You want a balance of speed and customization
- You're building a public-facing application
- You value developer experience

### Choose **Next.js + FastAPI** if:
- You're building a production SaaS product
- Your team has React experience
- You need maximum flexibility and customization
- You plan to scale to many users
- You need advanced features (SSR, infinite scroll, etc.)
- You want the largest ecosystem of components
- You're prepared for a more complex architecture

---

## Other Considerations

### 1. Hybrid Approach: Start with Streamlit, Migrate Later
**Strategy:**
- Build initial version with Streamlit for rapid validation
- Collect user feedback
- If successful, migrate to Svelte or Next.js for production

**Pros:**
- Fastest time to market
- Low initial investment
- Validate before committing to complex build

**Cons:**
- Potential rewrite costs
- Users may expect consistency
- May delay optimal UX

### 2. Backend-Only API: Let Users Choose Frontend
**Strategy:**
- Build robust FastAPI backend with excellent docs
- Provide OpenAPI schema for auto-generated clients
- Community can build frontends (React, Vue, etc.)

**Pros:**
- Focus on core value (data aggregation)
- Multiple frontend options
- Community engagement

**Cons:**
- No official UI out of the box
- Fragmented user experience
- Requires technical users

### 3. CLI + Dashboard: Best of Both Worlds
**Strategy:**
- Keep existing rich CLI for power users
- Add lightweight Streamlit dashboard for visualization
- Both use the same Python modules

**Pros:**
- Serve multiple user personas
- Minimal disruption to existing users
- Leverage existing codebase

**Cons:**
- Maintain two interfaces
- Feature parity challenges
- Split development effort

### 4. Progressive Enhancement
**Strategy:**
- Phase 1: Streamlit dashboard (quick win)
- Phase 2: Add FastAPI endpoints for mobile/integrations
- Phase 3: Rebuild frontend with Next.js incrementally

**Pros:**
- Gradual investment
- Always have working product
- Learn from each phase

**Cons:**
- Longer overall timeline
- Potential technical debt
- Context switching between phases

### 5. Static Site Generation (SSG)
**Strategy:**
- Use Next.js or Astro to generate static HTML pages
- Pre-render charts and tables at build time
- Update daily via CI/CD

**Pros:**
- Ultra-fast load times
- No backend needed for viewing
- Cheap hosting (Netlify, GitHub Pages)

**Cons:**
- No real-time updates
- Limited interactivity
- Requires build step for updates

---

## Final Recommendation

Based on the current state of your application, I recommend:

### **Phase 1: Streamlit (2-3 weeks)**
Build a functional dashboard quickly to validate the concept and gather user feedback. This lets you:
- Demonstrate value immediately
- Learn what users actually need
- Keep development simple

### **Phase 2: Evaluate & Decide**
After using Streamlit for 1-2 months:
- If users love it and customization is fine → enhance Streamlit
- If users need better mobile/UX → migrate to Svelte
- If you're building a product company → go Next.js

### **Phase 3: Production Build**
Based on Phase 2 learnings, either:
- **Option A:** Polish Streamlit with custom CSS and components
- **Option B:** Build production frontend with Svelte (balance) or Next.js (max power)

This approach minimizes risk while keeping options open for future growth.

---

## Appendix: Quick Start Commands

### Streamlit
```bash
uv add streamlit plotly
mkdir -p src/frontend/streamlit
streamlit run src/frontend/streamlit/app.py
```

### Svelte + FastAPI
```bash
# Backend
mkdir src/api && cd src/api
uv init && uv add fastapi uvicorn

# Frontend
npm create svelte@latest src/frontend
cd src/frontend && npm install
```

### Next.js + FastAPI
```bash
# Backend
mkdir src/api && cd src/api
uv init && uv add fastapi uvicorn

# Frontend
npx create-next-app@latest src/web --typescript --tailwind
cd src/web && npm install @tanstack/react-query recharts
```

---

**Document Version:** 1.0
**Last Updated:** 2026-01-10
**Next Review:** After frontend approach is selected
