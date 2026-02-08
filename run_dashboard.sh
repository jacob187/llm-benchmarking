#!/bin/bash
# Run the LLM Benchmark Streamlit Dashboard

# Navigate to project root
cd "$(dirname "$0")"

# Ensure dependencies are installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install uv first."
    exit 1
fi

# Sync dependencies to ensure everything is up to date
echo "Syncing dependencies..."
uv sync

# Run Streamlit
echo "Starting LLM Benchmark Dashboard..."
echo "Access at: http://localhost:8501"
echo ""

uv run streamlit run src/frontend/streamlit/app.py \
    --server.port 8501 \
    --server.address localhost \
    --browser.gatherUsageStats false
