#!/bin/bash
# Run the LLM Benchmark Streamlit Dashboard

# Navigate to project root
cd "$(dirname "$0")"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit not found. Installing dependencies..."
    uv sync
fi

# Run Streamlit
echo "Starting LLM Benchmark Dashboard..."
echo "Access at: http://localhost:8501"
echo ""

streamlit run src/frontend/streamlit/app.py \
    --server.port 8501 \
    --server.address localhost \
    --browser.gatherUsageStats false
