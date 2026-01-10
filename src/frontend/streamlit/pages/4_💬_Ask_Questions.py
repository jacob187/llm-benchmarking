"""
Q&A Chat Page

Ask questions about benchmark data with streaming LLM responses.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.frontend.streamlit.utils.data_loader import load_aggregator_cache

# Page configuration
st.set_page_config(
    page_title="Ask Questions | LLM Benchmarks",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Ask Questions About Benchmarks")
st.markdown("Chat with AI about model performance and benchmark data")

# Check Ollama availability
try:
    from llm_benchmarks.llm import check_ollama_available, BenchmarkAnalyzer

    ollama_available = check_ollama_available()
except ImportError as e:
    ollama_available = False
    st.error(f"Could not import LLM modules: {e}")

# Sidebar
with st.sidebar:
    st.header("Chat Settings")

    if ollama_available:
        st.success("Ollama connected")
    else:
        st.error("Ollama not available")
        st.markdown("""
        **To use this feature:**

        1. Install Ollama from [ollama.ai](https://ollama.ai)
        2. Start the Ollama service:
           ```bash
           ollama serve
           ```
        3. Pull a model (e.g., gemma3):
           ```bash
           ollama pull gemma3:4b
           ```
        4. Refresh this page
        """)

    st.markdown("---")

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    # Suggested questions
    st.subheader("Suggested Questions")

    suggestions = [
        "What are the top 5 coding models?",
        "Which model has the best average score?",
        "Compare Claude and GPT-4",
        "What benchmarks are most commonly tested?",
        "Which models improved recently?",
    ]

    for suggestion in suggestions:
        if st.button(suggestion, key=f"suggest_{suggestion[:20]}"):
            st.session_state.pending_question = suggestion
            st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# Check for data
aggregator = load_aggregator_cache()

if not aggregator or not aggregator.models:
    st.warning(
        "No benchmark data available. "
        "Run `llm-bench scrape` to collect data for the AI to analyze."
    )
    st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle pending question from sidebar
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate response
    if ollama_available:
        with st.chat_message("assistant"):
            try:
                analyzer = BenchmarkAnalyzer()

                # Stream response
                response_placeholder = st.empty()
                full_response = ""

                for chunk in analyzer.answer_question_stream(
                    question, aggregator.models
                ):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                })

            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })
    else:
        with st.chat_message("assistant"):
            msg = "Ollama is not available. Please start the Ollama service."
            st.warning(msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": msg,
            })

    st.rerun()

# Chat input
if question := st.chat_input(
    "Ask about benchmark data...",
    disabled=not ollama_available,
):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate response
    with st.chat_message("assistant"):
        try:
            analyzer = BenchmarkAnalyzer()

            # Stream response
            response_placeholder = st.empty()
            full_response = ""

            for chunk in analyzer.answer_question_stream(
                question, aggregator.models
            ):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
            })

        except ConnectionError:
            error_msg = (
                "Could not connect to Ollama. "
                "Please ensure the service is running."
            )
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
            })

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
            })

# Export conversation
if st.session_state.messages:
    st.markdown("---")

    with st.expander("Export Conversation"):
        # Format conversation
        conversation = []
        for msg in st.session_state.messages:
            role = msg["role"].title()
            content = msg["content"]
            conversation.append(f"**{role}:** {content}\n")

        markdown_content = "\n".join(conversation)

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="📥 Download as Markdown",
                data=markdown_content,
                file_name="chat_conversation.md",
                mime="text/markdown",
            )

        with col2:
            # Plain text version
            plain_text = "\n\n".join([
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in st.session_state.messages
            ])

            st.download_button(
                label="📥 Download as Text",
                data=plain_text,
                file_name="chat_conversation.txt",
                mime="text/plain",
            )

# Info section
if not st.session_state.messages:
    st.markdown("---")
    st.info(
        "💡 **Tip:** Ask questions about model performance, comparisons, "
        "benchmark trends, or any other aspect of the LLM benchmark data. "
        "The AI has access to all collected benchmark information."
    )

    st.markdown("""
    **Example questions:**
    - "Which models are best at coding tasks?"
    - "How does GPT-4 compare to Claude on reasoning benchmarks?"
    - "What are the latest trends in LLM performance?"
    - "Summarize the top 10 models and their strengths"
    """)
