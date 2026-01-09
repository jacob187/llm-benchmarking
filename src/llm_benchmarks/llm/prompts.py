"""Prompt templates for LLM analysis."""

from langchain_core.prompts import PromptTemplate


# System context for all prompts
SYSTEM_CONTEXT = """You are an expert AI analyst specializing in LLM benchmarks and performance evaluation.
Your role is to analyze benchmark data and provide clear, actionable insights."""


# Executive summary of all benchmarks
SUMMARY_TEMPLATE = PromptTemplate.from_template(
    """Analyze the following LLM benchmark data and provide a concise executive summary.

Benchmark Data:
{benchmark_data}

Provide a summary that includes:
1. Overall trends and patterns in the data
2. Top performing models and their strengths
3. Notable observations or surprises
4. Key takeaways for developers choosing an LLM

Keep the summary under 300 words and focus on actionable insights.
"""
)


# Model comparison analysis
COMPARISON_TEMPLATE = PromptTemplate.from_template(
    """Compare the following LLM models based on their benchmark performance.

Models to compare: {model_names}

Benchmark Data:
{benchmark_data}

Provide a detailed comparison that includes:
1. Direct performance comparison across available benchmarks
2. Strengths and weaknesses of each model
3. Use case recommendations for each model
4. Value proposition (performance vs. size/cost if known)

Keep the analysis under 400 words and be specific about performance differences.
"""
)


# Coding-focused analysis
CODING_TEMPLATE = PromptTemplate.from_template(
    """Analyze the following LLM benchmark data with a focus on coding capabilities.

Benchmark Data:
{benchmark_data}

Provide an analysis that includes:
1. Top models for code generation and understanding
2. Performance on coding-specific benchmarks (HumanEval, MBPP, etc.)
3. Recommendations for different coding tasks (code completion, debugging, explanation)
4. Trends in coding performance across model families

Keep the analysis under 300 words and focus on practical coding applications.
"""
)


# Blog content analysis
BLOG_ANALYSIS_TEMPLATE = PromptTemplate.from_template(
    """Analyze the following blog post content about LLMs and extract key insights.

Blog Content:
{blog_content}

Provide a summary that includes:
1. Main topics and themes discussed
2. Important developments or announcements mentioned
3. Expert opinions or predictions
4. Relevant insights for understanding current LLM trends

Keep the summary under 250 words and focus on the most important information.
"""
)


# Research paper analysis
RESEARCH_TEMPLATE = PromptTemplate.from_template(
    """Analyze the following research paper data about LLMs.

Research Data:
{research_data}

Provide a summary that includes:
1. Key findings and contributions
2. Methodological approaches
3. Implications for LLM development and usage
4. Connection to current benchmark trends

Keep the summary under 300 words and highlight practical implications.
"""
)


# Interactive question answering
QA_TEMPLATE = PromptTemplate.from_template(
    """You are an AI assistant helping users understand LLM benchmark data.

Available Benchmark Data:
{benchmark_data}

User Question: {question}

Provide a clear, accurate answer based on the benchmark data. If the data doesn't contain
enough information to answer the question fully, acknowledge this and provide what information is available.

Keep your answer concise and relevant to the user's question.
"""
)


def get_template(name: str) -> PromptTemplate:
    """
    Get a prompt template by name.

    Args:
        name: Template name (summary, comparison, coding, blog, research, qa)

    Returns:
        PromptTemplate

    Raises:
        ValueError: If template name is not recognized
    """
    templates = {
        "summary": SUMMARY_TEMPLATE,
        "comparison": COMPARISON_TEMPLATE,
        "coding": CODING_TEMPLATE,
        "blog": BLOG_ANALYSIS_TEMPLATE,
        "research": RESEARCH_TEMPLATE,
        "qa": QA_TEMPLATE,
    }

    if name not in templates:
        raise ValueError(
            f"Unknown template: {name}. "
            f"Available templates: {', '.join(templates.keys())}"
        )

    return templates[name]
