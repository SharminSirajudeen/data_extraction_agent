"""Data Extraction Agent - Main Agent Configuration.

This module creates and configures the data extraction agent using
the Deep Agents framework with custom extraction tools and multi-provider
LLM support (Groq, Together, OpenRouter, HuggingFace, Google, Anthropic).
"""

import os
from typing import Optional

from deepagents import create_deep_agent
from langchain_core.language_models import BaseChatModel

from data_extraction_agent.prompts import get_specialist_prompt, get_system_prompt
from data_extraction_agent.tools import extraction_tools
from data_extraction_agent.providers import (
    ProviderFactory,
    Provider,
    ModelRouter,
    TaskProfile,
    TaskComplexity,
    RoutingStrategy,
    FallbackChain,
    get_model,
    list_available_providers,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Agent limits
MAX_CONCURRENT_EXTRACTORS = 3
MAX_EXTRACTION_ITERATIONS = 3

# Default provider priority: Open-source first!
DEFAULT_PROVIDER_PRIORITY = [
    Provider.GROQ,  # Free tier, ultra-fast
    Provider.TOGETHER,  # Budget, open models
    Provider.OPENROUTER,  # Unified API, free models
    Provider.HUGGINGFACE,  # Free experimentation
    Provider.GOOGLE,  # Gemini
    Provider.ANTHROPIC,  # Commercial fallback
]


def get_best_available_model(
    model_type: str = "extraction",
    strategy: RoutingStrategy = RoutingStrategy.OPEN_SOURCE_ONLY,
) -> BaseChatModel:
    """Get the best available LLM based on configured API keys.

    Prioritizes open-source models (Groq, Together, etc.) over commercial ones.

    Args:
        model_type: Type of model (fast, balanced, extraction, reasoning)
        strategy: Routing strategy (default: open-source only)

    Returns:
        Configured LangChain chat model

    Raises:
        ValueError: If no providers are configured
    """
    # Try providers in priority order
    for provider in DEFAULT_PROVIDER_PRIORITY:
        if ProviderFactory.is_available(provider):
            try:
                return ProviderFactory.create(provider, model_type=model_type)
            except Exception:
                continue

    raise ValueError(
        "No LLM providers configured. Set at least one API key:\n"
        "- GROQ_API_KEY (recommended, free)\n"
        "- TOGETHER_API_KEY\n"
        "- OPENROUTER_API_KEY\n"
        "- GOOGLE_API_KEY\n"
        "- ANTHROPIC_API_KEY"
    )


# ============================================================================
# SUB-AGENT DEFINITIONS
# ============================================================================


def create_extraction_specialist(model: Optional[BaseChatModel] = None) -> dict:
    """Create the extraction specialist sub-agent configuration.

    Args:
        model: Optional model to use (defaults to best available)

    Returns:
        Sub-agent configuration dictionary
    """
    return {
        "name": "extraction-specialist",
        "description": """A specialized sub-agent for extracting data from specific sources.
        Delegate to this agent when you need to:
        - Extract data from a specific API or web source
        - Query a database and transform results
        - Process files (CSV, Excel, JSON, PDF)
        - Handle complex extraction that benefits from isolated context

        The specialist will return clean, structured data ready for integration.""",
        "instructions": get_specialist_prompt(),
        "tools": extraction_tools,
        "model": model,  # Uses same model as parent if None
    }


# ============================================================================
# AGENT CREATION
# ============================================================================


def create_data_extraction_agent(
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
    model_type: str = "extraction",
    strategy: RoutingStrategy = RoutingStrategy.OPEN_SOURCE_ONLY,
    max_concurrent: int = MAX_CONCURRENT_EXTRACTORS,
    max_iterations: int = MAX_EXTRACTION_ITERATIONS,
    use_fallback: bool = True,
):
    """Create a configured data extraction agent.

    Args:
        provider: Specific provider to use (groq, together, openrouter, etc.)
                 If None, uses best available based on strategy
        model_id: Specific model ID (overrides model_type)
        model_type: Type of model (fast, balanced, extraction, reasoning)
        strategy: Routing strategy (default: open-source only)
        max_concurrent: Maximum parallel sub-agents
        max_iterations: Maximum delegation rounds
        use_fallback: Whether to use fallback chain for resilience

    Returns:
        Configured Deep Agent graph
    """
    # Get the model
    if provider:
        llm = get_model(provider=provider, model_type=model_type, model_id=model_id)
    else:
        llm = get_best_available_model(model_type=model_type, strategy=strategy)

    # Generate system prompt with configuration
    system_prompt = get_system_prompt(
        max_concurrent_extractors=max_concurrent,
        max_extraction_iterations=max_iterations,
    )

    # Create extraction specialist
    specialist = create_extraction_specialist(model=llm)

    # Create the agent with Deep Agents framework
    agent = create_deep_agent(
        model=llm,
        tools=extraction_tools,
        system_prompt=system_prompt,
        subagents=[specialist],
    )

    return agent


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================


def create_groq_agent():
    """Create agent using Groq (free tier, ultra-fast).

    Best for: Quick extractions, cost-sensitive workloads
    """
    return create_data_extraction_agent(provider="groq", model_type="extraction")


def create_together_agent():
    """Create agent using Together AI (budget, open models).

    Best for: Balanced cost/quality, access to large open models
    """
    return create_data_extraction_agent(provider="together", model_type="extraction")


def create_openrouter_agent():
    """Create agent using OpenRouter (unified API).

    Best for: Model flexibility, access to many providers
    """
    return create_data_extraction_agent(provider="openrouter", model_type="extraction")


def create_lightweight_agent():
    """Create a lightweight agent for simple extraction tasks.

    Uses the fastest available free model.
    """
    return create_data_extraction_agent(
        model_type="fast",
        max_concurrent=1,
        max_iterations=2,
    )


def create_quality_agent():
    """Create a quality-focused agent for complex extraction.

    Uses the best available model regardless of cost.
    """
    return create_data_extraction_agent(
        strategy=RoutingStrategy.QUALITY_FIRST,
        model_type="reasoning",
        max_concurrent=5,
        max_iterations=5,
    )


# ============================================================================
# DEFAULT AGENT FOR LANGGRAPH SERVER
# ============================================================================

# Check available providers and create default agent
def _create_default_agent():
    """Create the default agent for LangGraph server deployment."""
    available = list_available_providers()

    if not available:
        # Return a placeholder that will error with helpful message
        raise ValueError(
            "No LLM providers configured!\n\n"
            "Set at least one of these environment variables:\n"
            "- GROQ_API_KEY (free at console.groq.com)\n"
            "- TOGETHER_API_KEY (together.ai)\n"
            "- OPENROUTER_API_KEY (openrouter.ai)\n"
            "- GOOGLE_API_KEY (aistudio.google.com)\n"
            "- ANTHROPIC_API_KEY (console.anthropic.com)"
        )

    print(f"Available providers: {list(available.keys())}")
    return create_data_extraction_agent()


# Lazy initialization for LangGraph server
agent = None


def get_agent():
    """Get or create the default agent (lazy initialization)."""
    global agent
    if agent is None:
        agent = _create_default_agent()
    return agent


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    import asyncio

    # Show available providers
    print("Checking available LLM providers...")
    available = list_available_providers()
    print(f"Available: {available}\n")

    if not available:
        print("No providers configured. Set API keys in .env file.")
        exit(1)

    async def main():
        # Create agent
        agent = create_data_extraction_agent()
        print(f"Agent created successfully!")

        # Example extraction request
        request = """
        Extract all product information from this API:
        https://fakestoreapi.com/products

        I need:
        - Product ID
        - Title
        - Price
        - Category
        - Rating

        Return as a clean JSON file.
        """

        print(f"\nRunning extraction: {request[:100]}...")

        # Stream the agent execution
        async for event in agent.astream({"messages": [{"role": "user", "content": request}]}):
            if "messages" in event:
                for msg in event["messages"]:
                    content = msg.content if hasattr(msg, "content") else str(msg)
                    print(f"[{type(msg).__name__}]: {content[:200]}...")

    asyncio.run(main())
