"""Model Router - Intelligent routing of tasks to appropriate models.

Routes extraction tasks to the best model based on:
- Task complexity
- Cost constraints
- Speed requirements
- Model capabilities
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from langchain_core.language_models import BaseChatModel

from data_extraction_agent.providers.factory import (
    Provider,
    ProviderFactory,
    PROVIDER_MODELS,
)


class TaskComplexity(str, Enum):
    """Task complexity levels for routing."""

    SIMPLE = "simple"  # Single field extraction, basic parsing
    MODERATE = "moderate"  # Multi-field extraction, API response handling
    COMPLEX = "complex"  # Schema inference, document parsing
    EXPERT = "expert"  # Complex reasoning, multi-source integration


class RoutingStrategy(str, Enum):
    """Model selection strategies."""

    COST_OPTIMIZED = "cost_optimized"  # Minimize cost, acceptable quality
    QUALITY_FIRST = "quality_first"  # Best quality, ignore cost
    BALANCED = "balanced"  # Balance quality and cost
    SPEED_FIRST = "speed_first"  # Fastest response time
    OPEN_SOURCE_ONLY = "open_source_only"  # Only use open-source models


@dataclass
class TaskProfile:
    """Profile describing an extraction task for routing."""

    description: str
    complexity: TaskComplexity = TaskComplexity.MODERATE
    requires_json_output: bool = True
    requires_tool_calling: bool = False
    max_input_tokens: int = 4000
    max_output_tokens: int = 2000
    tags: list[str] = field(default_factory=list)


# Provider rankings by strategy
STRATEGY_RANKINGS = {
    RoutingStrategy.COST_OPTIMIZED: [
        (Provider.GROQ, "fast"),  # Free tier
        (Provider.OPENROUTER, "fast"),  # Free models
        (Provider.HUGGINGFACE, "fast"),  # Free tier
        (Provider.TOGETHER, "fast"),  # $0.05/M tokens
        (Provider.GOOGLE, "fast"),  # Low cost
        (Provider.ANTHROPIC, "fast"),  # Haiku is cheap
    ],
    RoutingStrategy.QUALITY_FIRST: [
        (Provider.ANTHROPIC, "reasoning"),  # Best quality
        (Provider.GOOGLE, "reasoning"),  # Gemini Pro
        (Provider.TOGETHER, "reasoning"),  # DeepSeek R1
        (Provider.GROQ, "reasoning"),  # QwQ-32B
        (Provider.OPENROUTER, "reasoning"),
    ],
    RoutingStrategy.BALANCED: [
        (Provider.GROQ, "balanced"),  # Free + good quality
        (Provider.TOGETHER, "balanced"),  # Good price/perf
        (Provider.GOOGLE, "balanced"),
        (Provider.OPENROUTER, "balanced"),
        (Provider.ANTHROPIC, "balanced"),
    ],
    RoutingStrategy.SPEED_FIRST: [
        (Provider.GROQ, "fast"),  # 750 tokens/sec
        (Provider.GOOGLE, "fast"),  # Very fast
        (Provider.TOGETHER, "fast"),
        (Provider.ANTHROPIC, "fast"),  # Haiku
    ],
    RoutingStrategy.OPEN_SOURCE_ONLY: [
        (Provider.GROQ, "balanced"),  # Llama models
        (Provider.TOGETHER, "balanced"),  # Open models
        (Provider.HUGGINGFACE, "balanced"),
        (Provider.OLLAMA, "balanced"),  # Local
    ],
}

# Complexity to model type mapping
COMPLEXITY_MODEL_TYPES = {
    TaskComplexity.SIMPLE: "fast",
    TaskComplexity.MODERATE: "balanced",
    TaskComplexity.COMPLEX: "extraction",
    TaskComplexity.EXPERT: "reasoning",
}


class ModelRouter:
    """Routes tasks to the most appropriate LLM based on requirements."""

    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.COST_OPTIMIZED,
        preferred_providers: Optional[list[Provider]] = None,
        excluded_providers: Optional[list[Provider]] = None,
    ):
        """Initialize the router.

        Args:
            strategy: The routing strategy to use
            preferred_providers: Prioritize these providers if available
            excluded_providers: Never use these providers
        """
        self.strategy = strategy
        self.preferred_providers = preferred_providers or []
        self.excluded_providers = excluded_providers or []

    def get_model_for_task(
        self,
        task: TaskProfile,
        **model_kwargs,
    ) -> BaseChatModel:
        """Get the best model for a given task.

        Args:
            task: Task profile describing the extraction requirements
            **model_kwargs: Additional arguments passed to the model

        Returns:
            Configured LangChain chat model

        Raises:
            ValueError: If no suitable provider is available
        """
        # Get model type based on complexity
        model_type = COMPLEXITY_MODEL_TYPES.get(task.complexity, "balanced")

        # Get ranked providers for strategy
        rankings = STRATEGY_RANKINGS.get(self.strategy, STRATEGY_RANKINGS[RoutingStrategy.BALANCED])

        # Filter and reorder based on preferences
        candidates = []

        # Add preferred providers first
        for provider in self.preferred_providers:
            if provider not in self.excluded_providers and ProviderFactory.is_available(provider):
                candidates.append((provider, model_type))

        # Add strategy-ranked providers
        for provider, _ in rankings:
            if provider not in self.excluded_providers and provider not in self.preferred_providers:
                if ProviderFactory.is_available(provider):
                    candidates.append((provider, model_type))

        if not candidates:
            raise ValueError("No suitable LLM provider available. Configure at least one API key.")

        # Try first available provider
        provider, m_type = candidates[0]
        return ProviderFactory.create(provider, model_type=m_type, **model_kwargs)

    def get_model_chain(
        self,
        task: TaskProfile,
        max_providers: int = 3,
    ) -> list[tuple[Provider, str]]:
        """Get an ordered list of (provider, model_type) for fallback chain.

        Args:
            task: Task profile
            max_providers: Maximum providers in chain

        Returns:
            List of (provider, model_type) tuples
        """
        model_type = COMPLEXITY_MODEL_TYPES.get(task.complexity, "balanced")
        rankings = STRATEGY_RANKINGS.get(self.strategy, STRATEGY_RANKINGS[RoutingStrategy.BALANCED])

        chain = []
        for provider, _ in rankings:
            if provider not in self.excluded_providers:
                if ProviderFactory.is_available(provider):
                    chain.append((provider, model_type))
                    if len(chain) >= max_providers:
                        break

        return chain

    @staticmethod
    def estimate_complexity(
        description: str,
        input_size: int = 0,
        num_fields: int = 0,
    ) -> TaskComplexity:
        """Estimate task complexity from description and parameters.

        Args:
            description: Task description
            input_size: Estimated input size in characters
            num_fields: Number of fields to extract

        Returns:
            Estimated TaskComplexity
        """
        description_lower = description.lower()

        # Expert complexity indicators
        expert_keywords = ["analyze", "reason", "infer", "complex", "multiple sources", "integrate"]
        if any(kw in description_lower for kw in expert_keywords):
            return TaskComplexity.EXPERT

        # Complex indicators
        complex_keywords = ["schema", "document", "pdf", "parse", "structure", "transform"]
        if any(kw in description_lower for kw in complex_keywords) or input_size > 50000:
            return TaskComplexity.COMPLEX

        # Moderate indicators
        moderate_keywords = ["api", "database", "extract", "multiple", "fields"]
        if any(kw in description_lower for kw in moderate_keywords) or num_fields > 5:
            return TaskComplexity.MODERATE

        # Default to simple
        return TaskComplexity.SIMPLE
