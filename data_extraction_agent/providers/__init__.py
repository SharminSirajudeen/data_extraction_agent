"""Multi-Provider LLM Support - Registry Pattern.

Clean, extensible architecture for LLM providers.
Adding a new provider requires ONLY editing config.py - no other code changes.

Usage:
    from data_extraction_agent.providers import get_model, list_providers

    # Get a model
    model = get_model("groq", model_type="fast")

    # Get best available (prefers free, then fast)
    model = get_best_available_model()

    # List all providers
    providers = list_providers()
"""

from data_extraction_agent.providers.base import (
    ProviderConfig,
    ProviderRegistry,
    ProviderTier,
)
from data_extraction_agent.providers.factory import (
    get_model,
    get_best_available_model,
    get_fastest_provider,
    get_cheapest_provider,
    get_free_providers,
    get_provider_config,
    list_providers,
    list_available_providers,
    create_groq_model,
    create_cerebras_model,
    create_together_model,
    create_ollama_model,
)
from data_extraction_agent.providers.fallback import FallbackChain
from data_extraction_agent.providers.router import (
    ModelRouter,
    TaskProfile,
    TaskComplexity,
    RoutingStrategy,
)

__all__ = [
    # Core
    "ProviderConfig",
    "ProviderRegistry",
    "ProviderTier",
    # Factory functions
    "get_model",
    "get_best_available_model",
    "get_fastest_provider",
    "get_cheapest_provider",
    "get_free_providers",
    "get_provider_config",
    "list_providers",
    "list_available_providers",
    # Convenience
    "create_groq_model",
    "create_cerebras_model",
    "create_together_model",
    "create_ollama_model",
    # Routing
    "ModelRouter",
    "TaskProfile",
    "TaskComplexity",
    "RoutingStrategy",
    "FallbackChain",
]
