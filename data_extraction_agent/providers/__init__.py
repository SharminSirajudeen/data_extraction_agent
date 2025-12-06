"""Multi-Provider LLM Support.

This module provides a provider-agnostic LLM abstraction supporting:
- Groq (Free tier, ultra-fast Llama/Mixtral)
- Together AI (Budget, large models)
- OpenRouter (Unified API, free models)
- HuggingFace Inference API
- Google Gemini
- Anthropic Claude (Complex reasoning fallback)
"""

from data_extraction_agent.providers.factory import (
    ProviderFactory,
    get_provider,
    get_model,
    list_available_providers,
)
from data_extraction_agent.providers.router import (
    ModelRouter,
    TaskProfile,
    TaskComplexity,
    RoutingStrategy,
)
from data_extraction_agent.providers.fallback import FallbackChain

__all__ = [
    "ProviderFactory",
    "get_provider",
    "get_model",
    "list_available_providers",
    "ModelRouter",
    "TaskProfile",
    "TaskComplexity",
    "RoutingStrategy",
    "FallbackChain",
]
