"""Provider Factory - Clean API for getting LLM clients.

This module provides a simple interface to the provider registry.
All provider configurations are in config.py - this file contains
NO hardcoded provider logic.
"""

from typing import Any, Optional

from langchain_core.language_models import BaseChatModel

# Import config to register all providers
from data_extraction_agent.providers import config as _config  # noqa: F401
from data_extraction_agent.providers.base import (
    ProviderConfig,
    ProviderRegistry,
    ProviderTier,
)


def get_model(
    provider: str = "groq",
    model_type: str = "default",
    model_id: Optional[str] = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Get an LLM client for the specified provider.

    Args:
        provider: Provider name (groq, cerebras, together, etc.)
        model_type: Model type (default, fast, balanced, reasoning, extraction)
        model_id: Specific model ID (overrides model_type)
        **kwargs: Additional arguments passed to the client

    Returns:
        Configured LangChain chat model

    Example:
        >>> model = get_model("groq", model_type="fast")
        >>> model = get_model("cerebras")  # Uses default model
        >>> model = get_model("together", model_id="meta-llama/Llama-3.3-70B")
    """
    return ProviderRegistry.create_client(
        provider_name=provider,
        model_type=model_type,
        model_id=model_id,
        **kwargs,
    )


def list_providers() -> dict[str, dict]:
    """List all registered providers with their status.

    Returns:
        Dict mapping provider name to info dict with:
        - available: bool
        - tier: str
        - speed: int (tokens/sec)
        - models: list of model types
    """
    result = {}
    for name, config in ProviderRegistry.get_all().items():
        result[name] = {
            "available": config.is_available(),
            "tier": config.tier.value,
            "speed_tokens_per_sec": config.speed_tokens_per_sec,
            "models": list(config.models.keys()),
            "env_key": config.env_key,
        }
    return result


def list_available_providers() -> dict[str, list[str]]:
    """List available providers and their models.

    Returns:
        Dict mapping provider name to list of model IDs
    """
    result = {}
    for config in ProviderRegistry.get_available():
        result[config.name] = list(config.models.values())
    return result


def get_provider_config(name: str) -> Optional[ProviderConfig]:
    """Get the configuration for a provider."""
    return ProviderRegistry.get(name)


def get_fastest_provider() -> Optional[str]:
    """Get the name of the fastest available provider."""
    config = ProviderRegistry.get_fastest()
    return config.name if config else None


def get_cheapest_provider() -> Optional[str]:
    """Get the name of the cheapest available provider."""
    config = ProviderRegistry.get_cheapest()
    return config.name if config else None


def get_free_providers() -> list[str]:
    """Get names of all providers with free tiers."""
    return [c.name for c in ProviderRegistry.get_by_tier(ProviderTier.FREE)]


def get_best_available_model(
    model_type: str = "default",
    prefer_free: bool = True,
    prefer_speed: bool = False,
    **kwargs: Any,
) -> BaseChatModel:
    """Get the best available LLM based on preferences.

    Args:
        model_type: Type of model (default, fast, balanced, reasoning, extraction)
        prefer_free: Prefer free tier providers (default True)
        prefer_speed: Prefer faster providers over cheaper ones
        **kwargs: Additional arguments passed to the client

    Returns:
        Configured LangChain chat model

    Raises:
        ValueError: If no providers are available
    """
    available = ProviderRegistry.get_available()

    if not available:
        all_providers = ProviderRegistry.get_all()
        env_keys = [p.env_key for p in all_providers.values() if p.env_key]
        raise ValueError(
            "No LLM providers configured. Set at least one API key:\n"
            + "\n".join(f"- {key}" for key in env_keys[:5])
        )

    # Sort by preference
    if prefer_free:
        # Free tier first, then by speed
        available.sort(
            key=lambda p: (
                0 if p.tier == ProviderTier.FREE else 1,
                -p.speed_tokens_per_sec if prefer_speed else 0,
            )
        )
    elif prefer_speed:
        available.sort(key=lambda p: -p.speed_tokens_per_sec)

    # Try first available
    config = available[0]
    return ProviderRegistry.create_client(
        provider_name=config.name,
        model_type=model_type,
        **kwargs,
    )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_groq_model(model_type: str = "default", **kwargs) -> BaseChatModel:
    """Create a Groq model (free tier, fast)."""
    return get_model("groq", model_type=model_type, **kwargs)


def create_cerebras_model(model_type: str = "default", **kwargs) -> BaseChatModel:
    """Create a Cerebras model (free tier, FASTEST at 1800 tok/s)."""
    return get_model("cerebras", model_type=model_type, **kwargs)


def create_together_model(model_type: str = "default", **kwargs) -> BaseChatModel:
    """Create a Together AI model (budget, good model variety)."""
    return get_model("together", model_type=model_type, **kwargs)


def create_ollama_model(model_type: str = "default", **kwargs) -> BaseChatModel:
    """Create an Ollama model (local, free)."""
    return get_model("ollama", model_type=model_type, **kwargs)
