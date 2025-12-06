"""Base provider classes and registry pattern.

This module provides a clean, extensible architecture for LLM providers.
Adding a new provider requires NO changes to existing code - just add a config.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from langchain_core.language_models import BaseChatModel


class ProviderTier(str, Enum):
    """Provider pricing tiers."""
    FREE = "free"
    BUDGET = "budget"
    STANDARD = "standard"
    PREMIUM = "premium"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    model_id: str
    context_length: int = 8192
    supports_tools: bool = True
    supports_vision: bool = False
    cost_per_m_input: float = 0.0
    cost_per_m_output: float = 0.0


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider.

    This dataclass contains everything needed to create an LLM client.
    No code changes required to add new providers - just add config.
    """
    name: str
    env_key: Optional[str]  # None for local providers like Ollama
    base_url: Optional[str]  # For OpenAI-compatible APIs
    tier: ProviderTier = ProviderTier.BUDGET
    speed_tokens_per_sec: int = 100

    # Model configurations by type
    models: dict[str, str] = field(default_factory=dict)

    # Client factory - how to create the LangChain client
    # Format: "langchain_package:ClassName" or "openai_compatible"
    client_type: str = "openai_compatible"

    # Additional client kwargs
    client_kwargs: dict[str, Any] = field(default_factory=dict)

    def is_available(self) -> bool:
        """Check if this provider is configured (has API key if needed)."""
        if self.env_key is None:
            return True
        return bool(os.getenv(self.env_key))

    def get_api_key(self) -> Optional[str]:
        """Get the API key from environment."""
        if self.env_key is None:
            return None
        return os.getenv(self.env_key)

    def get_model_id(self, model_type: str = "default") -> str:
        """Get model ID for a given type."""
        return self.models.get(model_type, self.models.get("default", ""))


class ProviderRegistry:
    """Registry for LLM providers.

    Implements the Registry pattern - providers register themselves,
    and the factory uses the registry to create clients dynamically.
    """

    _providers: dict[str, ProviderConfig] = {}
    _client_factories: dict[str, Callable[..., BaseChatModel]] = {}

    @classmethod
    def register(cls, config: ProviderConfig) -> None:
        """Register a provider configuration."""
        cls._providers[config.name] = config

    @classmethod
    def register_client_factory(
        cls,
        client_type: str,
        factory: Callable[..., BaseChatModel]
    ) -> None:
        """Register a client factory function."""
        cls._client_factories[client_type] = factory

    @classmethod
    def get(cls, name: str) -> Optional[ProviderConfig]:
        """Get a provider configuration by name."""
        return cls._providers.get(name.lower())

    @classmethod
    def get_all(cls) -> dict[str, ProviderConfig]:
        """Get all registered providers."""
        return cls._providers.copy()

    @classmethod
    def get_available(cls) -> list[ProviderConfig]:
        """Get all providers that are configured and available."""
        return [p for p in cls._providers.values() if p.is_available()]

    @classmethod
    def get_by_tier(cls, tier: ProviderTier) -> list[ProviderConfig]:
        """Get providers by pricing tier."""
        return [p for p in cls.get_available() if p.tier == tier]

    @classmethod
    def get_fastest(cls) -> Optional[ProviderConfig]:
        """Get the fastest available provider."""
        available = cls.get_available()
        if not available:
            return None
        return max(available, key=lambda p: p.speed_tokens_per_sec)

    @classmethod
    def get_cheapest(cls) -> Optional[ProviderConfig]:
        """Get the cheapest available provider."""
        available = cls.get_available()
        if not available:
            return None
        # Prefer free tier, then sort by input cost
        free = [p for p in available if p.tier == ProviderTier.FREE]
        if free:
            return free[0]
        return min(available, key=lambda p: p.models.get("default", {}) if isinstance(p.models.get("default"), dict) else 0)

    @classmethod
    def create_client(
        cls,
        provider_name: str,
        model_type: str = "default",
        model_id: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Create an LLM client for the specified provider.

        This is the main factory method - it uses registered configs
        and factories to create clients without any hardcoded logic.
        """
        config = cls.get(provider_name)
        if config is None:
            available = list(cls._providers.keys())
            raise ValueError(f"Unknown provider: {provider_name}. Available: {available}")

        if not config.is_available():
            raise ValueError(
                f"Provider {provider_name} not configured. "
                f"Set environment variable: {config.env_key}"
            )

        # Get the model ID
        final_model_id = model_id or config.get_model_id(model_type)

        # Get the client factory
        factory = cls._client_factories.get(config.client_type)
        if factory is None:
            raise ValueError(f"Unknown client type: {config.client_type}")

        # Merge kwargs
        final_kwargs = {**config.client_kwargs, **kwargs}

        # Create and return the client
        return factory(
            config=config,
            model_id=final_model_id,
            **final_kwargs,
        )


# ============================================================================
# CLIENT FACTORIES
# ============================================================================

def _create_openai_compatible_client(
    config: ProviderConfig,
    model_id: str,
    **kwargs: Any,
) -> BaseChatModel:
    """Factory for OpenAI-compatible APIs (Cerebras, SambaNova, Fireworks, etc.)."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_id,
        openai_api_base=config.base_url,
        openai_api_key=config.get_api_key(),
        **kwargs,
    )


def _create_groq_client(
    config: ProviderConfig,
    model_id: str,
    **kwargs: Any,
) -> BaseChatModel:
    """Factory for Groq."""
    from langchain_groq import ChatGroq
    return ChatGroq(model=model_id, **kwargs)


def _create_together_client(
    config: ProviderConfig,
    model_id: str,
    **kwargs: Any,
) -> BaseChatModel:
    """Factory for Together AI."""
    from langchain_together import ChatTogether
    return ChatTogether(model=model_id, **kwargs)


def _create_anthropic_client(
    config: ProviderConfig,
    model_id: str,
    **kwargs: Any,
) -> BaseChatModel:
    """Factory for Anthropic Claude."""
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model_id, **kwargs)


def _create_google_client(
    config: ProviderConfig,
    model_id: str,
    **kwargs: Any,
) -> BaseChatModel:
    """Factory for Google Gemini."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model_id, **kwargs)


def _create_ollama_client(
    config: ProviderConfig,
    model_id: str,
    **kwargs: Any,
) -> BaseChatModel:
    """Factory for Ollama (local)."""
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model_id, **kwargs)


def _create_huggingface_client(
    config: ProviderConfig,
    model_id: str,
    **kwargs: Any,
) -> BaseChatModel:
    """Factory for HuggingFace."""
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    llm = HuggingFaceEndpoint(repo_id=model_id, task="text-generation", **kwargs)
    return ChatHuggingFace(llm=llm)


# Register all client factories
ProviderRegistry.register_client_factory("openai_compatible", _create_openai_compatible_client)
ProviderRegistry.register_client_factory("groq", _create_groq_client)
ProviderRegistry.register_client_factory("together", _create_together_client)
ProviderRegistry.register_client_factory("anthropic", _create_anthropic_client)
ProviderRegistry.register_client_factory("google", _create_google_client)
ProviderRegistry.register_client_factory("ollama", _create_ollama_client)
ProviderRegistry.register_client_factory("huggingface", _create_huggingface_client)
